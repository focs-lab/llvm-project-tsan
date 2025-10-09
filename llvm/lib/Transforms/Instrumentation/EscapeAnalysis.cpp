//===- EscapeAnalysis.cpp - Intraprocedural Escape Analysis Implementation ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the EscapeAnalysis helper class. It uses a worklist-
// based, backward dataflow analysis to determine if an allocation can escape.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation/EscapeAnalysis.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/TargetParser/ARMTargetParser.h"

#include <deque>

#define DEBUG_TYPE "escape-analysis"

using namespace llvm;

STATISTIC(NumAllocationsAnalyzed, "Number of allocation sites analyzed");
STATISTIC(NumAllocationsEscaped, "Number of allocation sites found to escape");

/// Per-allocation worklist cap (safety valve). If the number of processed
/// worklist nodes exceeds this limit, the analysis bails out conservatively and
/// considers the allocation as escaping.
static cl::opt<unsigned>
WorklistLimit("escape-analysis-worklist-limit", cl::init(10000), cl::Hidden,
              cl::desc("Max number of worklist nodes processed per allocation; "
                       "if exceeded, assume the allocation escapes"));

//===----------------------------------------------------------------------===//
// getUnderlyingObjectsThroughLoads Implementation
//===----------------------------------------------------------------------===//

template <typename PtrT, typename SetT, typename WorklistT>
static bool tryEnqueueIfNew(PtrT *P, SetT &Seen, WorklistT &WL) {
  if (P && Seen.insert(P).second) {
    WL.push_back(P);
    return true;
  }
  return false;
}

static bool tryValueTracking(const Value *Curr, LoopInfo *LI,
                             SmallVectorImpl<const Value *> &Work,
                             SmallPtrSetImpl<const Value *> &Enqueued) {
  SmallVector<const Value *, 4> Bases;
  if (!Curr->getType()->isPointerTy())
    return false; // Only pointers have underlying objects.

  // getUnderlyingObjects(..., MaxLookup = 0) is assumed to mean "unbounded".
  // If upstream changes semantics, this must be revisited.
  getUnderlyingObjects(Curr, Bases, LI, /*MaxLookup=*/ 0);

  if (Bases.empty() || (Bases.size() == 1 && Bases[0] == Curr))
    return false;

  for (const Value *B : Bases)
    tryEnqueueIfNew(B, Enqueued, Work);
  return true;
}

// Add incoming unvisited MemoryAccesses of a MemoryPhi to MAWorkList.
static void
appendUnvisitedIncomingMAs(const MemoryPhi *MP,
                           SmallPtrSetImpl<MemoryAccess *> &VisitedMA,
                           SmallVectorImpl<MemoryAccess *> &MAWorkList) {
  for (unsigned i = 0, N = MP->getNumIncomingValues(); i != N; ++i) {
    MemoryAccess *InMA = MP->getIncomingValue(i);
    tryEnqueueIfNew(InMA, VisitedMA, MAWorkList);
  }
}

///===- GetUnderlyingObjectsThroughLoads ---------------------------------===//
///
/// A stronger variant of `llvm::getUnderlyingObjects` that uses MemorySSA
/// to chase defining writes and, when possible, look through loads. This is
/// more precise (and potentially more expensive) than plain ValueTracking.
///
/// \param V        A pointer-typed value to analyze.
/// \param MSSA     A valid, up-to-date MemorySSA for the parent function of V.
///                 Must not be null.
/// \param Result   Output set that will be populated with the results. The set
///                 is not cleared; new elements are inserted into it.
/// \param LI       Optional LoopInfo used to improve reasoning about PHIs in
///                 loops in ValueTracking.
///
/// \post
///  - `Result` is augmented with zero or more pointer-typed "terminal sources"
///    for `V`.
///  - A "terminal source" is a pointer value where this analysis intentionally
///    stops. Typical terminals include:
///      * `AllocaInst`, `GlobalVariable`, `Argument`, `ConstantPointerNull`.
///      * An SSA pointer value such as a `LoadInst` (or `PHINode`/`Select`
///        as returned by ValueTracking) when MemorySSA cannot prove a precise
///        defining write for the loaded bytes (e.g., memory is liveOnEntry,
///        written by an opaque call, clobbered by memset/atomics/volatile,
///        etc.).
///      * A call result if ValueTracking stops at a call that returns a pointer
///        (i.e., the usual terminals that `getUnderlyingObjects` would
///        produce).
///
/// \note
///  - If the analysis cannot find any sound terminal sources (e.g., due to a
///    cycle or lack of proof), it is permitted to insert nothing. However, to
///    match the spirit of `getUnderlyingObjects` and to keep clients
///    predictable, when the walk stops at a pointer-typed SSA value (e.g., a
///    `LoadInst`), this API is encouraged to insert that SSA value to represent
///    an "unknown terminal".
///  - The result is a may-set: `Result` contains all terminal candidates the
///    analysis can conservatively identify, not a single precise source.
///
void getUnderlyingObjectsThroughLoads(const Value *V, MemorySSA *MSSA,
                                      SmallPtrSetImpl<const Value *> &Result,
                                      LoopInfo *LI,
                                      bool *IsComplete = nullptr,
                                      unsigned MaxSteps = 10000) {
  SmallPtrSet<const Value *, 32> VisitedWithVT; // 1st stage (ValueTracking)
  SmallPtrSet<const Value *, 32> Enqueued;      // Guard for enqueue (seen)
  SmallPtrSet<MemoryAccess *, 32> VisitedMA;
  SmallVector<const Value *, 32> Work;

  if (!V->getType()->isPointerTy())
    return; // Only pointers have underlying objects.

  auto markTerminal = [&](const Value *Term,
                          bool ForceIncompleteIfNotBase = true) {
    if (!Term || !Term->getType()->isPointerTy())
      return;

    const bool IsBase = isa<AllocaInst>(Term) || isa<GlobalVariable>(Term) ||
                        isa<GlobalAlias>(Term) || isa<Argument>(Term) ||
                        isa<ConstantPointerNull>(Term);

    Result.insert(Term);
    if (IsComplete && !IsBase && ForceIncompleteIfNotBase)
      *IsComplete = false;
  };

  tryEnqueueIfNew(V, Enqueued, Work);

  assert(MSSA && "MemorySSA is required for this analysis");
  MemorySSAWalker *Walker = MSSA->getWalker();

  unsigned StepCount = 0;
  if (IsComplete)
    *IsComplete = true;

  while (!Work.empty()) {
    const Value *CurrV = Work.pop_back_val();

    // Safety valve: if we exceed MaxSteps, bail out conservatively.
    if (++StepCount > MaxSteps) {
      markTerminal(CurrV);
      goto Bailout;
    }

    // Try ValueTracking first (only once per value)
    if (VisitedWithVT.insert(CurrV).second) {
      if (tryValueTracking(CurrV, LI, Work, Enqueued))
        continue; // Successfully expanded via ValueTracking;
    }

    const auto *Load = dyn_cast<LoadInst>(CurrV);
    if (!Load || Load->isVolatile() || Load->isAtomic()) {
      markTerminal(CurrV);
      continue;
    }

    MemoryAccess *MA = MSSA->getMemoryAccess(Load);
    if (!MA) {
      // If the instruction is not a memory access, we cannot go further.
      markTerminal(Load);
      continue;
    }

    // Use MemorySSA's API to get the clobbering MemoryAccess.
    MemoryAccess *FirstClobber = Walker->getClobberingMemoryAccess(MA);
    if (!FirstClobber) {
      markTerminal(Load);
      continue;
    }

    SmallVector<MemoryAccess *, 32> MAWorkList;
    if (!tryEnqueueIfNew(FirstClobber, VisitedMA, MAWorkList))
      continue;

    // Local accumulators for Load
    SmallPtrSet<const Value *, 8> LocalResult; // Terminals found for Load
    SmallVector<const Value *, 8> LocalWork;
    SmallPtrSet<const Value *, 8> LocalEnqueued;
    LocalEnqueued.insert(Load);
    bool Fallback = false;
    unsigned MAIterations = 0;
    const unsigned MAIterationLimit = MaxSteps / 2;

    while (!MAWorkList.empty()) {
      if (++MAIterations > MAIterationLimit) {
        Fallback = true;
        break;
      }
      MemoryAccess *CurrClobber = MAWorkList.pop_back_val();

      // If the defining access is live-on-entry, we conservatively treat the
      // load itself as a terminal (unknown source).
      if (MSSA->isLiveOnEntryDef(CurrClobber)) {
        LocalResult.insert(Load);
        continue; // We are done
      }

      if (auto *MD = dyn_cast<MemoryDef>(CurrClobber)) {
        // If clobber is a MemoryDef, inspect its instruction.
        const Instruction *ClobberI = MD->getMemoryInst();
        assert(ClobberI && "MemoryDef must have an instruction");

        // Stores: chase the stored pointer when safe; otherwise, conservatively stop.
        if (const auto *Store = dyn_cast<StoreInst>(ClobberI)) {
          // Volatile or atomic stores are opaque.
          if (Store->isVolatile() || Store->isAtomic()) {
            Fallback = true;
            break;
          }
          const Value *SV = Store->getValueOperand();
          if (SV->getType()->isPointerTy()) {
            tryEnqueueIfNew(SV, LocalEnqueued, LocalWork);
            continue;
          }
          // Non-pointer store into memory from which we later load a pointer:
          // treat as unknown/opaque write.
        }
        // NOTE: We intentionally don't consider the source in memintrinsics
        // such as memset/memcpy/memset as underlying objects, because it's
        // wrong semantics.

        // Fallback: unrecognized defining write, stop here conservatively.
        Fallback = true;
        break;
      }

      if (const auto *MP = dyn_cast<MemoryPhi>(CurrClobber)) {
        // Iterate the incoming accesses and process each incoming
        // MemoryAccess.
        appendUnvisitedIncomingMAs(MP, VisitedMA, MAWorkList);
      } else if (auto *MU = dyn_cast<MemoryUse>(CurrClobber)) {
        // If clobber is a MemoryUse (rarely but possible), get its defining.
        MemoryAccess *Def = Walker->getClobberingMemoryAccess(MU);
        if (!Def) {
          Fallback = true;
          break;
        }
        tryEnqueueIfNew(Def, VisitedMA, MAWorkList);
      } else {
#ifndef NDEBUG
        llvm_unreachable("Unexpected MemoryAccess kind");
#else
        // In release builds be conservative and fallback to sound result.
        Fallback = true;
        break;
#endif
      }
    } // end while for MAWorkList

    if (Fallback) {
      markTerminal(Load);
    } else {
      // Merge LocalResult and LocalWork into global sets
      for (const auto *T : LocalResult)
        markTerminal(T);
      for (const auto *WV : LocalWork)
        tryEnqueueIfNew(WV, Enqueued, Work);
    }
  } // end while for Work
  return;

Bailout:
  LLVM_DEBUG(dbgs() << "getUnderlyingObjectsThroughLoads: MaxSteps exceeded\n");
  // Conservative: mark all remaining items as terminals
  for (const Value *WV : Work)
    if (WV && WV->getType()->isPointerTy())
      Result.insert(WV);
}

//===----------------------------------------------------------------------===//
// EscapeCaptureTracker Implementation
//===----------------------------------------------------------------------===//

bool EscapeAnalysisInfo::EscapeCaptureTracker::shouldExplore(const Use *U) {
  // Always explore, but we can add optimizations here later
  return true;
}

bool EscapeAnalysisInfo::EscapeCaptureTracker::doesStoreDestinationEscape(
    const StoreInst *SI) {
  const Value *Dest = SI->getPointerOperand();

  // Find base objects for the storage location using our enhanced analysis
  SmallPtrSet<const Value *, 8> BaseObjects;
  bool IsComplete = false;
  getUnderlyingObjectsThroughLoads(Dest, EAI.MSSA, BaseObjects, EAI.LI,
                                   &IsComplete);

  if (BaseObjects.empty() || !IsComplete) {
    LLVM_DEBUG(dbgs() << "  Store destination unknown, escapes\n");
    return true;
  }

  // Check each base object
  for (const Value *Base : BaseObjects) {
    // Global variable - object escapes
    if (isa<GlobalVariable>(Base) || isa<GlobalAlias>(Base)) {
      LLVM_DEBUG(dbgs() << "  Stored to global variable, escapes\n");
      return true;
    }

    // Function argument - object may escape (unless nocapture)
    if (const auto *Arg = dyn_cast<Argument>(Base)) {
      if (!Arg->hasNoCaptureAttr()) {
        LLVM_DEBUG(dbgs() << "  Stored to non-nocapture argument, escapes\n");
        return true;
      }
      continue;
    }

    // If storing to another local allocation, recursively check if it escapes
    if (const auto *AI = dyn_cast<AllocaInst>(Base)) {
      // Prevent infinite recursion by checking if we're already processing this alloca
      if (ProcessingSet.count(AI)) {
        // Cyclic dependency - conservatively assume escape
        LLVM_DEBUG(dbgs() << "  Cyclic store dependency, escapes\n");
        return true;
      }

      // Check cache first
      if (auto CacheIt = EAI.Cache.find(AI); CacheIt != EAI.Cache.end()) {
        if (CacheIt->second) {
          LLVM_DEBUG(dbgs() << "  Stored to escaping alloca (cached), escapes\n");
          return true;
        }
        continue;
      }

      // Recursively analyze this allocation
      SmallPtrSet<const Value *, 32> RecursiveProcessingSet;
      RecursiveProcessingSet.insert(AI);

      if (EAI.solveEscapeFor(*AI, RecursiveProcessingSet)) {
        LLVM_DEBUG(dbgs() << "  Stored to escaping alloca, escapes\n");
        return true;
      }
      continue;
    }
    // Unknown base object (e.g., LoadInst, Call result)
    // Conservatively assume the object escapes
    LLVM_DEBUG(dbgs() << "  Stored to unknown location: " << *Base
                      << ", escapes\n");
    return true;
  }
  return false;
}

CaptureTracker::Action
EscapeAnalysisInfo::EscapeCaptureTracker::captured(const Use *U,
                                                    UseCaptureInfo CI) {
  Instruction *I = cast<Instruction>(U->getUser());

  // First, use the standard CaptureTracking analysis to filter obvious cases
  // CI contains UseCC (direct capture) and ResultCC (capture through return)

  // Handle return instructions specially for escape analysis
  if (isa<ReturnInst>(I)) {
    // Returning the pointer means it escapes the function
    LLVM_DEBUG(dbgs() << "  Returned from function, escapes\n");
    Escaped = true;
    return Stop;
  }

  // If CaptureTracking says it's not captured at all, we can continue
  if (capturesNothing(CI.UseCC)) {
    // But check if it's passed through (e.g., GEP, bitcast, phi, select)
    if (CI.isPassthrough()) {
      // Continue analyzing the result of this operation
      return Continue;
    }
    // Otherwise, this use doesn't cause escape
    return ContinueIgnoringReturn;
  }

  // Now handle special cases where CaptureTracking says it's captured,
  // but we need more sophisticated escape analysis

  // Special handling for Store instructions
  if (const auto *SI = dyn_cast<StoreInst>(I)) {
    // Check if we're storing the pointer (not storing to it)
    if (SI->getValueOperand() == U->get()) {
      // Use our enhanced MemorySSA-based analysis to check if the
      // destination itself escapes
      if (doesStoreDestinationEscape(SI)) {
        LLVM_DEBUG(dbgs() << "  Store to escaping destination, escapes\n");
        Escaped = true;
        return Stop;
      }
      // Store to local non-escaping location - doesn't cause escape
      return ContinueIgnoringReturn;
    }
  }

  // Special handling for Call/Invoke with nocapture arguments
  // if (const auto *CB = dyn_cast<CallBase>(I)) {
  //   if (!CB->isCallee(U)) {
  //     // Check if the argument is marked nocapture
  //     for (unsigned ArgNo = 0; ArgNo < CB->arg_size(); ++ArgNo) {
  //       if (CB->getArgOperand(ArgNo) == U->get()) {
  //         if (CB->getCalledFunction()->hasParamAttribute(
  //                 ArgNo, Attribute::AttrKind::NoCapture)) {
  //           // Argument is nocapture - doesn't escape
  //           LLVM_DEBUG(dbgs() << "  Passed to nocapture parameter\n");
  //           return ContinueIgnoringReturn;
  //         }
  //         break;
  //       }
  //     }
  //   }
  // }

  // For all other captures reported by CaptureTracking, we trust its judgment
  if (capturesAnything(CI.UseCC)) {
    LLVM_DEBUG(dbgs() << "  Captured by: " << *I << "\n");
    Escaped = true;
    return Stop;
  }

  // If ResultCC indicates the result may capture the pointer, continue
  // analyzing the result
  if (capturesAnything(CI.ResultCC))
    return Continue;

  return ContinueIgnoringReturn;
}

bool EscapeAnalysisInfo::analyzeStoreDestEscapes(
    SmallVector<const Value *, 32> Worklist,
    SmallPtrSet<const Value *, 32> Visited, const Value *Dest) {
  // Find base objects for the storage location
  SmallPtrSet<const Value *, 8> BaseObjects;
  bool IsComplete = false;
  getUnderlyingObjectsThroughLoads(Dest, MSSA, BaseObjects, LI,
                                   &IsComplete);

  if (BaseObjects.empty() || !IsComplete) {
    LLVM_DEBUG(dbgs() << "  Store destination unknown, escapes\n");
    return true;
  }

  // Check each base object
  for (const Value *Base : BaseObjects) {
    // Global variable - object escapes
    if (isa<GlobalVariable>(Base) || isa<GlobalAlias>(Base)) {
      LLVM_DEBUG(dbgs() << "  Stored to global variable, escapes\n");
      return true;
    }

    // Function argument - object may escape
    if (const auto *Arg = dyn_cast<Argument>(Base)) {
      LLVM_DEBUG(dbgs() << "  Stored to function argument, escapes\n");
      return true;
    }

    // If storing to another local allocation
    if (const auto *AI = dyn_cast<AllocaInst>(Base)) {
      // Recursively check if this allocation escapes
      if (Visited.count(AI) == 0) {
        // Avoid infinite recursion: check cache
        if (auto CacheIt = Cache.find(AI); CacheIt != Cache.end()) {
          if (CacheIt->second) {
            LLVM_DEBUG(dbgs() << "  Stored to escaping alloca, escapes\n");
            return true;
          }
          continue;
        }
        // Add to worklist for further analysis
        tryEnqueueIfNew(AI, Visited, Worklist);
      }
      continue;
    }

    // Unknown base object (e.g., LoadInst)
    // Conservatively assume the object escapes
    LLVM_DEBUG(dbgs() << "  Stored to unknown location: " << *Base
                      << ", escapes\n");
    return true;
  } // end for (const Value *Base : BaseObjects)
  return false;
}

//===----------------------------------------------------------------------===//
// EscapeAnalysis Core Implementation
//===----------------------------------------------------------------------===//

bool EscapeAnalysisInfo::solveEscapeFor(
    const Value &Allocation,
    SmallPtrSet<const Value *, 32> &ProcessingSet) {
  // Mark this allocation as being processed to prevent infinite recursion
  ProcessingSet.insert(&Allocation);

  // Create our custom tracker
  EscapeCaptureTracker Tracker(*this, ProcessingSet);

  // Use the CaptureTracking infrastructure to analyze the allocation
  // We set ReturnCaptures=true because returning a pointer means it escapes
  PointerMayBeCaptured(&Allocation, &Tracker,
                       /*MaxUsesToExplore=*/WorklistLimit);

  return Tracker.hasEscaped();
}

bool EscapeAnalysisInfo::isEscaping(const Value &Alloc) {
  // 1. Get the underlying object
  const Value *UnderlyingObj = getUnderlyingObjectAggressive(&Alloc);

  // 2. Check the cache for a previously computed result
  if (const auto CacheIt = Cache.find(UnderlyingObj); CacheIt != Cache.end())
    return CacheIt->second;

  // 3. If not in cache, run the analysis
  LLVM_DEBUG(dbgs() << "EscapeAnalysis: Analyzing " << *UnderlyingObj << "\n");
  NumAllocationsAnalyzed++;

  // Lazily get other analyses from the FAM
  if (!MSSA)
    MSSA = &FAM.getResult<MemorySSAAnalysis>(F).getMSSA();
  if (!LI)
    LI = &FAM.getResult<LoopAnalysis>(F);

  // Track allocations being processed to detect cycles
  SmallPtrSet<const Value *, 32> ProcessingSet;
  const bool Result = solveEscapeFor(*UnderlyingObj, ProcessingSet);

  if (Result) {
    NumAllocationsEscaped++;
    LLVM_DEBUG(dbgs() << "  -> Result: ESCAPES\n");
  } else {
    LLVM_DEBUG(dbgs() << "  -> Result: DOES NOT ESCAPE\n");
  }

  // 4. Store result in cache and return
  return Cache[UnderlyingObj] = Result;
}

/*
bool EscapeAnalysisInfo::solveEscapeFor(const Value &AllocationSite) {
  SmallVector<const Value *, 32> Worklist;
  SmallPtrSet<const Value *, 32> Visited;

  // Start analysis from the allocation site itself
  tryEnqueueIfNew(&AllocationSite, Visited, Worklist);

  unsigned StepCount = 0;

  while (!Worklist.empty()) {
    const Value *V = Worklist.pop_back_val();

    // Safety valve to prevent infinite loops
    if (++StepCount > WorklistLimit) {
      LLVM_DEBUG(dbgs() << "  Worklist limit exceeded, conservatively assuming escape\n");
      return true; // Conservatively assume the object escapes
    }

    // Check all uses of the current value
    for (const Use &U : V->uses()) {
      const auto *I = dyn_cast<Instruction>(U.getUser());

      if (!I) {
        LLVM_DEBUG(dbgs() << "  Non-instruction user, escapes\n");
        return true;
      }

      // Analyze different instruction types using switch
      switch (I->getOpcode()) {
      case Instruction::Ret:
        // Returning pointer from function - object escapes
        LLVM_DEBUG(dbgs() << "  Returned from function, escapes\n");
        return true;

      case Instruction::Store: {
        const auto *SI = cast<StoreInst>(I);
        // Volatile stores make the address observable
        if (SI->isVolatile()) {
          LLVM_DEBUG(dbgs() << "  Volatile store, escapes\n");
          return true;
        }

        // Store instruction: check where the pointer is being stored
        if (SI->getValueOperand() == V) {
          // Storing a pointer to our object somewhere (operand 0 is the value)
          if (analyzeStoreDestEscapes(Worklist, Visited,
                                      SI->getPointerOperand()))
            return true;
        }
        // Storing something else at the address of our object (operand 1).
        // This does not cause the object itself to escape
        break;
      }

      case Instruction::Load: {
        const auto *LI = cast<LoadInst>(I);
        // Volatile loads make the address observable
        if (LI->isVolatile()) {
          LLVM_DEBUG(dbgs() << "  Volatile load, escapes\n");
          return true;
        }

        // Load from our object - does not cause the object itself to escape
        // (unless the load result is a pointer that subsequently escapes)
        if (I->getType()->isPointerTy()) {
          // Load returns a pointer, need to check its uses
          tryEnqueueIfNew(I, Visited, Worklist);
        }
        break;
      }

      case Instruction::Call:
      case Instruction::Invoke: {
        const auto *CB = cast<CallBase>(I);

        // Calling a function pointer does not in itself cause the pointer
        // to be captured
        if (CB->isCallee(&U))
          break;

        // Check if our pointer is passed as an argument
        bool PassedAsArg = false;
        for (unsigned ArgNo = 0; ArgNo < CB->arg_size(); ++ArgNo) {
          if (CB->getArgOperand(ArgNo) == V) {
            // TODO: Check if argument has nocapture attribute
            PassedAsArg = true;
            break;
          }
        }

        if (PassedAsArg) {
          // For safe intrinsics (e.g., lifetime intrinsics)
          if (const auto *II = dyn_cast<IntrinsicInst>(I)) {
            switch (II->getIntrinsicID()) {
            case Intrinsic::lifetime_start:
            case Intrinsic::lifetime_end:
            case Intrinsic::invariant_start:
            case Intrinsic::invariant_end:
            case Intrinsic::dbg_declare:
            case Intrinsic::dbg_value:
            case Intrinsic::dbg_label:
              // These intrinsics do not cause escape
              continue;
            default:
              break;
            }
          }

          // Check for memory intrinsics with volatile flag
          if (const auto *MI = dyn_cast<MemIntrinsic>(CB)) {
            if (MI->isVolatile()) {
              LLVM_DEBUG(dbgs() << "  Volatile memory intrinsic, escapes\n");
              return true;
            }
          }

          // For other functions, conservatively assume the object escapes
          LLVM_DEBUG(dbgs()
                     << "  Passed to function call: " << *CB << ", escapes\n");
          return true;
        }
        break;
      }

      case Instruction::VAArg:
        // va_arg from a pointer does not cause it to be captured
        break;

      case Instruction::AtomicRMW: {
        const auto *ARMWI = cast<AtomicRMWInst>(I);
        // Volatile atomics make the address observable
        if (ARMWI->isVolatile()) {
          LLVM_DEBUG(dbgs() << "  Volatile AtomicRMW, escapes\n");
          return true;
        }
        // If the value being stored is our pointer (operand 1), it escapes
        if (U.getOperandNo() == 1) {
          LLVM_DEBUG(dbgs() << "  Stored via AtomicRMW, escapes\n");
          return true;
        }
        // Otherwise, the location being accessed does not cause escape
        break;
      }

      case Instruction::AtomicCmpXchg: {
        const auto *ACXI = cast<AtomicCmpXchgInst>(I);
        // Volatile atomics make the address observable
        if (ACXI->isVolatile()) {
          LLVM_DEBUG(dbgs() << "  Volatile AtomicCmpXchg, escapes\n");
          return true;
        }
        // If the value being compared or stored is our pointer (operands 1 or 2)
        if (U.getOperandNo() == 1 || U.getOperandNo() == 2) {
          LLVM_DEBUG(dbgs() << "  Stored via AtomicCmpXchg, escapes\n");
          return true;
        }
        // Otherwise, the location being accessed does not cause escape
        break;
      }

      case Instruction::GetElementPtr: {
        // GEP with vector type should be considered as capture
        if (I->getType()->isVectorTy()) {
          LLVM_DEBUG(dbgs() << "  GEP with vector type, escapes\n");
          return true;
        }
        // Simple pointer operation - continue analysis
        tryEnqueueIfNew(I, Visited, Worklist);
        break;
      }

      case Instruction::BitCast:
      case Instruction::AddrSpaceCast:
      case Instruction::Select:
      case Instruction::PHI:
        // Simple pointer operations - continue analysis
        tryEnqueueIfNew(I, Visited, Worklist);
        break;

      case Instruction::ICmp:
        // Pointer comparison - does not cause escape
        break;

      case Instruction::PtrToInt:
        // Converting pointer to integer - conservatively assume escape
        // (pointer can be reconstructed via inttoptr)
        LLVM_DEBUG(dbgs() << "  PtrToInt conversion, escapes\n");
        return true;

      case Instruction::IntToPtr:
        // This is less common in escape context, but handle conservatively
        if (I->getType()->isPointerTy())
          tryEnqueueIfNew(I, Visited, Worklist);
        break;

      default:
        // Unknown use type - conservatively assume escape
        LLVM_DEBUG(dbgs() << "  Unknown use: " << *I << ", escapes\n");
        return true;
      }
    }
  }
  return false; // If we checked all uses and found no escapes
}

bool EscapeAnalysisInfo::isEscaping(const Value &Alloc) {
  // 1. Get the underlying object. This handles bitcasts, GEPs, and
  //    simple cases of PHIs and selects pointing to the same object.
  const Value *UnderlyingObj = getUnderlyingObjectAggressive(&Alloc);

  // 2. Check the cache for a previously computed result.
  if (const auto CacheIt = Cache.find(UnderlyingObj); CacheIt != Cache.end())
    return CacheIt->second;

  // 3. If not in cache, run the analysis.
  LLVM_DEBUG(dbgs() << "EscapeAnalysis: Analyzing " << *UnderlyingObj << "\n");
  NumAllocationsAnalyzed++;

  // Lazily get other analyses from the FAM.
  if (!MSSA)
    MSSA = &FAM.getResult<MemorySSAAnalysis>(F).getMSSA();
  if (!LI)
    LI = &FAM.getResult<LoopAnalysis>(F);

  const bool Result = solveEscapeFor(*UnderlyingObj);

  if (Result) {
    NumAllocationsEscaped++;
    LLVM_DEBUG(dbgs() << "  -> Result: ESCAPES\n");
  } else {
    LLVM_DEBUG(dbgs() << "  -> Result: DOES NOT ESCAPE\n");
  }

  // 4. Store result in cache and return.
  return Cache[UnderlyingObj] = Result;
}
*/

bool EscapeAnalysisInfo::invalidate(Function &F, const PreservedAnalyses &PA,
                                    FunctionAnalysisManager::Invalidator &Inv) {
  if (!PA.getChecker<EscapeAnalysis>().preserved())
    return true;

  // If dependant analysis invalidated - invalidate too
  // if (Inv.invalidate<AAManager>(F, PA)) return true;
  return false;
}

AnalysisKey EscapeAnalysis::Key;

EscapeAnalysis::Result EscapeAnalysis::run(Function &F,
                                           FunctionAnalysisManager &FAM) {
  EscapeAnalysisInfo EAI(F, FAM);
  return EAI;
}

//===----------------------------------------------------------------------===//
// Printing Pass for Verification
//===----------------------------------------------------------------------===//

PreservedAnalyses
EscapeAnalysisPrinterPass::run(Function &F, FunctionAnalysisManager &AM) const {
  if (F.isDeclaration())
    return PreservedAnalyses::all();

  OS << "EscapeAnalysis for function: " << F.getName() << "\n";

  bool HasInterestingAllocs = false;
  auto &EA = AM.getResult<EscapeAnalysis>(F);
  auto &TLI = AM.getResult<TargetLibraryAnalysis>(F);

  for (Instruction &I : instructions(F)) {
    bool IsAllocation = false;
    if (isa<AllocaInst>(I)) {
      IsAllocation = true;
    } else if (const auto *CB = dyn_cast<CallBase>(&I)) {
      if (isAllocationFn(&I, &TLI) || isNewLikeFn(&I, &TLI))
        IsAllocation = true;
    }

    if (IsAllocation) {
      HasInterestingAllocs = true;
      const bool Escapes = EA.isEscaping(I);
      OS << "  Allocation " << I.getName() << ": "
             << (Escapes ? "ESCAPES" : "DOES NOT ESCAPE") << "\n";
    }
  }

  if (!HasInterestingAllocs)
    OS << "  No allocations to analyze.\n";

  return PreservedAnalyses::all();
}