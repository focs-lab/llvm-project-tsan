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
#include "llvm/ADT/SmallString.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#include <deque>

#define DEBUG_TYPE "escape-analysis"

using namespace llvm;

STATISTIC(NumAllocationsAnalyzed, "Number of allocation sites analyzed");
STATISTIC(NumAllocationsEscaped, "Number of allocation sites found to escape");

/// Per-allocation worklist cap (safety valve). If the number of processed
/// worklist nodes exceeds this limit, the analysis bails out conservatively and
/// considers the allocation as escaping.
static cl::opt<unsigned>
WorklistLimit("escape-analysis-worklist-limit", cl::init(1000), cl::Hidden,
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

void EscapeAnalysisInfo::getUnderlyingObjectsThroughLoads(
    const Value *V, MemorySSA *MSSA, SmallPtrSetImpl<const Value *> &Result,
    LoopInfo *LI, bool *IsComplete, unsigned MaxSteps) {
  SmallPtrSet<const Value *, 32> VisitedWithVT; // 1st stage (ValueTracking)
  SmallPtrSet<const Value *, 32> Enqueued;      // Guard for enqueue (seen)
  SmallPtrSet<MemoryAccess *, 32> VisitedMA;
  SmallVector<const Value *, 32> Work;

  if (!V->getType()->isPointerTy())
    return; // Only pointers have underlying objects.

  auto markTerminal = [&](const Value *Term,
                          bool ForceIncompleteIfNotBase = true) {
    if (!Term || !Term->getType()->isPointerTy()) return;

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

      if (auto *MDef = dyn_cast<MemoryDef>(CurrClobber)) {
        // If clobber is a MemoryDef, inspect its instruction.
        const Instruction *ClobberI = MDef->getMemoryInst();
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
        // (e.g. memmove/memcpy/memset) because they are not semantically
        // underlying objects

        // Fallback: unrecognized defining write, stop here conservatively.
        Fallback = true;
        break;
      }

      if (const auto *MPhi = dyn_cast<MemoryPhi>(CurrClobber)) {
        // Iterate the incoming accesses and process each incoming
        // MemoryAccess.
        appendUnvisitedIncomingMAs(MPhi, VisitedMA, MAWorkList);
      } else if (auto *MUse = dyn_cast<MemoryUse>(CurrClobber)) {
        // If clobber is a MemoryUse (rarely but possible), get its defining.
        MemoryAccess *Def = Walker->getClobberingMemoryAccess(MUse);
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
  if (IsComplete)
    *IsComplete = false;
  // Conservative: mark all remaining items as terminals
  for (const Value *WV : Work)
    markTerminal(WV);
}

//===----------------------------------------------------------------------===//
// EscapeCaptureTracker Implementation
//===----------------------------------------------------------------------===//

bool EscapeAnalysisInfo::EscapeCaptureTracker::shouldExplore(const Use *U) {
  // Always explore, but we can add optimizations here later
  return true;
}

bool EscapeAnalysisInfo::EscapeCaptureTracker::doesStoreDestinationEscape(
    const Value *Dest) {
  // Find base objects for the storage location
  SmallPtrSet<const Value *, 8> BaseObjects;
  bool IsComplete = false;
  getUnderlyingObjectsThroughLoads(Dest, EAI.MSSA, BaseObjects, EAI.LI,
                                   &IsComplete);

  // If bases are unknown or the walk is incomplete, be conservative.
  if (BaseObjects.empty() || !IsComplete) {
    LLVM_DEBUG(dbgs() << "  Store destination unknown/incomplete, escapes\n");
    return true;
  }

  for (const Value *Base : BaseObjects) {
    if (isa<GlobalVariable>(Base) || isa<GlobalAlias>(Base)) {
      LLVM_DEBUG(dbgs() << "  Stored to global, escapes\n");
      return true;
    }

    // Any memory reachable through a function Argument is externally visible.
    if (isa<Argument>(Base)) {
      LLVM_DEBUG(
          dbgs() << "  Stored to memory reachable from argument, escapes\n");
      return true;
    }

    // If storing to another local allocation, recursively check if it escapes
    if (const auto *Alloca = dyn_cast<AllocaInst>(Base)) {
      // Cycle in the current recursion indicates incomplete local reasoning.
      // Without a global fixpoint, treat as escape to remain conservative.
      if (ProcessingSet.count(Alloca)) {
        LLVM_DEBUG(dbgs() << "  Cyclic store dependency, escapes\n");
        return true;
      }

      // If cache says the alloca escapes, propagate escape.
      // TODO: maybe move Cache check to solveEscapeFor?
      if (auto CacheIt = EAI.Cache.find(Alloca); CacheIt != EAI.Cache.end()) {
        if (CacheIt->second) {
          LLVM_DEBUG(dbgs() << "  Stored to escaping (cached), escapes\n");
          return true;
        }
        continue; // Cached non-escape: keep checking other bases.
      }

      // Recurse to decide whether the target alloca itself escapes.
      // auto RecursiveProcessingSet = ProcessingSet;
      // RecursiveProcessingSet.insert(Alloca);
      if (EAI.solveEscapeFor(*Alloca, ProcessingSet)) {
        LLVM_DEBUG(dbgs() << "  Stored to escaping alloca, escapes\n");
        return true;
      }
      continue;
    }
    // Any other/unknown terminal means the destination is not proven local.
    LLVM_DEBUG(dbgs() << "  Stored to unknown location: " << *Base
                      << ", escapes\n");
    return true;
  }
  return false;
}

bool EscapeAnalysisInfo::EscapeCaptureTracker::doesStoredPointerEscapeViaLoads(
    const StoreInst *Store, SmallPtrSet<const Value *, 32> &ProcessingSet) {
  if (!EAI.MSSA) return true;
  auto *MDef = dyn_cast<MemoryDef>(EAI.MSSA->getMemoryAccess(Store));
  if (!MDef) return true;

  SmallVector<MemoryAccess *, 32> MAWorkList;
  SmallPtrSet<MemoryAccess *, 32> VisitedMA;
  MAWorkList.push_back(MDef);
  VisitedMA.insert(MDef);

  MemorySSAWalker *Walker = EAI.MSSA->getWalker();

  unsigned Steps = 0;
  while (!MAWorkList.empty()) {
    if (++Steps > WorklistLimit) return true;
    MemoryAccess *MA = MAWorkList.pop_back_val();

    for (User *U : MA->users()) {
      if (auto *MUse = dyn_cast<MemoryUse>(U)) {
        if (Walker->getClobberingMemoryAccess(MUse) != MA)
          continue;

        if (const auto *Load = dyn_cast<LoadInst>(MUse->getMemoryInst()); Load->getType.).) {
          LLVM_DEBUG(dbgs() << "LoadInst: " << *Load << "\n");
          if (!Load->getType()->isPointerTy())
            continue;
          // if (EAI.solveEscapeFor(*Load, ProcessingSet))
          //   return true;
          EscapeCaptureTracker LocalTracker(EAI, ProcessingSet);
          PointerMayBeCaptured(Load, &LocalTracker, WorklistLimit);
          if (LocalTracker.hasEscaped())
            return true;
        }
      } else if (auto *MPhi = dyn_cast<MemoryPhi>(U)) {
        if (VisitedMA.insert(MPhi).second)
          MAWorkList.push_back(MPhi);
      }
    }
  }
  return false;
}

CaptureTracker::Action
EscapeAnalysisInfo::EscapeCaptureTracker::captured(const Use *U,
                                                   UseCaptureInfo CI) {
  LLVM_DEBUG(dbgs() << "  Analyzing capture use: " << *U->get() << " in "
                    << *U->getUser() << "\n");
  const auto *I = cast<Instruction>(U->getUser());

  // If CaptureTracking says this use does not capture, continue exploring.
  if (capturesNothing(CI.UseCC)) {
    LLVM_DEBUG(dbgs() << "    Use doesn't capture, continue\n");
    return Continue; // CaptureTracking says it's not captured, continue
  }

  // Passthrough ops (gep/bitcast/select/phi..) should be explored transitively.
  if (CI.isPassthrough()) {
    LLVM_DEBUG(dbgs() << "    Passthrough operation, continue to result\n");
    return Continue;
  }

  // Now handle special cases where CaptureTracking says it's captured,
  // but we need more sophisticated escape analysis

  if (const auto *SI = dyn_cast<StoreInst>(I)) {
    // Check if we're storing the pointer (not storing to it)
    if (SI->getValueOperand() == U->get()) {
      LLVM_DEBUG(dbgs() << "    Storing pointer value, analyze destination\n");
      if (SI->isVolatile() || SI->isAtomic() ||
          doesStoreDestinationEscape(SI->getPointerOperand())) {
        LLVM_DEBUG(dbgs() << "  Store to escaping destination, escapes\n");
        Escaped = true;
        return Stop;
      }

      LLVM_DEBUG(dbgs() << "---- doesStoredPointerEscapeViaLoads ----\n");
      if (doesStoredPointerEscapeViaLoads(SI, ProcessingSet)) {
        LLVM_DEBUG(dbgs() << "  Stored to escaping alloca, escapes\n");
        Escaped = true;
        LLVM_DEBUG(dbgs() << "-----------------------------------------\n");
        return Stop;
      }
      LLVM_DEBUG(dbgs() << "-----------------------------------------\n");
      LLVM_DEBUG(dbgs() << "    Store to safe local, doesn't escape\n");
      return ContinueIgnoringReturn;
    }
    // If we are the destination pointer, this use does not capture the value.
    LLVM_DEBUG(dbgs() << "    Used as store destination, doesn't escape\n");
    return ContinueIgnoringReturn;
  }

  if (isa<ICmpInst>(I)) { // Pure comparisons of addresses do not cause escape.
    LLVM_DEBUG(dbgs() << "    Pointer comparison, doesn't escape\n");
    return ContinueIgnoringReturn;
  }

  // Default: if CaptureTracking still indicates capture, treat as escape.
  if (capturesAnything(CI.UseCC)) {
    LLVM_DEBUG(dbgs() << "  Captured by: " << *I << "\n");
    Escaped = true;
    return Stop;
  }

#ifndef NDEBUG
  llvm_unreachable("Unhandled case in EscapeCaptureTracker::captured");
#else
  return Continue;
#endif
}

//===----------------------------------------------------------------------===//
// EscapeAnalysis Core Implementation
//===----------------------------------------------------------------------===//

bool EscapeAnalysisInfo::solveEscapeFor(
    const Value &Allocation,
    SmallPtrSet<const Value *, 32> &ProcessingSet) {
  // Mark this allocation as being processed to prevent infinite recursion
  LLVM_DEBUG(dbgs() << "====================================\n";
             dbgs() << "solveEscapeFor " << Allocation << "\n";);
  ProcessingSet.insert(&Allocation);

  // const bool Captured = PointerMayBeCaptured(&Allocation, true);
  // dbgs() << "CT: Allocation " << Allocation
  //        << (Captured ? " may be CAPTURED\n" : " NOT CAPTURED\n");
  // dbgs() << "====================================\n";

  // Create our custom tracker
  EscapeCaptureTracker Tracker(*this, ProcessingSet);

  // Use the CaptureTracking infrastructure to analyze the allocation
  // We set ReturnCaptures=true because returning a pointer means it escapes
  PointerMayBeCaptured(&Allocation, &Tracker,
                       /*MaxUsesToExplore=*/WorklistLimit);
  Cache[&Allocation] = Tracker.hasEscaped();

  return Tracker.hasEscaped();
}

bool EscapeAnalysisInfo::isHeapAllocation(const CallBase *CB,
                                          const TargetLibraryInfo *TLI) {
  // Try standard path first (works for C++ new and modern IR with allockind)
  if (isAllocationFn(CB, TLI) || isNewLikeFn(CB, TLI))
    return true;

  // Fallback: check directly via TLI for malloc/calloc/etc
  const Function *Callee = CB->getCalledFunction();
  if (!Callee || !Callee->getReturnType()->isPointerTy())
    return false;

  LibFunc Func;
  if (!TLI->getLibFunc(*Callee, Func) || !TLI->has(Func))
    return false;

  // List of known heap allocation functions from libc
  switch (Func) {
  case LibFunc_malloc:
  case LibFunc_calloc:
  case LibFunc_realloc:
  case LibFunc_reallocf:
  case LibFunc_reallocarray:
  case LibFunc_valloc:
  case LibFunc_pvalloc:
  case LibFunc_aligned_alloc:
  case LibFunc_memalign:
  case LibFunc_vec_malloc:
  case LibFunc_vec_calloc:
  case LibFunc_vec_realloc:
  case LibFunc_strdup:
  case LibFunc_strndup:
    return true;
  default:
    return false;
  }
}
bool EscapeAnalysisInfo::isAllocationSite(const Value *V,
                                          const TargetLibraryInfo *TLI) {
  if (isa<AllocaInst>(V))
    return true;
  if (const auto *CB = dyn_cast<CallBase>(V))
    return isHeapAllocation(CB, TLI);
  return false;
}

bool EscapeAnalysisInfo::isEscaping(const Value &Alloc) {
  // Validate input
  auto &TLI = FAM.getResult<TargetLibraryAnalysis>(F);
  if (!isAllocationSite(&Alloc, &TLI)) {
    LLVM_DEBUG(dbgs() << "EscapeAnalysis: Not an allocation site: "
                      << &Alloc << "\n");
    return true; // Conservative: unknown things "escape"
  }

  // Check the cache for a previously computed result
  if (const auto CacheIt = Cache.find(&Alloc); CacheIt != Cache.end())
    return CacheIt->second;

  // If not in cache, run the analysis
  LLVM_DEBUG(dbgs() << "EscapeAnalysis: Analyzing " << Alloc << "\n");
  NumAllocationsAnalyzed++;

  if (!MSSA)
    MSSA = &FAM.getResult<MemorySSAAnalysis>(F).getMSSA();
  if (!LI)
    LI = &FAM.getResult<LoopAnalysis>(F);

  // Track allocations being processed to detect cycles
  SmallPtrSet<const Value *, 32> ProcessingSet;
  const bool IsEscaped = solveEscapeFor(Alloc, ProcessingSet);

  if (IsEscaped)
    NumAllocationsEscaped++;

  // 4. Store result in cache and return
  return Cache[(&Alloc)] = IsEscaped;
}

void EscapeAnalysisInfo::print(raw_ostream &OS) {
  auto &TLI = FAM.getResult<TargetLibraryAnalysis>(F);
  bool Any = false;
  unsigned UnnamedCount = 0;

  for (Instruction &I : instructions(F)) {
    LLVM_DEBUG(OS << "\nI: " << I << "\n");
    if (!isAllocationSite(&I, &TLI))
      continue;

    Any = true;

    // Stable symbol: use SSA name if exists, otherwise "unnamed#N".
    StringRef Name = I.hasName() ? I.getName() : StringRef();
    SmallString<32> Gen;
    if (Name.empty()) {
      ++UnnamedCount;
      Gen += "unnamed#";
      Gen += Twine(UnnamedCount).str();
      Name = Gen;
    }

    const bool Esc = isEscaping(I);
    OS << "  " << Name << " escapes: " << (Esc ? "yes" : "no") << "\n";
  }

  if (!Any)
    OS << "  none\n";
  OS << "\n";
}

bool EscapeAnalysisInfo::invalidate(Function &F, const PreservedAnalyses &PA,
                                    FunctionAnalysisManager::Invalidator &Inv) {
  if (Inv.invalidate<MemorySSAAnalysis>(F, PA) ||
      Inv.invalidate<LoopAnalysis>(F, PA))
    return true;
  if (!PA.getChecker<EscapeAnalysis>().preserved())
    return true;
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
EscapeAnalysisPrinterPass::run(Function &F, FunctionAnalysisManager &FAM) const {
  if (F.isDeclaration())
    return PreservedAnalyses::all();
  OS << "Printing analysis 'Escape Analysis' for function '" << F.getName()
     << "':\n";
  FAM.getResult<EscapeAnalysis>(F).print(OS);
  return PreservedAnalyses::all();
}