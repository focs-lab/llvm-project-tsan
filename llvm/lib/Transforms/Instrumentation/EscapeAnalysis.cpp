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
#include "llvm/ADT/SmallString.h"
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

namespace llvm {
/// Add P to Worklist if it doesn't exist in Seen
template <typename PtrT, typename SetT, typename WorklistT>
static bool tryEnqueueIfNew(PtrT *P, SetT &Seen, WorklistT &Worklist) {
  if (P && Seen.insert(P).second) {
    Worklist.push_back(P);
    return true;
  }
  return false;
}

/// Try to use ValueTracking to find underlying objects.
static bool tryValueTracking(const Value *V, LoopInfo *LI,
                             SmallVectorImpl<const Value *> &Work,
                             SmallPtrSetImpl<const Value *> &Enqueued) {
  SmallVector<const Value *, 4> Bases;
  if (!V->getType()->isPointerTy())
    return false; // Only pointers have underlying objects.

  getUnderlyingObjects(V, Bases, LI, VTMaxLookup);

  if (Bases.empty() || (Bases.size() == 1 && Bases[0] == V))
    return false;

  for (const Value *B : Bases)
    tryEnqueueIfNew(B, Enqueued, Work);
  return true;
}


void getUnderlyingObjectsThroughLoads(const Value *Ptr, MemorySSA *MSSA,
                                      AAResults *AA,
                                      SmallPtrSetImpl<const Value *> &Result,
                                      LoopInfo *LI, bool *IsComplete,
                                      unsigned MaxSteps) {
  LLVM_DEBUG(dbgs() << "\tgetUnderlyingObjectsThroughLoads: " << Ptr->getName()
                    << "\n");

  assert(MSSA && "MemorySSA is required for this analysis");
  MemorySSAWalker *Walker = MSSA->getSkipSelfWalker();

  if (!Ptr->getType()->isPointerTy()) {
    LLVM_DEBUG(dbgs() << "Input is not a pointer, early return: " << *Ptr
                      << "\n");
    return; // Only pointers have underlying objects.
  }

  auto addTerminal = [&](const Value *Term,
                         bool MarkIncompleteIfNotBase = true) {
    if (!Term || !Term->getType()->isPointerTy())
      return;
    const bool IsBase = isa<AllocaInst>(Term) || isa<GlobalVariable>(Term) ||
                        isa<GlobalAlias>(Term) || isa<Argument>(Term) ||
                        isa<ConstantPointerNull>(Term);
    LLVM_DEBUG(dbgs() << "\tMark terminal: " << *Term << " IsBase="
                      << (IsBase ? "yes" : "no") << "\n");
    Result.insert(Term);
    if (IsComplete && !IsBase && MarkIncompleteIfNotBase) {
      *IsComplete = false;
      LLVM_DEBUG(dbgs() << "\tMarking incomplete due to terminal\n");
    }
  };

  SmallPtrSet<const Value *, 32> SeenVT;    // 1st stage (ValueTracking)
  SmallPtrSet<const Value *, 32> Seen;      // Guard for enqueue
  SmallVector<const Value *, 32> Worklist;

  auto bail = [&]() {
    if (IsComplete)
      *IsComplete = false;
    for (const Value *WV : Worklist)
      addTerminal(WV);
  };

  tryEnqueueIfNew(Ptr, Seen, Worklist);

  unsigned Step = 0;
  if (IsComplete)
    *IsComplete = true;

  while (!Worklist.empty()) {
    const Value *CurrPtr = Worklist.pop_back_val();

    // Safety valve: if we exceed MaxSteps, bail out conservatively.
    if (++Step > MaxSteps) {
      LLVM_DEBUG(dbgs() << "MaxSteps exceeded at: " << *CurrPtr << "\n");
      addTerminal(CurrPtr);
      bail();
      return;
    }

    // Try ValueTracking first (only once per value)
    if (!isa<LoadInst>(CurrPtr) && SeenVT.insert(CurrPtr).second &&
        tryValueTracking(CurrPtr, LI, Worklist, Seen))
      continue; // Successfully expanded via ValueTracking;

    const auto *Load = dyn_cast<LoadInst>(CurrPtr);
    if (!Load || Load->isVolatile() || Load->isAtomic()) {
      addTerminal(CurrPtr);
      continue;
    }

    const auto LoadLoc = MemoryLocation::get(Load);

    // Use MemorySSA's API to get the clobbering MemoryAccess.
    MemoryAccess *Clobber = Walker->getClobberingMemoryAccess(Load);
    assert(Clobber && "Expected clobbering MemoryAccess");

    // Local accumulators for Load
    SmallVector<const Value *, 8> LocalWorklist;
    SmallPtrSet<const Value *, 8> LocalSeen;
    SmallPtrSet<MemoryAccess *, 32> VisitedMA;
    SmallVector<MemoryAccess *, 32> MAWorklist;

    LocalSeen.insert(Load);
    bool Fallback = false;
    unsigned MaxMAIterations = 0;
    const unsigned MAIterationLimit = std::max(1u, MaxSteps / 2);

    if (!tryEnqueueIfNew(Clobber, VisitedMA, MAWorklist))
      continue;

    while (!MAWorklist.empty()) {
      if (++MaxMAIterations > MAIterationLimit) {
        LLVM_DEBUG(dbgs() << "MA iteration limit exceeded, fallback at Load: "
                          << *Load << "\n");
        Fallback = true;
        break;
      }
      MemoryAccess *CurrClobber = MAWorklist.pop_back_val();

      if (MSSA->isLiveOnEntryDef(CurrClobber)) {
        // Try to get base objects from the current load.
        const Value *PtrOpnd = Load->getPointerOperand()->stripPointerCasts();
        tryEnqueueIfNew(PtrOpnd, Seen, Worklist);
        continue;
      }

      if (const auto *MDef = dyn_cast<MemoryDef>(CurrClobber)) {
        // If clobber is a MemoryDef, inspect its instruction.
        const Instruction *ClobberI = MDef->getMemoryInst();
        assert(ClobberI && "MemoryDef must have an instruction");

        // Stores: chase the stored pointer when safe
        if (const auto *Store = dyn_cast<StoreInst>(ClobberI)) {
          // Volatile or atomic stores are opaque.
          if (Store->isVolatile() || Store->isAtomic()) {
            Fallback = true;
            break;
          }

          const Value *SV = Store->getValueOperand();
          if (SV->getType()->isPointerTy()) {
            tryEnqueueIfNew(SV, LocalSeen, LocalWorklist);
            continue;
          }
          // Non-pointer store into memory from which we later load a pointer:
          // treat as unknown/opaque write.
          LLVM_DEBUG(dbgs()
                     << "Non-pointer store, fallback at: " << *Store << "\n");
        }
        // NOTE: We intentionally don't consider the source in memintrinsics
        // (e.g. memmove/memcpy/memset): they are not underlying objects
        LLVM_DEBUG(dbgs() << "Unrecognized defining write, fallback\n");
        // Fallback: unrecognized defining write, stop here conservatively.
        Fallback = true;
        break;
      }

      if (const auto *MPhi = dyn_cast<MemoryPhi>(CurrClobber))
        appendIncomingMAs(MPhi, VisitedMA, MAWorklist, LoadLoc, Walker);
      else
        llvm_unreachable("Unexpected MemoryAccess kind");
    } // end while for MAWorkList

    if (Fallback) {
      LLVM_DEBUG(dbgs() << "Fallback: mark Load as term: " << *Load << "\n");
      addTerminal(Load);
    } else {
      for (const auto *WV : LocalWorklist)
        tryEnqueueIfNew(WV, Seen, Worklist);
    }
  } // end while for Work
  LLVM_DEBUG(dbgs() << "\tgetUnderlyingObjectsThroughLoads: " << Ptr->getName()
                    << " -- end\n");
}

} // end namespace llvm

//===----------------------------------------------------------------------===//
// EscapeCaptureTracker Implementation
//===----------------------------------------------------------------------===//

bool EscapeAnalysisInfo::EscapeCaptureTracker::shouldExplore(const Use *U) {
  // Always explore, but we can add optimizations here later
  return true;
}

bool EscapeAnalysisInfo::isExternalObject(const Value *Base) {
  return isa<GlobalVariable>(Base) || isa<GlobalAlias>(Base) ||
         isa<Argument>(Base);
}

bool EscapeAnalysisInfo::EscapeCaptureTracker::doesStoreDestEscapes(
    const Value *Dest) {
  // Find base objects for the storage location
  SmallPtrSet<const Value *, 8> BaseObjects;
  bool IsComplete = false;
  getUnderlyingObjectsThroughLoads(Dest, EAI.MSSA, EAI.AA, BaseObjects, EAI.LI,
                                   &IsComplete);

  // If bases are unknown or the walk is incomplete, be conservative.
  if (BaseObjects.empty() || !IsComplete) {
    LLVM_DEBUG(dbgs() << "  Store destination unknown/incomplete, escapes\n");
    return true;
  }

  for (const Value *Base : BaseObjects) {
    if (isExternalObject(Base)) {
      LLVM_DEBUG(dbgs() << "  Stored to external object, escapes\n");
      return true;
    }

    // If storing to another local allocation, recursively check if it escapes
    if (const auto *Alloca = dyn_cast<AllocaInst>(Base)) {
      if (ProcessingSet.count(Alloca)) { // Cycle
        LLVM_DEBUG(dbgs() << Alloca->getName() << " is processing now, skip\n");
        continue;
      }

      // Recurse to decide whether the target alloca itself escapes.
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

/// Add incoming unvisited MemoryAccesses of a MemoryPhi to MAWorkList.
static void appendIncomingMAs(const MemoryPhi *MPhi,
                              SmallPtrSetImpl<MemoryAccess *> &VisitedMA,
                              SmallVectorImpl<MemoryAccess *> &MAWorkList,
                              MemoryLocation Loc, MemorySSAWalker *Walker) {
  for (unsigned i = 0, N = MPhi->getNumIncomingValues(); i != N; ++i) {
    MemoryAccess *InMA = MPhi->getIncomingValue(i);
    MemoryAccess *EdgeCl = Walker->getClobberingMemoryAccess(InMA, Loc);
    if (!EdgeCl) EdgeCl = InMA;
    tryEnqueueIfNew(EdgeCl, VisitedMA, MAWorkList);
  }
}

enum class EdgeWalkStep { Recurse, SkipSuccessors, Stop };

/// Walk edge clobbering definitions starting from Start MemoryAccess.
template <typename VisitT>
static bool walkEdgeClobbers(MemoryAccess *Start, MemorySSA *MSSA,
                             MemorySSAWalker *Walker, MemoryLocation Loc,
                             unsigned Limit, VisitT Visit, bool &IsComplete) {
  IsComplete = true;
  if (!Start) return true;

  SmallVector<MemoryAccess *, 32> MAWorklist;
  SmallPtrSet<MemoryAccess *, 32> MAVisited;
  tryEnqueueIfNew(Start, MAVisited, MAWorklist);
  unsigned Steps = 0;

  while (!MAWorklist.empty()) {
    if (++Steps > Limit) {
      IsComplete = false;
      return false;
    }

    MemoryAccess *MA = MAWorklist.pop_back_val();
    if (MSSA->isLiveOnEntryDef(MA))
      continue;

    EdgeWalkStep Act = Visit(MA);
    if (Act == EdgeWalkStep::Stop)
      return false;
    if (Act == EdgeWalkStep::SkipSuccessors)
      continue;

    if (auto *MD = dyn_cast<MemoryDef>(MA)) {
      MemoryAccess *EdgeCl = Walker->getClobberingMemoryAccess(MD, Loc);
      if (!EdgeCl) {
        IsComplete = false;
        return false;
      }
      tryEnqueueIfNew(EdgeCl, MAVisited, MAWorklist);
    } else if (const auto *MPhi = dyn_cast<MemoryPhi>(MA)) {
      appendIncomingMAs(MPhi, MAVisited, MAWorklist, Loc, Walker);
    } else {
      llvm_unreachable("Unexpected MemoryAccess kind");
    }
  }
  return true;
}

SmallVector<const LoadInst *, 32>
EscapeAnalysisInfo::EscapeCaptureTracker::collectLoadsReadingFromStore(
    const StoreInst *StartStore, bool &IsComplete) {
  IsComplete = true;
  auto *StartMDef = cast<MemoryDef>(EAI.MSSA->getMemoryAccess(StartStore));
  LLVM_DEBUG(dbgs() << "Collecting Loads reading from Store: " << *StartStore
                    << "\t" << *StartMDef << "\n");
  const MemoryLocation LocDest = MemoryLocation::get(StartStore);
  MemorySSAWalker *Walker = EAI.MSSA->getSkipSelfWalker();

  auto doesStemFrom = [&](MemoryAccess *Cl) -> bool {
    bool WalkComplete = false, Found = false;
    walkEdgeClobbers(
        Cl, EAI.MSSA, Walker, LocDest, WorklistLimit,
        [&](MemoryAccess *MA) {
          LLVM_DEBUG(dbgs() << "  IsFromStart: Inspecting MA: " << *MA << "\n");
          if (MA == StartMDef) {
            Found = true;
            return EdgeWalkStep::Stop;
          }
          return EdgeWalkStep::Recurse;
        },
        WalkComplete);
    if (!WalkComplete)
      IsComplete = false;
    return Found || !WalkComplete;
  };

  SmallVector<const LoadInst *, 32> Out;
  SmallVector<MemoryAccess *, 32> MAWorklist;
  SmallPtrSet<MemoryAccess *, 32> VisitedMA;
  tryEnqueueIfNew(StartMDef, VisitedMA, MAWorklist);

  unsigned Steps = 0;
  while (!MAWorklist.empty()) {
    if (++Steps > WorklistLimit) {
      IsComplete = false;
      return {};
    }

    MemoryAccess *MA = MAWorklist.pop_back_val();
    for (User *U : MA->users()) {
      if (auto *MU = dyn_cast<MemoryUse>(U)) {
        const auto *Load = dyn_cast<LoadInst>(MU->getMemoryInst());
        if (!Load)
          continue;
        LLVM_DEBUG(dbgs() << "Inspecting Load: " << *Load << "\n");

        MemoryAccess *Clobber =
            Walker->getClobberingMemoryAccess(MU->getDefiningAccess(), LocDest);
        if (!Clobber) {
          IsComplete = false;
          return {};
        }
        if (!doesStemFrom(Clobber))
          continue;

        LLVM_DEBUG(dbgs() << "Found Load reading from Store: " << *Load << "\n");
        Out.push_back(Load);
        continue;
      }

      tryEnqueueIfNew(cast<MemoryAccess>(U), VisitedMA, MAWorklist);
    }
  }
  return Out;
}

bool EscapeAnalysisInfo::EscapeCaptureTracker::
    doesStoredPointerEscapeViaLoads(const StoreInst *Store) {
  LLVM_DEBUG(dbgs() << "\n---- doesStoredPointerEscapeThroughLoads " << *Store
                    << " ----\n");
  bool IsComplete = false;
  const auto Loads = collectLoadsReadingFromStore(Store, IsComplete);
  if (!IsComplete)
    return true;
  for (const LoadInst *Load: Loads) {
    if (Load->isVolatile() || Load->isAtomic() ||
        !Load->getType()->isPointerTy())
      return true;
    if (EAI.solveEscapeFor(*Load, ProcessingSet)) {
      LLVM_DEBUG(dbgs() << "---- doesStoredPointerEscapeThroughLoads - end --> "
                           "escape\n\n");
      return true;
    }
  }
  LLVM_DEBUG(dbgs() << "---- doesStoredPointerEscapeThroughLoads - end --> "
                       "not escape\n\n");
  return false;
}

CaptureTracker::Action
EscapeAnalysisInfo::EscapeCaptureTracker::captured(const Use *U,
                                                   UseCaptureInfo CI) {
  LLVM_DEBUG(dbgs() << "\n--> Analyzing capture use: " << *U->get() << " in "
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

  if (const auto *Store = dyn_cast<StoreInst>(I)) {
    // Check if we're storing the pointer (not storing to it)
    if (Store->getValueOperand() == U->get()) {
      LLVM_DEBUG(dbgs() << "    Storing pointer value, analyze destination "
                        << *Store->getPointerOperand() << "\n");
      if (Store->isVolatile() || Store->isAtomic() ||
          doesStoreDestEscapes(Store->getPointerOperand())) {
        LLVM_DEBUG(dbgs() << "  Store to escaping destination, escapes\n");
        Escaped = true;
        return Stop;
      }

      LLVM_DEBUG(dbgs() << "\n---- doesStoredPointerEscape " << *Store
                        << " ----\n");
      if (doesStoredPointerEscapeViaLoads(Store)) {
        Escaped = true;
        return Stop;
      }
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
    const Value &Ptr, SmallPtrSet<const Value *, 32> &ProcessingSet) {
  // Mark this allocation as being processed to prevent infinite recursion
  LLVM_DEBUG(dbgs() << "============ solveEscapeFor(" << Ptr.getName()
                    << ") ===================\n";);
  ProcessingSet.insert(&Ptr);

  if (const auto CacheIt = Cache.find(&Ptr);
      CacheIt != Cache.end() && CacheIt->second) {
    LLVM_DEBUG(dbgs() << "  Stored to escaping (cached), escapes\n");
    return true;
  }

  // Create our custom tracker
  EscapeCaptureTracker Tracker(*this, ProcessingSet);

  // Use the CaptureTracking infrastructure to analyze the allocation
  // We set ReturnCaptures=true because returning a pointer means it escapes
  PointerMayBeCaptured(&Ptr, &Tracker,
                       /*MaxUsesToExplore=*/WorklistLimit);
  Cache[&Ptr] = Tracker.hasEscaped();

  LLVM_DEBUG(dbgs() << "============ solveEscapeFor(" << Ptr.getName()
                    << ") -- end --> "
                    << (Tracker.hasEscaped() ? " escaped " : " not escaped ")
                    << " ============\n";);
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

  LLVM_DEBUG(dbgs() << "EscapeAnalysis: Analyzing " << Alloc << "\n");
  NumAllocationsAnalyzed++;

  if (!MSSA)
    MSSA = &FAM.getResult<MemorySSAAnalysis>(F).getMSSA();
  if (!LI)
    LI = &FAM.getResult<LoopAnalysis>(F);
  if (!AA)
    AA = &FAM.getResult<AAManager>(F);

  // Track allocations being processed to detect cycles
  SmallPtrSet<const Value *, 32> ProcessingSet;
  const bool IsEscaped = solveEscapeFor(Alloc, ProcessingSet);

  if (IsEscaped)
    NumAllocationsEscaped++;

  return IsEscaped;
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