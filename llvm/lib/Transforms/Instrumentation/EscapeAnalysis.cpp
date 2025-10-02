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
// EscapeAnalysis Implementation
//===----------------------------------------------------------------------===//

void EscapeAnalysisInfo::applyTransferFunction(
    const Instruction *I, SmallVectorImpl<const Value *> &Worklist,
    DenseSet<const Value *> &EscapedSet) {
  // This is a backward analysis. We check if the instruction's *result* is in
  // the EscapedSet. If so, we propagate the "escaped" property to its operands.

  if (!EscapedSet.count(I))
    return; // This instruction doesn't produce an escaped value.

  // The value produced by I escapes. Remove it from the set and add its
  // relevant operands, propagating the escaped property backward.
  EscapedSet.erase(I);

  if (isa<GetElementPtrInst>(I) || isa<BitCastInst>(I) || isa<SelectInst>(I)) {
    // Simple propagation: if a GEP/cast/select result escapes, the base
    // pointer/operands escape.
    for (const Use &Op : I->operands()) {
      const Value *V = Op.get();
      if (V->getType()->isPointerTy())
        Worklist.push_back(V);
    }
  } else if (const PHINode *PN = dyn_cast<PHINode>(I)) {
    // For a PHI node, all incoming values are considered to escape.
    for (const Use &V : PN->incoming_values())
      if (V.get()->getType()->isPointerTy())
        Worklist.push_back(V);
  } else if (const LoadInst *LI = dyn_cast<LoadInst>(I)) {
    // If a loaded pointer escapes, the pointer it was loaded from also escapes.
    // This is a key part of handling indirect escapes.
    Worklist.push_back(LI->getPointerOperand());
  }
  // For other instructions (e.g., binary operators), we stop propagation.
}

// Helper function to check if value is of supported type for Result set
static bool isResultTypeValue(const Value *V) {
  return isa<AllocaInst>(V) || isa<GlobalVariable>(V) || isa<GlobalAlias>(V) ||
         isa<Argument>(V) || isa<ConstantPointerNull>(V) ||
         isa<UndefValue>(V) || isa<PoisonValue>(V);
}

static bool fastPathCollectUnderlying(const Value *Curr,
                                      SmallPtrSetImpl<const Value *> &Result,
                                      SmallVectorImpl<const Value *> &Work) {
  SmallVector<const Value *, 4> Bases;
  getUnderlyingObjects(Curr, Bases);

  if (Bases.empty() || (Bases.size() == 1 && Bases[0] == Curr)) {
    Result.insert(Curr);
    return false;
  }

  bool IsAllFinalTypes = true;
  for (const Value *B : Bases) {
    assert(B && "getUnderlyingObjects must return non-null pointers");
    if (isResultTypeValue(B)) {
      Result.insert(B);
    } else if (B != Curr) {
      Work.push_back(B);
      IsAllFinalTypes = false;
    }
  }
  return IsAllFinalTypes;
}

static bool handleMemoryInstr(const Instruction *I,
                              SmallVectorImpl<const Value *> &Work) {
  // If it's a store, follow the stored value.
  if (const auto *SI = dyn_cast<StoreInst>(I)) {
    if (SI->getValueOperand())
      Work.push_back(SI->getValueOperand());
    return true;
  }

  // If it's a memcpy/memmove, use the src operand as source of data.
  if (const auto *MTI = dyn_cast<MemTransferInst>(I)) {
    Work.push_back(MTI->getSource());
    return true;
  }

  // Memset writes repeated byte value; follow the value (often a constant).
  if (const auto *MSI = dyn_cast<MemSetInst>(I)) {
    if (const Value *V = MSI->getValue())
      Work.push_back(V);
    return true;
  }

  return false;
}

// Add incoming unvisited MemoryAccesses of a MemoryPhi to MAWorkList.
static void
appendUnvisitedIncomingMAs(const MemoryPhi *MP,
                           SmallPtrSetImpl<const MemoryAccess *> &VisitedMA,
                           SmallVectorImpl<const MemoryAccess *> &MAWorkList) {
  for (const auto &U : MP->incoming_values()) {
    const MemoryAccess *InMA = cast<MemoryAccess>(U);
    if (VisitedMA.insert(InMA).second)
      MAWorkList.push_back(InMA);
  }
}

/*
static void
fallbackToGetUnderlyingObjects(const Value *Ptr,
                               SmallPtrSetImpl<const Value *> &Result) {
  SmallVector<const Value *, 4> PtrBases;
  getUnderlyingObjects(Ptr, PtrBases);
  for (const Value *Pb : PtrBases) {
    if (isResultTypeValue(Pb))
      Result.insert(Pb);
    else
      ;
  }
}
*/

/// Return the set of possible underlying objects (alloca/global/arg/const/...)
/// for V. Uses llvm::getUnderlyingObjects for cheap cases and MemorySSA for
/// following stores through loads.
SmallPtrSet<const Value *, 16>
getUnderlyingObjectsThroughLoads(const Value *V, MemorySSA *MSSA) {
  SmallPtrSet<const Value *, 16> Result;
  SmallPtrSet<const Value *, 32> VisitedValues;
  SmallPtrSet<const MemoryAccess *, 32> VisitedMA;

  SmallVector<const Value *, 32> Work;
  Work.push_back(V);

  assert(MSSA && "MemorySSA is required for this analysis");
  MemorySSAWalker *Walker = MSSA->getWalker();

  while (!Work.empty()) {
    const Value *CurrVal = Work.pop_back_val();
    if (!VisitedValues.insert(CurrVal).second)
      continue;

    // Fast-path: try ValueTracking to strip casts/geps and get candidate bases.
    if (fastPathCollectUnderlying(CurrVal, Result, Work))
      continue;

    // If CurrVal is a LoadInst, use MemorySSA to find the clobbering access
    if (const auto *LI = dyn_cast<LoadInst>(CurrVal)) {
      MemoryAccess *MA = MSSA->getMemoryAccess(LI);
      if (!MA) {
        Result.insert(LI);
        continue;
      }

      // Use MemorySSA's API to get the clobbering MemoryAccess.
      const MemoryAccess *Clobber = Walker->getClobberingMemoryAccess(MA);
      assert(Clobber &&
             "getClobberingMemoryAccess must return valid MemoryAccess");

      SmallVector<const MemoryAccess *, 32> MAWorkList;
      if (VisitedMA.insert(Clobber).second)
        MAWorkList.push_back(Clobber);

      while (!MAWorkList.empty()) {
        const MemoryAccess *CurrClobber = MAWorkList.pop_back_val();

        if (MSSA->isLiveOnEntryDef(CurrClobber))
          continue; // We are done

        if (const MemoryDef *MD = dyn_cast<MemoryDef>(CurrClobber)) {
          // If clobber is a MemoryDef, inspect its instruction.
          const Instruction *MI = MD->getMemoryInst();
          assert(MI && "MemoryDef must have an instruction");

          if (handleMemoryInstr(MI, Work))
            continue;

          // If it's a Call/Invoke (CallBase), try to be smarter via attributes.
          if (const auto *CB = dyn_cast<CallBase>(MI)) {
            // If the call only reads/doesn't access memory, it can't be a
            // defining write -> ignore.
            if (!(CB->onlyReadsMemory() || CB->doesNotAccessMemory()))
              Result.insert(CB); // Cannot do anymore with the Call
            continue;
          }
          // If the clobber instruction wasn't handled specially, push it for
          // normal analysis (will be processed with fastPath or inserted).
          Work.push_back(MI);
        } else if (const auto *MP = dyn_cast<MemoryPhi>(CurrClobber)) {
          // Iterate the incoming accesses and process each incoming MemoryAccess.
          appendUnvisitedIncomingMAs(MP, VisitedMA, MAWorkList);
        } else {
          llvm_unreachable("getClobberingMemoryAccess must return either "
                           "MemoryDef or MemoryPhi");
        }
      } // end while for MAWorkList
    } else {
      // Non-load instruction, we've already tried fastPath; just insert it as is
      Result.insert(CurrVal);
    }
  } // end while for Work
  return Result;
}

bool EscapeAnalysisInfo::solveEscapeFor(const Value &AllocationSite) {
  return true;
}

bool EscapeAnalysisInfo::isEscaping(const Value &Alloc) {
  // 1. Get the underlying object. This handles bitcasts, GEPs, and
  //    simple cases of PHIs and selects pointing to the same object.
  const Value *UnderlyingObj = getUnderlyingObjectAggressive(&Alloc);

  // 2. Check cache for a previously computed result.
  const auto CacheIt = Cache.find(UnderlyingObj);
  if (CacheIt != Cache.end())
    return CacheIt->second;

  // 3. If not in cache, run the analysis.
  LLVM_DEBUG(dbgs() << "EscapeAnalysis: Analyzing " << *UnderlyingObj << "\n");
  NumAllocationsAnalyzed++;

  // Lazily get other analyses from the FAM.
  // AAResults &AA = FAM.getResult<AAManager>(F);

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

  dbgs() << "EscapeAnalysis for function: " << F.getName() << "\n";

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
      dbgs() << "  Allocation " << I.getName() << ": "
             << (Escapes ? "ESCAPES" : "DOES NOT ESCAPE") << "\n";
    }
  }

  if (!HasInterestingAllocs)
    dbgs() << "  No allocations to analyze.\n";

  return PreservedAnalyses::all();
}

