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
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

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

  if (isa<GetElementPtrInst>(I) || isa<BitCastInst>(I) ||
      isa<SelectInst>(I)) {
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

bool EscapeAnalysisInfo::solveEscapeFor(const Value &AllocationSite) {
  // Find all initial escape points for the allocation.
  // We use LLVM's built-in CaptureTracking as a powerful first-pass filter.
  // It's fast but not flow-sensitive, so it might miss indirect escapes that
  // our dataflow analysis will catch.
  SmallVector<const Value *, 16> Worklist;
  DenseSet<const Value *> EscapedSet;

  SmallVector<Use *, 16> Uses;
  for (const Use &U : AllocationSite.uses())
    Uses.push_back(const_cast<Use*>(&U));

  while (!Uses.empty()) {
    Use *U = Uses.pop_back_val();
    Instruction *User = dyn_cast<Instruction>(U->getUser());
    // Some users (e.g. ConstantExpr) are not Instructions; skip them safely.
    if (!User)
      continue;

    // 1. Direct escape points
    if (isa<ReturnInst>(User)) return true;
    if (isa<StoreInst>(User) && U->get() == cast<StoreInst>(User)->getValueOperand()) {
        const StoreInst *SI = cast<StoreInst>(User);
        if (isa<GlobalVariable>(getUnderlyingObject(SI->getPointerOperand())))
            return true; // Stored to a global.
    }
    if (CallBase *CB = dyn_cast<CallBase>(User)) {
      if (!CB->isArgOperand(U) || CB->doesNotCapture(CB->getArgOperandNo(U))) {
        // Not a captured argument, continue.
      } else {
        return true; // Passed to a capturing function.
      }
    }

    // 2. Add indirect uses to the worklist for dataflow analysis
    // If the allocation is stored, the address where it's stored becomes
    // a source of potential escapes.
    if (isa<StoreInst>(User) &&
        U->get() == cast<StoreInst>(User)->getValueOperand())
      Worklist.push_back(cast<StoreInst>(User)->getPointerOperand());

    // 3. Propagate through pointer-like instructions
    for (const Use &UserUse : User->uses())
      Uses.push_back(const_cast<Use *>(&UserUse));
  }

  // Now, run the backward dataflow analysis for indirect escapes.
  // The worklist is seeded with pointers that store our allocation.
  unsigned Iter = 0;
  while (!Worklist.empty()) {
    // Safety valve against pathological def-use graphs.
    if (++Iter > WorklistLimit) {
      LLVM_DEBUG(dbgs() << "[EA] worklist limit exceeded (" << WorklistLimit
                        << ") for allocation site: " << AllocationSite << "\n");
      // Too complex: conservatively assume it escapes.
      return true;
    }

    const Value *V = Worklist.pop_back_val();
    if (!V || !V->getType()->isPointerTy()) continue;
    if (isa<Constant>(V)) continue; // Constants can't be part of a def-use
                                    // chain we care about.
    if (!EscapedSet.insert(V).second) continue; // Already processed.

    if (const auto *Arg = dyn_cast<Argument>(V))
      // If an escaped value can be traced back to a function argument,
      // it means the allocation was stored in a location pointed to by
      // that argument. Since we can't know where that argument points,
      // we must conservatively assume it escapes.
      return true;

    if (const auto *I = dyn_cast<Instruction>(V))
      applyTransferFunction(I, Worklist, EscapedSet);
  }

  return false;
}

bool EscapeAnalysisInfo::isEscaping(const Value &Allocation) {
  // 1. Get the underlying object. This handles bitcasts and GEPs.
  const Value *UnderlyingObj = getUnderlyingObject(&Allocation);

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

  for (Instruction &I : instructions(F)) {
    bool IsAllocation = false;
    if (isa<AllocaInst>(I)) {
      IsAllocation = true;
    } else if (const auto *CB = dyn_cast<CallBase>(&I)) {
      if (const Function *Callee = CB->getCalledFunction())
        if (Callee->getName() == "malloc")
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

