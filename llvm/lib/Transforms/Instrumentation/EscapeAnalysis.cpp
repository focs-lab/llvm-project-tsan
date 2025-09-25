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

#include "EscapeAnalysis.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "escape-analysis"

using namespace llvm;

STATISTIC(NumAllocationsAnalyzed, "Number of allocation sites analyzed");
STATISTIC(NumAllocationsEscaped, "Number of allocation sites found to escape");
STATISTIC(NumWorklistIterations, "Number of worklist iterations");

// A debug option to limit the complexity of the analysis.
static cl::opt<unsigned>
WorklistLimit("escape-analysis-worklist-limit", cl::init(10000), cl::Hidden,
              cl::desc("Worklist limit for escape analysis solver"));

namespace {

/// EscapeAnalysisSolver - This class implements the actual backward dataflow
/// analysis for a single allocation site. It is hidden from the public API.
class EscapeAnalysisSolver {
  Function &F;
  AAResults &AA;

  SmallVector<const Value *, 16> Worklist;
  DenseSet<const Value *> EscapedSet;

public:
  EscapeAnalysisSolver(Function &F, AAResults &AA)
      : F(F), AA(AA) {}

  bool run(const Value &AllocationSite);

private:
  void applyTransferFunction(const Instruction &I);
  bool isMallocLike(const Value *V) const;
};

} // end anonymous namespace

/// The private implementation of the EscapeAnalysis class.
struct EscapeAnalysis::Implementation {
  Function &F;
  FunctionAnalysisManager &FAM;

  // Cache for analysis results. Key is the allocation site.
  DenseMap<const Value *, bool> Cache;

  Implementation(Function &F, FunctionAnalysisManager &FAM)
      : F(F), FAM(FAM) {}
};

EscapeAnalysis::EscapeAnalysis(Function &F, FunctionAnalysisManager &FAM)
    : Impl(new Implementation(F, FAM)) {}

EscapeAnalysis::~EscapeAnalysis() = default;

bool EscapeAnalysis::isEscaping(const Value &Allocation) {
  // 1. Get the underlying object. This handles bitcasts and GEPs.
  const Value *UnderlyingObject = getUnderlyingObject(&Allocation);

  // 2. Check cache for a previously computed result.
  auto CacheIt = Impl->Cache.find(UnderlyingObject);
  if (CacheIt != Impl->Cache.end()) {
    return CacheIt->second;
  }

  // 3. If not in cache, run the analysis.
  LLVM_DEBUG(dbgs() << "EscapeAnalysis: Analyzing " << *UnderlyingObject << "\n");
  NumAllocationsAnalyzed++;

  // Lazily get other analyses from the FAM.
  AAResults &AA = Impl->FAM.getResult<AAManager>(Impl->F);

  EscapeAnalysisSolver Solver(Impl->F, AA);
  bool Result = Solver.run(*UnderlyingObject);

  if (Result) {
    NumAllocationsEscaped++;
    LLVM_DEBUG(dbgs() << "  -> Result: ESCAPES\n");
  } else {
    LLVM_DEBUG(dbgs() << "  -> Result: DOES NOT ESCAPE\n");
  }

  // 4. Store result in cache and return.
  return Impl->Cache[UnderlyingObject] = Result;
}

//===----------------------------------------------------------------------===//
// EscapeAnalysisSolver Implementation
//===----------------------------------------------------------------------===//

bool EscapeAnalysisSolver::isMallocLike(const Value *V) const {
  if (const CallBase *CB = dyn_cast<CallBase>(V)) {
    if (Function *F = CB->getCalledFunction()) {
      if (F->getName() == "malloc" || F->getName() == "_Znwm" /* new */) {
        return true;
      }
    }
  }
  return false;
}

void EscapeAnalysisSolver::applyTransferFunction(const Instruction &I) {
  // This is a backward analysis. We check if the instruction's *result* is in
  // the EscapedSet. If so, we propagate the "escaped" property to its operands.

  if (!EscapedSet.count(&I)) {
    return; // This instruction doesn't produce an escaped value.
  }

  // The value produced by I escapes. Remove it from the set and add its
  // relevant operands, propagating the escaped property backward.
  EscapedSet.erase(&I);

  if (isa<GetElementPtrInst>(&I) || isa<BitCastInst>(&I) || isa<SelectInst>(&I)) {
    // Simple propagation: if a GEP/cast/select result escapes, the base
    // pointer/operands escape.
    for (const Value *Op : I.operands()) {
      if (Op->getType()->isPointerTy())
        Worklist.push_back(Op);
    }
  } else if (const PHINode *PN = dyn_cast<PHINode>(&I)) {
    // For a PHI node, all incoming values are considered to escape.
    for (const Value *V : PN->incoming_values()) {
        if (V->getType()->isPointerTy())
            Worklist.push_back(V);
    }
  } else if (const LoadInst *LI = dyn_cast<LoadInst>(&I)) {
    // If a loaded pointer escapes, the pointer it was loaded from also escapes.
    // This is a key part of handling indirect escapes.
    Worklist.push_back(LI->getPointerOperand());
  }
  // For other instructions (e.g., binary operators), we stop propagation.
}

bool EscapeAnalysisSolver::run(const Value &AllocationSite) {
  // Find all initial escape points for the allocation.
  // We use LLVM's built-in CaptureTracking as a powerful first-pass filter.
  // It's fast but not flow-sensitive, so it might miss indirect escapes that
  // our dataflow analysis will catch.

  SmallVector<Use *, 16> Uses;
  for (const Use &U : AllocationSite.uses())
    Uses.push_back(const_cast<Use*>(&U));

  while (!Uses.empty()) {
    Use *U = Uses.pop_back_val();
    Instruction *User = cast<Instruction>(U->getUser());

    // 1. Direct escape points
    if (isa<ReturnInst>(User)) return true;
    if (isa<StoreInst>(User) && U->get() == cast<StoreInst>(User)->getValueOperand()) {
        const StoreInst *SI = cast<StoreInst>(User);
        if (isa<GlobalVariable>(getUnderlyingObject(SI->getPointerOperand()))) {
            return true; // Stored to a global.
        }
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
    if (isa<StoreInst>(User) && U->get() == cast<StoreInst>(User)->getValueOperand()) {
      Worklist.push_back(cast<StoreInst>(User)->getPointerOperand());
    }

    // 3. Propagate through pointer-like instructions
    for (const Use &UserUse : User->uses()) {
        Uses.push_back(const_cast<Use*>(&UserUse));
    }
  }

  // Now, run the backward dataflow analysis for indirect escapes.
  // The worklist is seeded with pointers that store our allocation.
  while (!Worklist.empty()) {
    NumWorklistIterations++;
    if (NumWorklistIterations > WorklistLimit) {
        // Analysis is too complex, conservatively assume it escapes.
        return true;
    }

    const Value *V = Worklist.pop_back_val();
    if (!V || !V->getType()->isPointerTy()) continue;
    if (isa<Constant>(V)) continue; // Constants can't be part of a def-use chain we care about.
    if (!EscapedSet.insert(V).second) continue; // Already processed.

    if (const Argument *Arg = dyn_cast<Argument>(V)) {
        // If an escaped value can be traced back to a function argument,
        // it means the allocation was stored in a location pointed to by
        // that argument. Since we can't know where that argument points,
        // we must conservatively assume it escapes.
        return true;
    }

    if (const Instruction *I = dyn_cast<Instruction>(V)) {
      applyTransferFunction(*I);
    }
  }

  return false;
}