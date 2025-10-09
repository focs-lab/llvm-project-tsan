//===- EscapeAnalysis.h - Intraprocedural Escape Analysis -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the interface for a simple, conservative intraprocedural
// escape analysis. It is designed as a helper utility for other passes, like
// ThreadSanitizer, to determine if an allocation escapes the context of its
// containing function.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_ESCAPEANALYSIS_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_ESCAPEANALYSIS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

/// EscapeAnalysisInfo - This class implements the actual backward dataflow
/// analysis for a function; queries are per allocation site.
///
/// This is a lightweight, intraprocedural and conservative analysis intended
/// to help instrumentation passes (e.g. ThreadSanitizer) skip objects that do
/// not escape the function scope. The main query is \c isEscaping(Value&),
/// which answers whether an allocation site (alloca/malloc-like) may escape
/// the current function. Results are memoized per underlying object.
struct EscapeAnalysisInfo {
  /// Constructs an escape analysis utility for a given function.
  /// Requires a FunctionAnalysisManager to obtain other analyses like AA.
  EscapeAnalysisInfo(Function &F, FunctionAnalysisManager &FAM)
      : F(F), FAM(FAM) {};
  ~EscapeAnalysisInfo() = default;

  /// Return true if \p Allocation may escape the function.
  bool isEscaping(const Value &Alloc);

  bool invalidate(Function &F, const PreservedAnalyses &PA,
                  FunctionAnalysisManager::Invalidator &Inv);

private:
  Function &F;
  FunctionAnalysisManager &FAM;
  DenseMap<const Value *, bool> Cache;

  MemorySSA *MSSA = nullptr;
  LoopInfo *LI = nullptr;

  /// Custom CaptureTracker for escape analysis
  class EscapeCaptureTracker : public CaptureTracker {
  public:
    EscapeCaptureTracker(EscapeAnalysisInfo &EAI,
                         const SmallPtrSet<const Value *, 32> &ProcessingSet)
        : EAI(EAI), ProcessingSet(ProcessingSet) {}

    void tooManyUses() override { Escaped = true; }
    bool shouldExplore(const Use *U) override;
    Action captured(const Use *U, UseCaptureInfo CI) override;
    bool hasEscaped() const { return Escaped; }

  private:
    EscapeAnalysisInfo &EAI;
    SmallPtrSet<const Value *, 32> ProcessingSet;
    bool Escaped = false;

    /// Analyze if storing to destination causes escape
    bool doesStoreDestinationEscape(const StoreInst *SI);
  };

  bool analyzeStoreDestEscapes(SmallVector<const Value *, 32> Worklist,
                               SmallPtrSet<const Value *, 32> Visited,
                               const Value *Dest);

  /// Solve escape for a single allocation site using backward dataflow.
  bool solveEscapeFor(const Value &Allocation,
                      SmallPtrSet<const Value *, 32> &ProcessingSet);
};

/// EscapeAnalysisInfo wrapper for the new pass manager.
class EscapeAnalysis : public AnalysisInfoMixin<EscapeAnalysis> {
  friend AnalysisInfoMixin<EscapeAnalysis>;
  static AnalysisKey Key;

public:
  using Result = EscapeAnalysisInfo;
  static Result run(Function &F, FunctionAnalysisManager &FAM);
};

/// Printer pass for the \c EscapeAnalysis results.
class EscapeAnalysisPrinterPass
    : public PassInfoMixin<EscapeAnalysisPrinterPass> {
  raw_ostream &OS;

public:
  explicit EscapeAnalysisPrinterPass(raw_ostream &OS) : OS(OS) {}
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) const;
  static bool isRequired() { return true; }
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_INSTRUMENTATION_ESCAPEANALYSIS_H