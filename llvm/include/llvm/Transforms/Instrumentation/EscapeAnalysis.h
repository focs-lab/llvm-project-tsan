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
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"

#include <memory>

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

  /// Backward transfer for pointer-like instructions (GEP/cast/select,
  /// PHI, load) that propagate the "escaped" property to operands.
  void applyTransferFunction(const Instruction *I,
                             SmallVectorImpl<const Value *> &Worklist,
                             DenseSet<const Value *> &EscapedSet);

  /// Solve escape for a single allocation site using backward dataflow.
  bool solveEscapeFor(const Value &Allocation);
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