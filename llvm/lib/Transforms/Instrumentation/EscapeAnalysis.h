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

#ifndef LLVM_ESCAPE_ANALYSIS_H
#define LLVM_ESCAPE_ANALYSIS_H

#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"

#include <memory>

namespace llvm {

/// EscapeAnalysis - A utility class that performs escape analysis on function
/// allocations. The analysis is performed lazily when requested for a specific
/// value and the results are cached.
class EscapeAnalysis {
public:
  /// Constructs an escape analysis utility for a given function.
  /// Requires a FunctionAnalysisManager to obtain other analyses like AA.
  EscapeAnalysis(Function &F, FunctionAnalysisManager &FAM);
  ~EscapeAnalysis();

  /// Main API method. Returns true if the given value (which must be an
  /// allocation site like alloca or malloc) can escape the function.
  /// An allocation is considered to escape if it is:
  /// - Returned from the function.
  /// - Stored into global memory.
  /// - Passed as an argument to a function that may capture it.
  /// - Stored into another heap object or an escaping allocation.
  bool isEscaping(const Value &Allocation);

private:
  // Using the PImpl idiom to hide implementation details from the header.
  struct Implementation;
  std::unique_ptr<Implementation> Impl;
};

} // end namespace llvm

#endif // LLVM_ESCAPE_ANALYSIS_H