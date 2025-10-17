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

  /// Return true if \p Alloc may escape the function.
  /// \param Alloc - Must be an allocation site (AllocaInst or heap allocation
  ///                call). Passing GEPs/bitcasts is not supported; use the base
  ///                allocation.
  /// \returns true if the allocation escapes or if \p Alloc is not an
  /// allocation site.
  bool isEscaping(const Value &Alloc);

  /// Print escape information for all allocations in the function
  void print(raw_ostream &OS);

  bool invalidate(Function &F, const PreservedAnalyses &PA,
                  FunctionAnalysisManager::Invalidator &Inv);

private:
  Function &F;
  FunctionAnalysisManager &FAM;
  DenseMap<const Value *, bool> Cache;

  MemorySSA *MSSA = nullptr;
  LoopInfo *LI = nullptr;
  AAResults *AA = nullptr;

  // getUnderlyingObjects(..., MaxLookup = 0) is assumed to mean "unbounded".
  // If upstream changes semantics, this must be revisited.
  static const unsigned VTMaxLookup = 0;

  /// Add P to Worklist if it doesn't exist in Seen
  template <class PtrT, class SetT, class WorklistT>
  static bool tryEnqueueIfNew(PtrT *P, SetT &Seen, WorklistT &Worklist);

  /// Try to use ValueTracking to find underlying objects.
  static bool tryValueTracking(const Value *V, LoopInfo *LI,
                               SmallVectorImpl<const Value *> &Work,
                               SmallPtrSetImpl<const Value *> &Enqueued);

  /// Add incoming unvisited MemoryAccesses of a MemoryPhi to MAWorkList.
  static void appendIncomingMAs(const MemoryPhi *MPhi,
                                SmallPtrSetImpl<MemoryAccess *> &VisitedMA,
                                SmallVectorImpl<MemoryAccess *> &MAWorkList,
                                MemoryLocation Loc, MemorySSAWalker *Walker);
  bool isNonAtomicNonVolatile(const Value *V);

  ///===- GetUnderlyingObjectsThroughLoads
  ///---------------------------------===//
  ///
  /// A stronger variant of `llvm::getUnderlyingObjects` that uses MemorySSA
  /// to chase defining writes and, when possible, look through loads. This is
  /// more precise (and potentially more expensive) than plain ValueTracking.
  ///
  /// \param Ptr        A pointer-typed value to analyze.
  /// \param MSSA     A valid, up-to-date MemorySSA for V. Must not be null.
  /// \param AA
  /// \param Result   Output set that will be populated with the results.
  ///                 The set is not cleared; new elements are inserted into it.
  /// \param LI       Optional LoopInfo used to improve reasoning about PHIs in
  ///                 loops in ValueTracking.
  ///
  /// \post
  ///  - `Result` is augmented with zero or more pointer-typed "terminal
  ///  sources" for `V`.
  ///  - A "terminal source" is a pointer value where this analysis
  ///  intentionally stops. Typical terminals include:
  ///      * `AllocaInst`, `GlobalVariable`, `Argument`, `ConstantPointerNull`.
  ///      * An SSA pointer value such as a `LoadInst` (or `PHINode`/`Select`
  ///        as returned by ValueTracking) when MemorySSA cannot prove a precise
  ///        defining write for the loaded bytes (e.g., memory is liveOnEntry,
  ///        written by an opaque call, clobbered by memset/atomics/volatile,
  ///        etc.).
  ///      * A call result if ValueTracking stops at a call that returns a
  ///      pointer
  ///        (i.e., the usual terminals that `getUnderlyingObjects` would
  ///        produce).
  ///
  /// \note
  ///  - If the analysis cannot find any sound terminal sources (e.g., due to a
  ///    cycle or lack of proof), it is permitted to insert nothing. However, to
  ///    match the spirit of `getUnderlyingObjects` and to keep clients
  ///    predictable, when the walk stops at a pointer-typed SSA value (e.g., a
  ///    `LoadInst`), this API is encouraged to insert that SSA value to
  ///    represent an "unknown terminal".
  ///  - The result is a may-set: `Result` contains all terminal candidates the
  ///    analysis can conservatively identify, not a single precise source.
  ///
  static void getUnderlyingObjectsThroughLoads(
      const Value *Ptr, MemorySSA *MSSA, AAResults *AA,
      SmallPtrSetImpl<const Value *> &Result, LoopInfo *LI = nullptr,
      bool *IsComplete = nullptr, unsigned MaxSteps = 10000);

  /// Checks whether a base location is externally visible (thus escapes).
  static bool isExternalObject(const Value *Base);

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
    bool doesStoreDestinationEscape(const Value *Dest);

    /// Analyze whether the pointer value stored by `Store` can escape
    bool doesStoredPointerEscapeViaLoads(
        const StoreInst *Store, SmallPtrSet<const Value *, 32> &ProcessingSet);
  };

  /// Solve escape for a single allocation site using backward dataflow.
  bool solveEscapeFor(const Value &Alloca,
                      SmallPtrSet<const Value *, 32> &ProcessingSet);

  // Helper function to detect heap allocations.
  // Required because isAllocationFn() requires the 'allockind' attribute,
  // which older Clang versions don't generate for malloc/calloc/etc.
  static bool isHeapAllocation(const CallBase *CB,
                               const TargetLibraryInfo *TLI);

  /// Helper function to detect allocation sites (malloc/new-like)
  /// Returns true if V is an Alloca or a call to a known heap alloc function.
  static bool isAllocationSite(const Value *V, const TargetLibraryInfo *TLI);
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
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM) const;
  static bool isRequired() { return true; }
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_INSTRUMENTATION_ESCAPEANALYSIS_H