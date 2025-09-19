//===-- ThreadSanitizer.cpp - race detector -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer, a race detector.
//
// The tool is under development, for the details about previous versions see
// http://code.google.com/p/data-race-test
//
// The instrumentation phase is quite simple:
//   - Insert calls to run-time library before every memory access.
//      - Optimizations may apply to avoid instrumenting some of the accesses.
//   - Insert calls at function entry/exit.
// The rest is handled by the run-time library.
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation/ThreadSanitizer.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/EscapeEnumerator.h"
#include "llvm/Transforms/Utils/Instrumentation.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

#define DEBUG_TYPE "tsan"

static cl::opt<bool> ClInstrumentMemoryAccesses(
    "tsan-instrument-memory-accesses", cl::init(true),
    cl::desc("Instrument memory accesses"), cl::Hidden);
static cl::opt<bool>
    ClInstrumentFuncEntryExit("tsan-instrument-func-entry-exit", cl::init(true),
                              cl::desc("Instrument function entry and exit"),
                              cl::Hidden);
static cl::opt<bool> ClHandleCxxExceptions(
    "tsan-handle-cxx-exceptions", cl::init(true),
    cl::desc("Handle C++ exceptions (insert cleanup blocks for unwinding)"),
    cl::Hidden);
static cl::opt<bool> ClInstrumentAtomics("tsan-instrument-atomics",
                                         cl::init(true),
                                         cl::desc("Instrument atomics"),
                                         cl::Hidden);
static cl::opt<bool> ClInstrumentMemIntrinsics(
    "tsan-instrument-memintrinsics", cl::init(true),
    cl::desc("Instrument memintrinsics (memset/memcpy/memmove)"), cl::Hidden);
static cl::opt<bool> ClDistinguishVolatile(
    "tsan-distinguish-volatile", cl::init(false),
    cl::desc("Emit special instrumentation for accesses to volatiles"),
    cl::Hidden);
static cl::opt<bool> ClInstrumentReadBeforeWrite(
    "tsan-instrument-read-before-write", cl::init(false),
    cl::desc("Do not eliminate read instrumentation for read-before-writes"),
    cl::Hidden);
static cl::opt<bool> ClCompoundReadBeforeWrite(
    "tsan-compound-read-before-write", cl::init(false),
    cl::desc("Emit special compound instrumentation for reads-before-writes"),
    cl::Hidden);
static cl::opt<bool>
    ClOmitNonCaptured("tsan-omit-by-pointer-capturing", cl::init(true),
                      cl::desc("Omit accesses due to pointer capturing"),
                      cl::Hidden);
static cl::opt<bool>
    ClUseDominanceAnalysis("tsan-use-dominance-analysis", cl::init(false),
                           cl::desc("Eliminate duplicating instructions which "
                                    "(post)dominates given instruction"),
                           cl::Hidden);

STATISTIC(NumInstrumentedReads, "Number of instrumented reads");
STATISTIC(NumInstrumentedWrites, "Number of instrumented writes");
STATISTIC(NumOmittedReadsBeforeWrite,
          "Number of reads ignored due to following writes");
STATISTIC(NumAccessesWithBadSize, "Number of accesses with bad size");
STATISTIC(NumInstrumentedVtableWrites, "Number of vtable ptr writes");
STATISTIC(NumInstrumentedVtableReads, "Number of vtable ptr reads");
STATISTIC(NumOmittedReadsFromConstantGlobals,
          "Number of reads from constant globals");
STATISTIC(NumOmittedReadsFromVtable, "Number of vtable reads");
STATISTIC(NumOmittedNonCaptured, "Number of accesses ignored due to capturing");
STATISTIC(NumOmittedByDominance,
          "Number of accesses ignored due to dominance");
STATISTIC(NumOmittedByPostDominance,
          "Number of accesses ignored due to post-dominance");

const char kTsanModuleCtorName[] = "tsan.module_ctor";
const char kTsanInitName[] = "__tsan_init";

namespace {
// Internal Instruction wrapper that contains more information about the
// Instruction from prior analysis.
struct InstructionInfo {
  // Instrumentation emitted for this instruction is for a compounded set of
  // read and write operations in the same basic block.
  static constexpr unsigned kCompoundRW = (1U << 0);

  explicit InstructionInfo(Instruction *Inst) : Inst(Inst) {}

  Instruction *Inst;
  unsigned Flags = 0;
};

/// A helper class to encapsulate the logic for eliminating redundant
/// instrumentation based on dominance analysis.
///
/// This class takes a list of all memory access instructions that are candidates
/// for instrumentation. It prunes instructions that are (post-)dominated by
/// another access to the same memory location, provided that the path between
/// them is "clear" of any dangerous instructions (like function calls or
/// synchronization primitives).
class DominanceBasedElimination {
public:
  /// \param AllInstr The vector of instructions to analyze. This vector is
  ///                 modified in-place.
  /// \param DT The Dominator Tree for the current function.
  /// \param PDT The Post-Dominator Tree for the current function.
  /// \param AA The results of Alias Analysis.
  /// \param TLI Target Library Info, used to identify dangerous calls.
  DominanceBasedElimination(SmallVectorImpl<InstructionInfo> &AllInstr,
                            DominatorTree &DT, PostDominatorTree &PDT,
                            AAResults &AA)
      : AllInstr(AllInstr), DT(DT), PDT(PDT), AA(AA) {}
      // const TargetLibraryInfo &TLI), TLI(TLI) {}

  /// Runs the analysis and prunes redundant instructions.
  /// It sequentially applies elimination based on dominance and post-dominance.
  void run() {
    eliminate</*IsPostDom=*/false>(); // Dominance-based elimination
    eliminate</*IsPostDom=*/true>();  // Post-dominance-based elimination
  }

private:
  /// Checks if an instruction is "dangerous" from TSan's perspective.
  /// Dangerous instructions include function calls, atomics, and fences.
  ///
  /// \param Inst The instruction to check.
  /// \return true if the instruction is dangerous.
  static bool isInstrSafe(const Instruction *Inst);

  /// Checks if the path between two instructions is "clear", i.e., it does not
  /// contain any dangerous instructions that could alter the thread
  /// synchronization state.
  /// \param StartInst The starting (dominating) instruction.
  /// \param EndInst The ending (dominated) instruction.
  /// \return true if the path is clear.
  template <bool IsPostDom>
  static bool
  isPathClear(Instruction *StartInst, Instruction *EndInst,
              const DominatorTreeBase<BasicBlock, IsPostDom> *DTBase);

  /// A helper function to create a map from Instruction* to its index
  /// in the AllInstr vector for fast lookups.
  DenseMap<Instruction *, size_t> createInstrToIndexMap() const;

  /// Attempts to find a dominating instruction that can eliminate the need to
  /// instrument instruction i
  /// \param DTBase The dominator (post-dominator) tree being used
  /// \param InstrToIndexMap Maps instructions to their indices in the AllInstr
  /// \param ToRemove Vector tracking which instructions can be eliminated
  /// \returns true if a dominating instruction was found that eliminates i
  template <bool IsPostDom>
  bool findAndMarkDominatingInstr(
      size_t i, const DominatorTreeBase<BasicBlock, IsPostDom> *DTBase,
      const DenseMap<Instruction *, size_t> &InstrToIndexMap,
      SmallVectorImpl<bool> &ToRemove);

  /// The core elimination logic. Templated to work with both Dominators
  /// and Post-Dominators.
  template <bool IsPostDom>
  void eliminate();

  /// A reference to the vector of instructions that we modify.
  SmallVectorImpl<InstructionInfo> &AllInstr;

  /// References to the required analysis results.
  DominatorTree &DT;
  PostDominatorTree &PDT;
  AAResults &AA;
  // const TargetLibraryInfo &TLI;
};

/// ThreadSanitizer: instrument the code in module to find races.
///
/// Instantiating ThreadSanitizer inserts the tsan runtime library API function
/// declarations into the module if they don't exist already. Instantiating
/// ensures the __tsan_init function is in the list of global constructors for
/// the module.
struct ThreadSanitizer {
  ThreadSanitizer(const TargetLibraryInfo &TLI_, DominatorTree *DT_,
                  PostDominatorTree *PDT_, AAResults *AA_)
      : TLI(TLI_), DT(DT_), PDT(PDT_), AA(AA_) {
    // Check options and warn user.
    if (ClInstrumentReadBeforeWrite && ClCompoundReadBeforeWrite) {
      errs()
          << "warning: Option -tsan-compound-read-before-write has no effect "
             "when -tsan-instrument-read-before-write is set.\n";
    }
  }

  bool sanitizeFunction(Function &F);

private:

  void initialize(Module &M, const TargetLibraryInfo &TLI);
  bool instrumentLoadOrStore(const InstructionInfo &II, const DataLayout &DL);
  bool instrumentAtomic(Instruction *I, const DataLayout &DL);
  bool instrumentMemIntrinsic(Instruction *I);
  void chooseInstructionsToInstrument(SmallVectorImpl<Instruction *> &Local,
                                      SmallVectorImpl<InstructionInfo> &All,
                                      const DataLayout &DL);
  bool addrPointsToConstantData(Value *Addr);
  int getMemoryAccessFuncIndex(Type *OrigTy, Value *Addr, const DataLayout &DL);
  void InsertRuntimeIgnores(Function &F);

  const TargetLibraryInfo &TLI;
  DominatorTree *DT = nullptr;
  PostDominatorTree *PDT = nullptr;
  AAResults *AA = nullptr;

  Type *IntptrTy;
  FunctionCallee TsanFuncEntry;
  FunctionCallee TsanFuncExit;
  FunctionCallee TsanIgnoreBegin;
  FunctionCallee TsanIgnoreEnd;
  // Accesses sizes are powers of two: 1, 2, 4, 8, 16.
  static const size_t kNumberOfAccessSizes = 5;
  FunctionCallee TsanRead[kNumberOfAccessSizes];
  FunctionCallee TsanWrite[kNumberOfAccessSizes];
  FunctionCallee TsanUnalignedRead[kNumberOfAccessSizes];
  FunctionCallee TsanUnalignedWrite[kNumberOfAccessSizes];
  FunctionCallee TsanVolatileRead[kNumberOfAccessSizes];
  FunctionCallee TsanVolatileWrite[kNumberOfAccessSizes];
  FunctionCallee TsanUnalignedVolatileRead[kNumberOfAccessSizes];
  FunctionCallee TsanUnalignedVolatileWrite[kNumberOfAccessSizes];
  FunctionCallee TsanCompoundRW[kNumberOfAccessSizes];
  FunctionCallee TsanUnalignedCompoundRW[kNumberOfAccessSizes];
  FunctionCallee TsanAtomicLoad[kNumberOfAccessSizes];
  FunctionCallee TsanAtomicStore[kNumberOfAccessSizes];
  FunctionCallee TsanAtomicRMW[AtomicRMWInst::LAST_BINOP + 1]
                              [kNumberOfAccessSizes];
  FunctionCallee TsanAtomicCAS[kNumberOfAccessSizes];
  FunctionCallee TsanAtomicThreadFence;
  FunctionCallee TsanAtomicSignalFence;
  FunctionCallee TsanVptrUpdate;
  FunctionCallee TsanVptrLoad;
  FunctionCallee MemmoveFn, MemcpyFn, MemsetFn;
};

//-----------------------------------------------------------------------------
// DominanceBasedElimination Implementation
//-----------------------------------------------------------------------------

bool isTsanAtomic(const Instruction *I) {
  // TODO: Ask TTI whether synchronization scope is between threads.
  auto SSID = getAtomicSyncScopeID(I);
  if (!SSID)
    return false;
  if (isa<LoadInst>(I) || isa<StoreInst>(I))
    return *SSID != SyncScope::SingleThread;
  return true;
}

bool DominanceBasedElimination::isInstrSafe(const Instruction *Inst) {
  // 1. Atomic operations with inter-thread communication are the primary
  //    source of synchronization and are NEVER safe.
  if (isTsanAtomic(Inst))
    return false;

  // 2. Check function calls.
  if (const auto *CB = dyn_cast<CallBase>(Inst)) {
    if (const Function *Callee = CB->getCalledFunction()) {
      // If a function is known to be sync-free (e.g., @llvm.sqrt), then false.
      // Otherwise - true.
      if (Callee->doesNotAccessMemory() || Callee->onlyReadsMemory() ||
          Callee->hasNoSync())
        return true;
      // Conservatively: any non-intrinsic and not explicitly safe call is
      // dangerous
      return false;
    }
    // Indirect call - always dangerous for simplicity
    return false;
  }
  // 3. All other instructions are considered safe because they do not,
  // by themselves, create happens-before relationships
  return true;
}

template <bool IsPostDom>
bool DominanceBasedElimination::isPathClear(
    Instruction *StartInst, Instruction *EndInst,
    const DominatorTreeBase<BasicBlock, IsPostDom> *DTBase) {
  LLVM_DEBUG(dbgs() << "Checking path from " << *StartInst << " to " << *EndInst
                    << "\n");
  const BasicBlock *StartBB = StartInst->getParent();
  LLVM_DEBUG(dbgs() << "StartBB: " << StartBB->getName() << "\n");

  // 1. Check instructions in StartBB after StartInst
  for (const Instruction *I = StartInst->getNextNode();
       I && I->getParent() == StartBB; I = I->getNextNode()) {
    LLVM_DEBUG(dbgs() << "\tisPathClear -- Checking 1: " << *I << "\n");
    if (!isInstrSafe(I))
      return false;
    if (I == EndInst)
      // If both StartInst and EndInst belongs to the same BB, we are done.
      return true; // Reached target instruction in the same block
  }

  // 2. Check blocks on the dominance path between DomBB and CurrBB (excluding
  // DomBB, excluding CurrBB) Traverse up from CurrBB along the immediate
  // dominator tree until DomBB is reached
  const BasicBlock *EndBB = EndInst->getParent();
  const BasicBlock *CurBB = !IsPostDom ? EndBB : StartBB;
  const BasicBlock *DomBB = !IsPostDom ? StartBB : EndBB;
  const DomTreeNode *CurrNode = DTBase->getNode(CurBB);
  assert(CurrNode && "DomNode not found");

  for (DomTreeNode *Node = CurrNode->getIDom();
       Node && Node->getBlock() && Node->getBlock() != DomBB;
       Node = Node->getIDom()) {
    BasicBlock *IntermediateBB = Node->getBlock();
    LLVM_DEBUG(dbgs() << "Inter IDom BB " << IntermediateBB->getName() << "\n");
    for (const Instruction &InterI : *IntermediateBB) {
      LLVM_DEBUG(dbgs() << "\tisPathClear -- Checking 2: " << InterI << "\n");
      if (!isInstrSafe(&InterI))
        return false;
    }
  }

  // 3. Check instructions in CurrBB before CurrInst
  for (const Instruction &I : *EndBB) {
    LLVM_DEBUG(dbgs() << "\tisPathClear -- Checking 3: " << I << "\n");
    if (&I == EndInst)
      break;
    if (!isInstrSafe(&I))
      return false;
  }
  return true;
}

DenseMap<Instruction *, size_t>
DominanceBasedElimination::createInstrToIndexMap() const {
  DenseMap<Instruction *, size_t> InstrToIndexMap;
  for (size_t i = 0; i < AllInstr.size(); ++i)
    if (AllInstr[i].Inst) // Ensure the instruction hasn't been removed
      InstrToIndexMap[AllInstr[i].Inst] = i;
  return InstrToIndexMap;
}

template <bool IsPostDom>
bool DominanceBasedElimination::findAndMarkDominatingInstr(
    size_t i, const DominatorTreeBase<BasicBlock, IsPostDom> *DTBase,
    const DenseMap<Instruction *, size_t> &InstrToIndexMap,
    SmallVectorImpl<bool> &ToRemove) {
  const InstructionInfo &CurrII = AllInstr[i];
  Instruction *CurrInst = CurrII.Inst;
  const BasicBlock *CurrBB = CurrInst->getParent();
  Value *CurrAddr = getLoadStorePointerOperand(CurrInst);
  assert(CurrAddr && "Should not happen for load/store");
  const Value *CurrUnderlyingObj = getUnderlyingObject(CurrAddr);

  DomTreeNode *CurrDTNode = DTBase->getNode(CurrBB);
  if (!CurrDTNode)
    return false;

  // Traverse up the dominator tree
  const auto *IDomNode = CurrDTNode;
  while (IDomNode && IDomNode->getBlock()) {
    BasicBlock *DomBB = IDomNode->getBlock();

    // Look for a suitable dominating instrumented instruction in DomBB
    auto StartIt = DomBB->begin();
    auto EndIt = DomBB->end();
    if (CurrBB == DomBB) {
      if constexpr (IsPostDom)
        StartIt = CurrInst->getIterator();
      else
        EndIt = CurrInst->getIterator();
    }

    for (auto InstIt = StartIt; InstIt != EndIt; ++InstIt) {
      Instruction &PotentialDomInst = *InstIt;
      LLVM_DEBUG(dbgs() << "PotentialDomInst: " << PotentialDomInst << "\n");
      if (&PotentialDomInst == CurrInst)
        continue;

      // Check if PotentialDomInst is dominating and instrumented
      const auto It = InstrToIndexMap.find(&PotentialDomInst);
      if ((It == InstrToIndexMap.end()) || ToRemove[It->second])
        // Not found in AllInstr or already marked for removal
        continue;

      const size_t DomIndex = It->second;
      InstructionInfo &DomII = AllInstr[DomIndex];
      Instruction *DomInst = DomII.Inst;

      const Value *DomAddr = getLoadStorePointerOperand(DomInst);
      const Value *DomUnderlyingObj = getUnderlyingObject(DomAddr);
      if (CurrUnderlyingObj == DomUnderlyingObj ||
          (CurrUnderlyingObj && DomUnderlyingObj &&
           AA.isMustAlias(CurrAddr, DomAddr))) {
        auto isWriteOperation = [](const InstructionInfo &II) {
          return isa<StoreInst>(II.Inst) ||
                 (II.Flags & InstructionInfo::kCompoundRW);
        };
        const bool CurrIsWrite = isWriteOperation(CurrII);
        const bool DomIsWrite = isWriteOperation(DomII);

        // Check compatibility logic (DomInst covers CurrInst):
        // 1. If DomInst is a 'write', it covers both read and write of
        // CurrInst.
        // 2. If DomInst is a 'read', it only covers a read of CurrInst.
        if (DomIsWrite || !CurrIsWrite) {
          // Check the path to/from CurrInst from/to DomInst
          bool IsPathClear = false;
          if constexpr (IsPostDom)
            IsPathClear = this->isPathClear<true>(CurrInst, DomInst, DTBase);
          else
            IsPathClear = this->isPathClear<false>(DomInst, CurrInst, DTBase);

          if (IsPathClear) {
            LLVM_DEBUG(dbgs()
                       << "TSAN: Omitting instrumentation for: " << *CurrInst
                       << " ((post-)dominated and covered by: " << *DomInst
                       << ")\n");
            ToRemove[i] = true;
            // Found a (post)dominator, move to the next Inst
            return true;
          }
        }
      }
    }
    IDomNode = IDomNode->getIDom();
  }
  return false;
}

template <bool IsPostDom>
void DominanceBasedElimination::eliminate() {
  LLVM_DEBUG(dbgs() << "====================================\n=== Starting "
                    << (IsPostDom ? "post-" : "")
                    << "dominance-based analysis ===\n");
  if (AllInstr.empty())
    return;

  // DominatorTreeBase<BasicBlock, IsPostDom> &DTBase = IsPostDom ? PDT : DT;
  DominatorTreeBase<BasicBlock, IsPostDom> *DTBase;
  if constexpr (IsPostDom)
    DTBase = &PDT;
  else
    DTBase = &DT;

  SmallVector<bool, 16> ToRemove(AllInstr.size(), false);
  unsigned RemovedCount = 0;

  // Create a map from Instruction* to its index in the AllInstr vector.
  DenseMap<Instruction *, size_t> InstrToIndexMap = createInstrToIndexMap();

  for (size_t i = 0; i < AllInstr.size(); ++i) {
    if (ToRemove[i])
      continue;

    if (findAndMarkDominatingInstr<IsPostDom>(i, DTBase, InstrToIndexMap,
                                              ToRemove))
      RemovedCount++;
  }

  LLVM_DEBUG(
      dbgs() << "\n=== Final list of instructions and their status ===\n";
      for (size_t i = 0; i < AllInstr.size(); ++i) dbgs()
      << "[" << (ToRemove[i] ? "REMOVED" : "KEPT") << "]\t" << *AllInstr[i].Inst
      << "\n");

  if (RemovedCount > 0) {
    LLVM_DEBUG(dbgs() << "\n=== Updating final instruction list ===\n"
                      << "Original size: " << AllInstr.size() << "\n"
                      << "Instructions to remove: " << RemovedCount << "\n"
                      << "Remaining instructions: "
                      << (AllInstr.size() - RemovedCount) << "\n");
    auto ToRemoveIter = ToRemove.begin();
    erase_if(AllInstr, [&](const InstructionInfo &) {
      return *ToRemoveIter++;
    });

    // erase_if(AllInstr, [&](const InstructionInfo &II) {
    //   size_t Index = InstrIndexMap.lookup(II.Inst);
    //   return ToRemove[Index];
    // });

    // SmallVector<InstructionInfo, 8> NewAllInstr;
    // NewAllInstr.reserve(AllInstr.size() - RemovedCount);
    // for (size_t k = 0; k < AllInstr.size(); ++k)
    //   if (!ToRemove[k])
    //     NewAllInstr.push_back(AllInstr[k]);
    // AllInstr.swap(NewAllInstr);
    if constexpr (IsPostDom)
      NumOmittedByPostDominance += RemovedCount; // Statistics
    else
      NumOmittedByDominance += RemovedCount; // Statistics
  }
  LLVM_DEBUG(dbgs() << "=== Dominance analysis complete ===\n"
                    << "===========================================\n");
}

//-----------------------------------------------------------------------------
// ThreadSanitizer Implementation
//-----------------------------------------------------------------------------

void insertModuleCtor(Module &M) {
  getOrCreateSanitizerCtorAndInitFunctions(
      M, kTsanModuleCtorName, kTsanInitName, /*InitArgTypes=*/{},
      /*InitArgs=*/{},
      // This callback is invoked when the functions are created the first
      // time. Hook them into the global ctors list in that case:
      [&](Function *Ctor, FunctionCallee) { appendToGlobalCtors(M, Ctor, 0); });
}
}  // namespace

PreservedAnalyses ThreadSanitizerPass::run(Function &F,
                                           FunctionAnalysisManager &FAM) {
  auto *DT = ClUseDominanceAnalysis ? &FAM.getResult<DominatorTreeAnalysis>(F)
                                    : nullptr;
  auto *PDT = ClUseDominanceAnalysis
                  ? &FAM.getResult<PostDominatorTreeAnalysis>(F)
                  : nullptr;
  auto *AA = ClUseDominanceAnalysis ? &FAM.getResult<AAManager>(F) : nullptr;

  ThreadSanitizer TSan(FAM.getResult<TargetLibraryAnalysis>(F), DT, PDT, AA);
  if (TSan.sanitizeFunction(F))
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}

PreservedAnalyses ModuleThreadSanitizerPass::run(Module &M,
                                                 ModuleAnalysisManager &MAM) {
  // Return early if nosanitize_thread module flag is present for the module.
  if (checkIfAlreadyInstrumented(M, "nosanitize_thread"))
    return PreservedAnalyses::all();
  insertModuleCtor(M);
  return PreservedAnalyses::none();
}
void ThreadSanitizer::initialize(Module &M, const TargetLibraryInfo &TLI) {
  const DataLayout &DL = M.getDataLayout();
  LLVMContext &Ctx = M.getContext();
  IntptrTy = DL.getIntPtrType(Ctx);

  IRBuilder<> IRB(Ctx);
  AttributeList Attr;
  Attr = Attr.addFnAttribute(Ctx, Attribute::NoUnwind);
  // Initialize the callbacks.
  TsanFuncEntry = M.getOrInsertFunction("__tsan_func_entry", Attr,
                                        IRB.getVoidTy(), IRB.getPtrTy());
  TsanFuncExit =
      M.getOrInsertFunction("__tsan_func_exit", Attr, IRB.getVoidTy());
  TsanIgnoreBegin = M.getOrInsertFunction("__tsan_ignore_thread_begin", Attr,
                                          IRB.getVoidTy());
  TsanIgnoreEnd =
      M.getOrInsertFunction("__tsan_ignore_thread_end", Attr, IRB.getVoidTy());
  IntegerType *OrdTy = IRB.getInt32Ty();
  for (size_t i = 0; i < kNumberOfAccessSizes; ++i) {
    const unsigned ByteSize = 1U << i;
    const unsigned BitSize = ByteSize * 8;
    std::string ByteSizeStr = utostr(ByteSize);
    std::string BitSizeStr = utostr(BitSize);
    SmallString<32> ReadName("__tsan_read" + ByteSizeStr);
    TsanRead[i] = M.getOrInsertFunction(ReadName, Attr, IRB.getVoidTy(),
                                        IRB.getPtrTy());

    SmallString<32> WriteName("__tsan_write" + ByteSizeStr);
    TsanWrite[i] = M.getOrInsertFunction(WriteName, Attr, IRB.getVoidTy(),
                                         IRB.getPtrTy());

    SmallString<64> UnalignedReadName("__tsan_unaligned_read" + ByteSizeStr);
    TsanUnalignedRead[i] = M.getOrInsertFunction(
        UnalignedReadName, Attr, IRB.getVoidTy(), IRB.getPtrTy());

    SmallString<64> UnalignedWriteName("__tsan_unaligned_write" + ByteSizeStr);
    TsanUnalignedWrite[i] = M.getOrInsertFunction(
        UnalignedWriteName, Attr, IRB.getVoidTy(), IRB.getPtrTy());

    SmallString<64> VolatileReadName("__tsan_volatile_read" + ByteSizeStr);
    TsanVolatileRead[i] = M.getOrInsertFunction(
        VolatileReadName, Attr, IRB.getVoidTy(), IRB.getPtrTy());

    SmallString<64> VolatileWriteName("__tsan_volatile_write" + ByteSizeStr);
    TsanVolatileWrite[i] = M.getOrInsertFunction(
        VolatileWriteName, Attr, IRB.getVoidTy(), IRB.getPtrTy());

    SmallString<64> UnalignedVolatileReadName("__tsan_unaligned_volatile_read" +
                                              ByteSizeStr);
    TsanUnalignedVolatileRead[i] = M.getOrInsertFunction(
        UnalignedVolatileReadName, Attr, IRB.getVoidTy(), IRB.getPtrTy());

    SmallString<64> UnalignedVolatileWriteName(
        "__tsan_unaligned_volatile_write" + ByteSizeStr);
    TsanUnalignedVolatileWrite[i] = M.getOrInsertFunction(
        UnalignedVolatileWriteName, Attr, IRB.getVoidTy(), IRB.getPtrTy());

    SmallString<64> CompoundRWName("__tsan_read_write" + ByteSizeStr);
    TsanCompoundRW[i] = M.getOrInsertFunction(
        CompoundRWName, Attr, IRB.getVoidTy(), IRB.getPtrTy());

    SmallString<64> UnalignedCompoundRWName("__tsan_unaligned_read_write" +
                                            ByteSizeStr);
    TsanUnalignedCompoundRW[i] = M.getOrInsertFunction(
        UnalignedCompoundRWName, Attr, IRB.getVoidTy(), IRB.getPtrTy());

    Type *Ty = Type::getIntNTy(Ctx, BitSize);
    Type *PtrTy = PointerType::get(Ctx, 0);
    SmallString<32> AtomicLoadName("__tsan_atomic" + BitSizeStr + "_load");
    TsanAtomicLoad[i] =
        M.getOrInsertFunction(AtomicLoadName,
                              TLI.getAttrList(&Ctx, {1}, /*Signed=*/true,
                                              /*Ret=*/BitSize <= 32, Attr),
                              Ty, PtrTy, OrdTy);

    // Args of type Ty need extension only when BitSize is 32 or less.
    using Idxs = std::vector<unsigned>;
    Idxs Idxs2Or12   ((BitSize <= 32) ? Idxs({1, 2})       : Idxs({2}));
    Idxs Idxs34Or1234((BitSize <= 32) ? Idxs({1, 2, 3, 4}) : Idxs({3, 4}));
    SmallString<32> AtomicStoreName("__tsan_atomic" + BitSizeStr + "_store");
    TsanAtomicStore[i] = M.getOrInsertFunction(
        AtomicStoreName,
        TLI.getAttrList(&Ctx, Idxs2Or12, /*Signed=*/true, /*Ret=*/false, Attr),
        IRB.getVoidTy(), PtrTy, Ty, OrdTy);

    for (unsigned Op = AtomicRMWInst::FIRST_BINOP;
         Op <= AtomicRMWInst::LAST_BINOP; ++Op) {
      TsanAtomicRMW[Op][i] = nullptr;
      const char *NamePart = nullptr;
      if (Op == AtomicRMWInst::Xchg)
        NamePart = "_exchange";
      else if (Op == AtomicRMWInst::Add)
        NamePart = "_fetch_add";
      else if (Op == AtomicRMWInst::Sub)
        NamePart = "_fetch_sub";
      else if (Op == AtomicRMWInst::And)
        NamePart = "_fetch_and";
      else if (Op == AtomicRMWInst::Or)
        NamePart = "_fetch_or";
      else if (Op == AtomicRMWInst::Xor)
        NamePart = "_fetch_xor";
      else if (Op == AtomicRMWInst::Nand)
        NamePart = "_fetch_nand";
      else
        continue;
      SmallString<32> RMWName("__tsan_atomic" + itostr(BitSize) + NamePart);
      TsanAtomicRMW[Op][i] = M.getOrInsertFunction(
          RMWName,
          TLI.getAttrList(&Ctx, Idxs2Or12, /*Signed=*/true,
                          /*Ret=*/BitSize <= 32, Attr),
          Ty, PtrTy, Ty, OrdTy);
    }

    SmallString<32> AtomicCASName("__tsan_atomic" + BitSizeStr +
                                  "_compare_exchange_val");
    TsanAtomicCAS[i] = M.getOrInsertFunction(
        AtomicCASName,
        TLI.getAttrList(&Ctx, Idxs34Or1234, /*Signed=*/true,
                        /*Ret=*/BitSize <= 32, Attr),
        Ty, PtrTy, Ty, Ty, OrdTy, OrdTy);
  }
  TsanVptrUpdate =
      M.getOrInsertFunction("__tsan_vptr_update", Attr, IRB.getVoidTy(),
                            IRB.getPtrTy(), IRB.getPtrTy());
  TsanVptrLoad = M.getOrInsertFunction("__tsan_vptr_read", Attr,
                                       IRB.getVoidTy(), IRB.getPtrTy());
  TsanAtomicThreadFence = M.getOrInsertFunction(
      "__tsan_atomic_thread_fence",
      TLI.getAttrList(&Ctx, {0}, /*Signed=*/true, /*Ret=*/false, Attr),
      IRB.getVoidTy(), OrdTy);

  TsanAtomicSignalFence = M.getOrInsertFunction(
      "__tsan_atomic_signal_fence",
      TLI.getAttrList(&Ctx, {0}, /*Signed=*/true, /*Ret=*/false, Attr),
      IRB.getVoidTy(), OrdTy);

  MemmoveFn =
      M.getOrInsertFunction("__tsan_memmove", Attr, IRB.getPtrTy(),
                            IRB.getPtrTy(), IRB.getPtrTy(), IntptrTy);
  MemcpyFn =
      M.getOrInsertFunction("__tsan_memcpy", Attr, IRB.getPtrTy(),
                            IRB.getPtrTy(), IRB.getPtrTy(), IntptrTy);
  MemsetFn = M.getOrInsertFunction(
      "__tsan_memset",
      TLI.getAttrList(&Ctx, {1}, /*Signed=*/true, /*Ret=*/false, Attr),
      IRB.getPtrTy(), IRB.getPtrTy(), IRB.getInt32Ty(), IntptrTy);
}

static bool isVtableAccess(Instruction *I) {
  if (MDNode *Tag = I->getMetadata(LLVMContext::MD_tbaa))
    return Tag->isTBAAVtableAccess();
  return false;
}

// Do not instrument known races/"benign races" that come from compiler
// instrumentatin. The user has no way of suppressing them.
static bool shouldInstrumentReadWriteFromAddress(const Module *M, Value *Addr) {
  // Peel off GEPs and BitCasts.
  Addr = Addr->stripInBoundsOffsets();

  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(Addr)) {
    if (GV->hasSection()) {
      StringRef SectionName = GV->getSection();
      // Check if the global is in the PGO counters section.
      auto OF = M->getTargetTriple().getObjectFormat();
      if (SectionName.ends_with(
              getInstrProfSectionName(IPSK_cnts, OF, /*AddSegmentInfo=*/false)))
        return false;
    }
  }

  // Do not instrument accesses from different address spaces; we cannot deal
  // with them.
  if (Addr) {
    Type *PtrTy = cast<PointerType>(Addr->getType()->getScalarType());
    if (PtrTy->getPointerAddressSpace() != 0)
      return false;
  }

  return true;
}

bool ThreadSanitizer::addrPointsToConstantData(Value *Addr) {
  // If this is a GEP, just analyze its pointer operand.
  if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Addr))
    Addr = GEP->getPointerOperand();

  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(Addr)) {
    if (GV->isConstant()) {
      // Reads from constant globals can not race with any writes.
      NumOmittedReadsFromConstantGlobals++;
      return true;
    }
  } else if (LoadInst *L = dyn_cast<LoadInst>(Addr)) {
    if (isVtableAccess(L)) {
      // Reads from a vtable pointer can not race with any writes.
      NumOmittedReadsFromVtable++;
      return true;
    }
  }
  return false;
}

// Instrumenting some of the accesses may be proven redundant.
// Currently handled:
//  - read-before-write (within same BB, no calls between)
//  - not captured variables
//
// We do not handle some of the patterns that should not survive
// after the classic compiler optimizations.
// E.g. two reads from the same temp should be eliminated by CSE,
// two writes should be eliminated by DSE, etc.
//
// 'Local' is a vector of insns within the same BB (no calls between).
// 'All' is a vector of insns that will be instrumented.
void ThreadSanitizer::chooseInstructionsToInstrument(
    SmallVectorImpl<Instruction *> &Local,
    SmallVectorImpl<InstructionInfo> &All, const DataLayout &DL) {
  DenseMap<Value *, size_t> WriteTargets; // Map of addresses to index in All
  // Iterate from the end.
  for (Instruction *I : reverse(Local)) {
    const bool IsWrite = isa<StoreInst>(*I);
    Value *Addr = IsWrite ? cast<StoreInst>(I)->getPointerOperand()
                          : cast<LoadInst>(I)->getPointerOperand();

    if (!shouldInstrumentReadWriteFromAddress(I->getModule(), Addr))
      continue;

    if (!IsWrite) {
      const auto WriteEntry = WriteTargets.find(Addr);
      if (!ClInstrumentReadBeforeWrite && WriteEntry != WriteTargets.end()) {
        auto &WI = All[WriteEntry->second];
        // If we distinguish volatile accesses and if either the read or write
        // is volatile, do not omit any instrumentation.
        const bool AnyVolatile =
            ClDistinguishVolatile && (cast<LoadInst>(I)->isVolatile() ||
                                      cast<StoreInst>(WI.Inst)->isVolatile());
        if (!AnyVolatile) {
          // We will write to this temp, so no reason to analyze the read.
          // Mark the write instruction as compound.
          WI.Flags |= InstructionInfo::kCompoundRW;
          NumOmittedReadsBeforeWrite++;
          continue;
        }
      }

      if (addrPointsToConstantData(Addr)) {
        // Addr points to some constant data -- it can not race with any writes.
        continue;
      }
    }

    const AllocaInst *AI = findAllocaForValue(Addr);
    // Instead of Addr, we should check whether its base pointer is captured.
    if (AI && !PointerMayBeCaptured(AI, /*ReturnCaptures=*/true) &&
        ClOmitNonCaptured) {
      // The variable is addressable but not captured, so it cannot be
      // referenced from a different thread and participate in a data race
      // (see llvm/Analysis/CaptureTracking.h for details).
      NumOmittedNonCaptured++;
      continue;
    }

    // Instrument this instruction.
    All.emplace_back(I);
    if (IsWrite) {
      // For read-before-write and compound instrumentation we only need one
      // write target, and we can override any previous entry if it exists.
      WriteTargets[Addr] = All.size() - 1;
    }
  }
  Local.clear();
}

void ThreadSanitizer::InsertRuntimeIgnores(Function &F) {
  InstrumentationIRBuilder IRB(&F.getEntryBlock(),
                               F.getEntryBlock().getFirstNonPHIIt());
  IRB.CreateCall(TsanIgnoreBegin);
  EscapeEnumerator EE(F, "tsan_ignore_cleanup", ClHandleCxxExceptions);
  while (IRBuilder<> *AtExit = EE.Next()) {
    InstrumentationIRBuilder::ensureDebugInfo(*AtExit, F);
    AtExit->CreateCall(TsanIgnoreEnd);
  }
}

bool ThreadSanitizer::sanitizeFunction(Function &F) {
  // This is required to prevent instrumenting call to __tsan_init from within
  // the module constructor.
  if (F.getName() == kTsanModuleCtorName)
    return false;
  // Naked functions can not have prologue/epilogue
  // (__tsan_func_entry/__tsan_func_exit) generated, so don't instrument them at
  // all.
  if (F.hasFnAttribute(Attribute::Naked))
    return false;

  // __attribute__(disable_sanitizer_instrumentation) prevents all kinds of
  // instrumentation.
  if (F.hasFnAttribute(Attribute::DisableSanitizerInstrumentation))
    return false;

  initialize(*F.getParent(), TLI);
  SmallVector<InstructionInfo, 8> AllLoadsAndStores;
  SmallVector<Instruction*, 8> LocalLoadsAndStores;
  SmallVector<Instruction*, 8> AtomicAccesses;
  SmallVector<Instruction*, 8> MemIntrinCalls;
  bool Res = false;
  bool HasCalls = false;
  bool SanitizeFunction = F.hasFnAttribute(Attribute::SanitizeThread);
  const DataLayout &DL = F.getDataLayout();

  // Traverse all instructions, collect loads/stores/returns, check for calls.
  for (auto &BB : F) {
    for (auto &Inst : BB) {
      // Skip instructions inserted by another instrumentation.
      if (Inst.hasMetadata(LLVMContext::MD_nosanitize))
        continue;
      if (isTsanAtomic(&Inst))
        AtomicAccesses.push_back(&Inst);
      else if (isa<LoadInst>(Inst) || isa<StoreInst>(Inst))
        LocalLoadsAndStores.push_back(&Inst);
      else if (isa<CallInst>(Inst) || isa<InvokeInst>(Inst)) {
        if (CallInst *CI = dyn_cast<CallInst>(&Inst))
          maybeMarkSanitizerLibraryCallNoBuiltin(CI, &TLI);
        if (isa<MemIntrinsic>(Inst))
          MemIntrinCalls.push_back(&Inst);
        HasCalls = true;
        chooseInstructionsToInstrument(LocalLoadsAndStores, AllLoadsAndStores,
                                       DL);
      }
    }
    chooseInstructionsToInstrument(LocalLoadsAndStores, AllLoadsAndStores, DL);
  }

  if (ClUseDominanceAnalysis && DT && PDT && AA) {
    // DominanceBasedElimination DBE(AllLoadsAndStores, *DT, *PDT, *AA, TLI);
    DominanceBasedElimination DBE(AllLoadsAndStores, *DT, *PDT, *AA);
    DBE.run();
  }

  // We have collected all loads and stores.
  // FIXME: many of these accesses do not need to be checked for races
  // (e.g. variables that do not escape, etc).

  // Instrument memory accesses only if we want to report bugs in the function.
  if (ClInstrumentMemoryAccesses && SanitizeFunction)
    for (const auto &II : AllLoadsAndStores) {
      Res |= instrumentLoadOrStore(II, DL);
    }

  // Instrument atomic memory accesses in any case (they can be used to
  // implement synchronization).
  if (ClInstrumentAtomics)
    for (auto *Inst : AtomicAccesses) {
      Res |= instrumentAtomic(Inst, DL);
    }

  if (ClInstrumentMemIntrinsics && SanitizeFunction)
    for (auto *Inst : MemIntrinCalls) {
      Res |= instrumentMemIntrinsic(Inst);
    }

  if (F.hasFnAttribute("sanitize_thread_no_checking_at_run_time")) {
    assert(!F.hasFnAttribute(Attribute::SanitizeThread));
    if (HasCalls)
      InsertRuntimeIgnores(F);
  }

  // Instrument function entry/exit points if there were instrumented accesses.
  if ((Res || HasCalls) && ClInstrumentFuncEntryExit) {
    InstrumentationIRBuilder IRB(&F.getEntryBlock(),
                                 F.getEntryBlock().getFirstNonPHIIt());
    Value *ReturnAddress =
        IRB.CreateIntrinsic(Intrinsic::returnaddress, IRB.getInt32(0));
    IRB.CreateCall(TsanFuncEntry, ReturnAddress);

    EscapeEnumerator EE(F, "tsan_cleanup", ClHandleCxxExceptions);
    while (IRBuilder<> *AtExit = EE.Next()) {
      InstrumentationIRBuilder::ensureDebugInfo(*AtExit, F);
      AtExit->CreateCall(TsanFuncExit, {});
    }
    Res = true;
  }
  return Res;
}

bool ThreadSanitizer::instrumentLoadOrStore(const InstructionInfo &II,
                                            const DataLayout &DL) {
  InstrumentationIRBuilder IRB(II.Inst);
  const bool IsWrite = isa<StoreInst>(*II.Inst);
  Value *Addr = IsWrite ? cast<StoreInst>(II.Inst)->getPointerOperand()
                        : cast<LoadInst>(II.Inst)->getPointerOperand();
  Type *OrigTy = getLoadStoreType(II.Inst);

  // swifterror memory addresses are mem2reg promoted by instruction selection.
  // As such they cannot have regular uses like an instrumentation function and
  // it makes no sense to track them as memory.
  if (Addr->isSwiftError())
    return false;

  int Idx = getMemoryAccessFuncIndex(OrigTy, Addr, DL);
  if (Idx < 0)
    return false;
  if (IsWrite && isVtableAccess(II.Inst)) {
    LLVM_DEBUG(dbgs() << "  VPTR : " << *II.Inst << "\n");
    Value *StoredValue = cast<StoreInst>(II.Inst)->getValueOperand();
    // StoredValue may be a vector type if we are storing several vptrs at once.
    // In this case, just take the first element of the vector since this is
    // enough to find vptr races.
    if (isa<VectorType>(StoredValue->getType()))
      StoredValue = IRB.CreateExtractElement(
          StoredValue, ConstantInt::get(IRB.getInt32Ty(), 0));
    if (StoredValue->getType()->isIntegerTy())
      StoredValue = IRB.CreateIntToPtr(StoredValue, IRB.getPtrTy());
    // Call TsanVptrUpdate.
    IRB.CreateCall(TsanVptrUpdate, {Addr, StoredValue});
    NumInstrumentedVtableWrites++;
    return true;
  }
  if (!IsWrite && isVtableAccess(II.Inst)) {
    IRB.CreateCall(TsanVptrLoad, Addr);
    NumInstrumentedVtableReads++;
    return true;
  }

  const Align Alignment = IsWrite ? cast<StoreInst>(II.Inst)->getAlign()
                                  : cast<LoadInst>(II.Inst)->getAlign();
  const bool IsCompoundRW =
      ClCompoundReadBeforeWrite && (II.Flags & InstructionInfo::kCompoundRW);
  const bool IsVolatile = ClDistinguishVolatile &&
                          (IsWrite ? cast<StoreInst>(II.Inst)->isVolatile()
                                   : cast<LoadInst>(II.Inst)->isVolatile());
  assert((!IsVolatile || !IsCompoundRW) && "Compound volatile invalid!");

  const uint32_t TypeSize = DL.getTypeStoreSizeInBits(OrigTy);
  FunctionCallee OnAccessFunc = nullptr;
  if (Alignment >= Align(8) || (Alignment.value() % (TypeSize / 8)) == 0) {
    if (IsCompoundRW)
      OnAccessFunc = TsanCompoundRW[Idx];
    else if (IsVolatile)
      OnAccessFunc = IsWrite ? TsanVolatileWrite[Idx] : TsanVolatileRead[Idx];
    else
      OnAccessFunc = IsWrite ? TsanWrite[Idx] : TsanRead[Idx];
  } else {
    if (IsCompoundRW)
      OnAccessFunc = TsanUnalignedCompoundRW[Idx];
    else if (IsVolatile)
      OnAccessFunc = IsWrite ? TsanUnalignedVolatileWrite[Idx]
                             : TsanUnalignedVolatileRead[Idx];
    else
      OnAccessFunc = IsWrite ? TsanUnalignedWrite[Idx] : TsanUnalignedRead[Idx];
  }
  IRB.CreateCall(OnAccessFunc, Addr);
  if (IsCompoundRW || IsWrite)
    NumInstrumentedWrites++;
  if (IsCompoundRW || !IsWrite)
    NumInstrumentedReads++;
  return true;
}

static ConstantInt *createOrdering(IRBuilder<> *IRB, AtomicOrdering ord) {
  uint32_t v = 0;
  switch (ord) {
    case AtomicOrdering::NotAtomic:
      llvm_unreachable("unexpected atomic ordering!");
    case AtomicOrdering::Unordered:              [[fallthrough]];
    case AtomicOrdering::Monotonic:              v = 0; break;
    // Not specified yet:
    // case AtomicOrdering::Consume:                v = 1; break;
    case AtomicOrdering::Acquire:                v = 2; break;
    case AtomicOrdering::Release:                v = 3; break;
    case AtomicOrdering::AcquireRelease:         v = 4; break;
    case AtomicOrdering::SequentiallyConsistent: v = 5; break;
  }
  return IRB->getInt32(v);
}

// If a memset intrinsic gets inlined by the code gen, we will miss races on it.
// So, we either need to ensure the intrinsic is not inlined, or instrument it.
// We do not instrument memset/memmove/memcpy intrinsics (too complicated),
// instead we simply replace them with regular function calls, which are then
// intercepted by the run-time.
// Since tsan is running after everyone else, the calls should not be
// replaced back with intrinsics. If that becomes wrong at some point,
// we will need to call e.g. __tsan_memset to avoid the intrinsics.
bool ThreadSanitizer::instrumentMemIntrinsic(Instruction *I) {
  InstrumentationIRBuilder IRB(I);
  if (MemSetInst *M = dyn_cast<MemSetInst>(I)) {
    Value *Cast1 = IRB.CreateIntCast(M->getArgOperand(1), IRB.getInt32Ty(), false);
    Value *Cast2 = IRB.CreateIntCast(M->getArgOperand(2), IntptrTy, false);
    IRB.CreateCall(
        MemsetFn,
        {M->getArgOperand(0),
         Cast1,
         Cast2});
    I->eraseFromParent();
  } else if (MemTransferInst *M = dyn_cast<MemTransferInst>(I)) {
    IRB.CreateCall(
        isa<MemCpyInst>(M) ? MemcpyFn : MemmoveFn,
        {M->getArgOperand(0),
         M->getArgOperand(1),
         IRB.CreateIntCast(M->getArgOperand(2), IntptrTy, false)});
    I->eraseFromParent();
  }
  return false;
}

// Both llvm and ThreadSanitizer atomic operations are based on C++11/C1x
// standards.  For background see C++11 standard.  A slightly older, publicly
// available draft of the standard (not entirely up-to-date, but close enough
// for casual browsing) is available here:
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2011/n3242.pdf
// The following page contains more background information:
// http://www.hpl.hp.com/personal/Hans_Boehm/c++mm/

bool ThreadSanitizer::instrumentAtomic(Instruction *I, const DataLayout &DL) {
  InstrumentationIRBuilder IRB(I);
  if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
    Value *Addr = LI->getPointerOperand();
    Type *OrigTy = LI->getType();
    int Idx = getMemoryAccessFuncIndex(OrigTy, Addr, DL);
    if (Idx < 0)
      return false;
    Value *Args[] = {Addr,
                     createOrdering(&IRB, LI->getOrdering())};
    Value *C = IRB.CreateCall(TsanAtomicLoad[Idx], Args);
    Value *Cast = IRB.CreateBitOrPointerCast(C, OrigTy);
    I->replaceAllUsesWith(Cast);
    I->eraseFromParent();
  } else if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
    Value *Addr = SI->getPointerOperand();
    int Idx =
        getMemoryAccessFuncIndex(SI->getValueOperand()->getType(), Addr, DL);
    if (Idx < 0)
      return false;
    const unsigned ByteSize = 1U << Idx;
    const unsigned BitSize = ByteSize * 8;
    Type *Ty = Type::getIntNTy(IRB.getContext(), BitSize);
    Value *Args[] = {Addr,
                     IRB.CreateBitOrPointerCast(SI->getValueOperand(), Ty),
                     createOrdering(&IRB, SI->getOrdering())};
    IRB.CreateCall(TsanAtomicStore[Idx], Args);
    SI->eraseFromParent();
  } else if (AtomicRMWInst *RMWI = dyn_cast<AtomicRMWInst>(I)) {
    Value *Addr = RMWI->getPointerOperand();
    int Idx =
        getMemoryAccessFuncIndex(RMWI->getValOperand()->getType(), Addr, DL);
    if (Idx < 0)
      return false;
    FunctionCallee F = TsanAtomicRMW[RMWI->getOperation()][Idx];
    if (!F)
      return false;
    const unsigned ByteSize = 1U << Idx;
    const unsigned BitSize = ByteSize * 8;
    Type *Ty = Type::getIntNTy(IRB.getContext(), BitSize);
    Value *Val = RMWI->getValOperand();
    Value *Args[] = {Addr, IRB.CreateBitOrPointerCast(Val, Ty),
                     createOrdering(&IRB, RMWI->getOrdering())};
    Value *C = IRB.CreateCall(F, Args);
    I->replaceAllUsesWith(IRB.CreateBitOrPointerCast(C, Val->getType()));
    I->eraseFromParent();
  } else if (AtomicCmpXchgInst *CASI = dyn_cast<AtomicCmpXchgInst>(I)) {
    Value *Addr = CASI->getPointerOperand();
    Type *OrigOldValTy = CASI->getNewValOperand()->getType();
    int Idx = getMemoryAccessFuncIndex(OrigOldValTy, Addr, DL);
    if (Idx < 0)
      return false;
    const unsigned ByteSize = 1U << Idx;
    const unsigned BitSize = ByteSize * 8;
    Type *Ty = Type::getIntNTy(IRB.getContext(), BitSize);
    Value *CmpOperand =
      IRB.CreateBitOrPointerCast(CASI->getCompareOperand(), Ty);
    Value *NewOperand =
      IRB.CreateBitOrPointerCast(CASI->getNewValOperand(), Ty);
    Value *Args[] = {Addr,
                     CmpOperand,
                     NewOperand,
                     createOrdering(&IRB, CASI->getSuccessOrdering()),
                     createOrdering(&IRB, CASI->getFailureOrdering())};
    CallInst *C = IRB.CreateCall(TsanAtomicCAS[Idx], Args);
    Value *Success = IRB.CreateICmpEQ(C, CmpOperand);
    Value *OldVal = C;
    if (Ty != OrigOldValTy) {
      // The value is a pointer, so we need to cast the return value.
      OldVal = IRB.CreateIntToPtr(C, OrigOldValTy);
    }

    Value *Res =
      IRB.CreateInsertValue(PoisonValue::get(CASI->getType()), OldVal, 0);
    Res = IRB.CreateInsertValue(Res, Success, 1);

    I->replaceAllUsesWith(Res);
    I->eraseFromParent();
  } else if (FenceInst *FI = dyn_cast<FenceInst>(I)) {
    Value *Args[] = {createOrdering(&IRB, FI->getOrdering())};
    FunctionCallee F = FI->getSyncScopeID() == SyncScope::SingleThread
                           ? TsanAtomicSignalFence
                           : TsanAtomicThreadFence;
    IRB.CreateCall(F, Args);
    FI->eraseFromParent();
  }
  return true;
}

int ThreadSanitizer::getMemoryAccessFuncIndex(Type *OrigTy, Value *Addr,
                                              const DataLayout &DL) {
  assert(OrigTy->isSized());
  if (OrigTy->isScalableTy()) {
    // FIXME: support vscale.
    return -1;
  }
  uint32_t TypeSize = DL.getTypeStoreSizeInBits(OrigTy);
  if (TypeSize != 8  && TypeSize != 16 &&
      TypeSize != 32 && TypeSize != 64 && TypeSize != 128) {
    NumAccessesWithBadSize++;
    // Ignore all unusual sizes.
    return -1;
  }
  size_t Idx = llvm::countr_zero(TypeSize / 8);
  assert(Idx < kNumberOfAccessSizes);
  return Idx;
}
