//===-- tsan_ownership_profile.cpp ---------------------------------------===//

#include "tsan_ownership_profile.h"

#include "sanitizer_common/sanitizer_atomic.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_file.h"
#include "tsan_platform.h"
#include "tsan_rtl.h"

namespace __tsan {

bool ownership_profile_enabled;

namespace {

constexpr u64 kNoteMask = 0xffffffffull;
constexpr uptr kOwnerShift = 32;
constexpr u64 kOwnerMask = ((1ull << 28) - 1) << kOwnerShift;
constexpr u64 kTouched = 1ull << 60;
constexpr u64 kNoteFailed = 1ull << 61;
constexpr u64 kShared = 1ull << 62;

constexpr u32 kPageAbsorbed = 1u << 0;
constexpr u32 kPageFallback = 1u << 1;
constexpr uptr kPageCountShift = 2;
constexpr uptr kBitmapWordsPerPage = 4;

struct Counters {
  u64 callbacks;
  u64 callback_reads;
  u64 callback_writes;
  u64 callback_atomics;
  u64 cell_touches;
  u64 first_touches;
  u64 private_first_touches;
  u64 owner_hits;
  u64 transitions;
  u64 shared_hits;
  u64 private_csa_hits;
  u64 shared_csa_hits;
  u64 cell_reads;
  u64 cell_writes;
  u64 cell_atomics;
  u64 shadow_slots[5];
  u64 note_claims;
  u64 note_exact_hits;
  u64 note_cover_hits;
  u64 note_merges;
  u64 note_owner_fallbacks;
  u64 note_merge_fallbacks;
  u64 note_ineligible_fallbacks;
  u64 note_fallback_hits;
  u64 note_absorbed_csa_hits;
  u64 note_fallback_csa_hits;
  u64 shadow_exact_eligible;
  u64 shadow_exact_slot0_hits;
  u64 shadow_exact_late_hits;
  u64 shadow_exact_misses;
  u64 page_absorbed_seen;
  u64 page_fallback_seen;
  u64 page_reached_1;
  u64 page_reached_2;
  u64 page_reached_65;
  u64 page_reached_129;
  u64 page_reached_193;
  u64 page_reached_256;
  u64 allocation_cells_tagged;
  u64 allocation_ranges;
};

struct ProfileThread {
  ProfileThread *next;
  u32 owner;
  Counters counters;
};

atomic_uintptr_t threads;
atomic_uint32_t next_owner;
atomic_uintptr_t focus_beg;
atomic_uintptr_t focus_end;
atomic_uint32_t focus_stopped;
uptr profile_beg;
uptr profile_end;
uptr cell_profile_end;
uptr page_bitmap_beg;
uptr page_bitmap_end;
uptr page_profile_beg;
uptr page_profile_end;

uptr CellCount() {
  return (ShadowEnd() - ShadowBeg()) / (kShadowCnt * kShadowSize);
}

uptr ShadowPageCount() {
  return (ShadowEnd() - ShadowBeg()) / GetPageSizeCached();
}

uptr CellProfileSize() {
  return RoundUp(CellCount() * sizeof(u64), GetPageSizeCached());
}

uptr PageBitmapSize() {
  return RoundUp(ShadowPageCount() * kBitmapWordsPerPage * sizeof(u64),
                 GetPageSizeCached());
}

uptr PageProfileSize() {
  return RoundUp(ShadowPageCount() * sizeof(u32), GetPageSizeCached());
}

uptr ProfileSize() {
  return CellProfileSize() + PageBitmapSize() + PageProfileSize();
}

uptr TryGap(uptr beg, uptr end, uptr size) {
  beg = RoundUp(beg, GetPageSizeCached());
  return end >= beg && end - beg >= size ? beg : 0;
}

uptr FindProfileBeg(uptr size) {
  uptr beg = TryGap(ShadowEnd(), MetaShadowBeg(), size);
  if (beg)
    return beg;

  uptr after_meta = ~uptr{};
  const uptr app_beg[] = {MidAppMemBeg(), HeapMemBeg(), HiAppMemBeg()};
  for (uptr candidate : app_beg)
    if (candidate > MetaShadowEnd())
      after_meta = Min(after_meta, candidate);
  if (after_meta != ~uptr{}) {
    beg = TryGap(MetaShadowEnd(), after_meta, size);
    if (beg)
      return beg;
  }

  return TryGap(LoAppMemEnd(), ShadowBeg(), size);
}

atomic_uint64_t *MemToProfile(uptr addr) {
  uptr shadow = reinterpret_cast<uptr>(MemToShadow(addr));
  uptr index = (shadow - ShadowBeg()) / (kShadowCnt * kShadowSize);
  return reinterpret_cast<atomic_uint64_t *>(profile_beg) + index;
}

ProfileThread *GetProfileThread(ThreadState *thr) {
  return reinterpret_cast<ProfileThread *>(thr->ownership_profile);
}

u32 NewOwner() {
  u32 owner = atomic_fetch_add(&next_owner, 1, memory_order_relaxed) + 1;
  if (owner == 0 || owner > (kOwnerMask >> kOwnerShift)) {
    Printf("FATAL: ThreadSanitizer ownership profile exhausted owner IDs\n");
    Die();
  }
  return owner;
}

void Add(Counters *dst, const Counters &src) {
#define ADD_FIELD(field) dst->field += src.field
  ADD_FIELD(callbacks);
  ADD_FIELD(callback_reads);
  ADD_FIELD(callback_writes);
  ADD_FIELD(callback_atomics);
  ADD_FIELD(cell_touches);
  ADD_FIELD(first_touches);
  ADD_FIELD(private_first_touches);
  ADD_FIELD(owner_hits);
  ADD_FIELD(transitions);
  ADD_FIELD(shared_hits);
  ADD_FIELD(private_csa_hits);
  ADD_FIELD(shared_csa_hits);
  ADD_FIELD(cell_reads);
  ADD_FIELD(cell_writes);
  ADD_FIELD(cell_atomics);
  ADD_FIELD(note_claims);
  ADD_FIELD(note_exact_hits);
  ADD_FIELD(note_cover_hits);
  ADD_FIELD(note_merges);
  ADD_FIELD(note_owner_fallbacks);
  ADD_FIELD(note_merge_fallbacks);
  ADD_FIELD(note_ineligible_fallbacks);
  ADD_FIELD(note_fallback_hits);
  ADD_FIELD(note_absorbed_csa_hits);
  ADD_FIELD(note_fallback_csa_hits);
  ADD_FIELD(shadow_exact_eligible);
  ADD_FIELD(shadow_exact_slot0_hits);
  ADD_FIELD(shadow_exact_late_hits);
  ADD_FIELD(shadow_exact_misses);
  ADD_FIELD(page_absorbed_seen);
  ADD_FIELD(page_fallback_seen);
  ADD_FIELD(page_reached_1);
  ADD_FIELD(page_reached_2);
  ADD_FIELD(page_reached_65);
  ADD_FIELD(page_reached_129);
  ADD_FIELD(page_reached_193);
  ADD_FIELD(page_reached_256);
  ADD_FIELD(allocation_cells_tagged);
  ADD_FIELD(allocation_ranges);
#undef ADD_FIELD
  for (uptr i = 0; i < ARRAY_SIZE(dst->shadow_slots); ++i)
    dst->shadow_slots[i] += src.shadow_slots[i];
}

uptr OccupiedSlots(RawShadow *shadow_mem) {
  uptr occupied = 0;
  for (uptr i = 0; i < kShadowCnt; ++i)
    occupied += LoadShadow(&shadow_mem[i]) != Shadow::kEmpty;
  return occupied;
}

bool TryExactMerge(u32 a, u32 b, u32 *merged) {
  constexpr u32 kAccessMask = 0x000000ff;
  constexpr u32 kSidMask = 0x0000ff00;
  constexpr u32 kEpochMask = 0x3fff0000;
  constexpr u32 kTypeMask = 0xc0000000;
  if ((a ^ b) & kSidMask)
    return false;

  const bool a_dominates_b =
      ((b & kAccessMask) & ~(a & kAccessMask)) == 0 &&
      (a & kEpochMask) >= (b & kEpochMask) &&
      ((a & kTypeMask) & ~(b & kTypeMask)) == 0;
  const bool b_dominates_a =
      ((a & kAccessMask) & ~(b & kAccessMask)) == 0 &&
      (b & kEpochMask) >= (a & kEpochMask) &&
      ((b & kTypeMask) & ~(a & kTypeMask)) == 0;
  const bool same_epoch_and_type = ((a ^ b) & (kEpochMask | kTypeMask)) == 0;
  if (!a_dominates_b && !b_dominates_a && !same_epoch_and_type)
    return false;

  *merged = (a & kSidMask) | ((a | b) & kAccessMask) |
            Max(a & kEpochMask, b & kEpochMask) | ((a & b) & kTypeMask);
  return true;
}

int ExactHitSlot(RawShadow *shadow_mem, RawShadow current, AccessType typ) {
  for (uptr i = 0; i < kShadowCnt; ++i) {
    RawShadow old = LoadShadow(&shadow_mem[i]);
    if (!(typ & kAccessRead)) {
      if (old == current)
        return i;
      continue;
    }
    RawShadow masked = static_cast<RawShadow>(
        static_cast<u32>(old) | static_cast<u32>(Shadow::kRodata));
    if (masked == current ||
        (!(typ & kAccessNoRodata) && !SANITIZER_GO &&
         old == Shadow::kRodata))
      return i;
  }
  return -1;
}

bool MarkCellSeen(RawShadow *shadow_mem) {
  const uptr shadow = reinterpret_cast<uptr>(shadow_mem);
  const uptr page = GetPageSizeCached();
  const uptr page_index = (shadow - ShadowBeg()) / page;
  const uptr cell_index = (shadow % page) / (kShadowCnt * kShadowSize);
  auto *word = reinterpret_cast<atomic_uint64_t *>(page_bitmap_beg) +
               page_index * kBitmapWordsPerPage + cell_index / 64;
  const u64 bit = 1ull << (cell_index % 64);
  u64 old = atomic_load(word, memory_order_relaxed);
  while (!(old & bit)) {
    if (atomic_compare_exchange_weak(word, &old, old | bit,
                                     memory_order_relaxed))
      return true;
  }
  return false;
}

void MarkPage(Counters &c, RawShadow *shadow_mem, bool absorbed) {
  const uptr page = GetPageSizeCached();
  const uptr page_index =
      (reinterpret_cast<uptr>(shadow_mem) - ShadowBeg()) / page;
  auto *state = reinterpret_cast<atomic_uint32_t *>(page_profile_beg) +
                page_index;
  const bool first_cell = MarkCellSeen(shadow_mem);
  const u32 path = absorbed ? kPageAbsorbed : kPageFallback;
  u32 old = atomic_load(state, memory_order_relaxed);
  for (;;) {
    u32 next = (old | path) +
               (first_cell ? static_cast<u32>(1 << kPageCountShift) : 0);
    if (next == old || atomic_compare_exchange_weak(state, &old, next,
                                                    memory_order_relaxed)) {
      if (!(old & path)) {
        if (absorbed)
          ++c.page_absorbed_seen;
        else
          ++c.page_fallback_seen;
      }
      if (first_cell) {
        uptr count = old >> kPageCountShift;
        if (count == 0)
          ++c.page_reached_1;
        else if (count == 1)
          ++c.page_reached_2;
        else if (count == 64)
          ++c.page_reached_65;
        else if (count == 128)
          ++c.page_reached_129;
        else if (count == 192)
          ++c.page_reached_193;
        else if (count == 255)
          ++c.page_reached_256;
      }
      return;
    }
  }
}

void SetRange(uptr addr, uptr size, u32 value) {
  if (!ownership_profile_enabled || size == 0)
    return;
  addr = STRIP_MTE_TAG(addr);
  uptr beg = RoundDown(addr, kShadowCell);
  uptr end = RoundUp(addr + size, kShadowCell);
  if (!IsAppMem(beg) || !IsAppMem(end - 1))
    return;

  atomic_uint64_t *first = MemToProfile(beg);
  atomic_uint64_t *last = MemToProfile(end - kShadowCell) + 1;
  uptr bytes = reinterpret_cast<uptr>(last) - reinterpret_cast<uptr>(first);
  const u64 stored = static_cast<u64>(value) << kOwnerShift;
  if (value != 0 || bytes <= common_flags()->clear_shadow_mmap_threshold) {
    for (; first != last; ++first)
      atomic_store(first, stored, memory_order_relaxed);
    return;
  }

  const uptr page = GetPageSizeCached();
  auto *middle_beg = reinterpret_cast<atomic_uint64_t *>(
      RoundUp(reinterpret_cast<uptr>(first), page));
  auto *middle_end = reinterpret_cast<atomic_uint64_t *>(
      RoundDown(reinterpret_cast<uptr>(last), page));
  for (; first < Min(last, middle_beg); ++first)
    atomic_store(first, 0, memory_order_relaxed);
  if (middle_beg < middle_end &&
      !MmapFixedSuperNoReserve(reinterpret_cast<uptr>(middle_beg),
                               reinterpret_cast<uptr>(middle_end) -
                                   reinterpret_cast<uptr>(middle_beg),
                               "ownership profile"))
    Die();
  for (auto *p = Max(first, middle_end); p < last; ++p)
    atomic_store(p, 0, memory_order_relaxed);
}

}  // namespace

void InitializeOwnershipProfile() {
  if (!flags()->ownership_profile)
    return;
  if (GetPageSizeCached() != 4096) {
    Printf("FATAL: ThreadSanitizer ownership profile requires 4 KiB pages\n");
    Die();
  }
  if (flags()->one_word_filter) {
    Printf("FATAL: ownership_profile and one_word_filter cannot be combined\n");
    Die();
  }

  uptr size = ProfileSize();
  profile_beg = FindProfileBeg(size);
  profile_end = profile_beg + size;
  cell_profile_end = profile_beg + CellProfileSize();
  page_bitmap_beg = cell_profile_end;
  page_bitmap_end = page_bitmap_beg + PageBitmapSize();
  page_profile_beg = page_bitmap_end;
  page_profile_end = page_profile_beg + PageProfileSize();
  if (!profile_beg ||
      !MmapFixedSuperNoReserve(profile_beg, size, "ownership profile")) {
    Printf("FATAL: ThreadSanitizer cannot map ownership profile memory\n");
    Die();
  }
  DontDumpShadow(profile_beg, size);
  atomic_store(&focus_stopped, flags()->ownership_profile_wait_for_range,
               memory_order_relaxed);
  ownership_profile_enabled = true;
  Printf("TSAN_OWNERSHIP_MAP shadow=%zx-%zx profile=%zx-%zx page_size=%zu\n",
         ShadowBeg(), ShadowEnd(), profile_beg, profile_end,
         GetPageSizeCached());
}

void OwnershipProfileThreadStart(ThreadState *thr) {
  if (!ownership_profile_enabled)
    return;
  uptr size = RoundUp(sizeof(ProfileThread), GetPageSizeCached());
  auto *thread =
      static_cast<ProfileThread *>(MmapOrDie(size, "ownership profile thread"));
  internal_memset(thread, 0, sizeof(*thread));
  thread->owner = NewOwner();

  uptr old = atomic_load(&threads, memory_order_relaxed);
  do {
    thread->next = reinterpret_cast<ProfileThread *>(old);
  } while (!atomic_compare_exchange_weak(
      &threads, &old, reinterpret_cast<uptr>(thread), memory_order_release));
  thr->ownership_profile = thread;
}

void OwnershipProfileBeginAccess(ThreadState *thr, AccessType typ) {
  if (!ownership_profile_enabled || thr->fast_state.GetIgnoreBit())
    return;
  ProfileThread *thread = GetProfileThread(thr);
  CHECK(thread);
  thr->ownership_profile_seen = 0;
  if (thr->ownership_profile_retry) {
    thr->ownership_profile_retry = false;
    return;
  }
  thr->ownership_profile_skip = 0;
  ++thread->counters.callbacks;
  if (typ & kAccessAtomic)
    ++thread->counters.callback_atomics;
  else if (typ & kAccessRead)
    ++thread->counters.callback_reads;
  else
    ++thread->counters.callback_writes;
}

void OwnershipProfileRestartAccess(ThreadState *thr) {
  if (!ownership_profile_enabled)
    return;
  thr->ownership_profile_skip = thr->ownership_profile_seen;
  thr->ownership_profile_retry = true;
}

void OwnershipProfileAccess(ThreadState *thr, uptr addr,
                            RawShadow *shadow_mem, AccessType typ,
                            RawShadow current, bool contains_same_access,
                            bool note_eligible) {
  if (!ownership_profile_enabled || thr->fast_state.GetIgnoreBit())
    return;
  ++thr->ownership_profile_seen;
  if (thr->ownership_profile_skip) {
    --thr->ownership_profile_skip;
    return;
  }

  if (atomic_load(&focus_stopped, memory_order_acquire))
    return;
  uptr selected_beg = atomic_load(&focus_beg, memory_order_acquire);
  uptr selected_end = atomic_load(&focus_end, memory_order_relaxed);
  if (selected_beg && (addr < selected_beg || addr >= selected_end))
    return;

  ProfileThread *thread = GetProfileThread(thr);
  Counters &c = thread->counters;
  ++c.cell_touches;
  if (typ & kAccessAtomic)
    ++c.cell_atomics;
  else if (typ & kAccessRead)
    ++c.cell_reads;
  else
    ++c.cell_writes;
  ++c.shadow_slots[OccupiedSlots(shadow_mem)];

  note_eligible &= !(typ & ~kAccessRead);
  if (note_eligible) {
    ++c.shadow_exact_eligible;
    int slot = ExactHitSlot(shadow_mem, current, typ);
    if (slot == 0)
      ++c.shadow_exact_slot0_hits;
    else if (slot > 0)
      ++c.shadow_exact_late_hits;
    else
      ++c.shadow_exact_misses;
  }

  atomic_uint64_t *cell = MemToProfile(addr);
  u64 old = atomic_load(cell, memory_order_relaxed);
  for (;;) {
    bool touched = old & kTouched;
    bool shared = old & kShared;
    u32 owner = (old & kOwnerMask) >> kOwnerShift;
    u64 next = old;
    enum Result { kPrivateFirst, kOwnerHit, kTransition, kSharedHit } result;
    enum NoteResult {
      kClaim,
      kExactHit,
      kCoverHit,
      kMerge,
      kOwnerFallback,
      kMergeFallback,
      kIneligibleFallback,
      kFallbackHit
    } note_result;

    if (shared) {
      result = kSharedHit;
    } else if (owner == 0 || owner == thread->owner) {
      result = touched ? kOwnerHit : kPrivateFirst;
      next = (next & ~kOwnerMask) |
             (static_cast<u64>(thread->owner) << kOwnerShift) | kTouched;
    } else {
      result = kTransition;
      next = kShared | kTouched;
    }

    if (old & kNoteFailed) {
      note_result = kFallbackHit;
    } else if (!note_eligible) {
      note_result = kIneligibleFallback;
      next |= kNoteFailed;
    } else if (result == kTransition || result == kSharedHit) {
      note_result = kOwnerFallback;
      next |= kNoteFailed;
    } else if (!touched) {
      note_result = kClaim;
      next = (next & ~kNoteMask) | static_cast<u32>(current);
    } else {
      u32 note = old & kNoteMask;
      u32 incoming = static_cast<u32>(current);
      if (note == incoming) {
        note_result = kExactHit;
      } else {
        u32 merged;
        if (!TryExactMerge(note, incoming, &merged)) {
          note_result = kMergeFallback;
          next |= kNoteFailed;
        } else if (merged == note) {
          note_result = kCoverHit;
        } else {
          note_result = kMerge;
          next = (next & ~kNoteMask) | merged;
        }
      }
    }

    if (next != old && !atomic_compare_exchange_weak(
                           cell, &old, next, memory_order_relaxed))
      continue;

    if (!touched)
      ++c.first_touches;
    if (result == kPrivateFirst)
      ++c.private_first_touches;
    else if (result == kOwnerHit)
      ++c.owner_hits;
    else if (result == kTransition)
      ++c.transitions;
    else
      ++c.shared_hits;

    bool absorbed = false;
    if (note_result == kClaim) {
      ++c.note_claims;
      absorbed = true;
    } else if (note_result == kExactHit) {
      ++c.note_exact_hits;
      absorbed = true;
    } else if (note_result == kCoverHit) {
      ++c.note_cover_hits;
      absorbed = true;
    } else if (note_result == kMerge) {
      ++c.note_merges;
      absorbed = true;
    } else if (note_result == kOwnerFallback) {
      ++c.note_owner_fallbacks;
    } else if (note_result == kMergeFallback) {
      ++c.note_merge_fallbacks;
    } else if (note_result == kIneligibleFallback) {
      ++c.note_ineligible_fallbacks;
    } else {
      ++c.note_fallback_hits;
    }

    if (contains_same_access) {
      if (result == kPrivateFirst || result == kOwnerHit)
        ++c.private_csa_hits;
      else
        ++c.shared_csa_hits;
      if (absorbed)
        ++c.note_absorbed_csa_hits;
      else
        ++c.note_fallback_csa_hits;
    }
    MarkPage(c, shadow_mem, absorbed);
    return;
  }
}

void OwnershipProfileSetOwner(ThreadState *thr, uptr addr, uptr size) {
  if (!ownership_profile_enabled || !flags()->ownership_profile_allocations ||
      size == 0)
    return;
  ProfileThread *thread = GetProfileThread(thr);
  if (!thread)
    return;
  uptr beg = RoundDown(addr, kShadowCell);
  uptr end = RoundUp(addr + size, kShadowCell);
  SetRange(beg, end - beg, thread->owner);
  thread->counters.allocation_cells_tagged += (end - beg) / kShadowCell;
  ++thread->counters.allocation_ranges;
}

void OwnershipProfileResetRange(uptr addr, uptr size) {
  SetRange(addr, size, 0);
}

void OwnershipProfileSetFocusRange(uptr addr, uptr size) {
  if (!ownership_profile_enabled)
    return;
  if (size == 0) {
    atomic_store(&focus_stopped, 1, memory_order_release);
    return;
  }
  if (addr + size < addr ||
      !IsAppMem(addr) || !IsAppMem(addr + size - 1))
    return;
  atomic_store(&focus_end, addr + size, memory_order_relaxed);
  atomic_store(&focus_beg, addr, memory_order_release);
  atomic_store(&focus_stopped, 0, memory_order_release);
}

void PrintOwnershipProfile(ThreadState *) {
  if (!ownership_profile_enabled)
    return;
  static atomic_uint32_t printed;
  if (atomic_exchange(&printed, 1, memory_order_relaxed))
    return;

  static Counters total;
  auto *thread = reinterpret_cast<ProfileThread *>(
      atomic_load(&threads, memory_order_acquire));
  for (; thread; thread = thread->next)
    Add(&total, thread->counters);

  u64 private_accesses = total.private_first_touches + total.owner_hits;
  u64 shared_accesses = total.transitions + total.shared_hits;
  u64 private_cells = total.first_touches - total.transitions;
  u64 note_absorbed = total.note_claims + total.note_exact_hits +
                      total.note_cover_hits + total.note_merges;
  u64 note_fallback = total.cell_touches - note_absorbed;
  u64 pages = total.page_reached_1;
  u64 avoidable_pages = pages - total.page_fallback_seen;
  u64 fallback_only_pages = pages - total.page_absorbed_seen;
  u64 mixed_pages = total.page_absorbed_seen + total.page_fallback_seen - pages;
  u64 pages_1 = total.page_reached_1 - total.page_reached_2;
  u64 pages_2_64 = total.page_reached_2 - total.page_reached_65;
  u64 pages_65_128 = total.page_reached_65 - total.page_reached_129;
  u64 pages_129_192 = total.page_reached_129 - total.page_reached_193;
  u64 pages_193_255 = total.page_reached_193 - total.page_reached_256;
  u64 pages_256 = total.page_reached_256;
  static char json[8192];
  uptr pos = 0;
#define APPEND(...)                                                         \
  pos += internal_snprintf(json + pos, sizeof(json) - pos, __VA_ARGS__)
  APPEND("{\"schema\":2,\"cell_bytes\":8,\"allocation_ownership\":%s,"
         "\"callbacks\":%llu,\"callback_reads\":%llu,"
         "\"callback_writes\":%llu,\"callback_atomics\":%llu,",
         flags()->ownership_profile_allocations ? "true" : "false",
         total.callbacks, total.callback_reads, total.callback_writes,
         total.callback_atomics);
  APPEND("\"cell_touches\":%llu,\"first_touches\":%llu,"
         "\"private_cells\":%llu,\"shared_cells\":%llu,"
         "\"private_accesses\":%llu,\"shared_accesses\":%llu,",
         total.cell_touches, total.first_touches, private_cells,
         total.transitions, private_accesses, shared_accesses);
  APPEND("\"private_first_touches\":%llu,\"owner_hits\":%llu,"
         "\"transitions\":%llu,\"shared_hits\":%llu,"
         "\"private_csa_hits\":%llu,\"shared_csa_hits\":%llu,",
         total.private_first_touches, total.owner_hits, total.transitions,
         total.shared_hits, total.private_csa_hits, total.shared_csa_hits);
  APPEND("\"cell_reads\":%llu,\"cell_writes\":%llu,"
         "\"cell_atomics\":%llu,\"shadow_slots_0\":%llu,"
         "\"shadow_slots_1\":%llu,\"shadow_slots_2\":%llu,"
         "\"shadow_slots_3\":%llu,\"shadow_slots_4\":%llu,",
         total.cell_reads, total.cell_writes, total.cell_atomics,
         total.shadow_slots[0], total.shadow_slots[1], total.shadow_slots[2],
         total.shadow_slots[3], total.shadow_slots[4]);
  APPEND("\"note_absorbed_accesses\":%llu,\"note_fallback_accesses\":%llu,"
         "\"note_claims\":%llu,\"note_exact_hits\":%llu,"
         "\"note_cover_hits\":%llu,\"note_merges\":%llu,",
         note_absorbed, note_fallback, total.note_claims, total.note_exact_hits,
         total.note_cover_hits, total.note_merges);
  APPEND("\"note_owner_fallbacks\":%llu,\"note_merge_fallbacks\":%llu,"
         "\"note_ineligible_fallbacks\":%llu,"
         "\"note_fallback_hits\":%llu,"
         "\"note_absorbed_csa_hits\":%llu,"
         "\"note_fallback_csa_hits\":%llu,",
         total.note_owner_fallbacks, total.note_merge_fallbacks,
         total.note_ineligible_fallbacks, total.note_fallback_hits,
         total.note_absorbed_csa_hits, total.note_fallback_csa_hits);
  APPEND("\"shadow_exact_eligible\":%llu,"
         "\"shadow_exact_slot0_hits\":%llu,"
         "\"shadow_exact_late_hits\":%llu,"
         "\"shadow_exact_misses\":%llu,"
         "\"shadow_pages\":%llu,\"shadow_pages_absorbed_seen\":%llu,"
         "\"shadow_pages_fallback_seen\":%llu,",
         total.shadow_exact_eligible, total.shadow_exact_slot0_hits,
         total.shadow_exact_late_hits, total.shadow_exact_misses, pages,
         total.page_absorbed_seen, total.page_fallback_seen);
  APPEND("\"potentially_avoidable_shadow_pages\":%llu,"
         "\"mixed_path_shadow_pages\":%llu,"
         "\"fallback_only_shadow_pages\":%llu,"
         "\"shadow_pages_1_cell\":%llu,\"shadow_pages_2_64_cells\":%llu,"
         "\"shadow_pages_65_128_cells\":%llu,",
         avoidable_pages, mixed_pages, fallback_only_pages, pages_1, pages_2_64,
         pages_65_128);
  APPEND("\"shadow_pages_129_192_cells\":%llu,"
         "\"shadow_pages_193_255_cells\":%llu,"
         "\"shadow_pages_256_cells\":%llu,"
         "\"allocation_cells_tagged\":%llu,\"allocation_ranges\":%llu,",
         pages_129_192, pages_193_255, pages_256,
         total.allocation_cells_tagged, total.allocation_ranges);
  APPEND("\"shadow_beg\":%llu,\"shadow_end\":%llu,"
         "\"profile_beg\":%llu,\"profile_end\":%llu,"
         "\"cell_profile_end\":%llu,",
         static_cast<u64>(ShadowBeg()), static_cast<u64>(ShadowEnd()),
         static_cast<u64>(profile_beg), static_cast<u64>(profile_end),
         static_cast<u64>(cell_profile_end));
  APPEND("\"page_bitmap_beg\":%llu,\"page_bitmap_end\":%llu,"
         "\"page_profile_beg\":%llu,\"page_profile_end\":%llu,"
         "\"focus_beg\":%llu,\"focus_end\":%llu,\"page_size\":%zu}\n",
         static_cast<u64>(page_bitmap_beg), static_cast<u64>(page_bitmap_end),
         static_cast<u64>(page_profile_beg), static_cast<u64>(page_profile_end),
         static_cast<u64>(atomic_load(&focus_beg, memory_order_relaxed)),
         static_cast<u64>(atomic_load(&focus_end, memory_order_relaxed)),
         GetPageSizeCached());
#undef APPEND

  const char *path = flags()->ownership_profile_path;
  if (path && path[0]) {
    fd_t fd = OpenFile(path, WrOnly);
    if (fd == kInvalidFd || !WriteToFile(fd, json, internal_strlen(json))) {
      Printf("ThreadSanitizer: failed to write ownership profile '%s'\n", path);
    } else {
      CloseFile(fd);
    }
  } else {
    Printf("TSAN_OWNERSHIP_PROFILE %s", json);
  }
}

}  // namespace __tsan
