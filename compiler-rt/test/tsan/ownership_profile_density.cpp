// RUN: %clangxx_tsan -O1 -mllvm -tsan-instrument-memory-accesses=0 %s -o %t
// RUN: env TSAN_OPTIONS=ownership_profile=1:ownership_profile_allocations=0:ownership_profile_wait_for_range=1 %run %t 2>&1 | FileCheck %s

#include <pthread.h>
#include <stdint.h>
#include <sys/mman.h>

extern "C" void __tsan_read8(void *);
extern "C" void __tsan_write1(void *);
extern "C" void __tsan_write8(void *);
extern "C" void __tsan_write16(void *);
extern "C" void __tsan_ownership_profile_set_range(void *, uintptr_t);

constexpr uintptr_t kAppBytesPerShadowPage = 2048;
constexpr int kPages = 60;
char *memory;

void *ChangeOwner(void *) {
  for (int page = 30; page < 40; ++page)
    __tsan_write8(memory + page * kAppBytesPerShadowPage);
  return nullptr;
}

int main() {
  memory = static_cast<char *>(
      mmap(nullptr, kPages * kAppBytesPerShadowPage, PROT_READ | PROT_WRITE,
           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
  __tsan_ownership_profile_set_range(memory, kPages * kAppBytesPerShadowPage);

  for (int page = 0; page < 10; ++page)
    for (int cell = 0; cell < 256; ++cell)
      __tsan_write8(memory + page * kAppBytesPerShadowPage + cell * 8);
  for (int page = 10; page < 15; ++page)
    for (int cell = 0; cell < 193; ++cell)
      __tsan_write8(memory + page * kAppBytesPerShadowPage + cell * 8);
  for (int page = 15; page < 20; ++page)
    for (int cell = 0; cell < 129; ++cell)
      __tsan_write8(memory + page * kAppBytesPerShadowPage + cell * 8);
  for (int page = 20; page < 25; ++page)
    for (int cell = 0; cell < 65; ++cell)
      __tsan_write8(memory + page * kAppBytesPerShadowPage + cell * 8);

  for (int page = 25; page < 30; ++page)
    if (page != 28)
      __tsan_write8(memory + page * kAppBytesPerShadowPage);
  __tsan_write8(memory + 26 * kAppBytesPerShadowPage);
  __tsan_read8(memory + 27 * kAppBytesPerShadowPage);
  __tsan_write1(memory + 28 * kAppBytesPerShadowPage);
  __tsan_write1(memory + 28 * kAppBytesPerShadowPage + 1);

  for (int page = 30; page < 40; ++page)
    __tsan_write8(memory + page * kAppBytesPerShadowPage);
  pthread_t thread;
  pthread_create(&thread, nullptr, ChangeOwner, nullptr);
  pthread_join(thread, nullptr);

  for (int page = 40; page < 50; ++page)
    __tsan_write1(memory + page * kAppBytesPerShadowPage);
  static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
  pthread_mutex_lock(&mutex);
  pthread_mutex_unlock(&mutex);
  for (int page = 40; page < 50; ++page) {
    __tsan_write1(memory + page * kAppBytesPerShadowPage + 1);
    __tsan_write1(memory + page * kAppBytesPerShadowPage + 1);
  }

  for (int page = 50; page < 60; ++page)
    __tsan_write16(memory + page * kAppBytesPerShadowPage);

  __tsan_ownership_profile_set_range(nullptr, 0);
}

// CHECK: TSAN_OWNERSHIP_PROFILE {"schema":2
// CHECK-SAME: "cell_touches":4573
// CHECK-SAME: "first_touches":4540
// CHECK-SAME: "private_cells":4530
// CHECK-SAME: "shared_cells":10
// CHECK-SAME: "note_absorbed_accesses":4523
// CHECK-SAME: "note_fallback_accesses":50
// CHECK-SAME: "note_claims":4520
// CHECK-SAME: "note_exact_hits":1
// CHECK-SAME: "note_cover_hits":1
// CHECK-SAME: "note_merges":1
// CHECK-SAME: "note_owner_fallbacks":10
// CHECK-SAME: "note_merge_fallbacks":10
// CHECK-SAME: "note_ineligible_fallbacks":20
// CHECK-SAME: "note_fallback_hits":10
// CHECK-SAME: "note_absorbed_csa_hits":{{[1-9][0-9]*}}
// CHECK-SAME: "note_fallback_csa_hits":10
// CHECK-SAME: "shadow_exact_eligible":4553
// CHECK-SAME: "shadow_exact_slot0_hits":{{[1-9][0-9]*}}
// CHECK-SAME: "shadow_exact_late_hits":10
// CHECK-SAME: "shadow_exact_misses":{{[1-9][0-9]*}}
// CHECK-SAME: "shadow_pages":60
// CHECK-SAME: "potentially_avoidable_shadow_pages":30
// CHECK-SAME: "mixed_path_shadow_pages":20
// CHECK-SAME: "fallback_only_shadow_pages":10
// CHECK-SAME: "shadow_pages_1_cell":25
// CHECK-SAME: "shadow_pages_2_64_cells":10
// CHECK-SAME: "shadow_pages_65_128_cells":5
// CHECK-SAME: "shadow_pages_129_192_cells":5
// CHECK-SAME: "shadow_pages_193_255_cells":5
// CHECK-SAME: "shadow_pages_256_cells":10
