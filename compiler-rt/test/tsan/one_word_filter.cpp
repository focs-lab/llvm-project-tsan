// RUN: %clangxx_tsan -O1 -mllvm -tsan-one-word-filter %s -o %t
// RUN: env TSAN_OPTIONS=one_word_filter=1 %deflake %run %t race 2>&1 | FileCheck %s --check-prefix=RACE
// RUN: env TSAN_OPTIONS=one_word_filter=0 %deflake %run %t race 2>&1 | FileCheck %s --check-prefix=RACE
// RUN: env TSAN_OPTIONS=one_word_filter=1 %run %t private 2>&1 | FileCheck %s --check-prefix=PRIVATE
// RUN: env TSAN_OPTIONS=one_word_filter=1 %run %t state 2>&1 | FileCheck %s --check-prefix=STATE

#include "test.h"

#include <atomic>
#include <stdint.h>
#include <string.h>

struct alignas(8) Cell {
  volatile uint64_t value;
};

std::atomic<Cell *> published;
std::atomic<int> next_epoch;

extern "C" uintptr_t __tsan_one_word_lo_beg;
extern "C" uintptr_t __tsan_one_word_lo_end;
extern "C" uintptr_t __tsan_one_word_heap_beg;
extern "C" uintptr_t __tsan_one_word_shadow_beg;

__attribute__((disable_sanitizer_instrumentation)) uint32_t
load_note(Cell *cell) {
  uintptr_t addr = reinterpret_cast<uintptr_t>(cell);
  uintptr_t index;
  if (addr < __tsan_one_word_lo_end) {
    index = (addr - __tsan_one_word_lo_beg) / 8;
  } else {
    index = (__tsan_one_word_lo_end - __tsan_one_word_lo_beg) / 8;
    index += (addr - __tsan_one_word_heap_beg) / 8;
  }
  auto *note = reinterpret_cast<uint32_t *>(__tsan_one_word_shadow_beg);
  return __atomic_load_n(note + index, __ATOMIC_ACQUIRE);
}

__attribute__((noinline)) void write_private(Cell *cell, uint64_t value);

void *producer(void *) {
  Cell *cell = static_cast<Cell *>(malloc(sizeof(Cell)));
  next_epoch.store(1, std::memory_order_release);
  write_private(cell, 1);
  published.store(cell, std::memory_order_relaxed);
  return cell;
}

void *consumer(void *) {
  Cell *cell;
  while ((cell = published.load(std::memory_order_relaxed)) == nullptr)
    sched_yield();
  write_private(cell, 2);
  return nullptr;
}

void *revoke(void *arg) {
  write_private(static_cast<Cell *>(arg), 3);
  return nullptr;
}

__attribute__((noinline)) void write_private(Cell *cell, uint64_t value) {
  cell->value = value;
}

__attribute__((noinline)) void write_private_loop(Cell *cell, int iterations) {
  for (int i = 0; i != iterations; ++i)
    cell->value = i;
}

int main(int argc, char **argv) {
  if (strcmp(argv[1], "state") == 0) {
    Cell *cell = static_cast<Cell *>(malloc(sizeof(Cell)));
    fprintf(stderr, "allocated %s\n", load_note(cell) ? "exclusive" : "shared");
    write_private(cell, 1);
    fprintf(stderr, "owner %s\n", load_note(cell) ? "exclusive" : "shared");
    pthread_t thread;
    pthread_create(&thread, nullptr, revoke, cell);
    pthread_join(thread, nullptr);
    fprintf(stderr, "revoked %s\n", load_note(cell) ? "exclusive" : "shared");
    free(cell);
    return 0;
  }

  if (strcmp(argv[1], "private") == 0) {
    Cell *cell = static_cast<Cell *>(malloc(sizeof(Cell)));
    int iterations = argc > 2 ? atoi(argv[2]) : 1000;
    write_private_loop(cell, iterations);
    fprintf(stderr, "private done\n");
    free(cell);
    return 0;
  }

  pthread_t first;
  pthread_t second;
  pthread_create(&first, nullptr, producer, nullptr);
  pthread_create(&second, nullptr, consumer, nullptr);
  void *cell;
  pthread_join(first, &cell);
  pthread_join(second, nullptr);
  free(cell);
  return 0;
}

// RACE: WARNING: ThreadSanitizer: data race
// RACE-COUNT-2: write_private
// PRIVATE-NOT: WARNING: ThreadSanitizer
// PRIVATE: private done
// STATE-NOT: WARNING: ThreadSanitizer
// STATE: allocated exclusive
// STATE: owner exclusive
// STATE: revoked shared
