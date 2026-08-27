// REQUIRES: linux
// RUN: %clangxx_tsan -O1 -mllvm -tsan-shadow-resident-filter %s -o %t
// RUN: %run %t private 2>&1 | FileCheck %s --check-prefix=PRIVATE
// RUN: %deflake %run %t race 2>&1 | FileCheck %s --check-prefix=RACE

#include "test.h"

#include <string.h>

long Global;

__attribute__((noinline)) void PrivateLoop(volatile long *cell) {
  for (long i = 0; i != 100000; ++i)
    *cell = i;
}

__attribute__((noinline)) void WriteGlobal(long value) { Global = value; }

void *RaceThread(void *) {
  barrier_wait(&barrier);
  WriteGlobal(1);
  return nullptr;
}

int main(int argc, char **argv) {
  if (argc == 2 && strcmp(argv[1], "private") == 0) {
    auto *cell = new long;
    PrivateLoop(cell);
    delete cell;
    fprintf(stderr, "private done\n");
    return 0;
  }

  barrier_init(&barrier, 2);
  pthread_t thread;
  pthread_create(&thread, nullptr, RaceThread, nullptr);
  barrier_wait(&barrier);
  WriteGlobal(2);
  pthread_join(thread, nullptr);
  return 0;
}

// PRIVATE: private done
// RACE: WARNING: ThreadSanitizer: data race
