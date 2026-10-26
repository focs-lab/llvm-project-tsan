// RUN: %clangxx_tsan -O1 -Xclang -disable-lifetime-markers -mllvm -tsan-use-escape-analysis %s -o %t && %deflake %run %t 2>&1 | FileCheck %s
#include "test.h"

int Shared;

// Keep Local eligible for the runtime guard that skips instrumentation.
__attribute__((noinline)) void store_selected(bool UseShared,
                                              int *SharedAddress) {
  int Local = 0;
  int *Address = UseShared ? SharedAddress : &Local;
  *Address = 1;
}

void *Thread(void *Arg) {
  barrier_wait(&barrier);
  store_selected(true, static_cast<int *>(Arg));
  return nullptr;
}

int main() {
  // Want to ensure that we do not just directly instrument it
  store_selected(false, &Shared);

  barrier_init(&barrier, 2);
  pthread_t Threads[2];
  pthread_create(&Threads[0], nullptr, Thread, &Shared);
  pthread_create(&Threads[1], nullptr, Thread, &Shared);
  pthread_join(Threads[0], nullptr);
  pthread_join(Threads[1], nullptr);
  return 0;
}

// CHECK: WARNING: ThreadSanitizer: data race
// CHECK: #0 store_selected