// The lock-order detector keeps 4096 mutexes in its graph by default; the
// 4097th flushes the graph, and every edge learned before the flush is gone.
// deadlock_detector_large_table=1 keeps 16384. Here a -> b is learned, then
// 6000 more mutexes get nodes (each locked under a), then b -> a closes the
// cycle. With the default table the flush has forgotten a -> b and nothing is
// reported; with the large one the inversion is. The two runs differ in the
// flag alone.
// RUN: %clangxx_tsan -O1 %s -o %t
// The small-table runs say so explicitly, so a harness that turns the large
// table on for every test cannot flip them.
// RUN: %env_tsan_opts=deadlock_detector_large_table=0 %run %t 2>&1 | FileCheck %s --check-prefix=SMALL --implicit-check-not=lock-order-inversion
// RUN: %env_tsan_opts=deadlock_detector_large_table=1 not %run %t 2>&1 | FileCheck %s --check-prefix=LARGE
// The pre-lock shortcut (deadlock_detector_exact_prelock=1) changes nothing
// here, with either table: the same report or the same silence.
// RUN: %env_tsan_opts=deadlock_detector_exact_prelock=1:deadlock_detector_large_table=0 %run %t 2>&1 | FileCheck %s --check-prefix=SMALL --implicit-check-not=lock-order-inversion
// RUN: %env_tsan_opts=deadlock_detector_exact_prelock=1:deadlock_detector_large_table=1 not %run %t 2>&1 | FileCheck %s --check-prefix=LARGE

#include <pthread.h>
#include <stdio.h>

static pthread_mutex_t a = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t b = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t many[6000];

int main() {
  pthread_mutex_lock(&a);
  pthread_mutex_lock(&b);
  pthread_mutex_unlock(&b);
  pthread_mutex_unlock(&a);
  for (int i = 0; i < 6000; i++) {
    pthread_mutex_init(&many[i], nullptr);
    pthread_mutex_lock(&a);
    pthread_mutex_lock(&many[i]);
    pthread_mutex_unlock(&many[i]);
    pthread_mutex_unlock(&a);
  }
  pthread_mutex_lock(&b);
  pthread_mutex_lock(&a);
  pthread_mutex_unlock(&a);
  pthread_mutex_unlock(&b);
  fprintf(stderr, "DONE\n");
  return 0;
}

// SMALL: DONE
// LARGE: WARNING: ThreadSanitizer: lock-order-inversion
// LARGE: DONE
