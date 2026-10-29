// What deadlock_detector_large_table=1 can miss. A recursive read lock of a
// rwlock already held puts held -> r into the graph without a cycle check (the
// detector's FIXME path). The default table forgets those edges at its next
// flush and checks them again when they are re-added, so it reports the
// inversion later; the large table keeps them, its fast path
// returns, and it never reports. With 5000 mutexes the default table flushes
// and the large one does not.
//   1. rdlock(r); lock(x); rdlock(r) again; unlock all: r -> x checked,
//      x -> r unchecked, nothing reported in either table.
//   2. 5000 more mutexes get nodes (each locked under z).
//   3. rdlock(r); lock(x): the default table re-learns r -> x.
//   4. lock(x); rdlock(r): the default table checks x -> r and reports; the
//      large table already has every edge and returns.
// The test pins both outcomes; the second is the documented loss. The
// detector statistics show that the default table flushed and the large one
// did not.
// RUN: %clangxx_tsan -O1 %s -o %t
// RUN: %env_tsan_opts=deadlock_detector_large_table=0:print_deadlock_detector_stats=1 not %run %t 2>&1 | FileCheck %s --check-prefix=SMALL
// RUN: %env_tsan_opts=deadlock_detector_large_table=1:print_deadlock_detector_stats=1 %run %t 2>&1 | FileCheck %s --check-prefix=LARGE --implicit-check-not=lock-order-inversion
#include <pthread.h>
#include <stdio.h>

static pthread_rwlock_t r = PTHREAD_RWLOCK_INITIALIZER;
static pthread_mutex_t x = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t z = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t many[5000];

int main() {
  pthread_rwlock_rdlock(&r);
  pthread_mutex_lock(&x);
  pthread_rwlock_rdlock(&r);
  pthread_rwlock_unlock(&r);
  pthread_mutex_unlock(&x);
  pthread_rwlock_unlock(&r);
  fprintf(stderr, "step 1 done\n");
  pthread_mutex_lock(&z);
  for (int i = 0; i < 5000; i++) {
    pthread_mutex_init(&many[i], NULL);
    pthread_mutex_lock(&many[i]);
    pthread_mutex_unlock(&many[i]);
  }
  pthread_mutex_unlock(&z);
  pthread_rwlock_rdlock(&r);
  pthread_mutex_lock(&x);
  pthread_mutex_unlock(&x);
  pthread_rwlock_unlock(&r);
  pthread_mutex_lock(&x);
  pthread_rwlock_rdlock(&r);
  pthread_rwlock_unlock(&r);
  pthread_mutex_unlock(&x);
  fprintf(stderr, "DONE\n");
  return 0;
}

// SMALL: step 1 done
// SMALL: WARNING: ThreadSanitizer: lock-order-inversion
// SMALL: DONE
// SMALL: graph flushes={{[1-9]}}
// LARGE: step 1 done
// LARGE: DONE
// LARGE: graph flushes=0,
