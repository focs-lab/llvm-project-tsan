// The second deterministic path by which deadlock_detector_large_table=1 can
// miss what the default table reports: pthread_rwlock_timedrdlock
// has no pre-lock and no try flag, so its post-lock adds held -> r to the
// graph without a cycle check. As with the recursive read lock
// (deadlock_detector_large_table_lost.cpp), the default table forgets that
// edge at a flush and reports when the inversion is taken again; the large
// table keeps it and never reports.
//   1. rdlock(r); lock(x): r -> x, checked.
//   2. lock(x); timedrdlock(r): x -> r, unchecked: no report in either table.
//      With "plain", step 2 uses pthread_rwlock_rdlock instead, is checked,
//      and both tables report there: the control that step 2 is the cycle.
//   3. 5000 more mutexes get nodes (each locked under z): the default table
//      flushes, the large one does not.
//   4. rdlock(r); lock(x), then lock(x); rdlock(r): the default table
//      re-learns r -> x, checks x -> r and reports; the large table returns.
// RUN: %clangxx_tsan -O1 %s -o %t
// RUN: %env_tsan_opts=deadlock_detector_large_table=0:print_deadlock_detector_stats=1 not %run %t 2>&1 | FileCheck %s --check-prefix=SMALL
// RUN: %env_tsan_opts=deadlock_detector_large_table=1:print_deadlock_detector_stats=1 %run %t 2>&1 | FileCheck %s --check-prefix=LARGE --implicit-check-not=lock-order-inversion
// RUN: %env_tsan_opts=deadlock_detector_large_table=0 not %run %t plain 2>&1 | FileCheck %s --check-prefix=PLAIN
// RUN: %env_tsan_opts=deadlock_detector_large_table=1 not %run %t plain 2>&1 | FileCheck %s --check-prefix=PLAIN
#include <pthread.h>
#include <stdio.h>
#include <time.h>

static pthread_rwlock_t r = PTHREAD_RWLOCK_INITIALIZER;
static pthread_mutex_t x = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t z = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t many[5000];

int main(int argc, char **argv) {
  pthread_rwlock_rdlock(&r);
  pthread_mutex_lock(&x);
  pthread_mutex_unlock(&x);
  pthread_rwlock_unlock(&r);
  fprintf(stderr, "step 1 done\n");
  pthread_mutex_lock(&x);
  if (argc > 1) {
    pthread_rwlock_rdlock(&r);
  } else {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_sec += 60;
    if (pthread_rwlock_timedrdlock(&r, &ts) != 0)
      return 1;
  }
  pthread_rwlock_unlock(&r);
  pthread_mutex_unlock(&x);
  fprintf(stderr, "step 2 done\n");
  if (argc > 1)
    return 0;
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
// SMALL-NOT: lock-order-inversion
// SMALL: step 2 done
// SMALL: WARNING: ThreadSanitizer: lock-order-inversion
// SMALL: DONE
// SMALL: graph flushes={{[1-9]}}
// LARGE: step 2 done
// LARGE: DONE
// LARGE: graph flushes=0,
// PLAIN: step 1 done
// PLAIN: WARNING: ThreadSanitizer: lock-order-inversion
// PLAIN: step 2 done
