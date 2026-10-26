// ALS (atomic_load_skip): an acquire-order atomic load skips the acquire when
// this thread has already acquired the sync object's clock in its present
// state. Each mode is run with the skip off, on, and on with the verify mode
// (a hit does the acquire anyway and dies if the clock changes); the reports
// must be the same.
//
//   hit      main stores and loads its own atomic: the loads after the first
//            are hits.
//   store    W writes a plain global and then stores the atomic (release);
//            main spins on acquire loads until it sees the value, then reads
//            the global. The load that sees W's value must acquire: no race.
//   join     the same with W's release a fetch_add (a join into the sync
//            object's clock, not a copy). Under the mutant no-bump-join the
//            join leaves the version, main's load is a hit, the acquire is
//            skipped and the read of the global is reported as a race.
//   cas      the same with a compare-exchange.
//   reuse    the atomic lives in a heap block that is freed and allocated
//            again at the same address before W's store.
//   reset    a global reset (__tsan_flush_memory) between main's loads and
//            W's store.
//   rmwrel   as join, with a release-only fetch_add.
//   annot    as join, with W's release an AnnotateHappensBefore on the
//            atomic's address (the function Release) and a relaxed store.
//   casfail  main's own failed compare-exchange between its loads: the
//            version is put back, the next load is a hit and verify agrees.
//   steal    (audit A84 F1) T loads the atomic, then waits without calling
//            the runtime; main starts and joins 300 threads, more than there
//            are slots, so that T's slot is taken; T loads once more. A hit
//            takes no slot lock, where T would learn of the loss and attach
//            anew, so that load must not be a hit: the census counts one
//            miss for a lost slot. Under the mutant no-slot-check it is a
//            hit (2 hits, no such miss) by a thread that goes on under a
//            (sid, epoch) another thread has since run under.
//            What this mode does NOT show is a lost race: T then writes a
//            global that a thread started afterwards writes too, and stock
//            itself reports nothing for that pair once the slot was taken
//            (3 runs of 3, with 300 and with 600 threads; with 250 or fewer,
//            no slot taken, the race is reported every time). So the check
//            is pinned on the load's outcome, not on a report.
//   empty    main stores (release), a global reset drops every sync clock,
//            main loads: the sync object is there and its clock is empty, so
//            every load is a hit with nothing cached (a join with an empty
//            clock changes nothing). Under the mutant no-empty a reset leaves
//            "no version" and none of them hits. That a release after the
//            reset is still acquired is the mode reset.
//   cache    128 neighbouring atomics, each stored (release) and then loaded
//            in turn, twice. With the default cache of 64 entries, mapped by
//            address, two of them share every entry and each load evicts
//            what the other needs: the first round acquires, the second
//            finds nothing cached (the census shows only the 9 hits of the
//            ten loads of x that every mode makes first). With
//            atomic_load_skip_cache=256 the second round is 128 hits more
//            (137). Either way no report and verify
//            agrees: the size changes how often a load hits, nothing else.
//   unsync   positive control for "no race": W never releases (a relaxed
//            store), so the read of the global IS a race, skip on or off.
//
// RUN: %clang_tsan -O1 %s -o %t
// RUN: %run %t hit 2>&1 | FileCheck %s --check-prefix=CLEAN
// RUN: %env_tsan_opts=atomic_load_skip=1:atomic_load_census=1 %run %t hit 2>&1 | FileCheck %s --check-prefixes=CLEAN,HITS
// RUN: %env_tsan_opts=atomic_load_census=1 %run %t hit 2>&1 | FileCheck %s --check-prefixes=CLEAN,HITS
// RUN: %env_tsan_opts=atomic_load_skip=1:atomic_load_skip_verify=1 %run %t hit 2>&1 | FileCheck %s --check-prefix=CLEAN
// RUN: %run %t store 2>&1 | FileCheck %s --check-prefix=CLEAN
// RUN: %env_tsan_opts=atomic_load_skip=1:atomic_load_census=1 %run %t store 2>&1 | FileCheck %s --check-prefixes=CLEAN,HITS
// RUN: %env_tsan_opts=atomic_load_skip=1:atomic_load_skip_verify=1 %run %t store 2>&1 | FileCheck %s --check-prefix=CLEAN
// RUN: %run %t join 2>&1 | FileCheck %s --check-prefix=CLEAN
// RUN: %env_tsan_opts=atomic_load_skip=1:atomic_load_census=1 %run %t join 2>&1 | FileCheck %s --check-prefixes=CLEAN,HITS
// RUN: %env_tsan_opts=atomic_load_skip=1:atomic_load_skip_verify=1 %run %t join 2>&1 | FileCheck %s --check-prefix=CLEAN
// RUN: %env_tsan_opts=atomic_load_skip=1:atomic_load_mutant=no-bump-join %deflake %run %t join 2>&1 | FileCheck %s --check-prefix=RACE
// RUN: %env_tsan_opts=atomic_load_skip=1:atomic_load_skip_verify=1:atomic_load_mutant=no-bump-join not %run %t join 2>&1 | FileCheck %s --check-prefix=VERIFY
// RUN: %run %t cas 2>&1 | FileCheck %s --check-prefix=CLEAN
// RUN: %env_tsan_opts=atomic_load_skip=1:atomic_load_skip_verify=1 %run %t cas 2>&1 | FileCheck %s --check-prefix=CLEAN
// RUN: %env_tsan_opts=atomic_load_skip=1:atomic_load_mutant=no-bump-join %deflake %run %t cas 2>&1 | FileCheck %s --check-prefix=RACE
// RUN: %run %t rmwrel 2>&1 | FileCheck %s --check-prefix=CLEAN
// RUN: %env_tsan_opts=atomic_load_skip=1:atomic_load_skip_verify=1 %run %t rmwrel 2>&1 | FileCheck %s --check-prefix=CLEAN
// RUN: %env_tsan_opts=atomic_load_skip=1:atomic_load_mutant=no-bump-join %deflake %run %t rmwrel 2>&1 | FileCheck %s --check-prefix=RACE
// RUN: %run %t annot 2>&1 | FileCheck %s --check-prefix=CLEAN
// RUN: %env_tsan_opts=atomic_load_skip=1:atomic_load_skip_verify=1 %run %t annot 2>&1 | FileCheck %s --check-prefix=CLEAN
// RUN: %env_tsan_opts=atomic_load_skip=1:atomic_load_mutant=no-bump-join %deflake %run %t annot 2>&1 | FileCheck %s --check-prefix=RACE
// RUN: %env_tsan_opts=atomic_load_skip=1:atomic_load_skip_verify=1:atomic_load_census=1 %run %t casfail 2>&1 | FileCheck %s --check-prefixes=CLEAN,HITS
// RUN: %env_tsan_opts=atomic_load_census=1 %run %t steal 2>&1 | FileCheck %s --check-prefixes=DONE,SLOT
// RUN: %env_tsan_opts=atomic_load_skip=1:atomic_load_census=1 %run %t steal 2>&1 | FileCheck %s --check-prefixes=DONE,SLOT
// RUN: %env_tsan_opts=atomic_load_skip=1:atomic_load_skip_verify=1:atomic_load_census=1 %run %t steal 2>&1 | FileCheck %s --check-prefixes=DONE,SLOT
// RUN: %env_tsan_opts=atomic_load_skip=1:atomic_load_census=1:atomic_load_mutant=no-slot-check %run %t steal 2>&1 | FileCheck %s --check-prefixes=DONE,NOSLOT
// RUN: %env_tsan_opts=atomic_load_skip=1:atomic_load_skip_verify=1:atomic_load_census=1 %run %t empty 2>&1 | FileCheck %s --check-prefixes=CLEAN,EMPTY
// RUN: %env_tsan_opts=atomic_load_census=1 %run %t empty 2>&1 | FileCheck %s --check-prefixes=CLEAN,EMPTY
// RUN: %env_tsan_opts=atomic_load_skip=1:atomic_load_census=1:atomic_load_mutant=no-empty %run %t empty 2>&1 | FileCheck %s --check-prefixes=CLEAN,NOEMPTY
// RUN: %env_tsan_opts=atomic_load_skip=1:atomic_load_skip_verify=1:atomic_load_census=1 %run %t cache 2>&1 | FileCheck %s --check-prefixes=CLEAN,CACHE64
// RUN: %env_tsan_opts=atomic_load_skip=1:atomic_load_skip_verify=1:atomic_load_census=1:atomic_load_skip_cache=256 %run %t cache 2>&1 | FileCheck %s --check-prefixes=CLEAN,CACHE256
// RUN: %run %t reuse 2>&1 | FileCheck %s --check-prefix=CLEAN
// RUN: %env_tsan_opts=atomic_load_skip=1:atomic_load_skip_verify=1 %run %t reuse 2>&1 | FileCheck %s --check-prefix=CLEAN
// RUN: %run %t reset 2>&1 | FileCheck %s --check-prefix=CLEAN
// RUN: %env_tsan_opts=atomic_load_skip=1:atomic_load_skip_verify=1 %run %t reset 2>&1 | FileCheck %s --check-prefix=CLEAN
// RUN: %deflake %run %t unsync 2>&1 | FileCheck %s --check-prefix=RACE
// RUN: %env_tsan_opts=atomic_load_skip=1 %deflake %run %t unsync 2>&1 | FileCheck %s --check-prefix=RACE
//
// CLEAN-NOT:  WARNING: ThreadSanitizer
// CLEAN-NOT:  ALS verify
// CLEAN:      DONE
// HITS:       ALS census{{.*}} {{[1-9][0-9]*}} hits
// RACE:       WARNING: ThreadSanitizer: data race
// EMPTY:      ALS census{{.*}} 10 hits on an empty clock
// NOEMPTY:    ALS census{{.*}} 0 hits on an empty clock
// CACHE64:    ALS census{{.*}} 9 hits on a clock the thread had acquired
// CACHE256:   ALS census{{.*}} 137 hits on a clock the thread had acquired
// DONE-NOT:   ALS verify
// DONE:       DONE
// SLOT:       ALS census{{.*}} 1 hits{{.*}} 1 the thread no longer owned its slot
// NOSLOT:     ALS census{{.*}} 2 hits{{.*}} 0 the thread no longer owned its slot
// VERIFY:     ALS verify: a hit on
#include "test.h"
#include <string.h>

void __tsan_flush_memory(void);
void AnnotateHappensBefore(const char *f, int l, void *addr);

static long x_static;
static long *x = &x_static;
static long g; // W's plain write, main's plain read
static const char *mode;
static int is(const char *m) { return !strcmp(mode, m); }

__attribute__((noinline)) static long load(void) {
  return __atomic_load_n(x, __ATOMIC_ACQUIRE);
}

static void *writer(void *arg) {
  barrier_wait(&barrier); // main has loaded: its cache holds the old version
  g = 42;
  if (is("join")) {
    __atomic_fetch_add(x, 1, __ATOMIC_ACQ_REL);
  } else if (is("rmwrel")) {
    __atomic_fetch_add(x, 1, __ATOMIC_RELEASE);
  } else if (is("annot")) {
    AnnotateHappensBefore(__FILE__, __LINE__, x);
    __atomic_store_n(x, 1, __ATOMIC_RELAXED);
  } else if (is("cas")) {
    long e = 0;
    __atomic_compare_exchange_n(x, &e, 1, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
  } else if (is("unsync")) {
    __atomic_store_n(x, 1, __ATOMIC_RELAXED);
  } else {
    __atomic_store_n(x, 1, __ATOMIC_RELEASE);
  }
  return 0;
}

// steal: signals TSan does not see.
static volatile int sflag[3];
__attribute__((disable_sanitizer_instrumentation)) static void sset(int i) {
  sflag[i] = 1;
}
__attribute__((disable_sanitizer_instrumentation)) static int sget(int i) {
  return sflag[i];
}
static long gx; // steal: written by T and by main, unordered
static void *nop_thread(void *arg) { return 0; }
static void *steal_second(void *arg) {
  gx = 1; // races with T's write
  return 0;
}
__attribute__((disable_sanitizer_instrumentation)) static void swait(int i) {
  while (!sflag[i]) // no call into the runtime: nothing that takes the slot
    __asm__ volatile("" ::: "memory"); // lock, and no trace event either
}
static void *steal_loader(void *arg) {
  long acc = load(); // acquires: the address is in this thread's cache
  acc += load();     // a hit
  sset(0);
  swait(1);
  acc += load(); // the slot is lost: this one must not be a hit
  gx = acc;
  sset(2);
  return 0;
}

int main(int argc, char **argv) {
  mode = argv[1];
  barrier_init(&barrier, 2);
  if (is("steal")) {
    __atomic_store_n(x, 0, __ATOMIC_RELEASE);
    pthread_t t, w;
    pthread_create(&t, 0, steal_loader, 0);
    while (!sget(0))
      usleep(100);
    for (int i = 0; i < 300; i++) { // every slot is taken once, T's among them
      pthread_create(&w, 0, nop_thread, 0);
      pthread_join(w, 0);
    }
    sset(1);
    while (!sget(2))
      usleep(100);
    // The second write, by a thread started now (see the note at "steal": no
    // report is expected or checked for this pair).
    pthread_create(&w, 0, steal_second, 0);
    pthread_join(w, 0);
    pthread_join(t, 0);
    // gx is read: a static that is only written is deleted by the optimizer,
    // and the race with it.
    fprintf(stderr, "DONE %ld\n", gx);
    return 0;
  }
  if (is("reuse"))
    x = (long *)malloc(sizeof(long));
  __atomic_store_n(x, 0, __ATOMIC_RELEASE); // the address has a sync object
  long acc = 0;
  for (int i = 0; i < 10; i++)
    acc += load(); // the first acquires, the others are hits
  if (is("hit")) {
    fprintf(stderr, "DONE %ld\n", acc);
    return 0;
  }
  if (is("empty")) {
    __tsan_flush_memory(); // every sync clock is dropped
    for (int i = 0; i < 10; i++)
      acc += load();
    fprintf(stderr, "DONE %ld\n", acc);
    return 0;
  }
  if (is("cache")) {
    static long many[128];
    for (int i = 0; i < 128; i++)
      __atomic_store_n(&many[i], i, __ATOMIC_RELEASE);
    acc = 0;
    for (int round = 0; round < 2; round++)
      for (int i = 0; i < 128; i++)
        acc += __atomic_load_n(&many[i], __ATOMIC_ACQUIRE);
    fprintf(stderr, "DONE %ld\n", acc);
    return 0;
  }
  if (is("casfail")) {
    long e = 7; // not the value: the exchange fails
    __atomic_compare_exchange_n(x, &e, 1, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
    for (int i = 0; i < 10; i++)
      acc += load(); // hits on the version put back
    fprintf(stderr, "DONE %ld\n", acc);
    return 0;
  }
  if (is("reuse")) {
    void *old = x;
    free(x);
    x = (long *)malloc(sizeof(long));
    if (x != old)
      fprintf(stderr, "note: the block moved\n");
    __atomic_store_n(x, 0, __ATOMIC_RELEASE);
    acc += load();
  }
  if (is("reset")) {
    __tsan_flush_memory();
    acc += load();
  }
  pthread_t t;
  pthread_create(&t, 0, writer, 0);
  barrier_wait(&barrier);
  while (load() != 1) // hits until W's release; then the one that acquires
    usleep(100);
  acc += g; // ordered after W's write only through that acquire
  pthread_join(t, 0);
  fprintf(stderr, "DONE %ld\n", acc);
  return 0;
}
