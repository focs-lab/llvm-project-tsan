// race_suite.cpp
//
// Build separately per race with -DRACE_CASE=N, e.g.:
//   clang++ -fsanitize=thread -mllvm -tsan-enable-l0=false -DRACE_CASE=1 -O1 -g
//   race_suite.cpp -o race1_base clang++ -fsanitize=thread -mllvm
//   -tsan-enable-l0=true  -DRACE_CASE=1 -O1 -g race_suite.cpp -o race1_l0
//
// Each case is a genuinely racy program (undefined behavior under the C++
// memory model), intentionally left unsynchronized so TSan should flag it.

#include <chrono>
#include <cstdio>
#include <thread>

#if RACE_CASE == 1
// Simple write/write race on a single global — the exact pattern your
// interpose test showed collapsing to near-zero real calls after warmup.
volatile long g_counter = 0;
void worker() {
  for (int i = 0; i < 100000; ++i)
    g_counter++;
}
int main() {
  std::thread t1(worker), t2(worker);
  t1.join();
  t2.join();
  printf("counter=%ld\n", (long)g_counter);
}

#elif RACE_CASE == 2
// Write/read race, staggered timing so it should trigger close to 100% of runs.
volatile int g_flag = 0;
void writer() {
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  g_flag = 42;
}
void reader() {
  volatile int x = g_flag; // unsynchronized read while writer may be mid-write
  (void)x;
}
int main() {
  std::thread t1(writer), t2(reader);
  t1.join();
  t2.join();
}

#elif RACE_CASE == 3
// Race on a heap-allocated address (different allocation pattern than
// a global/static — tests whether cache-slot aliasing behaves differently
// for heap addresses).
volatile long *g_heap_val;
void init() { g_heap_val = new long(0); }
void worker() {
  for (int i = 0; i < 50000; ++i)
    (*g_heap_val)++;
}
int main() {
  init();
  std::thread t1(worker), t2(worker);
  t1.join();
  t2.join();
  printf("val=%ld\n", (long)*g_heap_val);
  delete g_heap_val;
}

#elif RACE_CASE == 4
// Race across an array — many distinct addresses, all contended by both
// threads, exercising cache-slot collisions across a wider address range.
volatile long g_arr[16] = {0};
void worker(int id) {
  for (int i = 0; i < 20000; ++i)
    g_arr[i % 16]++;
}
int main() {
  std::thread t1(worker, 0), t2(worker, 1);
  t1.join();
  t2.join();
}

#elif RACE_CASE == 5
// Single-shot test targeting the epoch-never-bumps bug specifically.
//
// Thread A touches X once (a "warmup" write — guaranteed cache miss, since
// it's A's first-ever access to X), then after a delay writes X *again*,
// unsynchronized, with no loop repetition to mask what happens.
//
// If the L0 cache is sound, that second write should still be reported as
// racing with B's write. If the epoch never advances, A's second write will
// silently hit the cache (same address, same write-bit, same never-changed
// epoch) and never reach TSan's real runtime at all — meaning the only
// thing that *can* be detected is a race against A's first "warmup" write,
// not the actual later one. Check which source line TSan reports for
// thread A's access: line ~X (warmup) vs line ~Y (the real second write)
// tells you which one actually got recorded.
volatile long g_x = 0;

void thread_a() {
  g_x = 1; // warmup write (line A1)
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  g_x = 2; // the intended racy write (line A2)
}

void thread_b() {
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  g_x = 3; // B's write, timed to land inside A's sleep
}

int main() {
  std::thread t1(thread_a), t2(thread_b);
  t1.join();
  t2.join();
  printf("g_x=%ld\n", (long)g_x);
}

#endif