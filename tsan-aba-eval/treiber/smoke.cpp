// Smoke test for the Treiber stack in compiler-rt/test/tsan/Inputs/aba_treiber.h
// (the leak variant: popped nodes are never freed, so no ABA is possible).
//
// 1. Single thread: LIFO order and empty-pop behaviour.
// 2. Four threads pushing and popping concurrently: every pushed value must
//    come out exactly once (nothing lost, nothing duplicated).
//
// Every check prints "expected ... actual ...", whether it passes or fails.
// Under TSan this must print "done" with no warnings at all: no data race,
// no ABA report.
#include "aba_treiber.h"

#include <cstdio>
#include <cstdlib>
#include <latch>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int kThreads = 4;
constexpr int kPerThread = 10'000;
constexpr int kTotal = kThreads * kPerThread;
constexpr int kMaxListed = 10;  // how many lost/duplicated values to print

std::string show(std::optional<int> v) {
  return v ? std::to_string(*v) : std::string("empty");
}

[[noreturn]] void fail() {
  std::fprintf(stderr, "FAILED\n");
  std::exit(1);
}

// Pops once and compares with what the stack should return.
void expect_pop(aba_treiber::TreiberStack &s, std::optional<int> expected,
                const char *what) {
  std::optional<int> actual = s.pop();
  bool ok = actual == expected;
  std::fprintf(stderr, "  [%s] %s: expected %s, actual %s\n",
               ok ? "ok" : "FAIL", what, show(expected).c_str(),
               show(actual).c_str());
  if (!ok)
    fail();
}

void expect_eq(long actual, long expected, const char *what) {
  bool ok = actual == expected;
  std::fprintf(stderr, "  [%s] %s: expected %ld, actual %ld\n",
               ok ? "ok" : "FAIL", what, expected, actual);
  if (!ok)
    fail();
}

void print_values(const char *label, const std::vector<int> &vals) {
  if (vals.empty())
    return;
  std::fprintf(stderr, "  %s (%zu total), first ones:", label, vals.size());
  for (size_t i = 0; i < vals.size() && i < kMaxListed; ++i)
    std::fprintf(stderr, " %d", vals[i]);
  std::fprintf(stderr, "\n");
}

void test_single_thread() {
  std::fprintf(stderr, "single-thread:\n");
  aba_treiber::TreiberStack s;
  expect_pop(s, std::nullopt, "pop on a new stack");

  s.push(1);
  s.push(2);
  s.push(3);
  expect_pop(s, 3, "first pop after push 1,2,3");
  expect_pop(s, 2, "second pop");
  expect_pop(s, 1, "third pop");
  expect_pop(s, std::nullopt, "pop on an emptied stack");
}

void test_concurrent() {
  std::fprintf(stderr, "concurrent (%d threads x %d pushes):\n", kThreads,
               kPerThread);
  aba_treiber::TreiberStack s;
  std::vector<std::vector<int>> popped(kThreads);  // one vector per thread
  std::latch start(kThreads);                      // release all threads together

  {
    std::vector<std::jthread> threads;
    for (int t = 0; t < kThreads; ++t) {
      threads.emplace_back([&, t] {
        start.arrive_and_wait();
        for (int i = 0; i < kPerThread; ++i) {
          s.push(t * kPerThread + i);              // values are unique across threads
          if (i % 2 == 1) {                        // pop roughly every other step
            if (std::optional<int> v = s.pop())
              popped[t].push_back(*v);
          }
        }
      });
    }
  }  // jthreads join here

  std::vector<int> leftover;
  while (std::optional<int> v = s.pop())
    leftover.push_back(*v);

  long popped_total = static_cast<long>(leftover.size());
  for (const auto &vec : popped)
    popped_total += static_cast<long>(vec.size());

  std::vector<int> seen(kTotal, 0);
  std::vector<int> out_of_range;
  auto count = [&](int v) {
    if (v < 0 || v >= kTotal)
      out_of_range.push_back(v);
    else
      ++seen[v];
  };
  for (const auto &vec : popped)
    for (int v : vec) count(v);
  for (int v : leftover) count(v);

  std::vector<int> lost, duplicated;
  for (int v = 0; v < kTotal; ++v) {
    if (seen[v] == 0) lost.push_back(v);
    if (seen[v] > 1) duplicated.push_back(v);
  }

  print_values("lost values", lost);
  print_values("duplicated values", duplicated);
  print_values("out-of-range values", out_of_range);

  expect_eq(popped_total, kTotal, "values popped in total");
  expect_eq(static_cast<long>(lost.size()), 0, "values never popped (lost)");
  expect_eq(static_cast<long>(duplicated.size()), 0, "values popped more than once");
  expect_eq(static_cast<long>(out_of_range.size()), 0, "values never pushed");
}

}  // namespace

int main() {
  test_single_thread();
  test_concurrent();
  std::fprintf(stderr, "done\n");
  return 0;
}
