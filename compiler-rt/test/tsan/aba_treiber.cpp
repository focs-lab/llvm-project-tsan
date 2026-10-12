// RUN: %clangxx_tsan -std=c++20 -O1 -I %S/Inputs -DABA_TREIBER_VARIANT=FreeNow %s -o %t-free && %run %t-free 2>&1 | FileCheck %s --check-prefix=ABA
// RUN: %clangxx_tsan -std=c++20 -O1 -I %S/Inputs -DABA_TREIBER_VARIANT=Leak %s -o %t-leak && %run %t-leak 2>&1 | FileCheck %s --check-prefix=SILENT
// RUN: %clangxx_tsan -std=c++20 -O1 -I %S/Inputs -DABA_TREIBER_VARIANT=RacyRead %s -o %t-racy && %run %t-racy 2>&1 | FileCheck %s --check-prefix=SILENT
// RUN: %clangxx_tsan -std=c++20 -O1 -I %S/Inputs -DABA_TREIBER_VARIANT=RelaxedPush %s -o %t-relaxed && %run %t-relaxed 2>&1 | FileCheck %s --check-prefix=SILENT
// RUN: %clangxx_tsan -std=c++20 -O1 -I %S/Inputs -DABA_TREIBER_VARIANT=StoreNotCas %s -o %t-store && %run %t-store 2>&1 | FileCheck %s --check-prefix=SILENT

#include "aba_treiber.h"
#include <thread>
#include <semaphore>
#include <cassert>
#include <cstdio>
#include <optional>
using namespace aba_treiber;

TreiberStack stack;
std::binary_semaphore go_t2{0};
std::binary_semaphore go_t1{0};
std::thread::id t1_id;
bool paused = false;
const void* A = nullptr;
const void* Y = nullptr;
std::optional<int> t1_result;

void pause_thread1(Op op) {
    if (op == Op::Pop && std::this_thread::get_id() == t1_id && !paused) {
        paused = true;
        go_t2.release();
        go_t1.acquire();
    }
}

void T1() {
    stack.push(1);
    stack.push(2);
    stack.push(3);
    A = stack.top_for_testing();
    aba_treiber::between_load_and_cas = pause_thread1;
    t1_id = std::this_thread::get_id();
    t1_result = stack.pop();
}

void T2() {
    go_t2.acquire();
    static_cast<void>(stack.pop());
    static_cast<void>(stack.pop());
    auto volatile decoy = new Node{4, nullptr}; // decoy: never pushed, takes B's freed address
    static_cast<void>(decoy);
    stack.push(5);
    Y = stack.top_for_testing();
    go_t1.release();           
}

// Relies on TSan's allocator reusing the most recently freed block
// first (per-thread LIFO cache). The asserts below catch it if not.

int main() {
    
    std::thread t1{T1};
    std::thread t2{T2};
    t1.join();
    t2.join();
    aba_treiber::between_load_and_cas = nullptr;
    std::fprintf(stderr, "Y == A: %d\n", Y == A);
    if constexpr (kVariant == Variant::FreeNow)
        assert(Y == A);
    else
        assert(Y != A);

    std::fprintf(stderr, "T1 popped %d\n", *t1_result);
    

    std::fprintf(stderr, "remaining:");
    while (auto v = stack.pop()) {
        std::fprintf(stderr, " %d", *v);
    }
    std::fprintf(stderr, "\n");
    std::fprintf(stderr, "done\n");
    return 0;
}

// ABA: WARNING: ThreadSanitizer: ABA detected
// ABA: remaining: 4
// ABA: done

// SILENT-NOT: ABA detected
// SILENT: remaining: 1
// SILENT: done