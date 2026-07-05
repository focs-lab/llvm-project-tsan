#include <iostream>
#include <cstdint>
#include <cassert>
#include <cstring>
#include <random>

using namespace std;

constexpr uint32_t CACHE_SIZE = 4096;
constexpr uint64_t CACHE_MASK = 0xFFF; // 12 bits for index (4096 slots)

// Address Tag: Bits 15 to 57
constexpr uint64_t ADDR_TAG_MASK = 0x03FFFFFFFFFF8000ULL;

// Write Flag: Bit 63
constexpr uint64_t WRITE_BIT_MASK = 0x8000000000000000ULL;
constexpr uint64_t IGNORE_WRITE_BIT_MASK = 0x7FFFFFFFFFFFFFFFULL;

uint64_t L0Filter[CACHE_SIZE] = { 0 };
uint64_t local_epoch = 0;

bool check_read_hit(uint64_t raw_addr) {
    uint32_t index = (raw_addr >> 3) & CACHE_MASK;
    uint64_t cached_entry = L0Filter[index];
    uint64_t expected_read = (raw_addr & ADDR_TAG_MASK) | local_epoch;
    // Mask out the Write Bit because a previous write also satisfies a read check
    if (((cached_entry ^ expected_read) & IGNORE_WRITE_BIT_MASK) == 0) {
        return true; // Cache Hit: Skip TSan
    }

    // Cache Miss: Update the slot and fallback to TSan
    L0Filter[index] = expected_read;
    return false;
}

bool check_write_hit(uint64_t raw_addr) {
    uint32_t index = (raw_addr >> 3) & CACHE_MASK;
    uint64_t cached_entry = L0Filter[index];
    uint64_t expected_write = (raw_addr & ADDR_TAG_MASK) | local_epoch | WRITE_BIT_MASK;
    if ((cached_entry ^ expected_write) == 0) {
        return true; // Cache Hit: Skip TSan
    }

    // Cache Miss: Update the slot and fallback to TSan
    L0Filter[index] = expected_write;
    return false;
}

void simulate_synchronization() {
    local_epoch++;
    if (local_epoch > 0x7FFF) [[unlikely]] { // 15-bit rollover
        std::cout << "  [Runtime] Epoch rollover (32767 reached)! Vector-flushing cache...\n";
        std::memset(L0Filter, 0, sizeof(L0Filter));
        local_epoch = 0;
    }
}

void reset_state() {
    std::memset(L0Filter, 0, sizeof(L0Filter));
    local_epoch = 1; // Start at 1 for clean testing
}

// ==============================================================================
// 3. ADVANCED TESTING SUITE
// ==============================================================================

void test_tbi_and_pointer_aliasing() {
    std::cout << "[TEST] Top-Byte Ignore (TBI) & Pointer Aliasing... ";
    reset_state();

    // Simulating an ARM pointer with metadata in the highest byte (0xAB)
    uint64_t CLEAN_ADDR = 0x00007FFEDCBA9870ULL;
    uint64_t POLLUTED_ARM = 0xAB007FFEDCBA9870ULL;

    assert(check_read_hit(CLEAN_ADDR) == false); // Cold miss
    // The polluted address MUST hit, proving the math ignores the 0xAB metadata
    assert(check_read_hit(POLLUTED_ARM) == true && "Failed to mask out TBI metadata bits.");

    std::cout << "PASSED\n";
}

void test_exhaustive_alignment() {
    std::cout << "[TEST] Exhaustive 8-Byte Alignment Boundary... ";
    reset_state();

    uint64_t BASE_ADDR = 0x00001000ULL;

    // 1. Cold miss on the base
    assert(check_read_hit(BASE_ADDR) == false);

    // 2. The next 7 bytes MUST all result in a cache hit (same 8-byte chunk)
    for (int i = 1; i < 8; i++) {
        assert(check_read_hit(BASE_ADDR + i) == true && "Alignment grouping failed.");
    }

    // 3. The 8th byte crosses the boundary into the next cache slot. It MUST miss.
    assert(check_read_hit(BASE_ADDR + 8) == false && "Boundary spill failed. Byte 8 should miss.");

    std::cout << "PASSED\n";
}

void test_fuzzing() {
    std::cout << "[TEST] Fuzz Testing (1,000,000 Random Addresses)... ";
    reset_state();

    std::mt19937 rng(1337); // Fixed seed for reproducibility
    std::uniform_int_distribution<uint64_t> dist(0, 0xFFFFFFFFFFFFFFFFULL);

    for (int i = 0; i < 1000000; i++) {
        uint64_t random_addr = dist(rng);

        // Cold access must miss
        assert(check_read_hit(random_addr) == false);
        // Immediate second access MUST hit
        assert(check_read_hit(random_addr) == true);
    }

    std::cout << "PASSED\n";
}

void test_micro_benchmark() {
    std::cout << "[BENCHMARK] Profiling Hot-Path Latency...\n";
    reset_state();

    uint64_t HOT_ADDR = 0x00005555ABCD1000ULL;
    check_read_hit(HOT_ADDR); // Warm up the cache slot

    const int ITERATIONS = 100000000; // 100 million hits

    auto start = std::chrono::high_resolution_clock::now();

    // Force the compiler NOT to optimize the loop away
    volatile int dummy = 0;
    for (int i = 0; i < ITERATIONS; i++) {
        if (check_read_hit(HOT_ADDR)) {
            dummy++;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    double ns_per_op = (elapsed.count() * 1000000.0) / ITERATIONS;

    std::cout << "  -> " << ITERATIONS << " cache hits processed in " << elapsed.count() << " ms.\n";
    std::cout << "  -> Average latency per check: " << ns_per_op << " nanoseconds.\n";
}

// ==============================================================================
// MAIN RUNNER
// ==============================================================================
int main() {
    std::cout << "=================================================\n";
    std::cout << " L0 FILTER - ADVANCED VALIDATION SUITE\n";
    std::cout << "=================================================\n\n";

    test_tbi_and_pointer_aliasing();
    test_exhaustive_alignment();
    test_fuzzing();

    std::cout << "\n";
    test_micro_benchmark();

    std::cout << "\n=================================================\n";
    std::cout << " [SUCCESS] ALL ADVANCED TESTS PASSED.\n";
    std::cout << "=================================================\n";

    return 0;
}