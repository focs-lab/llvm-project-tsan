#pragma once

#ifndef ABA_TREIBER_VARIANT
#define ABA_TREIBER_VARIANT Leak
#endif

#include <atomic>
#include <optional>

namespace aba_treiber {
    enum class Variant { Leak, FreeNow, Hazard, RacyRead, RelaxedPush, StoreNotCas };
    static constexpr Variant kVariant = Variant::ABA_TREIBER_VARIANT;
    enum class Op { Push, Pop };
    inline void (*between_load_and_cas)(Op) = nullptr;

    class Node { 
    public:
        int value;
        Node* next; 
    };

    class TreiberStack {
    private:
        std::atomic<Node*> head_{nullptr};
        static void retire(Node* n) {
            if constexpr (kVariant == Variant::FreeNow) delete n;
        }

        Node* load_head() const {
            if constexpr (kVariant == Variant::RacyRead)
                return *reinterpret_cast<Node* const*>(&head_);   // plain, non-atomic read: the bug
            else
                return head_.load(std::memory_order_acquire);
        }

        static constexpr std::memory_order kPushSuccess =
            kVariant == Variant::RelaxedPush ? std::memory_order_relaxed
                                     : std::memory_order_release;

    public: 
        TreiberStack() = default;
        TreiberStack(const TreiberStack& other) = delete;
        TreiberStack& operator=(const TreiberStack& other) = delete;
        TreiberStack(TreiberStack&& other) = delete;
        TreiberStack& operator=(TreiberStack&& other) = delete;

        const void* top_for_testing() const { return head_.load(); }

        ~TreiberStack() {
            while (Node* node = head_.load(std::memory_order_acquire)) {
                head_.store(node->next, std::memory_order_release);
                delete node;
            }
        }

        void push(int value) {
            Node* n = new Node{value, head_.load(std::memory_order_relaxed)};
            if constexpr (kVariant == Variant::StoreNotCas) {
                n->next = head_.load(std::memory_order_acquire);
                if (between_load_and_cas) between_load_and_cas(Op::Push);
                head_.store(n, std::memory_order_release);
                return;
            }
            while (true) {
                if (between_load_and_cas) between_load_and_cas(Op::Push);
                if (head_.compare_exchange_weak(n->next, n,
                        kPushSuccess,
                        std::memory_order_relaxed))
                    break;
            }
        }

        [[nodiscard]] std::optional<int> pop() {
            Node* old = load_head();
            while (true) {
                if (!old) return std::nullopt;
                Node* next = old->next;
                if (between_load_and_cas)
                    between_load_and_cas(Op::Pop);
                if (head_.compare_exchange_weak(old, next,
                        std::memory_order_acquire,
                        std::memory_order_acquire))
                    break;
            }
            int value = old->value;
            retire(old);
            return value;
        }
    };
}