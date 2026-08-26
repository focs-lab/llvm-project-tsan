; RUN: opt < %s -passes=tsan -tsan-one-word-filter -S | FileCheck %s

target datalayout = "e-p:64:64:64-i64:64:64-n8:16:32:64"
target triple = "aarch64-unknown-linux-gnu"

define i32 @read4(ptr %addr) sanitize_thread {
; CHECK-LABEL: define i32 @read4(
; CHECK: call ptr @__tsan_one_word_fast_state()
; CHECK: load i32, ptr @__tsan_one_word_enabled
; CHECK: br i1 {{.*}}, label %tsan.one_word, label %tsan.runtime
; CHECK: tsan.one_word:
; CHECK: load atomic i32, ptr {{.*}} monotonic, align 4
; CHECK: br i1 {{.*}}, label %tsan.cont, label %tsan.runtime
; CHECK: tsan.runtime:
; CHECK: call void @__tsan_read4(ptr %addr)
; CHECK: tsan.cont:
; CHECK: load i32, ptr %addr, align 4
  %value = load i32, ptr %addr, align 4
  ret i32 %value
}

define i32 @unaligned_read4(ptr %addr) sanitize_thread {
; CHECK-LABEL: define i32 @unaligned_read4(
; CHECK-NOT: tsan.one_word
; CHECK: call void @__tsan_unaligned_read4(ptr %addr)
  %value = load i32, ptr %addr, align 1
  ret i32 %value
}

define i32 @read_after_call(ptr %addr) sanitize_thread {
; CHECK-LABEL: define i32 @read_after_call(
; CHECK-NOT: __tsan_one_word_fast_state
; CHECK: call void @__tsan_read4(ptr %addr)
  call void @foo()
  %value = load i32, ptr %addr, align 4
  ret i32 %value
}

declare void @foo()
