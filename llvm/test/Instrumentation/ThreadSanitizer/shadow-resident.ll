; RUN: opt < %s -passes='function(tsan),module(tsan-module)' \
; RUN:   -tsan-shadow-resident-filter -S | FileCheck %s

target datalayout = "e-p:64:64-i64:64-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

define void @write8(ptr %p, i64 %value) sanitize_thread {
; CHECK-LABEL: define void @write8(
; CHECK:         call void @__tsan_func_entry
; CHECK:         call ptr @__tsan_shadow_resident_fast_state()
; CHECK:         load i64, ptr @__tsan_shadow_address_mask
; CHECK:         load i64, ptr @__tsan_shadow_xor
; CHECK:         load i64, ptr @__tsan_shadow_add
; CHECK:         load atomic i32
; CHECK:         br i1 {{.*}}, label %tsan.cont, label %tsan.shadow.runtime
; CHECK:       tsan.shadow.runtime:
; CHECK-NEXT:    call void @__tsan_write8(ptr %p)
; CHECK-NEXT:    br label %tsan.cont
; CHECK:       tsan.cont:
; CHECK:         store i64 %value, ptr %p, align 8
  store i64 %value, ptr %p, align 8
  ret void
}

define i64 @read8(ptr %p) sanitize_thread {
; CHECK-LABEL: define i64 @read8(
; CHECK:         or i32 {{.*}}, 1073741824
; CHECK:       tsan.shadow.runtime:
; CHECK-NEXT:    call void @__tsan_read8(ptr %p)
; CHECK:       tsan.cont:
; CHECK:         load i64, ptr %p, align 8
  %value = load i64, ptr %p, align 8
  ret i64 %value
}

declare void @escape()

define void @with_call(ptr %p, i64 %value) sanitize_thread {
; CHECK-LABEL: define void @with_call(
; CHECK-NOT:     __tsan_shadow_resident_fast_state
; CHECK:         call void @__tsan_write8(ptr %p)
  call void @escape()
  store i64 %value, ptr %p, align 8
  ret void
}
