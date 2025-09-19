; RUN: opt -passes=tsan -tsan-use-dominance-analysis < %s -S | FileCheck %s

; This file contains tests for the TSan dominance-based optimization.
; We check that redundant instrumentation is removed when one access
; dominates/post-dominates another, and is NOT removed when the path between
; them is "dirty".

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; --- Global variables for testing (minimized to two) ---
@g1 = common global i32 0, align 4
@g2 = common global i32 0, align 4

; --- External Function Declarations for Tests ---
declare void @some_external_call()
declare void @llvm.donothing() #0

; =============================================================================
; INTRA-BLOCK DOMINANCE TESTS
; =============================================================================

define void @test_intra_block_write_write() nounwind uwtable sanitize_thread {
entry:
  store i32 1, ptr @g1, align 4
  store i32 2, ptr @g1, align 4
  ret void
}
; CHECK-LABEL: define void @test_intra_block_write_write
; CHECK:      call void @__tsan_write4(ptr @g1)
; The second write is dominated and should NOT be instrumented.
; CHECK-NOT:  call void @__tsan_write4(ptr @g1)
; CHECK:      ret void

define void @test_intra_block_write_read() nounwind uwtable sanitize_thread {
entry:
  store i32 1, ptr @g1, align 4
  %val = load i32, ptr @g1, align 4
  ret void
}
; CHECK-LABEL: define void @test_intra_block_write_read
; CHECK:      call void @__tsan_write4(ptr @g1)
; The read is dominated and should NOT be instrumented.
; CHECK-NOT:  call void @__tsan_read4(ptr @g1)
; CHECK:      ret void

define void @test_intra_block_read_read() nounwind uwtable sanitize_thread {
entry:
  %val1 = load i32, ptr @g1, align 4
  %val2 = load i32, ptr @g1, align 4
  ret void
}
; CHECK-LABEL: define void @test_intra_block_read_read
; CHECK:      call void @__tsan_read4(ptr @g1)
; The second read is dominated and should NOT be instrumented.
; CHECK-NOT:  call void @__tsan_read4(ptr @g1)
; CHECK:      ret void

; =============================================================================
; PATH CLEARNESS TESTS
; =============================================================================

define void @test_path_not_clear_call() nounwind uwtable sanitize_thread {
entry:
  store i32 1, ptr @g1, align 4
  call void @some_external_call()
  store i32 2, ptr @g1, align 4
  ret void
}
; CHECK-LABEL: define void @test_path_not_clear_call
; CHECK:      call void @__tsan_write4(ptr @g1)
; An unsafe call makes the path dirty. Optimization must NOT trigger.
; CHECK:      call void @__tsan_write4(ptr @g1)
; CHECK:      ret void

define void @test_path_clear_safe_call() nounwind uwtable sanitize_thread {
entry:
  store i32 1, ptr @g1, align 4
  call void @llvm.donothing()
  store i32 2, ptr @g1, align 4
  ret void
}
; CHECK-LABEL: define void @test_path_clear_safe_call
; CHECK:      call void @__tsan_write4(ptr @g1)
; A safe intrinsic call should not block the optimization.
; CHECK-NOT:  call void @__tsan_write4(ptr @g1)
; CHECK:      ret void

; =============================================================================
; INTER-BLOCK DOMINANCE TESTS
; =============================================================================

define void @test_inter_block_dom(i1 %cond) nounwind uwtable sanitize_thread {
entry:
  store i32 1, ptr @g1, align 4
  br i1 %cond, label %if.then, label %if.else
if.then:
  store i32 2, ptr @g1, align 4
  br label %if.end
if.else:
  store i32 3, ptr @g1, align 4
  br label %if.end
if.end:
  ret void
}
; CHECK-LABEL: define void @test_inter_block_dom
; CHECK:      call void @__tsan_write4(ptr @g1)
; CHECK:      if.then:
; CHECK-NOT:  call void @__tsan_write4(ptr @g1)
; CHECK:      if.else:
; CHECK-NOT:  call void @__tsan_write4(ptr @g1)
; CHECK:      ret void

; =============================================================================
; POST-DOMINANCE TESTS
; =============================================================================

define void @test_post_dom(i1 %cond) nounwind uwtable sanitize_thread {
entry:
  br i1 %cond, label %if.then, label %if.else
if.then:
  store i32 2, ptr @g1, align 4
  br label %if.end
if.else:
  store i32 3, ptr @g1, align 4
  br label %if.end
if.end:
  store i32 4, ptr @g1, align 4
  ret void
}
; CHECK-LABEL: define void @test_post_dom
; CHECK:      if.then:
; CHECK-NOT:  call void @__tsan_write4(ptr @g1)
; CHECK:      if.else:
; CHECK-NOT:  call void @__tsan_write4(ptr @g1)
; CHECK:      if.end:
; CHECK:      call void @__tsan_write4(ptr @g1)
; CHECK:      ret void

; =============================================================================
; ALIAS ANALYSIS TESTS
; =============================================================================

define void @test_no_alias() nounwind uwtable sanitize_thread {
entry:
  store i32 1, ptr @g1, align 4
  store i32 2, ptr @g2, align 4
  ret void
}
; CHECK-LABEL: define void @test_no_alias
; CHECK:      call void @__tsan_write4(ptr @g1)
; Different addresses. The optimization must NOT trigger.
; CHECK:      call void @__tsan_write4(ptr @g2)
; CHECK:      ret void

; Attributes for the "safe" intrinsic
attributes #0 = { nounwind readnone }