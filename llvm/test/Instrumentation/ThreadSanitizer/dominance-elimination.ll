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

; =============================================================================
; BRANCHING WITH MULTIPLE PATHS (one path dirty)
; =============================================================================

; Case A: inter-BB with a diamond where one branch is dirty.
; Path entry -> then (unsafe) -> merge, and entry -> else (safe) -> merge.
define void @multi_path_inter_dirty(i1 %cond) nounwind uwtable sanitize_thread {
entry:
  store i32 1, ptr @g1, align 4
  br i1 %cond, label %then, label %else

then:
  call void @some_external_call()
  br label %merge

else:
  call void @llvm.donothing()
  br label %merge

merge:
  %v = load i32, ptr @g1, align 4
  ret void
}
; CHECK-LABEL: define void @multi_path_inter_dirty
; CHECK:       entry:
; CHECK:       call void @__tsan_write4(ptr @g1)
; Dirty along one path => must instrument at merge.
; CHECK:       merge:
; CHECK:       call void @__tsan_read4(ptr @g1)
; CHECK:       ret void

; Case B: inter-BB where both branches are safe (no dangerous instr). Should eliminate.
define void @multi_path_inter_clean(i1 %cond) nounwind uwtable sanitize_thread {
entry:
  store i32 1, ptr @g1, align 4
  br i1 %cond, label %then, label %else

then:
  call void @llvm.donothing()
  br label %merge

else:
  call void @llvm.donothing()
  br label %merge

merge:
  %v = load i32, ptr @g1, align 4
  ret void
}
; CHECK-LABEL: define void @multi_path_inter_clean
; CHECK:       entry:
; CHECK:       call void @__tsan_write4(ptr @g1)
; Both paths clean => dominated read at merge should be removed.
; CHECK:       merge:
; CHECK-NOT:   call void @__tsan_read4(ptr @g1)

; =============================================================================
; MIXED: intra-BB safe suffix vs. inter-BB dirty path
; =============================================================================
define void @mixed_intra_inter(i1 %cond) nounwind uwtable sanitize_thread {
entry:
  ; intra-BB suffix between store and next store is safe (no calls)
  store i32 1, ptr @g1, align 4
  store i32 2, ptr @g1, align 4
  br i1 %cond, label %dirty, label %clean

dirty:
  ; dangerous call on one path
  call void @some_external_call()
  br label %merge

clean:
  ; safe on other path
  call void @llvm.donothing()
  br label %merge

merge:
  ; must keep because one incoming path is dirty
  store i32 3, ptr @g1, align 4
  ret void
}
; CHECK-LABEL: define void @mixed_intra_inter
; First store instruments.
; CHECK:       entry:
; CHECK:       call void @__tsan_write4(ptr @g1)
; Second store in same BB is dominated by the first and safe => removed.
; CHECK-NOT:   call void @__tsan_write4(ptr @g1)
; Final store must remain due to dirty path.
; CHECK:       merge:
; CHECK:       call void @__tsan_write4(ptr @g1)

; =============================================================================
; POST-DOM with dirty suffix at start BB blocks elimination (renamed BBs)
; =============================================================================
define void @postdom_dirty_start_suffix(i1 %cond) nounwind uwtable sanitize_thread {
entry:
  ; Initial write
  store i32 1, ptr @g1, align 4
  ; Dirty suffix in the start block blocks elimination
  call void @some_external_call()
  br i1 %cond, label %path_then, label %path_else

path_then:
  br label %merge

path_else:
  br label %merge

merge:
  ; Despite post-dominance, path is not clear due to dirty suffix in entry
  %v = load i32, ptr @g1, align 4
  ret void
}
; CHECK-LABEL: define void @postdom_dirty_start_suffix
; CHECK:       entry:
; CHECK:       call void @__tsan_write4(ptr @g1)
; CHECK:       call void @some_external_call()
; CHECK:       merge:
; CHECK:       call void @__tsan_read4(ptr @g1)
; CHECK:       ret void

; =============================================================================
; DIRTY PREFIX IN END BB blocks elimination (prefixSafe)
; =============================================================================
define void @dirty_prefix_in_end_bb() nounwind uwtable sanitize_thread {
entry:
  store i32 1, ptr @g1, align 4
  br label %end

end:
  ; Dirty prefix in the end block before the target access
  call void @some_external_call()
  %v = load i32, ptr @g1, align 4
  ret void
}
; CHECK-LABEL: define void @dirty_prefix_in_end_bb
; CHECK:       entry:
; CHECK:       call void @__tsan_write4(ptr @g1)
; CHECK:       end:
; CHECK:       call void @some_external_call()
; CHECK:       call void @__tsan_read4(ptr @g1)
; CHECK:       ret void

; =============================================================================
; IRRELEVANT DIRTY PATH NOT REACHING EndBB should not block elimination
; =============================================================================
define void @dirty_unrelated_cone(i1 %cond) nounwind uwtable sanitize_thread {
entry:
  store i32 1, ptr @g1, align 4
  br i1 %cond, label %to_end, label %to_dead

to_end:
  br label %end

to_dead:
  ; Dirty path that does NOT reach %end at all
  call void @some_external_call()
  br label %dead

dead:
  ret void

end:
  %v = load i32, ptr @g1, align 4
  ret void
}
; CHECK-LABEL: define void @dirty_unrelated_cone
; CHECK:       entry:
; CHECK:       call void @__tsan_write4(ptr @g1)
; The dirty path is outside the cone to %end, so read can be eliminated.
; CHECK:       end:
; CHECK-NOT:   call void @__tsan_read4(ptr @g1)
; CHECK:       ret void