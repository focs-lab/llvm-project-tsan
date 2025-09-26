; RUN: opt -passes='print<escape-analysis>' -disable-output < %s 2>&1 | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

define void @no_escape() {
entry:
  %p = alloca i32, align 4
  store i32 10, i32* %p, align 4
  %val = load i32, i32* %p, align 4
  ret void
}

; CHECK-LABEL: EscapeAnalysis for function: no_escape
; CHECK-NEXT:    Allocation p: DOES NOT ESCAPE