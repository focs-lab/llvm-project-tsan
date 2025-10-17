; RUN: opt -passes='print<escape-analysis>' -disable-output %s 2>&1 | FileCheck %s

; NOTE:
; - The printer emits:
;   "EscapeAnalysis for function: <func>"
;   "<alloc-name> escapes: yes|no" per allocation site (alloca/malloc-like).
;   "EA: none" if no allocations in the function.
; - Names are taken from SSA. We avoid relying on "unnamed#N" in tests.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

@G = global ptr null
@GPtr = dso_local global ptr null, align 8

declare noalias ptr @malloc(i64)
declare noalias ptr @external(i64)

; ----------------------------------------
; No allocations
; ----------------------------------------
define void @no_allocs() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'no_allocs':
; CHECK-NEXT: none
  ret void
}

; ----------------------------------------
; Local alloca that does not escape
; ----------------------------------------
define void @local_alloc_no_escape() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'local_alloc_no_escape':
; CHECK: a escapes: no
  %a = alloca i8, align 1
  ret void
}

; ----------------------------------------
; Returning alloca pointer -> escape
; ----------------------------------------
define ptr @return_alloca_escape() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'return_alloca_escape':
; CHECK: a escapes: yes
  %a = alloca i8, align 1
  ret ptr %a
}

; ----------------------------------------
; Store to global -> escape
; ----------------------------------------
define void @store_to_global_escape() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'store_to_global_escape':
; CHECK: a escapes: yes
  %a = alloca i8, align 1
  store ptr %a, ptr @G
  ret void
}

; ----------------------------------------
; Store to argument -> escape
; ----------------------------------------
define void @store_to_arg_escape(ptr %out) {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'store_to_arg_escape':
; CHECK: a escapes: yes
  %a = alloca i8, align 1
  store ptr %a, ptr %out
  ret void
}

; ----------------------------------------
; Safe store to local memory -> no escape
; Also checks we print both allocas
; ----------------------------------------
define void @store_to_local_ok() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'store_to_local_ok':
; CHECK: a escapes: no
; CHECK: p escapes: no
  %a = alloca i8, align 1
  %p = alloca ptr, align 8
  store ptr %a, ptr %p
  ret void
}

; ----------------------------------------
; Passthrough via phi/select-like -> no escape
; ----------------------------------------
define void @passthrough_phi_no_escape(i1 %c) {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'passthrough_phi_no_escape':
; CHECK: a escapes: no
entry:
  %a = alloca i8, align 1
  br i1 %c, label %t, label %f
t:
  br label %merge
f:
  br label %merge
merge:
  %p = phi ptr [ %a, %t ], [ %a, %f ]
  ret void
}

; ----------------------------------------
; Using pointer in icmp -> no escape
; ----------------------------------------
define void @icmp_no_escape() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'icmp_no_escape':
; CHECK: a escapes: no
  %a = alloca i8, align 1
  %cmp = icmp eq ptr %a, null
  ret void
}

; ----------------------------------------
; Volatile store of pointer -> treated as escape
; Also check destination alloca itself does not escape
; ----------------------------------------
define void @volatile_store_escape() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'volatile_store_escape':
; CHECK: a escapes: yes
; CHECK: p escapes: no
  %a = alloca i8, align 1
  %p = alloca ptr, align 8
  store volatile ptr %a, ptr %p
  ret void
}

; ----------------------------------------
; Malloc-like allocation that does not escape
; ----------------------------------------
define void @malloc_local_no_escape() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'malloc_local_no_escape':
; CHECK: m escapes: no
  %m = call ptr @malloc(i64 16)
  ret void
}

; ----------------------------------------
; Cyclic dependency between local allocas -> conservative escape
; a <-> b cycle through stores
; ----------------------------------------
define void @cycle_allocas_escape() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'cycle_allocas_escape':
; CHECK: a escapes: yes
; CHECK: b escapes: yes
  %a = alloca ptr, align 8
  %b = alloca ptr, align 8
  store ptr %a, ptr %b
  store ptr %b, ptr %a
  ret void
}

; ----------------------------------------
; Store to the pointer, returned by an external function -> escape
; ----------------------------------------
define void @store_to_unknown_ret_escape() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'store_to_unknown_ret_escape':
; CHECK: a escapes: yes
  %a = alloca i8, align 1
  %p = call ptr @external(i64 16)
  store ptr %a, ptr %p
  ret void
}

; ----------------------------------------
; Escape through pointer loaded from local alloca and stored to global -> escape
; ----------------------------------------
define dso_local void @escape_through_pointer1() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'escape_through_pointer1':
; CHECK: x escapes: yes
; CHECK: p escapes: no
  %x = alloca i32, align 4
  %p = alloca ptr, align 8
  store ptr %x, ptr %p, align 8
  %1 = load ptr, ptr %p, align 8
  store ptr %1, ptr @GPtr, align 8
  ret void
}

; ----------------------------------------
; Escape when a pointer from global overwrites a local pointer containing x (conservative) -> escape
; ----------------------------------------
define dso_local void @escape_through_pointer2() #0 {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'escape_through_pointer2':
; CHECK: x escapes: yes
; CHECK: p escapes: no
  %x = alloca i32, align 4
  %p = alloca ptr, align 8
  store ptr %x, ptr %p, align 8
  %1 = load ptr, ptr @GPtr, align 8
  store ptr %1, ptr %p, align 8
  ret void
}
