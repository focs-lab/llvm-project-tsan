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
@GPtrPtr = dso_local global ptr null, align 8
@GPtrPtrPtr = dso_local global ptr null, align 8
@GAlias = alias ptr, ptr @GPtr

%S = type { ptr, ptr }
@GS = dso_local global %S zeroinitializer, align 8
@GArr = dso_local global [2 x %S] zeroinitializer, align 8

declare noalias ptr @malloc(i64)
declare noalias ptr @external(ptr)

; ============================================================================ ;
; Basics and locals
; ============================================================================ ;

; No allocations
define void @no_allocs() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'no_allocs':
; CHECK-NEXT: none
  ret void
}

; Local alloca that does not escape
define void @local_alloc_no_escape() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'local_alloc_no_escape':
; CHECK: a escapes: no
  %a = alloca i8, align 1
  ret void
}

; Using pointer in icmp -> no escape
define void @icmp_no_escape() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'icmp_no_escape':
; CHECK: a escapes: no
  %a = alloca i8, align 1
  %cmp = icmp eq ptr %a, null
  ret void
}

; Passthrough via phi/select-like -> no escape
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

; Safe store to local memory -> no escape.
define void @store_to_local_ok() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'store_to_local_ok':
; CHECK: a escapes: no
; CHECK: p escapes: no
  %a = alloca i8, align 1
  %p = alloca ptr, align 8
  store ptr %a, ptr %p
  ret void
}

; Chain through local pointer (double indirection) remains local:
; %x is stored in %p, %p in %pp -> no escape.
define void @double_ptr_local_ok() sanitize_thread {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'double_ptr_local_ok':
; CHECK:  x escapes: no
; CHECK:  p escapes: no
; CHECK:  pp escapes: no
  %x  = alloca i32, align 4
  %p  = alloca ptr, align 8
  %pp = alloca ptr, align 8
  store ptr %x, ptr %p
  store ptr %p, ptr %pp
  store i32 1, ptr %x
  %lv = load i32, ptr %x
  ret void
}

; ============================================================================ ;
; Returns and heap allocations
; ============================================================================ ;

; Returning alloca pointer -> escape
define ptr @return_alloca_escape() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'return_alloca_escape':
; CHECK: a escapes: yes
  %a = alloca i8, align 1
  ret ptr %a
}

; Malloc-like allocations
define ptr @malloc_local_no_escape() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'malloc_local_no_escape':
; CHECK: m1 escapes: no
; CHECK: m2 escapes: yes
  %m1 = call ptr @malloc(i64 16)
  %m2 = call ptr @malloc(i64 32)
  ret ptr %m2
}

; Escape of malloc calls
define dso_local void @malloc_escape() #0 {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'malloc_escape':
; CHECK:   p escapes: no
; CHECK:   call escapes: yes
; CHECK:   call1 escapes: yes
entry:
  %p = alloca ptr, align 8
  %call = call noalias ptr @malloc(i64 noundef 4) #2
  store ptr %call, ptr @GPtr, align 8
  %call1 = call noalias ptr @malloc(i64 noundef 4) #2
  store ptr %call1, ptr %p, align 8
  %0 = load ptr, ptr %p, align 8
  store ptr %0, ptr @GPtr, align 8
  ret void
}

; ============================================================================ ;
; Globals, arguments, and mixed destinations
; ============================================================================ ;

; Store to global, global alias -> escape
define void @store_to_global_escape() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'store_to_global_escape':
; CHECK: a escapes: yes
; CHECK: b escapes: yes
  %a = alloca i8, align 1
  %b = alloca i8, align 1
  store ptr %a, ptr @G
  store ptr %b, ptr @GAlias
  ret void
}

; Store to argument -> escape
define void @store_to_arg_escape(ptr %out) {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'store_to_arg_escape':
; CHECK: a escapes: yes
  %a = alloca i8, align 1
  store ptr %a, ptr %out
  ret void
}

; Store to the pointer returned by an external function -> escape
define void @store_to_unknown_ret_escape() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'store_to_unknown_ret_escape':
; CHECK: a escapes: yes
; CHECK: b escapes: yes
  %a = alloca i8, align 1
  %b = alloca i8, align 1
  %p = call ptr @external(ptr %b)
  store ptr %a, ptr %p
  ret void
}

; Cyclic dependency between local allocas, one stored to global -> both escape
define void @cycle_allocas_escape() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'cycle_allocas_escape':
; CHECK: a escapes: yes
; CHECK: b escapes: yes
  %a = alloca ptr, align 8
  %b = alloca ptr, align 8
  store ptr %a, ptr %b
  store ptr %b, ptr %a
  store ptr %a, ptr @G
  ret void
}

; Destination via phi mixing local and global -> escape
define void @phi_mixed_dest_escape(i1 %c) {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'phi_mixed_dest_escape':
; CHECK: a escapes: yes
; CHECK: p escapes: no
  %a = alloca i8, align 1
  %p = alloca ptr, align 8
  br i1 %c, label %t, label %f
t:
  br label %m
f:
  br label %m
m:
  %dst = phi ptr [ %p, %t ], [ @GPtr, %f ]
  store ptr %a, ptr %dst, align 8
  ret void
}

; Store to local pointer and then store to global pointer -> escape
define dso_local void @store_ptr_store_to_global_escape() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'store_ptr_store_to_global_escape':
; CHECK: x escapes: yes
; CHECK: p escapes: no
  %x = alloca i32, align 4
  %p = alloca ptr, align 8
  store ptr %x, ptr %p, align 8
  %1 = load ptr, ptr %p, align 8
  store ptr %1, ptr @GPtr, align 8
  ret void
}

; Store to local pointer and then store from global pointer -> no escape
define dso_local void @store_ptr_store_from_global_no_escape() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'store_ptr_store_from_global_no_escape':
; CHECK: x escapes: no
; CHECK: p escapes: no
  %x = alloca i32, align 4
  %p = alloca ptr, align 8
  store ptr %x, ptr %p, align 8
  %1 = load ptr, ptr @GPtr, align 8
  store ptr %1, ptr %p, align 8
  ret void
}

; ============================================================================ ;
; Loaded destination patterns
; ============================================================================ ;

; Store through pointer loaded from argument (LiveOnEntry) -> escape
define void @store_through_loaded_arg_escape(ptr %out) {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'store_through_loaded_arg_escape':
; CHECK: a escapes: yes
  %a = alloca i8, align 1
  %l = load ptr, ptr %out, align 8
  store ptr %a, ptr %l, align 8
  ret void
}

; Loaded destination with MemoryPhi (two stores) -> no escape
define void @loaded_dest_memphi_local_ok(i1 %c) {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'loaded_dest_memphi_local_ok':
; CHECK: x escapes: no
; CHECK: p escapes: no
; CHECK: s1 escapes: no
; CHECK: s2 escapes: no
entry:
  %x = alloca i32, align 4
  %p = alloca ptr, align 8
  %s1 = alloca ptr, align 8
  %s2 = alloca ptr, align 8
  br i1 %c, label %t, label %f
t:
  store ptr %s1, ptr %p, align 8
  br label %m
f:
  store ptr %s2, ptr %p, align 8
  br label %m
m:
  %l = load ptr, ptr %p, align 8
  store ptr %x, ptr %l, align 8
  ret void
}

; ============================================================================ ;
; Arrays and GEP
; ============================================================================ ;

; Store to local array element via GEP -> no escape
define void @store_to_gep_local_ok() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'store_to_gep_local_ok':
; CHECK: a escapes: no
; CHECK: arr escapes: no
  %a = alloca i8, align 1
  %arr = alloca [2 x ptr], align 8
  %elem = getelementptr inbounds [2 x ptr], ptr %arr, i64 0, i64 1
  store ptr %a, ptr %elem, align 8
  ret void
}

; Stack array element holds address of local; then that element is stored to a
; global slot -> the local escapes, array remains local
define void @array_element_stack_escape_via_global() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'array_element_stack_escape_via_global':
; CHECK: a escapes: yes
; CHECK: arr escapes: no
  %a = alloca i8, align 1
  %arr = alloca [2 x ptr], align 8
  %elem = getelementptr inbounds [2 x ptr], ptr %arr, i64 0, i64 0
  store ptr %a, ptr %elem, align 8
  %loaded = load ptr, ptr %elem, align 8
  store ptr %loaded, ptr @GPtr, align 8
  ret void
}

; Escape through heap array element: store pointer to local into malloc'ed array
; element, then read back and store to global -> local escapes; the malloc escapes.
define dso_local void @escape_through_heap_array_element() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'escape_through_heap_array_element':
; CHECK:  x escapes: yes
; CHECK:  p escapes: no
; CHECK:  call escapes: yes
entry:
  %x = alloca i32, align 4
  %p = alloca ptr, align 8
  %call = call noalias ptr @malloc(i64 noundef 800) #2
  store ptr %call, ptr %p, align 8
  %0 = load ptr, ptr %p, align 8
  %arrayidx = getelementptr inbounds ptr, ptr %0, i64 33
  store ptr %x, ptr %arrayidx, align 8
  %1 = load ptr, ptr %p, align 8
  %arrayidx1 = getelementptr inbounds ptr, ptr %1, i64 11
  %2 = load ptr, ptr %arrayidx1, align 8
  store ptr %2, ptr @GPtr, align 8
  ret void
}

; Escape through stack array element read from another index: local escapes,
; array remains local (element copied out to global).
define dso_local void @escape_through_array_element() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'escape_through_array_element':
; CHECK:   x escapes: yes
; CHECK:   p escapes: no
entry:
  %x = alloca i32, align 4
  %p = alloca [100 x ptr], align 16
  %arrayidx = getelementptr inbounds [100 x ptr], ptr %p, i64 0, i64 33
  store ptr %x, ptr %arrayidx, align 8
  %arrayidx1 = getelementptr inbounds [100 x ptr], ptr %p, i64 0, i64 0
  %0 = load ptr, ptr %arrayidx1, align 16
  store ptr %0, ptr @GPtr, align 8
  ret void
}

; Whole array (stack): leak address of an element itself -> the array escapes
define dso_local void @escape_whole_array() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'escape_whole_array':
; CHECK:   a1 escapes: yes
entry:
  %a1 = alloca [100 x i32], align 16
  %arrayidx = getelementptr inbounds [100 x i32], ptr %a1, i64 0, i64 33
  store ptr %arrayidx, ptr @GPtr, align 8
  ret void
}

; Whole array (heap): leak address of an element; also keep a local alloca with
; the malloc pointer to ensure both are reported.
define dso_local void @escape_whole_array_heap() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'escape_whole_array_heap':
; CHECK:   a2 escapes: yes
; CHECK:   call escapes: yes
entry:
  %a2 = alloca ptr, align 8
  %call = call noalias ptr @malloc(i64 noundef 400) #2
  store ptr %call, ptr %a2, align 8
  store ptr %a2, ptr @GPtrPtr, align 8
  ret void
}

; Struct (stack): leak address of a field -> the struct itself escapes
define void @struct_stack_self_escape_via_field_addr() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'struct_stack_self_escape_via_field_addr':
; CHECK: s escapes: yes
  %s = alloca %S, align 8
  %f0 = getelementptr inbounds %S, ptr %s, i64 0, i32 0
  store ptr %f0, ptr @GPtr, align 8
  ret void
}

; Struct (heap-esque via malloc): leak address of a field -> the heap object escapes
define void @struct_heap_self_escape_via_field_addr() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'struct_heap_self_escape_via_field_addr':
; CHECK: m escapes: yes
  %m = call ptr @malloc(i64 16)
  %f0 = getelementptr inbounds %S, ptr %m, i64 0, i32 0
  store ptr %f0, ptr @GPtr, align 8
  ret void
}

; ============================================================================ ;
; Structs and struct fields
; ============================================================================ ;

; Store into field of a local struct -> no escape
define void @struct_field_local_ok() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'struct_field_local_ok':
; CHECK: x escapes: no
; CHECK: s escapes: no
  %x = alloca i8, align 1
  %s = alloca %S, align 8
  %f0 = getelementptr inbounds %S, ptr %s, i64 0, i32 0
  store ptr %x, ptr %f0, align 8
  ret void
}

; Store into field of a global struct -> escape
define void @struct_field_global_escape() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'struct_field_global_escape':
; CHECK: x escapes: yes
  %x = alloca i8, align 1
  %f0 = getelementptr inbounds %S, ptr @GS, i64 0, i32 0
  store ptr %x, ptr %f0, align 8
  ret void
}

; Loaded-dest via field of a local struct -> no escape
define void @loaded_dest_struct_local_ok() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'loaded_dest_struct_local_ok':
; CHECK: x escapes: no
; CHECK: q escapes: no
; CHECK: s escapes: no
  %x = alloca i8, align 1
  %q = alloca ptr, align 8
  %s = alloca %S, align 8
  %f1 = getelementptr inbounds %S, ptr %s, i64 0, i32 1
  store ptr %q, ptr %f1, align 8
  %l = load ptr, ptr %f1, align 8
  store ptr %x, ptr %l, align 8
  ret void
}

; Local struct holds pointer to local; then the pointer is stored to global
; -> local escapes, struct remains local.
define void @struct_field_local_escape_via_global() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'struct_field_local_escape_via_global':
; CHECK: x escapes: yes
; CHECK: s escapes: no
  %x = alloca i8, align 1
  %s = alloca %S, align 8
  %f0 = getelementptr inbounds %S, ptr %s, i64 0, i32 0
  store ptr %x, ptr %f0, align 8
  %loaded = load ptr, ptr %f0, align 8
  store ptr %loaded, ptr @GPtr, align 8
  ret void
}

; Global struct stores the address of a local slot ->
; x escapes via escaping slot; q escapes
define void @loaded_dest_struct_global_escape() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'loaded_dest_struct_global_escape':
; CHECK: x escapes: yes
; CHECK: q escapes: yes
  %x = alloca i8, align 1
  %q = alloca ptr, align 8
  store ptr %q, ptr getelementptr inbounds (%S, ptr @GS, i32 0, i32 1), align 8
  %l = load ptr, ptr %q, align 8
  store ptr %x, ptr %l, align 8
  ret void
}

; Loaded destination: struct field points to a global slot -> escape
define void @loaded_dest_struct_global_ptr_escape() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'loaded_dest_struct_global_ptr_escape':
; CHECK: x escapes: yes
; CHECK: s escapes: no
  %x = alloca i8, align 1
  %s = alloca %S, align 8
  %f1 = getelementptr inbounds %S, ptr %s, i64 0, i32 1
  store ptr @GPtr, ptr %f1, align 8
  %l = load ptr, ptr %f1, align 8
  store ptr %x, ptr %l, align 8
  ret void
}

; Select between local and global struct as container -> escape
define void @select_struct_dest_mixed_escape(i1 %c) {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'select_struct_dest_mixed_escape':
; CHECK: x escapes: yes
; CHECK: s escapes: no
  %x = alloca i8, align 1
  %s = alloca %S, align 8
  %dst = select i1 %c, ptr %s, ptr @GS
  %f0 = getelementptr inbounds %S, ptr %dst, i64 0, i32 0
  store ptr %x, ptr %f0, align 8
  ret void
}

; Return a struct containing a pointer to a local -> escape
define %S @return_struct_containing_ptr_escape() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'return_struct_containing_ptr_escape':
; CHECK: x escapes: yes
  %x = alloca i8, align 1
  %u = insertvalue %S undef, ptr %x, 0
  %u2 = insertvalue %S %u, ptr null, 1
  ret %S %u2
}

; ============================================================================ ;
; Atomics and volatile
; ============================================================================ ;

; Atomic store of pointer -> treated as escape
define void @atomic_store_escape() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'atomic_store_escape':
; CHECK: a escapes: yes
; CHECK: p escapes: no
  %a = alloca i8, align 1
  %p = alloca ptr, align 8
  store atomic ptr %a, ptr %p seq_cst, align 8
  ret void
}

; Volatile store of pointer -> treated as escape
define void @volatile_store_escape() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'volatile_store_escape':
; CHECK: a escapes: yes
; CHECK: p escapes: no
  %a = alloca i8, align 1
  %p = alloca ptr, align 8
  store volatile ptr %a, ptr %p
  ret void
}

; ============================================================================ ;
; Casts
; ============================================================================ ;

; PtrToInt cast -> escape
define void @worklist_limit_bailout() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'worklist_limit_bailout':
; CHECK: a escapes: yes
  %a = alloca i8, align 1
  %c1 = icmp ne ptr %a, null
  %c2 = icmp eq ptr %a, null
  %sel = select i1 %c1, ptr %a, ptr %a
  %use = ptrtoint ptr %sel to i64
  ret void
}

; ============================================================================ ;
; Escape through double pointers
; ============================================================================ ;

; Store pp to global triple pointer -> all three allocations escape
define dso_local void @esc_thorugh_double_ptr1() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'esc_thorugh_double_ptr1':
; CHECK:  x escapes: yes
; CHECK:  p escapes: yes
; CHECK:  pp escapes: yes
entry:
  %x = alloca i32, align 4
  %p = alloca ptr, align 8
  %pp = alloca ptr, align 8
  store ptr %x, ptr %p, align 8
  store ptr %p, ptr %pp, align 8
  store ptr %pp, ptr @GPtrPtrPtr, align 8
  ret void
}

; Store p (loaded from pp) to global double pointer -> x and p escape, pp stays local
define dso_local void @esc_thorugh_double_ptr2() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'esc_thorugh_double_ptr2':
; CHECK:  x escapes: yes
; CHECK:  p escapes: yes
; CHECK:  pp escapes: no
entry:
  %x = alloca i32, align 4
  %p = alloca ptr, align 8
  %pp = alloca ptr, align 8
  store ptr %x, ptr %p, align 8
  store ptr %p, ptr %pp, align 8
  %0 = load ptr, ptr %pp, align 8
  store ptr %0, ptr @GPtrPtr, align 8
  ret void
}

; Load through pp and use it to store x somewhere,
; then leak pp’s loaded value to global double pointer
define dso_local void @esc_thorugh_double_ptr3() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'esc_thorugh_double_ptr3':
; CHECK:  x escapes: yes
; CHECK:  pp escapes: no
entry:
  %x = alloca i32, align 4
  %pp = alloca ptr, align 8
  %0 = load ptr, ptr %pp, align 8
  store ptr %x, ptr %0, align 8
  %1 = load ptr, ptr %pp, align 8
  store ptr %1, ptr @GPtrPtr, align 8
  ret void
}