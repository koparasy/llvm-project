// Verify that -emit-cir-pre-lowering stops before target / C++ ABI /
// lowering-prepare passes, and that the resulting .cir round-trips back
// through cc1 for the rest of the pipeline. This is the artifact that
// LTO/Combine flows would ship between cc1 invocations.

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:   -emit-cir-pre-lowering %s -o %t.pre.cir
// RUN: FileCheck --input-file=%t.pre.cir %s --check-prefix=PRE

// Same source through the post-lowering -emit-cir, for contrast.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:   -emit-cir %s -o %t.post.cir
// RUN: FileCheck --input-file=%t.post.cir %s --check-prefix=POST

// Pre-lowering CIR fed back into cc1 must complete the pipeline and
// produce a valid object.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -x cir \
// RUN:   %t.pre.cir -emit-obj -o %t.o
// RUN: llvm-nm %t.o | FileCheck %s --check-prefix=OBJ

struct WithCtor { WithCtor(); int x; };
int test() {
  static WithCtor sl;
  return sl.x;
}

// Pre-lowering form keeps the high-level static-local init op and never
// emits the Itanium guard runtime calls.
// PRE: cir.local_init static_local
// PRE-NOT: __cxa_guard_acquire
// PRE-NOT: __cxa_guard_release

// Post-lowering form has been expanded into the guarded init sequence.
// POST: __cxa_guard_acquire
// POST: __cxa_guard_release

// OBJ: T _Z4testv
