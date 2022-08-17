// RUN: %clang_cc1 -no-opaque-pointers -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -no-opaque-pointers -verify -fopenmp -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s
// expected-no-diagnostics

extern int printf(const char *, ...);

extern "C" {
struct S {
  int a;
  float b;
};

// CHECK: define{{.*}}void @test(%struct.S* noundef byval(%struct.S) align {{[0-9]+}} [[arg:%[0-9a-zA-Z]+]])
// CHECK: [[g:%[0-9a-zA-Z]+]] = call align {{[0-9]+}} i8* @__kmpc_alloc_shared
// CHECK: bitcast i8* [[g]] to %struct.S*
// CHECK: bitcast %struct.S* [[arg]] to i8**
// CHECK: call void [[cc:@__copy_constructor[_0-9a-zA-Z]+]]
void test(S s) {
#pragma omp parallel for
  for (int i = 0; i < s.a; ++i) {
    printf("%d : %d : %f\n", i, s.a, s.b);
  }
}
}

void foo() {
  #pragma omp target teams num_teams(1)
  {
    S s;
    s.a = 7;
    s.b = 11;
    test(s);
  }
}

struct BB;

struct SS {
  int a;
  double b;
  BB *c;
  SS() = default;
  SS(const SS &);
  SS(SS &&) = delete;
};

extern "C" {
// CHECK: define{{.*}}void @test2(%struct.SS* noundef byval(%struct.SS) align {{[0-9]+}} [[arg:%[0-9a-zA-Z]+]])
// CHECK: [[g:%[0-9a-zA-Z]+]] = call align {{[0-9]+}} i8* @__kmpc_alloc_shared
// CHECK: bitcast i8* [[g]] to %struct.SS*
// CHECK: bitcast %struct.SS* [[arg]] to i8**
// CHECK: call void [[cc2:@__copy_constructor[_0-9a-zA-Z]+]]
void test2(SS s) {
#pragma omp parallel for
  for (int i = 0; i < s.a; ++i) {
    printf("%d : %d : %f\n", i, s.a, s.b);
  }
}
}

void bar() {
  #pragma omp target teams num_teams(1)
  {
    SS s;
    s.a = 7;
    s.b = 11;
    test2(s);
  }
}

// CHECK: void [[cc]]
// CHECK: void [[cc2]]
