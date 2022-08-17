//===------- LibC.cpp - Simple implementation of libc functions --- C++ ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LibC.h"
#include "Debug.h"
#include "Memory.h"
#include "Synchronization.h"
#include "Utils.h"

#include "HostRPC.h"

using namespace _OMP;

#pragma omp begin declare target device_type(nohost)

HostRPCDescriptor *omptarget_hostrpc_descriptor
    __attribute__((used, retain, weak, visibility("protected")));
uint32_t *omptarget_hostrpc_futex
    __attribute__((used, retain, weak, visibility("protected")));
char *omptarget_hostrpc_memory_buffer
    __attribute__((used, retain, weak, visibility("protected")));
size_t omptarget_hostrpc_memory_buffer_size
    __attribute__((used, retain, weak, visibility("protected")));

namespace impl {
int32_t omp_vprintf(const char *Format, void *Arguments, uint32_t);
}

#pragma omp begin declare variant match(                                       \
    device = {arch(nvptx, nvptx64)}, implementation = {extension(match_any)})
extern "C" int32_t vprintf(const char *, void *);
namespace impl {
int32_t omp_vprintf(const char *Format, void *Arguments, uint32_t) {
  return vprintf(Format, Arguments);
}
} // namespace impl
#pragma omp end declare variant

// We do not have a vprintf implementation for AMD GPU yet so we use a stub.
#pragma omp begin declare variant match(device = {arch(amdgcn)})
namespace impl {
int32_t omp_vprintf(const char *Format, void *Arguments, uint32_t) {
  return -1;
}
} // namespace impl
#pragma omp end declare variant

namespace {
size_t HostRPCMemoryBufferCurrentPosition = 0;
constexpr const size_t Alignment = 16;

// FIXME: For now we only allow one thread requesting host RPC.
mutex::TicketLock HostRPCLock;

void *HostRPCMemAlloc(size_t Size) {
  Size = utils::align_up(Size, Alignment);

  if (Size + HostRPCMemoryBufferCurrentPosition <
      omptarget_hostrpc_memory_buffer_size) {
    void *R =
        omptarget_hostrpc_memory_buffer + HostRPCMemoryBufferCurrentPosition;
    atomic::add(&HostRPCMemoryBufferCurrentPosition, Size, __ATOMIC_SEQ_CST);
    return R;
  }

  printf("out of host RPC memory!\n");

  __builtin_trap();

  return nullptr;
}

void HostRPCMemFree(void *) {

}

class HostRPCDescriptorWrapper {
  HostRPCDescriptor *SD = nullptr;
  bool IsValid = false;
  unsigned CurrentArgId = 0;

public:
  HostRPCDescriptorWrapper(SyscallId Id, unsigned NumArgs) {
    if (!omptarget_hostrpc_descriptor || !omptarget_hostrpc_futex)
      return;

    SD = omptarget_hostrpc_descriptor;

    HostRPCLock.lock();

    SD->Id = Id;
    SD->NumArgs = NumArgs;
    SD->Status = EXEC_STAT_CREATED;
    SD->ReturnValue = nullptr;
    SD->RVSize = sizeof(void *);
    SD->Args = (ArgTy *)HostRPCMemAlloc(sizeof(ArgTy) * NumArgs);
    if (!SD->Args)
      return;

    IsValid = true;
  }

  bool isValid() const { return IsValid; }

  template <typename Ty> void addArg(Ty Arg, ArgType Type, size_t Size) {
    ASSERT(IsValid && "SD is not valid");
    ASSERT(CurrentArgId < SD->NumArgs && "adding more arg than expected");

    SD->Args[CurrentArgId].Arg = (void *)Arg;
    SD->Args[CurrentArgId].Type = Type;
    SD->Args[CurrentArgId].Size = Size;

    ++CurrentArgId;
  }

  ~HostRPCDescriptorWrapper() {
    void *Args = SD->Args;
    HostRPCLock.unlock();
    if (Args)
      HostRPCMemFree(Args);
  }

  template <typename ReturnType> ReturnType getReturnValue() const {
    return (ReturnType)SD->ReturnValue;
  }

  bool sendAndWait() const {
    if (CurrentArgId != SD->NumArgs)
      return false;

    printf("send syscall=%d to host\n", SD->Id);

    atomic::add(omptarget_hostrpc_futex, 1U, __ATOMIC_SEQ_CST);

    // A system fence is required to make sure futex on the host is also
    // updated if USM is supported.
    fence::system(__ATOMIC_SEQ_CST);

    unsigned NS = 8;

    while (atomic::load(omptarget_hostrpc_futex, __ATOMIC_SEQ_CST)) {
      asm volatile("nanosleep.u32 %0;" : : "r"(NS));
      if (NS < 256)
        NS *= 2;
    }

    return true;
  }
};

} // namespace

extern "C" {

int errno __attribute__((used, retain, weak, visibility("protected")));

int memcmp(const void *lhs, const void *rhs, size_t count) {
  auto *L = reinterpret_cast<const unsigned char *>(lhs);
  auto *R = reinterpret_cast<const unsigned char *>(rhs);

  for (size_t I = 0; I < count; ++I)
    if (L[I] != R[I])
      return (int)L[I] - (int)R[I];

  return 0;
}

/// printf() calls are rewritten by CGGPUBuiltin to __llvm_omp_vprintf
int32_t __llvm_omp_vprintf(const char *Format, void *Arguments, uint32_t Size) {
  return impl::omp_vprintf(Format, Arguments, Size);
}

FILE *fopen(const char *filename, const char *mode) {
  HostRPCDescriptorWrapper Wrapper(SYSCALLID_fopen, 2);
  if (!Wrapper.isValid())
    return nullptr;

  size_t Len1 = strlen(filename) + 1;
  size_t Len2 = strlen(mode) + 1;

  char *FileName = (char *)HostRPCMemAlloc(Len1);
  char *Mode = (char *)HostRPCMemAlloc(Len2);

  __builtin_memcpy(FileName, filename, Len1);
  __builtin_memcpy(Mode, mode, Len2);

  Wrapper.addArg(FileName, ARG_POINTER, Len1);
  Wrapper.addArg(Mode, ARG_POINTER, Len2);

  if (!Wrapper.sendAndWait())
    return nullptr;

  return Wrapper.getReturnValue<FILE *>();
}

size_t fread(void *buffer, size_t size, size_t count, FILE *stream) {
  HostRPCDescriptorWrapper Wrapper(SYSCALLID_fread, 4);
  if (!Wrapper.isValid())
    return 0;

  void *Buffer = HostRPCMemAlloc(size * count);

  Wrapper.addArg(Buffer, ARG_LITERAL, sizeof(buffer));
  Wrapper.addArg(size, ARG_LITERAL, sizeof(size));
  Wrapper.addArg(count, ARG_LITERAL, sizeof(count));
  Wrapper.addArg(stream, ARG_LITERAL, sizeof(stream));

  if (!Wrapper.sendAndWait())
    return 0;

  __builtin_memcpy(buffer, Buffer, size * count);

  return Wrapper.getReturnValue<size_t>();
}

int stat(const char *path, struct stat *buf) {
  HostRPCDescriptorWrapper Wrapper(SYSCALLID_stat, 2);
  if (!Wrapper.isValid())
    return 0;

  size_t Len = strlen(path) + 1;
  size_t Size = sizeof(struct stat *);

  char *Path = (char *)HostRPCMemAlloc(Len);
  struct stat *Buf = (struct stat *)HostRPCMemAlloc(Size);
  __builtin_memcpy(Path, path, Len);

  Wrapper.addArg(Path, ARG_POINTER, Len);
  Wrapper.addArg(Buf, ARG_LITERAL, sizeof(struct stat *));

  if (!Wrapper.sendAndWait())
    return 0;

  return Wrapper.getReturnValue<int64_t>();
}

FILE *popen(const char *command, const char *type) {
  HostRPCDescriptorWrapper Wrapper(SYSCALLID_popen, 2);
  if (!Wrapper.isValid())
    return nullptr;

  size_t Len1 = strlen(command) + 1;
  size_t Len2 = strlen(type) + 1;

  char *Command = (char *)HostRPCMemAlloc(Len1);
  char *Type = (char *)HostRPCMemAlloc(Len2);

  __builtin_memcpy(Command, command, Len1);
  __builtin_memcpy(Type, type, Len2);

  Wrapper.addArg(Command, ARG_POINTER, Len1);
  Wrapper.addArg(Type, ARG_POINTER, Len2);

  if (!Wrapper.sendAndWait())
    return nullptr;

  return Wrapper.getReturnValue<FILE *>();
}

size_t fwrite(const void *buffer, size_t size, size_t count, FILE *stream) {
  HostRPCDescriptorWrapper Wrapper(SYSCALLID_fwrite, 4);
  if (!Wrapper.isValid())
    return 0;

  void *Buffer = HostRPCMemAlloc(size * count);
  __builtin_memcpy(Buffer, buffer, size * count);

  Wrapper.addArg(Buffer, ARG_POINTER, size * count);
  Wrapper.addArg(size, ARG_LITERAL, sizeof(size));
  Wrapper.addArg(count, ARG_LITERAL, sizeof(count));
  Wrapper.addArg(stream, ARG_LITERAL, sizeof(stream));

  if (!Wrapper.sendAndWait())
    return 0;

  return Wrapper.getReturnValue<size_t>();
}

int fclose(FILE *stream) {
  HostRPCDescriptorWrapper Wrapper(SYSCALLID_fclose, 1);
  if (!Wrapper.isValid())
    return 0;

  Wrapper.addArg(stream, ARG_LITERAL, sizeof(stream));

  if (!Wrapper.sendAndWait())
    return 0;

  return Wrapper.getReturnValue<int64_t>();
}

int getc(FILE *stream) {
  HostRPCDescriptorWrapper Wrapper(SYSCALLID_getc, 1);
  if (!Wrapper.isValid())
    return 0;

  Wrapper.addArg(stream, ARG_LITERAL, sizeof(stream));

  if (!Wrapper.sendAndWait())
    return 0;

  return Wrapper.getReturnValue<int64_t>();
}

int pclose(FILE *stream) {
  HostRPCDescriptorWrapper Wrapper(SYSCALLID_pclose, 1);
  if (!Wrapper.isValid())
    return 0;

  Wrapper.addArg(stream, ARG_LITERAL, sizeof(stream));

  if (!Wrapper.sendAndWait())
    return 0;

  return Wrapper.getReturnValue<int64_t>();
}

int fflush(FILE *stream) {
  HostRPCDescriptorWrapper Wrapper(SYSCALLID_fflush, 1);
  if (!Wrapper.isValid())
    return 0;

  Wrapper.addArg(stream, ARG_LITERAL, sizeof(stream));

  if (!Wrapper.sendAndWait())
    return 0;

  return Wrapper.getReturnValue<int64_t>();
}

void rewind(FILE *stream) {
  HostRPCDescriptorWrapper Wrapper(SYSCALLID_rewind, 1);
  if (!Wrapper.isValid())
    return;

  Wrapper.addArg(stream, ARG_LITERAL, sizeof(stream));

  if (!Wrapper.sendAndWait())
    return;
}

int fseek(FILE *stream, long offset, int origin) {
  HostRPCDescriptorWrapper Wrapper(SYSCALLID_fseek, 3);
  if (!Wrapper.isValid())
    return 0;

  Wrapper.addArg(stream, ARG_LITERAL, sizeof(stream));
  Wrapper.addArg(offset, ARG_LITERAL, sizeof(offset));
  Wrapper.addArg(origin, ARG_LITERAL, sizeof(origin));

  if (!Wrapper.sendAndWait())
    return 0;

  return Wrapper.getReturnValue<int64_t>();
}

long ftell(FILE *stream) {
  HostRPCDescriptorWrapper Wrapper(SYSCALLID_ftell, 1);
  if (!Wrapper.isValid())
    return 0;

  Wrapper.addArg(stream, ARG_LITERAL, sizeof(stream));

  if (!Wrapper.sendAndWait())
    return 0;

  return Wrapper.getReturnValue<long>();
}

int feof(FILE *stream) {
  HostRPCDescriptorWrapper Wrapper(SYSCALLID_feof, 1);
  if (!Wrapper.isValid())
    return 0;

  Wrapper.addArg(stream, ARG_LITERAL, sizeof(stream));

  if (!Wrapper.sendAndWait())
    return 0;

  return Wrapper.getReturnValue<int64_t>();
}

int gettimeofday(struct timeval *tv, struct timezone *tz) {
  HostRPCDescriptorWrapper Wrapper(SYSCALLID_gettimeofday, 2);
  if (!Wrapper.isValid())
    return 0;

  // sizeof(struct timeval)=16
  struct timeval *TV =
      (struct timeval *)HostRPCMemAlloc(16);
  // sizeof(struct timezone)=8
  struct timezone *TZ =(struct timezone *)HostRPCMemAlloc(8);

  Wrapper.addArg(TV, ARG_LITERAL, sizeof(tv));
  Wrapper.addArg(TZ, ARG_LITERAL, sizeof(tz));

  if (!Wrapper.sendAndWait())
    return 0;

  __builtin_memcpy(tv, TV, 16);
  __builtin_memcpy(tz, TZ, 8);

  return Wrapper.getReturnValue<int64_t>();
}

char *fgets(char *str, int count, FILE *stream) {
  HostRPCDescriptorWrapper Wrapper(SYSCALLID_fgets, 3);
  if (!Wrapper.isValid())
    return nullptr;

  char *Str = (char *)HostRPCMemAlloc(count);

  Wrapper.addArg(Str, ARG_LITERAL, sizeof(char *));
  Wrapper.addArg(count, ARG_LITERAL, sizeof(count));
  Wrapper.addArg(stream, ARG_LITERAL, sizeof(stream));

  if (!Wrapper.sendAndWait())
    return nullptr;

  __builtin_memcpy(str, Str, count);

  return Wrapper.getReturnValue<char *>();
}

size_t strftime(char *str, size_t count, const char *format,
                const struct tm *time) {
  HostRPCDescriptorWrapper Wrapper(SYSCALLID_strftime, 4);
  if (!Wrapper.isValid())
    return 0;

  char *Str = (char *)HostRPCMemAlloc(count);
  char *Format = (char *)HostRPCMemAlloc(strlen(format) + 1);
  // FIXME: sizeof(struct tm) = 56.
  struct tm *Time = (struct tm *)HostRPCMemAlloc(56);

  __builtin_memcpy(Format, format, strlen(format) + 1);
  // FIXME: sizeof(struct tm) = 56.
  __builtin_memcpy(Time, time, 56);

  Wrapper.addArg(Str, ARG_LITERAL, sizeof(Str));
  Wrapper.addArg(count, ARG_LITERAL, sizeof(count));
  Wrapper.addArg(Format, ARG_POINTER, strlen(format) + 1);
  // FIXME: sizeof(struct tm) = 56.
  Wrapper.addArg(Time, ARG_POINTER, 56);

  if (!Wrapper.sendAndWait())
    return 0;

  size_t R = Wrapper.getReturnValue<size_t>();

  printf("strftime returns %lu on the device.\n", R);

  __builtin_memcpy(str, Str, R);
  str[R] = '\0';

  printf("strftime gets %s.\n", str);

  return R;
}

struct tm *gmtime(const time_t *timer) {
  HostRPCDescriptorWrapper Wrapper(SYSCALLID_gmtime, 1);
  if (!Wrapper.isValid())
    return nullptr;

  time_t *Timer = (time_t *)HostRPCMemAlloc(sizeof(time_t));
  __builtin_memcpy(Timer, timer, sizeof(time_t));

  Wrapper.addArg(Timer, ARG_POINTER, sizeof(time_t));

  if (!Wrapper.sendAndWait())
    return nullptr;

  return Wrapper.getReturnValue<struct tm *>();
}

int __llvm_omp_fprintf(FILE *stream, const char *format, void *buffer,
                       int size) {
  HostRPCDescriptorWrapper Wrapper(SYSCALLID_fprintf, 4);
  if (!Wrapper.isValid())
    return 0;

  size_t Len = strlen(format) + 1;

  void *Buffer = HostRPCMemAlloc(size);
  char *Format = (char *)HostRPCMemAlloc(Len);
  __builtin_memcpy(Buffer, buffer, size);
  __builtin_memcpy(Format, format, Len);

  Wrapper.addArg(stream, ARG_LITERAL, sizeof(FILE *));
  Wrapper.addArg(Format, ARG_LITERAL, Len);
  Wrapper.addArg(Buffer, ARG_LITERAL, size);
  Wrapper.addArg(size, ARG_LITERAL, sizeof(int));

  if (!Wrapper.sendAndWait())
    return 0;

  return Wrapper.getReturnValue<int64_t>();
}

// -----------------------------------------------------------------------------

long strtol(const char *str, char **str_end, int base) {
  long r = 0;
  while (str && *str != '\0') {
    r = r * 10 + (*str - '0');
    ++str;
  }

  return r;
}

int strcmp(const char *lhs, const char *rhs) {
  while (*lhs != '\0' && *rhs != '\0') {
    if (*lhs == *rhs) {
      ++lhs;
      ++rhs;
    }
    return *lhs - *rhs;
  }
  if (*lhs != '\0')
    return 1;

  return -1;
}

void *calloc(size_t num, size_t size) {
  size_t bits = num * size;
  char *p = (char *)malloc(bits);
  if (!p)
    return p;
  char *q = (char *)p;
  while (q - p < bits) {
    *(int *)q = 0;
    q += sizeof(int);
  }
  while (q - p < bits) {
    *q = 0;
    q++;
  }
  return p;
}

int strcasecmp(const char *string1, const char *string2) {
  return strcmp(string1, string2);
}

void exit(int exit_code) { asm volatile("exit;"); }

size_t strlen(const char *str) {
  size_t r = 0;
  while (*str == ' ')
    ++str;

  while (*str != '\0') {
    ++r;
    ++str;
  }

  return r;
}

int atoi(const char *str) {
  int r = 0;
  while (*str != '\0')
    r = r * 10 + (*str) - '0';
  return r;
}

#define isdigit(c) (c >= '0' && c <= '9')

float atof(const char *s){
  float a = 0.0;
  int e = 0;
  int c;
  while ((c = *s++) != '\0' && isdigit(c)) {
    a = a*10.0 + (c - '0');
  }
  if (c == '.') {
    while ((c = *s++) != '\0' && isdigit(c)) {
      a = a*10.0 + (c - '0');
      e = e-1;
    }
  }
  if (c == 'e' || c == 'E') {
    int sign = 1;
    int i = 0;
    c = *s++;
    if (c == '+')
      c = *s++;
    else if (c == '-') {
      c = *s++;
      sign = -1;
    }
    while (isdigit(c)) {
      i = i*10 + (c - '0');
      c = *s++;
    }
    e += i*sign;
  }
  while (e > 0) {
    a *= 10.0;
    e--;
  }
  while (e < 0) {
    a *= 0.1;
    e++;
  }
  return a;
}

int fputs(const char *str, FILE *stream) {
  printf("%s", str);
  return 1;
}

int fprintf(FILE *stream, const char *format, ...) { return 0; }

int sprintf(char *buffer, const char *format, ...) { return 0; }

int sscanf(const char *buffer, const char *format, ...) { return 1; }

int fscanf(FILE *stream, const char *format, ...) { return 1; }

char *strcpy(char *dest, const char *src) {
  char *pd = dest;
  const char *ps = src;

  while (*ps != '\0')
    *(pd++) = *(ps++);

  *pd = '\0';

  return dest;
}

int *__errno_location() { return &errno; }

char *strcat(char *dest, const char *src) {
  char *pd = dest;
  const char *ps = src;

  while (*pd != '\0')
    ++pd;

  while (*ps != '\0')
    *(pd++) = *(ps++);

  *pd = '\0';

  return dest;
}

void perror(const char *s) { printf("%s", s); }

int strncmp(const char *lhs, const char *rhs, size_t count) {
  size_t c = 0;
  while (*lhs != '\0' && *rhs != '\0' && c < count) {
    if (*lhs == *rhs) {
      ++lhs;
      ++rhs;
      ++c;
    } else {
      return *lhs - *rhs;
    }
  }

  return 0;
}

char *strncpy(char *dest, const char *src, size_t count) {
  char *pd = dest;
  const char *ps = src;
  size_t c = 0;

  while (*ps != '\0' && c < count) {
    *(pd++) = *(ps++);
    ++c;
  }

  if (c < count)
    *pd = '\0';

  return dest;
}

char *strchr(const char *str, int ch) {
  while (*str != '\0') {
    if (*str == (char)ch)
      return const_cast<char *>(str);
  }
  if (*str == ch)
    return const_cast<char *>(str);

  return nullptr;
}

char *strtok(char *str, const char *delim) { return nullptr; }
double strtod(const char *nptr, char **endptr) { return 1.0; };

const unsigned short **__ctype_b_loc() { return nullptr; }

clock_t clock(void) { return 0; }

void *realloc(void *ptr, size_t new_size) { return malloc(new_size); }

void qsort(void *const pbase, size_t total_elems, size_t size,
           int (*comp)(const void *, const void *)) {}

int32_t **__ctype_tolower_loc(void) { return nullptr; }

time_t time(time_t *arg) { return 0; }

char *__xpg_basename(const char *path) { return const_cast<char *>(path); }

int clock_gettime(clockid_t clk_id, struct timespec *tp) { return 0; }

void srand(unsigned seed) {}

int rand() { return 1024; }

int abs(int n) { return n > 0 ? n : -n; }

void *memcpy(void *dest, const void *src, size_t count) {
  __builtin_memcpy(dest, src, count);
  return dest;
}

unsigned long strtoul(const char *str, char **str_end, int base) {
  unsigned long res = 0;
  while (*str != '\0') {
    if (*str == ' ') {
      ++str;
      continue;
    }
    if (*str >= '0' && *str <= '9') {
      res = res * 10 + *str - '0';
      ++str;
      continue;
    }
    break;
  }

  if (*str_end)
    *str_end = const_cast<char *>(str);
  return res;
}
}

#pragma omp end declare target
