//===------- Libc.c - Simple implementation of libc functions ----- C -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma omp begin declare target device_type(nohost)

#include <stdio.h>

#include "HostRPC.h"

volatile SyscallDescriptor *omptarget_syscall_request
    __attribute__((used, retain, weak, visibility("protected")));

typedef unsigned long size_t;

extern void *malloc(size_t size);

long strtol(const char *str, char **str_end, int base) {
  long r = 0;
  while (str && *str != '\0') {
    r = r * 10 + (*str - '0');
    ++str;
  }

  return r;
}

int strcmp(const char *lhs, const char *rhs) {
  while (*lhs != '\0' && *rhs != '0') {
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

void exit(int exit_code) {}

FILE *fopen(const char *filename, const char *mode) {
  struct SyscallDescriptor *SD =
      (struct SyscallDescriptor *)malloc(sizeof(struct SyscallDescriptor));
  if (SD == NULL)
    return NULL;

  SD->Id = SYSCALLID_FOPEN;
  SD->NumArgs = 2;
  SD->Status = EXEC_STAT_CREATED;
  SD->ReturnValue = NULL;
  SD->RVSize = sizeof(void *);

  struct ArgTy *Args = (struct ArgTy *)malloc(sizeof(struct ArgTy) * 2);
  if (Args == NULL)
    return NULL;

  Args[0].Arg = filename;
  Args[1].Arg = mode;
  Args[0].Size = strlen(filename);
  Args[1].Size = strlen(mode);

  SD->Args = Args;

  volatile struct SyscallDescriptor **GlobalSD = &omptarget_syscall_request;

  if (!*GlobalSD)
    *GlobalSD = SD;

  while (SD->Status == EXEC_STAT_CREATED)
    ;

  *GlobalSD = NULL;

  while (SD->Status == EXEC_STAT_RECEIVED)
    ;

  if (SD->Status == EXEC_STAT_FAILED)
    return NULL;

  FILE *R = (FILE *)SD->ReturnValue;

  free(Args);
  free(SD);

  return R;
}

size_t fwrite(const void *restrict buffer, size_t size, size_t count,
              FILE *restrict stream) {
  return size;
}

int fclose(FILE *stream) { return 0; }

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

int atoi(const char *str) { return 0; }

#include <bits/types/FILE.h>

int fputs(const char *str, FILE *stream) {
  printf("%s", str);
  return 1;
}

#pragma omp end declare target
