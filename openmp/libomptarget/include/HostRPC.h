//===------- HostRPC.h - Host RPC ---------------------------- C++ --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OPENMP_LIBOMPTARGET_INCLUDE_HOSTRPC_H
#define OPENMP_LIBOMPTARGET_INCLUDE_HOSTRPC_H

enum SyscallId {
  SYSCALLID_fopen = 0,
  SYSCALLID_fread = 1,
  SYSCALLID_stat = 2,
  SYSCALLID_popen = 3,
  SYSCALLID_fwrite = 4,
  SYSCALLID_fclose = 5,
  SYSCALLID_getc = 6,
  SYSCALLID_pclose = 7,
  SYSCALLID_fflush = 8,
  SYSCALLID_rewind = 9,
  SYSCALLID_fseek = 10,
  SYSCALLID_ftell = 11,
  SYSCALLID_feof = 12,
  SYSCALLID_gettimeofday = 13,
  SYSCALLID_fgets = 14,
  SYSCALLID_strftime = 15,
  SYSCALLID_gmtime = 16,
  SYSCALLID_fprintf = 17,
};

enum ExecutionStatus {
  EXEC_STAT_CREATED = 0,
  EXEC_STAT_DONE = 1,
};

// uint64_t has to be defined ahead of time.

enum ArgType {
  ARG_LITERAL = 0,
  ARG_POINTER = 1,
};

struct ArgTy {
  void *Arg;
  ArgType Type;
  uint64_t Size;
};

struct HostRPCDescriptor {
  enum SyscallId Id;
  struct ArgTy *Args;
  uint64_t NumArgs;
  volatile enum ExecutionStatus Status;
  volatile void *ReturnValue;
  uint64_t RVSize;
};

#endif
