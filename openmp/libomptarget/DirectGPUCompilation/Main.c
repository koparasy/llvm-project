//===------- Main.c - Direct compilation program start point ------ C -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <string.h>

extern int user_main(int, char *[]);

int main(int argc, char *argv[]) {
#pragma omp target enter data map(to: argv[:argc])

  for (int I = 0; I < argc; ++I) {
    size_t Len = strlen(argv[I]);
#pragma omp target enter data map(to: argv[I][:Len])
  }

  int Ret;
#pragma omp target teams num_teams(1) map(from: Ret) thread_limit(1024)
  { Ret = user_main(argc, argv); }

  return Ret;
}
