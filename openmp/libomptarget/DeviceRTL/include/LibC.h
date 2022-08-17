//===--------- LibC.h - Simple implementation of libc functions --- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_LIBC_H
#define OMPTARGET_LIBC_H

#include "Types.h"

extern "C" {

struct FILE;
struct stat;
struct timeval;
struct timezone;
struct timespec;
struct tm;

int memcmp(const void *lhs, const void *rhs, size_t count);

int printf(const char *format, ...);

long strtol(const char *str, char **str_end, int base);

int strcmp(const char *lhs, const char *rhs);

void *calloc(size_t num, size_t size);

int strcasecmp(const char *string1, const char *string2);

void exit(int exit_code);

FILE *fopen(const char *filename, const char *mode);

size_t fread(void *buffer, size_t size, size_t count, FILE *stream);

size_t fwrite(const void *buffer, size_t size, size_t count, FILE *stream);

int fclose(FILE *stream);

size_t strlen(const char *str);

int atoi(const char *str);

int fputs(const char *str, FILE *stream);

int fprintf(FILE *stream, const char *format, ...);

int sprintf(char *buffer, const char *format, ...);

int sscanf(const char *buffer, const char *format, ...);

int fscanf(FILE *stream, const char *format, ...);

char *strcpy(char *dest, const char *src);

int stat(const char *path, struct stat *buf);

FILE *popen(const char *command, const char *type);

int *__errno_location();

char *strcat(char *dest, const char *src);

void perror(const char *s);

int getc(FILE *stream);

int strncmp(const char *lhs, const char *rhs, size_t count);

char *strncpy(char *dest, const char *src, size_t count);

int pclose(FILE *stream);

int fflush(FILE *stream);

void rewind(FILE *stream);

char *strchr(const char *str, int ch);

char *strtok(char *str, const char *delim);

const unsigned short **__ctype_b_loc(void);

clock_t clock(void);

void *realloc(void *ptr, size_t new_size);

void qsort(void *const pbase, size_t total_elems, size_t size,
           int (*comp)(const void *, const void *));

int fseek(FILE *stream, long offset, int origin);

long ftell(FILE *stream);

int feof(FILE *stream);

int32_t **__ctype_tolower_loc(void);

int gettimeofday(struct timeval *tv, struct timezone *tz);

time_t time(time_t *arg);

struct tm *gmtime(const time_t *timer);

size_t strftime(char *str, size_t count, const char *format,
                const struct tm *time);

char *__xpg_basename(const char *path);

char *fgets(char *str, int count, FILE *stream);

int clock_gettime(clockid_t clk_id, struct timespec *tp);

void srand(unsigned seed);

int rand();

int abs(int n);
}

#endif
