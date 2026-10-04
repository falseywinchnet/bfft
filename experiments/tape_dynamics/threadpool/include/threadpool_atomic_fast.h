// Copyright (c) 2026 Joshuah Rainstar
// SPDX-License-Identifier: MIT

#ifndef FILEMAN_THREADPOOL_ATOMIC_FAST_H
#define FILEMAN_THREADPOOL_ATOMIC_FAST_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct threadpool_t threadpool_t;

typedef struct {
    void (*function)(void *);
    void *argument;
} threadpool_task_t;

typedef enum {
    THREADPOOL_OK = 0,
    THREADPOOL_INVALID_ARGUMENT = 1,
    THREADPOOL_ALLOCATION_FAILED = 2,
    THREADPOOL_SYSTEM_ERROR = 3,
    THREADPOOL_INVALID_TASK = 4
} threadpool_result_t;

/* Checked construction. `thread_count` must be non-zero. */
threadpool_result_t threadpool_create_checked(uint32_t thread_count,
                                              threadpool_t **output);

/* Synchronous batch. Concurrent calls are serialized. Callbacks must return. */
threadpool_result_t threadpool_run_checked(threadpool_t *pool,
                                           const threadpool_task_t *tasks,
                                           uint32_t count);

uint32_t threadpool_thread_count(const threadpool_t *pool);

/* Safe lifetime still requires no new API call to begin during destruction. */
void threadpool_destroy(threadpool_t *pool);

/* Source-compatible conveniences retained from the supplied control. */
threadpool_t *threadpool_create(uint32_t thread_count);
void threadpool_run(threadpool_t *pool, threadpool_task_t *tasks, uint32_t count);

#ifdef __cplusplus
}
#endif

#endif
