// Copyright (c) 2026 Joshuah Rainstar
// SPDX-License-Identifier: MIT

#pragma once

#include "threadpool_atomic_fast.h"

#include <cstdint>
#include <span>
#include <stdexcept>
#include <utility>

namespace fileman::worker {

class atomic_thread_pool final {
public:
    explicit atomic_thread_pool(std::uint32_t thread_count) {
        const auto result = threadpool_create_checked(thread_count, &pool_);
        if (result != THREADPOOL_OK) {
            throw std::runtime_error("threadpool_create_checked failed: " +
                                     std::to_string(static_cast<int>(result)));
        }
    }

    ~atomic_thread_pool() { threadpool_destroy(pool_); }
    atomic_thread_pool(const atomic_thread_pool&) = delete;
    atomic_thread_pool& operator=(const atomic_thread_pool&) = delete;

    atomic_thread_pool(atomic_thread_pool&& other) noexcept
        : pool_(std::exchange(other.pool_, nullptr)) {}

    atomic_thread_pool& operator=(atomic_thread_pool&& other) noexcept {
        if (this != &other) {
            threadpool_destroy(pool_);
            pool_ = std::exchange(other.pool_, nullptr);
        }
        return *this;
    }

    [[nodiscard]] std::uint32_t thread_count() const noexcept {
        return threadpool_thread_count(pool_);
    }

    void run(std::span<threadpool_task_t> tasks) {
        if (tasks.size() > UINT32_MAX) {
            throw std::length_error("atomic thread-pool batch exceeds u32");
        }
        const auto result = threadpool_run_checked(
            pool_, tasks.data(), static_cast<std::uint32_t>(tasks.size()));
        if (result != THREADPOOL_OK) {
            throw std::runtime_error("threadpool_run_checked failed: " +
                                     std::to_string(static_cast<int>(result)));
        }
    }

private:
    threadpool_t* pool_{};
};

} // namespace fileman::worker
