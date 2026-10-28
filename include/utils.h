#pragma once
#include <algorithm>
#include <memory>
#include <random>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <string>
#include <vector>
#include <sys/sysinfo.h>
#include "types.h"

inline std::shared_ptr<spdlog::logger> g_logger = nullptr;

inline void setup_logger(const std::string& log_file_path, spdlog::level::level_enum console_level = spdlog::level::info) {
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_level(console_level);
    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_file_path, true);
    file_sink->set_level(spdlog::level::debug);
    std::vector<spdlog::sink_ptr> sinks{console_sink, file_sink};
    g_logger = std::make_shared<spdlog::logger>("multi_sink", sinks.begin(), sinks.end());
    g_logger->set_level(spdlog::level::debug);
    g_logger->set_pattern("[%Y-%m-%d %H:%M:%S] [%l] %v");
    spdlog::set_default_logger(g_logger);
}


template<typename T>
std::vector<float> vec_to_float(const std::vector<T>& v) {
    std::vector<float> res;
    res.reserve(v.size());
    for (const auto& x : v) res.push_back(static_cast<float>(x));
    return res;
}

template<typename T>
std::vector<float> flatten_to_float(const std::vector<std::vector<T>>& vv) {
    if (vv.empty()) return {};
    size_t dim = vv[0].size();
    std::vector<float> out(vv.size() * dim);
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < vv.size(); ++i)
        for (size_t j = 0; j < dim; ++j)
            out[i*dim + j] = static_cast<float>(vv[i][j]);
    return out;
}


#define BATCH_SIZE 1000000

template<typename T>
int estimate_batch_size(int dim) {
    struct sysinfo info;
    sysinfo(&info);
    size_t avail_bytes = info.freeram / 2;
    size_t bytes_per_elem = sizeof(T);
    int batch = static_cast<int>(avail_bytes / (dim * bytes_per_elem));
    return batch;
}

template<typename T>
int estimate_batch_size(int dim,size_t mem_limit) {
    struct sysinfo info;
    sysinfo(&info);
    size_t avail_bytes = info.freeram / 2;
    avail_bytes = std::min(avail_bytes, mem_limit);
    size_t bytes_per_elem = sizeof(T);
    int batch = static_cast<int>(avail_bytes / (dim * bytes_per_elem));
    return batch;
}


inline std::vector<GID_Type> random_k_unique(GL_Type length,GL_Type k) {
    std::vector<GID_Type> all(length);
    for (GL_Type i = 0; i < length; ++i) all[i] = i;
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(all.begin(), all.end(), g);
    if (k > length) k = length;
    return std::vector<GID_Type>(all.begin(), all.begin() + k);
}
