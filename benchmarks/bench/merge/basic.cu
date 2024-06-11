// MIT License
//
// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Benchmark utils
#include "../../bench_utils/bench_utils.hpp"

// rocThrust
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/merge.h>
#include <thrust/sort.h>

// Google Benchmark
#include <benchmark/benchmark.h>

// STL
#include <cstdlib>
#include <string>
#include <vector>

struct basic
{
    template <typename T, typename Policy = thrust::detail::device_t>
    float64_t run(const std::size_t elements,
                  const std::string entropy_str,
                  const std::size_t input_size_ratio)
    {
        const auto entropy = bench_utils::str_to_entropy(entropy_str);
        const auto elements_in_lhs
            = static_cast<std::size_t>(static_cast<double>(input_size_ratio * elements) / 100.0f);

        thrust::device_vector<T> input = bench_utils::generate(elements, entropy);
        thrust::device_vector<T> output(elements);

        thrust::sort(input.begin(), input.begin() + elements_in_lhs);
        thrust::sort(input.begin() + elements_in_lhs, input.end());

        bench_utils::gpu_timer d_timer;

        d_timer.start(0);
        thrust::merge(Policy {},
                      input.cbegin(),
                      input.cbegin() + elements_in_lhs,
                      input.cbegin() + elements_in_lhs,
                      input.cend(),
                      output.begin());
        d_timer.stop(0);

        return d_timer.get_duration();
    }
};

template <class Benchmark, class T>
void run_benchmark(benchmark::State& state,
                   const std::size_t elements,
                   const std::string entropy_str,
                   const std::size_t input_size_ratio)
{
    // Benchmark object
    Benchmark benchmark {};

    // GPU times
    std::vector<double> gpu_times;

    for(auto _ : state)
    {
        float64_t duration = benchmark.template run<T>(elements, entropy_str, input_size_ratio);
        state.SetIterationTime(duration);
        gpu_times.push_back(duration);
    }

    // BytesProcessed include read and written bytes, so when the BytesProcessed/s are reported
    // it will actually be the global memory bandwidth gotten.
    state.SetBytesProcessed(state.iterations() * 2 * elements * sizeof(T));
    state.SetItemsProcessed(state.iterations() * elements);

    const double gpu_cv = bench_utils::StatisticsCV(gpu_times);
    state.SetLabel(std::to_string(gpu_cv));
}

#define CREATE_BENCHMARK(T, Elements, EntropyStr, InputSizeRatio)                         \
    benchmark::RegisterBenchmark(bench_utils::bench_naming::format_name(                  \
                                     "{algo:merge,subalgo:" + name + ",input_type:" #T    \
                                     + ",elements:" #Elements + ",entropy:" + #EntropyStr \
                                     + ",input_size_ratio:" #InputSizeRatio)              \
                                     .c_str(),                                            \
                                 run_benchmark<Benchmark, T>,                             \
                                 Elements,                                                \
                                 EntropyStr,                                              \
                                 InputSizeRatio)

#define BENCHMARK_INPUT_SIZE_RATIO(type, elements, entropy_str) \
    CREATE_BENCHMARK(type, elements, entropy_str, 25),          \
        CREATE_BENCHMARK(type, elements, entropy_str, 50),      \
        CREATE_BENCHMARK(type, elements, entropy_str, 75)

#define BENCHMARK_TYPE_ENTROPY(type, entropy_str)               \
    BENCHMARK_INPUT_SIZE_RATIO(type, 1 << 16, entropy_str),     \
        BENCHMARK_INPUT_SIZE_RATIO(type, 1 << 20, entropy_str), \
        BENCHMARK_INPUT_SIZE_RATIO(type, 1 << 24, entropy_str), \
        BENCHMARK_INPUT_SIZE_RATIO(type, 1 << 28, entropy_str)

template <class Benchmark>
void add_benchmarks(const std::string&                            name,
                    std::vector<benchmark::internal::Benchmark*>& benchmarks)
{
    const std::string entropy_strs[] = {"1.000", "0.201"};

    for(std::string entropy_str : entropy_strs)
    {
        std::vector<benchmark::internal::Benchmark*> bs
            = {BENCHMARK_TYPE_ENTROPY(int8_t, entropy_str),
               BENCHMARK_TYPE_ENTROPY(int16_t, entropy_str),
               BENCHMARK_TYPE_ENTROPY(int32_t, entropy_str),
               BENCHMARK_TYPE_ENTROPY(int64_t, entropy_str),
               BENCHMARK_TYPE_ENTROPY(float, entropy_str),
               BENCHMARK_TYPE_ENTROPY(double, entropy_str)};
        benchmarks.insert(benchmarks.end(), bs.begin(), bs.end());
    }
}

int main(int argc, char* argv[])
{
    benchmark::Initialize(&argc, argv);
    bench_utils::bench_naming::set_format("human"); /* either: json,human,txt*/

    // Benchmark info
    bench_utils::add_common_benchmark_info();

    // Add benchmark
    std::vector<benchmark::internal::Benchmark*> benchmarks;
    add_benchmarks<basic>("basic", benchmarks);

    // Use manual timing
    for(auto& b : benchmarks)
    {
        b->UseManualTime();
        b->Unit(benchmark::kMicrosecond);
        b->MinTime(0.5); // in seconds
    }

    // Run benchmarks
    benchmark::RunSpecifiedBenchmarks(new bench_utils::CustomReporter);

    // Finish
    benchmark::Shutdown();
    return 0;
}
