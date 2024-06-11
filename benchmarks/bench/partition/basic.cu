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
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/reverse_iterator.h>

// Google Benchmark
#include <benchmark/benchmark.h>

// STL
#include <cstdlib>
#include <string>
#include <vector>

template <class T>
struct less_then_t
{
    T m_val;

    __host__ __device__ bool operator()(const T& val) const
    {
        return val < m_val;
    }
};

struct basic
{
    template <typename T, typename Policy = thrust::detail::device_t>
    float64_t
    run(const std::size_t elements, const std::string seed_type, const std::string entropy_str)
    {
        using select_op_t = less_then_t<T>;

        const bit_entropy entropy = bench_utils::str_to_entropy(entropy_str);

        T val = bench_utils::value_from_entropy<T>(bench_utils::entropy_to_probability(entropy));
        select_op_t select_op {val};

        thrust::device_vector<T> input = bench_utils::generate(elements, seed_type);
        thrust::device_vector<T> output(elements);

        bench_utils::gpu_timer d_timer;

        d_timer.start(0);
        thrust::copy_if(Policy {},
                        input.cbegin(),
                        input.cend(),
                        output.begin(),
                        thrust::make_reverse_iterator(output.begin() + elements),
                        select_op);
        d_timer.stop(0);

        return d_timer.get_duration();
    }
};

template <class Benchmark, class T>
void run_benchmark(benchmark::State& state,
                   const std::size_t elements,
                   const std::string seed_type,
                   const std::string entropy_str)
{
    // Benchmark object
    Benchmark benchmark {};

    // GPU times
    std::vector<double> gpu_times;

    for(auto _ : state)
    {
        float64_t duration = benchmark.template run<T>(elements, seed_type, entropy_str);
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

#define CREATE_BENCHMARK(T, Elements, EntropyStr)                                          \
    benchmark::RegisterBenchmark(bench_utils::bench_naming::format_name(                   \
                                     "{algo:partition,subalgo:" + name + ",input_type:" #T \
                                     + ",elements:" #Elements + ",entropy:" + #EntropyStr) \
                                     .c_str(),                                             \
                                 run_benchmark<Benchmark, T>,                              \
                                 Elements,                                                 \
                                 seed_type,                                                \
                                 EntropyStr)

#define BENCHMARK_TYPE_ENTROPY(type, entropy_str)                                               \
    CREATE_BENCHMARK(type, 1 << 16, entropy_str), CREATE_BENCHMARK(type, 1 << 20, entropy_str), \
        CREATE_BENCHMARK(type, 1 << 24, entropy_str), CREATE_BENCHMARK(type, 1 << 28, entropy_str)

template <class Benchmark>
void add_benchmarks(const std::string&                            name,
                    std::vector<benchmark::internal::Benchmark*>& benchmarks,
                    const std::string                             seed_type)
{
    const std::string entropy_strs[] = {"1.000", "0.544", "0.000"};

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

    // Benchmark parameters
    const std::string seed_type = "random";

    // Benchmark info
    bench_utils::add_common_benchmark_info();
    benchmark::AddCustomContext("seed", seed_type);

    // Add benchmark
    std::vector<benchmark::internal::Benchmark*> benchmarks;
    add_benchmarks<basic>("basic", benchmarks, seed_type);

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
