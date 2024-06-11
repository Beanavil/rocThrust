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

#ifndef ROCTHRUST_BASE_HPP_
#define ROCTHRUST_BASE_HPP_

// Benchmark utils
#include "../../bench_utils/bench_utils.hpp"

// rocThrust
#include <thrust/device_vector.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

// Google Benchmark
#include <benchmark/benchmark.h>

// STL
#include <cstdlib>
#include <string>
#include <vector>

struct basic
{
    template <typename T, typename OpT, typename Policy = thrust::detail::device_t>
    float64_t run(benchmark::State& state,
                  const std::size_t elements,
                  const OpT         op,
                  const std::string entropy_str,
                  const std::size_t input_size_ratio)
    {
        const bit_entropy entropy = bench_utils::str_to_entropy(entropy_str);

        const auto elements_in_A
            = static_cast<std::size_t>(static_cast<double>(input_size_ratio * elements) / 100.0f);

        thrust::device_vector<T> input = bench_utils::generate(elements, entropy);
        thrust::device_vector<T> output(elements);

        thrust::sort(input.begin(), input.begin() + elements_in_A);
        thrust::sort(input.begin() + elements_in_A, input.end());

        const std::size_t elements_in_AB = thrust::distance(output.begin(),
                                                            op(Policy {},
                                                               input.cbegin(),
                                                               input.cbegin() + elements_in_A,
                                                               input.cbegin() + elements_in_A,
                                                               input.cend(),
                                                               output.begin()));

        // BytesProcessed include read and written bytes, so when the BytesProcessed/s are reported
        // it will actually be the global memory bandwidth gotten.
        state.SetBytesProcessed(state.bytes_processed() + (elements + elements_in_AB) * sizeof(T));

        bench_utils::gpu_timer d_timer;

        d_timer.start(0);
        op(Policy {},
           input.cbegin(),
           input.cbegin() + elements_in_A,
           input.cbegin() + elements_in_A,
           input.cend(),
           output.begin());
        d_timer.stop(0);

        return d_timer.get_duration();
    }
};

template <class T, class OpT>
void run_benchmark(benchmark::State& state,
                   const std::size_t elements,
                   const std::string entropy_str,
                   const std::size_t input_size_ratio)
{
    // Benchmark object
    basic benchmark {};

    // GPU times
    std::vector<double> gpu_times;

    for(auto _ : state)
    {
        float64_t duration = benchmark.template run<T, OpT>(
            state, elements, OpT {}, entropy_str, input_size_ratio);
        state.SetIterationTime(duration);
        gpu_times.push_back(duration);
    }

    state.SetItemsProcessed(state.iterations() * elements);

    const double gpu_cv = bench_utils::StatisticsCV(gpu_times);
    state.SetLabel(std::to_string(gpu_cv));
}

#define CREATE_BENCHMARK(T, Elements, EntropyStr, InputSizeRatio)                                \
    benchmark::RegisterBenchmark(bench_utils::bench_naming::format_name(                         \
                                     "{algo:" + algo_name + ",subalgo:basic" + ",input_type:" #T \
                                     + ",elements:" #Elements + ",entropy:" + #EntropyStr        \
                                     + ",input_size_ratio:" #InputSizeRatio)                     \
                                     .c_str(),                                                   \
                                 run_benchmark<T, OpT>,                                          \
                                 Elements,                                                       \
                                 EntropyStr,                                                     \
                                 InputSizeRatio)

#define BENCHMARK_ELEMENTS(type, elements, entropy_str)    \
    CREATE_BENCHMARK(type, elements, entropy_str, 25),     \
        CREATE_BENCHMARK(type, elements, entropy_str, 50), \
        CREATE_BENCHMARK(type, elements, entropy_str, 75)

#define BENCHMARK_TYPE_ENTROPY(type, entropy_str)       \
    BENCHMARK_ELEMENTS(type, 1 << 16, entropy_str),     \
        BENCHMARK_ELEMENTS(type, 1 << 20, entropy_str), \
        BENCHMARK_ELEMENTS(type, 1 << 24, entropy_str), \
        BENCHMARK_ELEMENTS(type, 1 << 28, entropy_str)

template <class OpT>
void add_benchmarks(const std::string&                            algo_name,
                    std::vector<benchmark::internal::Benchmark*>& benchmarks)
{
    const std::string entropy_strs[] = {"1.000", "0.201"};

    for(std::string entropy_str : entropy_strs)
    {
        std::vector<benchmark::internal::Benchmark*> bs
            = {BENCHMARK_TYPE_ENTROPY(int8_t, entropy_str),
               BENCHMARK_TYPE_ENTROPY(int16_t, entropy_str),
               BENCHMARK_TYPE_ENTROPY(int32_t, entropy_str),
               BENCHMARK_TYPE_ENTROPY(int64_t, entropy_str)};

        benchmarks.insert(benchmarks.end(), bs.begin(), bs.end());
    }
}

#endif // ROCTHRUST_BASE_HPP_
