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

#ifndef ROCTHRUST_BENCHMARKS_BENCH_UTILS_GENERATION_UTILS_HPP_
#define ROCTHRUST_BENCHMARKS_BENCH_UTILS_GENERATION_UTILS_HPP_

// Utils
#include "common/types.hpp"

// Thrust
#include <thrust/copy.h>
#include <thrust/detail/config.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/find.h>
#include <thrust/generate.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/tuple.h>

// rocRAND
#include <rocrand/rocrand.h>

// rocPRIM/CUB
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_HIP
#include <rocprim/rocprim.hpp>
#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
#include <cub/device/device_copy.cuh>
#endif

// Google Benchmark
#include <benchmark/benchmark.h>

// STL
#include <algorithm>
#include <cstdint>
#include <limits>
#include <random>
#include <string>

namespace bench_utils
{
/// \brief Provides a sequence of seeds.
class managed_seed
{
public:
    /// \param[in] seed_string Either "random" to get random seeds,
    ///   or an unsigned integer to get (a sequence) of deterministic seeds.
    managed_seed(const std::string& seed_string)
    {
        is_random = seed_string == "random";
        if(!is_random)
        {
            const unsigned int seed = std::stoul(seed_string);
            std::seed_seq      seq {seed};
            seq.generate(seeds.begin(), seeds.end());
        }
    }

    managed_seed()
        : managed_seed("random") {};

    unsigned int get_0() const
    {
        return is_random ? std::random_device {}() : seeds[0];
    }

    unsigned int get_1() const
    {
        return is_random ? std::random_device {}() : seeds[1];
    }

    unsigned int get_2() const
    {
        return is_random ? std::random_device {}() : seeds[2];
    }

private:
    std::array<unsigned int, 3> seeds;
    bool                        is_random;
};

namespace detail
{

    static const std::map<std::string, bit_entropy> string_entropy_map
        = {{"0.000", bit_entropy::_0_000},
           {"0.201", bit_entropy::_0_201},
           {"0.337", bit_entropy::_0_337},
           {"0.544", bit_entropy::_0_544},
           {"0.811", bit_entropy::_0_811},
           {"1.000", bit_entropy::_1_000}};

    static const std::map<bit_entropy, double> entropy_probability_map
        = {{bit_entropy::_0_000, 0.0},
           {bit_entropy::_0_811, 0.811},
           {bit_entropy::_0_544, 0.544},
           {bit_entropy::_0_337, 0.337},
           {bit_entropy::_0_201, 0.201},
           {bit_entropy::_1_000, 1.0}};

} // namespace detail

template <typename T>
T value_from_entropy(double percentage)
{
    if(percentage == 1)
    {
        return std::numeric_limits<T>::max();
    }

    const auto max_val = static_cast<double>(std::numeric_limits<T>::max());
    const auto min_val = static_cast<double>(std::numeric_limits<T>::lowest());
    const auto result  = min_val + percentage * max_val - percentage * min_val;
    return static_cast<T>(result);
}

bit_entropy str_to_entropy(const std::string& str)
{
    auto it = detail::string_entropy_map.find(str);
    if(it != detail::string_entropy_map.end())
    {
        return it->second;
    }

    throw std::runtime_error("Can't convert string to bit entropy");
}

double entropy_to_probability(bit_entropy entropy)
{
    auto it = detail::entropy_probability_map.find(entropy);
    if(it != detail::entropy_probability_map.end())
    {
        return it->second;
    }

    // Default case (for unknown entropy values)
    return 0.0;
}

namespace detail
{
    // Unary operators

    template <typename T>
    struct random_to_item_t
    {
        double m_min;
        double m_max;

        __host__ __device__ random_to_item_t(T min, T max)
            : m_min(static_cast<double>(min))
            , m_max(static_cast<double>(max))
        {
        }

        __host__ __device__ T operator()(double random_value) const
        {
            if(std::is_floating_point<T>::value)
            {
                return static_cast<T>((m_max - m_min) * random_value + m_min);
            }
            else
            {
                return static_cast<T>(std::floor((m_max - m_min + 1) * random_value + m_min));
            }
        }
    };

    struct and_t
    {
        template <class T>
        __host__ __device__ T operator()(T a, T b) const
        {
            return a & b;
        }

        __host__ __device__ float operator()(float a, float b) const
        {
            const std::uint32_t result
                = reinterpret_cast<std::uint32_t&>(a) & reinterpret_cast<std::uint32_t&>(b);
            return reinterpret_cast<const float&>(result);
        }

        __host__ __device__ double operator()(double a, double b) const
        {
            const std::uint64_t result
                = reinterpret_cast<std::uint64_t&>(a) & reinterpret_cast<std::uint64_t&>(b);
            return reinterpret_cast<const double&>(result);
        }

        __host__ __device__ std::complex<float> operator()(std::complex<float> a,
                                                           std::complex<float> b) const
        {
            double a_real = a.real();
            double a_imag = a.imag();

            double b_real = b.real();
            double b_imag = b.imag();

            const std::uint64_t result_real = reinterpret_cast<std::uint64_t&>(a_real)
                                              & reinterpret_cast<std::uint64_t&>(b_real);

            const std::uint64_t result_imag = reinterpret_cast<std::uint64_t&>(a_imag)
                                              & reinterpret_cast<std::uint64_t&>(b_imag);

            return {static_cast<float>(reinterpret_cast<const double&>(result_real)),
                    static_cast<float>(reinterpret_cast<const double&>(result_imag))};
        }
    };

    struct random_to_probability_t
    {
        double m_probability;

        __host__ __device__ bool operator()(double random_value) const
        {
            return random_value < m_probability;
        }
    };

    // Generators

    class device_generator
    {
    public:
        device_generator()
        {
            rocrand_create_generator(&m_gen, ROCRAND_RNG_PSEUDO_DEFAULT);
        }

        ~device_generator()
        {
            rocrand_destroy_generator(m_gen);
        }

        double* new_uniform_distribution(seed_t seed, std::size_t num_items)
        {
            m_distribution.resize(num_items);
            m_distribution.clear();
            double* d_distribution = thrust::raw_pointer_cast(m_distribution.data());
            rocrand_set_seed(m_gen, seed);
            rocrand_generate_uniform_double(m_gen, d_distribution, num_items);

            return d_distribution;
        }

    private:
        thrust::device_vector<double> m_distribution;
        rocrand_generator             m_gen;
    };

    static device_generator default_generator;
    static seed_t           default_seed {0};

    template <typename T>
    auto generate_impl(device_generator&            generator,
                       seed_t&                      seed,
                       thrust::device_vector<bool>& buff,
                       std::size_t                  elements,
                       bit_entropy                  entropy,
                       bool /*min*/,
                       bool /*max*/
                       ) -> std::enable_if_t<std::is_same<bool, T>::value, void>
    {
        switch(entropy)
        {
        case bit_entropy::_0_000: {
            thrust::fill(buff.data(), buff.data() + elements, false);
        }
        case bit_entropy::_1_000: {
            thrust::fill(buff.data(), buff.data() + elements, true);
        }
        default: {
            double* unf_dist = generator.new_uniform_distribution(seed, buff.size());
            thrust::transform(unf_dist,
                              unf_dist + buff.size(),
                              buff.data(),
                              random_to_probability_t {entropy_to_probability(entropy)});
        };
        }
        seed++;
    }

    template <typename T>
    auto generate_impl(device_generator&         generator,
                       seed_t&                   seed,
                       thrust::device_vector<T>& buff,
                       std::size_t               elements,
                       bit_entropy               entropy,
                       T                         min,
                       T max) -> std::enable_if_t<!std::is_same<bool, T>::value>
    {

        switch(entropy)
        {
        case bit_entropy::_0_000: {
            std::mt19937                          std_generator(seed);
            std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
            T randomValue = random_to_item_t<T>(min, max)(distribution(std_generator));
            thrust::fill(buff.data(), buff.data() + elements, randomValue);
            break;
        }
        case bit_entropy::_1_000: {
            double* unf_dist = generator.new_uniform_distribution(seed, elements);
            thrust::transform(
                unf_dist, unf_dist + elements, buff.data(), random_to_item_t<T>(min, max));
            break;
        }
        default: {
        }
        };
        seed++;
    }

    template <typename T>
    thrust::device_vector<T>
    generate(const std::size_t elements, const bit_entropy entropy, T min, T max)
    {
        thrust::device_vector<T> output(elements);

        // Prevent inf overlow which set all values in data to inf.
        if(std::is_same<double, T>::value || std::is_same<long double, T>::value)
        {
            min = std::pow(std::numeric_limits<float>::lowest(),
                           3); // use power of 3 to keep negative valueAanwa
            max = std::pow(std::numeric_limits<float>::max(), 3);
        }

        switch(entropy)
        {
        case bit_entropy::_0_000:
        case bit_entropy::_1_000: {
            detail::generate_impl(detail::default_generator,
                                  detail::default_seed,
                                  output,
                                  elements,
                                  entropy,
                                  min,
                                  max);
            break;
        }
        default: {
            if(std::is_same<bool, T>::value)
            {
                detail::generate_impl(detail::default_generator,
                                      detail::default_seed,
                                      output,
                                      elements,
                                      entropy,
                                      min,
                                      max);
            }
            else
            {
                detail::generate_impl(detail::default_generator,
                                      detail::default_seed,
                                      output,
                                      elements,
                                      bit_entropy::_1_000,
                                      min,
                                      max);
                const int                epochs = static_cast<int>(entropy);
                thrust::device_vector<T> epoch_output(elements);

                for(int i = 0; i < epochs; i++)
                {
                    detail::generate_impl(detail::default_generator,
                                          detail::default_seed,
                                          epoch_output,
                                          elements,
                                          bit_entropy::_1_000,
                                          min,
                                          max);
                    thrust::transform(output.data(),
                                      output.data() + elements,
                                      epoch_output.data(),
                                      output.data(),
                                      and_t {});
                }
            }
            break;
        }
        }

        return output;
    }

    template <class T>
    struct geq_t
    {
        T val;

        __host__ __device__ bool operator()(T x)
        {
            return x >= val;
        }
    };

    template <typename T>
    std::size_t gen_uniform_offsets(thrust::device_vector<T>& segment_offsets,
                                    const std::size_t         min_segment_size,
                                    const std::size_t         max_segment_size)
    {
        const T elements = segment_offsets.size() - 2;

        segment_offsets = bench_utils::detail::generate(segment_offsets.size(),
                                                        bit_entropy::_1_000,
                                                        static_cast<T>(min_segment_size),
                                                        static_cast<T>(max_segment_size));

        // Find the range of contiguous offsets starting from index 0 which sum is greater or
        // equal than 'elements'.
        auto tail = [&]() {
            const thrust::detail::device_t policy {};

            // Add the offset 'elements + 1' to the array of segment offsets to make sure that
            // there is at least one offset greater than 'elements'.
            thrust::fill_n(policy, segment_offsets.data() + elements, 1, elements + 1);

            // Perform an exclusive prefix sum scan with first value 0, so what we compute is
            // scan[i + 1] = \sum_{i=0}^{i} segment_offsets[i] for i \in [0, elements+1]
            // and scan[0] = 0;
            thrust::exclusive_scan(policy,
                                   segment_offsets.data(),
                                   segment_offsets.data() + segment_offsets.size(),
                                   segment_offsets.data() /*, thrust::plus<>{}*/);

            // Find first sum of offsets greater than 'elements', we are sure that there is
            // going to be one because we added elements + 1 at the end of the segment_offsets.
            auto iter = thrust::find_if(policy,
                                        segment_offsets.data(),
                                        segment_offsets.data() + segment_offsets.size(),
                                        geq_t<T> {elements});

            // Compute the element's index.
            auto dist = thrust::distance(segment_offsets.data(), iter);
            // Fill next item with 'elements'.
            thrust::fill_n(policy, segment_offsets.data() + dist, 1, elements);
            // Return next item's index.
            return dist + 1;
        };

        return tail();
    }

    // Temporal approach for generation of key segments withouth using iterators.
    template <typename T>
    void gen_key_segments(thrust::device_vector<T>&      keys,
                          thrust::device_vector<size_t>& segment_offsets)
    {
        const std::size_t total_segments = segment_offsets.size() - 1;

        thrust::device_vector<T*>          d_srcs(total_segments);
        thrust::device_vector<T*>          d_dsts(total_segments);
        thrust::device_vector<std::size_t> d_sizes(total_segments);

        for(std::size_t segment = 0; segment < total_segments; ++segment)
        {
            d_sizes[segment]              = segment_offsets[segment + 1] - segment_offsets[segment];
            const std::size_t        size = d_sizes[segment];
            thrust::device_vector<T> seq(size);
            thrust::sequence(seq.begin(), seq.end());
            d_srcs[segment] = thrust::raw_pointer_cast(&seq[0]);
            d_dsts[segment] = thrust::raw_pointer_cast(&keys[segment_offsets[segment]]);
        }

        std::uint8_t* d_temp_storage     = nullptr;
        std::size_t   temp_storage_bytes = 0;

        rocprim::batch_copy(d_temp_storage,
                            temp_storage_bytes,
                            thrust::raw_pointer_cast(d_srcs.data()),
                            thrust::raw_pointer_cast(d_dsts.data()),
                            thrust::raw_pointer_cast(d_sizes.data()),
                            total_segments);

        thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
        d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

        rocprim::batch_copy(d_temp_storage,
                            temp_storage_bytes,
                            thrust::raw_pointer_cast(d_srcs.data()),
                            thrust::raw_pointer_cast(d_dsts.data()),
                            thrust::raw_pointer_cast(d_sizes.data()),
                            total_segments);
        hipDeviceSynchronize();
    }

    // TODO: use this approach when rocPRIM allows it.
    // template <class T>
    // struct offset_to_iterator_t
    // {
    //     T* base_it;

    //     __host__ __device__ __forceinline__ T* operator()(std::size_t offset) const
    //     {
    //         return base_it + offset;
    //     }
    // };

    // template <class T>
    // struct repeat_index_t
    // {
    //     __host__ __device__ __forceinline__ thrust::constant_iterator<T> operator()(std::size_t i)
    //     {
    //         return thrust::constant_iterator<T>(static_cast<T>(i));
    //     }
    // };

    // struct offset_to_size_t
    // {
    //     std::size_t* offsets = nullptr;

    //     __host__ __device__ __forceinline__ std::size_t operator()(std::size_t i)
    //     {
    //         return offsets[i + 1] - offsets[i];
    //     }
    // };

    // template <typename T>
    // void gen_key_segments(thrust::device_vector<T>&           keys,
    //                       thrust::device_vector<std::size_t>& segment_offsets)
    // {

    //     const std::size_t total_segments = segment_offsets.size() - 1;

    //     thrust::counting_iterator<int> iota(0);
    //     repeat_index_t<T>       src_transform_op {};
    //     offset_to_iterator_t<T> dst_transform_op {thrust::raw_pointer_cast(keys.data())};
    //     offset_to_size_t size_transform_op {thrust::raw_pointer_cast(segment_offsets.data())};

    //     auto d_range_srcs = thrust::make_transform_iterator(iota, src_transform_op);
    //     auto d_range_dsts
    //         = thrust::make_transform_iterator(segment_offsets.begin(), dst_transform_op);
    //     auto d_range_sizes = thrust::make_transform_iterator(iota, size_transform_op);

    //     std::uint8_t*     d_temp_storage     = nullptr;
    //     std::size_t       temp_storage_bytes = 0;

    //     rocprim::batch_copy(d_temp_storage,
    //                         temp_storage_bytes,
    //                         d_range_srcs,
    //                         d_range_dsts,
    //                         d_range_sizes,
    //                         total_segments);

    //     thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
    //     d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

    //     rocprim::batch_copy(d_temp_storage,
    //                         temp_storage_bytes,
    //                         d_range_srcs,
    //                         d_range_dsts,
    //                         d_range_sizes,
    //                         total_segments);
    //     hipDeviceSynchronize();
    // }

    struct device_generator_base_t
    {
        const std::size_t elements {0};
        const bit_entropy entropy {bit_entropy::_1_000};

        device_generator_base_t(std::size_t m_elements, bit_entropy m_entropy)
            : elements(m_elements)
            , entropy(m_entropy)
        {
        }

        template <typename T>
        thrust::device_vector<T> generate(T min, T max)
        {
            return bench_utils::detail::generate(elements, entropy, min, max);
        }
    };

    template <class T>
    struct device_vector_generator_t : device_generator_base_t
    {
        const T min {std::numeric_limits<T>::min()};
        const T max {std::numeric_limits<T>::max()};

        device_vector_generator_t(std::size_t m_elements, bit_entropy m_entropy, T m_min, T m_max)
            : device_generator_base_t(m_elements, m_entropy)
            , min(m_min)
            , max(m_max)
        {
        }

        operator thrust::device_vector<T>()
        {
            return device_generator_base_t::generate(min, max);
        }
    };

    template <>
    struct device_vector_generator_t<void> : device_generator_base_t
    {
        device_vector_generator_t(std::size_t m_elements, bit_entropy m_entropy)
            : device_generator_base_t(m_elements, m_entropy)
        {
        }

        template <typename T>
        operator thrust::device_vector<T>()
        {
            return device_generator_base_t::generate(std::numeric_limits<T>::min(),
                                                     std::numeric_limits<T>::max());
        }
    };

    struct device_uniform_key_segments_generator_t
    {
        const std::size_t elements {0};
        const std::size_t min_segment_size {0};
        const std::size_t max_segment_size {0};

        device_uniform_key_segments_generator_t(std::size_t       m_elements,
                                                const std::size_t m_min_segment_size,
                                                const std::size_t m_max_segment_size)
            : elements(m_elements)
            , min_segment_size(m_min_segment_size)
            , max_segment_size(m_max_segment_size)
        {
        }

        template <class KeyT>
        operator thrust::device_vector<KeyT>()
        {
            thrust::device_vector<KeyT> keys(elements);

            thrust::device_vector<std::size_t> segment_offsets(keys.size() + 2);
            const std::size_t                  offsets_size
                = gen_uniform_offsets(segment_offsets, min_segment_size, max_segment_size);
            segment_offsets.resize(offsets_size);

            gen_key_segments(keys, segment_offsets);

            return keys;
        }
    };

    struct gen_uniform_key_segments_t
    {
        device_uniform_key_segments_generator_t operator()(const std::size_t elements,
                                                           const std::size_t min_segment_size,
                                                           const std::size_t max_segment_size) const
        {
            return {elements, min_segment_size, max_segment_size};
        }
    };

    struct gen_uniform_t
    {
        gen_uniform_key_segments_t key_segments {};
    };

    struct gen_t
    {
        template <class T>
        device_vector_generator_t<T> operator()(std::size_t       elements,
                                                const bit_entropy entropy = bit_entropy::_1_000,
                                                T                 min = std::numeric_limits<T>::min,
                                                T max = std::numeric_limits<T>::max()) const
        {
            return {elements, entropy, min, max};
        }

        device_vector_generator_t<void>
        operator()(std::size_t elements, const bit_entropy entropy = bit_entropy::_1_000) const
        {
            return {elements, entropy};
        }

        gen_uniform_t uniform {};
    };
} // namespace detail

detail::gen_t generate;

} // namespace bench_utils
#endif // ROCTHRUST_BENCHMARKS_BENCH_UTILS_GENERATION_UTILS_HPP_
