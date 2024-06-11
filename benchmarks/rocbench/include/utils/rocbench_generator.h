#ifndef ROCBENCH_GENERATOR_H
#define ROCBENCH_GENERATOR_H

#include <thrust/device_vector.h>
#include <rocrand/rocrand.h>

#include <limits>
#include <type_traits>

#include <cmath> // std::floor
#include <random> // std::mt19937
#include <complex>
#include <cstdlib> // std::rand

#include "rocbench_types.h"
#include "rocbench_entropy.h"
#include "rocbench_unary.h"

namespace rocbench{
namespace detail{

    template <typename T>
    struct is_complex :
        std::integral_constant<bool,
            std::is_same<std::complex<float>, T>::value ||
            std::is_same<std::complex<double>, T>::value
        > {};

    class device_generator {
    public:
        device_generator() {
            rocrand_create_generator(&m_gen, ROCRAND_RNG_PSEUDO_DEFAULT);
        }

        ~device_generator() {
            rocrand_destroy_generator(m_gen);
        }

        double* new_uniform_distribution(seed_t seed, std::size_t num_items){
            m_distribution.resize(num_items);
            m_distribution.clear();
            double* d_distribution = thrust::raw_pointer_cast(m_distribution.data());
            rocrand_set_seed(m_gen, seed);
            rocrand_generate_uniform_double(m_gen, d_distribution, num_items);
            hipDeviceSynchronize();
            return d_distribution;
        }

    private:
        thrust::device_vector<double> m_distribution;
        rocrand_generator m_gen;
    };

    static device_generator default_generator;
    static seed_t default_seed{0};

    template<typename T>
    auto generate_input_impl(
        device_generator& generator,
        seed_t& seed,
        thrust::device_vector<bool>& buff,
        std::size_t elements,
        bit_entropy entropy,
        bool /*min*/,
        bool /*max*/
    ) -> std::enable_if_t<std::is_same<bool, T>::value, void> {
        switch(entropy){
            case bit_entropy::_0_000 : {
                thrust::fill(buff.data(), buff.data() + elements, false);
            }
            case bit_entropy::_1_000 : {
                thrust::fill(buff.data(), buff.data() + elements, true);
            }
            default: {
                double* unf_dist = generator.new_uniform_distribution(seed, buff.size());
                thrust::transform(
                    unf_dist, unf_dist + buff.size(), buff.data(),
                    random_to_probability_t{entropy_to_probability(entropy)}
                );
            };
        }
        seed++;
    }

    template<typename T>
    auto generate_input_impl(
        device_generator& generator,
        seed_t& seed,
        thrust::device_vector<T>& buff,
        std::size_t elements,
        bit_entropy entropy,
        T min,
        T max
    ) -> std::enable_if_t<is_complex<T>::value, void> {
        switch(entropy){
            case bit_entropy::_0_000 : {
                std::mt19937 std_generator(seed);
                std::uniform_real_distribution<double> distribution(0.0f, 1.0f);
                const float random_imag = random_to_item_t<double>(min.imag(), max.imag())(distribution(std_generator));
                const float random_real = random_to_item_t<double>(min.imag(), max.imag())(distribution(std_generator));
                thrust::fill(buff.data(), buff.data() + elements, T{random_real, random_imag});
            }
            case bit_entropy::_1_000 : {
                double* unf_dist_real = generator.new_uniform_distribution(seed++, elements);
                double* unf_dist_imag = generator.new_uniform_distribution(seed++, elements);
                thrust::for_each_n(
                    thrust::make_counting_iterator(0), elements, set_real_t{min, max, buff.data(), unf_dist_real}
                );
                thrust::for_each_n(
                    thrust::make_counting_iterator(0), elements, set_imag_t{min, max, buff.data(), unf_dist_imag}
                );
            }
            default: {}
        };
        seed++;
    }
    
    template<typename T>
    auto generate_input_impl(
        device_generator& generator,
        seed_t& seed,
        thrust::device_vector<T>& buff,
        std::size_t elements,
        bit_entropy entropy,
        T min,
        T max
    ) -> std::enable_if_t<!std::is_same<bool, T>::value && !is_complex<T>::value, void> {

        switch(entropy){
            case bit_entropy::_0_000 : {
                std::mt19937 std_generator(seed);
                std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
                T randomValue = random_to_item_t<T>(min, max)(distribution(std_generator));
                thrust::fill(buff.data(), buff.data() + elements, randomValue);
                break;
            }
            case bit_entropy::_1_000 : {
                double* unf_dist = generator.new_uniform_distribution(seed, elements);
                thrust::transform(unf_dist, unf_dist + elements, buff.data(), random_to_item_t<T>(min, max));
                break;
            }
            default: {}
        };
        seed++;
        
    }

    struct random_data_generator {
        const std::size_t elements_{0};
        const bit_entropy entropy_{bit_entropy::_1_000};

        random_data_generator(std::size_t elements, bit_entropy entropy) : 
            elements_(elements),
            entropy_(entropy)
        {}

        template<typename T>
        thrust::device_vector<T> generate_input(
            T min_bound               = std::numeric_limits<T>::lowest(),
            T max_bound               = std::numeric_limits<T>::max()
        ){  
            thrust::device_vector<T> output(elements_);

            T min = min_bound;
            T max = max_bound;

            // Prevent inf overlow which set all values in data to inf.
            if(std::is_same<double, T>::value || std::is_same<long double, T>::value){
                min = std::pow(std::numeric_limits<float>::lowest(), 3); // use power of 3 to keep negative valueAanwa
                max = std::pow(std::numeric_limits<float>::max(), 3);
            }

            switch(entropy_){
                case bit_entropy::_0_000 : 
                case bit_entropy::_1_000 : {
                    detail::generate_input_impl(detail::default_generator, detail::default_seed, output, elements_, entropy_, min, max);
                    break;
                }
                default : {
                    if(std::is_same<bool, T>::value){
                        detail::generate_input_impl(detail::default_generator, detail::default_seed, output, elements_, entropy_, min, max);
                    }else{
                        detail::generate_input_impl(detail::default_generator, detail::default_seed, output, elements_, bit_entropy::_1_000, min, max);
                        const int epochs = static_cast<int>(entropy_);
                        thrust::device_vector<T> epoch_output(elements_);

                        for(int i = 0; i < epochs; i++){
                            detail::generate_input_impl(detail::default_generator, detail::default_seed, epoch_output, elements_, bit_entropy::_1_000, min, max);
                            thrust::transform(output.data(), output.data() + elements_, epoch_output.data(), output.data(), and_t{});
                        }
                    }
                    break;
                }
            }

            return output;
        }

    };

    template <typename T>
    struct thrust_device_vector_generator_impl {
        const std::size_t elements_;
        const bit_entropy entropy_;
        const T min_bound_;
        const T max_bound_;

        thrust_device_vector_generator_impl(
            std::size_t elements, bit_entropy entropy,
            T min_bound, T max_bound
        ):
            elements_(elements),
            entropy_(entropy),
            min_bound_(min_bound),
            max_bound_(max_bound)
        {}

        operator thrust::device_vector<T>() const {
            random_data_generator random_generator(elements_, entropy_);
            return random_generator.generate_input<T>(min_bound_, max_bound_);
        }
    };

    struct thrust_device_vector_generator {
        std::size_t elements_;
        bit_entropy entropy_;
        long double min_bound_;
        long double max_bound_;

        thrust_device_vector_generator(
            std::size_t elements, bit_entropy entropy,
            long double min_bound, long double max_bound
        ):
            elements_(elements),
            entropy_(entropy),
            min_bound_(min_bound),
            max_bound_(max_bound)
        {}

        template<typename T>
        operator thrust::device_vector<T>() const {
            return thrust_device_vector_generator_impl<T>(
                elements_, entropy_,
                min_bound_ == std::numeric_limits<long double>::infinity() ? std::numeric_limits<T>::lowest() : static_cast<T>(min_bound_),
                max_bound_ == std::numeric_limits<long double>::infinity() ? std::numeric_limits<T>::max() : static_cast<T>(max_bound_)
            );
        }

    };

}} // namespace rocbench::detail


#endif // ROCBENCH_GENERATOR_H