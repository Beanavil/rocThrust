#ifndef ROCBENCH_UNARY_H
#define ROCBENCH_UNARY_H

#include <type_traits>

#include <cmath> // std::floor
#include <random> // std::mt19937
#include <complex>

namespace rocbench{
    namespace detail{

        template <typename T>
        struct random_to_item_t{
            double m_min;
            double m_max;

            __host__ __device__ random_to_item_t(T min, T max)
                : m_min(static_cast<double>(min))
                , m_max(static_cast<double>(max))
            {}

            __host__ __device__ T operator()(double random_value) const
            {
                if(std::is_floating_point<T>::value) {
                    return static_cast<T>((m_max - m_min) * random_value + m_min);
                }
                else {
                    return static_cast<T>(std::floor((m_max - m_min + 1) * random_value + m_min));
                }
            }
        };

        struct and_t {
        template <class T>
            __host__ __device__ T operator()(T a, T b) const {
                return a & b;
            }

            __host__ __device__ float operator()(float a, float b) const {
                const std::uint32_t result = reinterpret_cast<std::uint32_t&>(a) & reinterpret_cast<std::uint32_t&>(b);
                return reinterpret_cast<const float&>(result);
            }

            __host__ __device__ double operator()(double a, double b) const {
                const std::uint64_t result = reinterpret_cast<std::uint64_t&>(a) & reinterpret_cast<std::uint64_t&>(b);
                return reinterpret_cast<const double&>(result);
            }

            __host__ __device__ std::complex<float> operator()(std::complex<float> a, std::complex<float> b) const {
                double a_real = a.real();
                double a_imag = a.imag();

                double b_real = b.real();
                double b_imag = b.imag();

                const std::uint64_t result_real =
                reinterpret_cast<std::uint64_t&>(a_real) & reinterpret_cast<std::uint64_t&>(b_real);

                const std::uint64_t result_imag =
                reinterpret_cast<std::uint64_t&>(a_imag) & reinterpret_cast<std::uint64_t&>(b_imag);

                return {static_cast<float>(reinterpret_cast<const double&>(result_real)),
                        static_cast<float>(reinterpret_cast<const double&>(result_imag))};
            }
        };

        struct set_real_t {
            std::complex<float> m_min{};
            std::complex<float> m_max{};
            std::complex<float>* m_d_in{};
            const double* m_d_tmp{};

            __host__ __device__ void operator()(std::size_t i) const {
                m_d_in[i].real(random_to_item_t<double>{m_min.real(), m_max.real()}(m_d_tmp[i]));
            }
        };

        struct set_imag_t {
            std::complex<float> m_min{};
            std::complex<float> m_max{};
            std::complex<float>* m_d_in{};
            const double* m_d_tmp{};

            __host__ __device__ void operator()(std::size_t i) const {
                m_d_in[i].imag(random_to_item_t<double>{m_min.imag(), m_max.imag()}(m_d_tmp[i]));
            }
        };

        struct random_to_probability_t {
            double m_probability;

            __host__ __device__ bool operator()(double random_value) const {
                return random_value < m_probability;
            }
        };



    }
}


#endif //ROCBENCH_UNARY_H