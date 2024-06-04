#ifndef ROCBENCH_HELPER_CUH
#define ROCBENCH_HELPER_CUH

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <rocrand/rocrand.h>

#include <cinttypes>
#include <functional>
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <limits>
#include <type_traits>
#include <cmath> // std::floor
#include <random> // std::mt19937
#include <complex>

#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || __cplusplus >= 201703L) //C++ 17+ available
    #include <any>
#else
    #include "rocbench_cpp14_compat.h"
#endif

namespace rocbench {

    namespace detail{
        constexpr auto bit_entropy_default = (1 << 12) + (200 - 96);
    }

    enum class exec_tag : uint8_t {
        sync
    };

    enum class bit_entropy {
        _0_000 = detail::bit_entropy_default,
        _0_201 = 4,
        _0_337 = 3,
        _0_544 = 2,
        _0_811 = 1,
        _1_000 = 0
    };

    class launch {};

    template <typename... Ts>
    struct type_list {};

    class state {
    public:
        void add_element_count(const std::size_t& elements){ std::cout << "state::add_element_count::elements = "  << elements << std::endl; };

        // These functions are currently not used,  but to supress warnings their output are logged. 
        template<typename T>
        void add_global_memory_reads( const T& elements ) { std::cout << "state::add_global_memory_reads::elements = "  << static_cast<std::int64_t>(elements) << std::endl; }

        template<typename T>
        void add_global_memory_writes( const T& elements ) { std::cout << "state::add_global_memory_writes::elements = "  << static_cast<std::int64_t>(elements) << std::endl; }

        void exec(const exec_tag& tag, std::function<void(launch&)> task){
            auto dummy_launch = launch{};
            
            switch(tag){
                case exec_tag::sync : {
                    task(dummy_launch);
                }
                default : {}
            }
        }

        int64_t get_int64(const std::string& name){
            return std::any_cast<int64_t>(cache_[name]);
        }

        template<typename T>
        void add_cache(const std::string& name, T elem){
            cache_[name] = elem;
        }

    private:
        std::map<std::string, std::any> cache_;
        
    };

    namespace detail {
        template<typename T>
        class benchmark_executor_impl{};

        template<typename... Ts>
        struct benchmark_executor_impl<type_list<Ts...>> {
            template<typename Func>
            static void execute(rocbench::state& state, Func func) {
                using swallow = int[]; // Trick to expand the parameter pack
                (void)swallow{0, (func(state, type_list<Ts>{}), void(), 0)...}; // Execute the function for each type in TypeList
            }
        };

        static const std::map<std::string, bit_entropy> string_entropy_map = {
            {"0.000", bit_entropy::_0_000},
            {"0.201", bit_entropy::_0_201},
            {"0.337", bit_entropy::_0_337},
            {"0.544", bit_entropy::_0_544},
            {"0.811", bit_entropy::_0_811},
            {"1.000", bit_entropy::_1_000}
        };

        static const std::map<bit_entropy, double> entropy_probability_map = {
            {bit_entropy::_0_000, 0.0},
            {bit_entropy::_0_811, 0.811},
            {bit_entropy::_0_544, 0.544},
            {bit_entropy::_0_337, 0.337},
            {bit_entropy::_0_201, 0.201},
            {bit_entropy::_1_000, 1.0}
        };

        using seed_t = uint64_t;
        class device_generator_t {
        public:
            device_generator_t() {
                // Create the random number generator
                rocrand_create_generator(&m_gen, ROCRAND_RNG_PSEUDO_DEFAULT);
            }

            ~device_generator_t() {
                // Destroy the random number generator
                rocrand_destroy_generator(m_gen);
            }

            const double* new_uniform_distribution(seed_t seed, std::size_t num_items){
                // Resize the distribution vector to hold num_items elements
                m_distribution.resize(num_items);

                // Get raw pointer to the device vector's data
                double* d_distribution = thrust::raw_pointer_cast(m_distribution.data());

                // Set the seed for the random number generator
                rocrand_set_seed(m_gen, seed);

                // Generate uniform double values on the device
                rocrand_generate_uniform_double(m_gen, d_distribution, num_items);

                // Return the raw pointer to the device distribution array
                return d_distribution;
            }

        private:
            thrust::device_vector<double> m_distribution;
            rocrand_generator m_gen;
        };

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

        // template<typename T>
        // struct is_complex : std::false_type {};

        // template<typename T>
        // struct is_complex<std::complex<T>> : std::true_type {};

        template <typename T>
        struct is_complex :
            std::integral_constant<bool,
                std::is_same<std::complex<float>, T>::value ||
                std::is_same<std::complex<double>, T>::value
            > {};
        
        
        template<typename T>
        auto generate_input_impl(
            seed_t& seed,
            thrust::device_vector<T>& buff,
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
                default: {};
            }
            seed++;
        }

        template<typename T>
        auto generate_input_impl(
            seed_t& seed,
            thrust::device_vector<T>& buff,
            std::size_t elements,
            bit_entropy entropy,
            T min,
            T max
        ) -> std::enable_if_t<std::is_same<std::complex<float>, T>::value, void> {
            switch(entropy){
                case bit_entropy::_0_000 : {
                    std::mt19937 generator(seed);
                    std::uniform_real_distribution<double> distribution(0.0f, 1.0f);
                    const float random_imag = random_to_item_t<double>(min.imag(), max.imag())(distribution(generator));
                    const float random_real = random_to_item_t<double>(min.imag(), max.imag())(distribution(generator));
                    thrust::fill(buff.data(), buff.data() + elements, T{random_real, random_imag});
                }
                case bit_entropy::_1_000 : {
                    const double* unf_dist_real = device_generator_t{}.new_uniform_distribution(seed++, elements);
                    const double* unf_dist_imag = device_generator_t{}.new_uniform_distribution(seed++, elements);
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
            seed_t& seed,
            thrust::device_vector<T>& buff,
            std::size_t elements,
            bit_entropy entropy,
            T min,
            T max
        ) -> std::enable_if_t<!std::is_same<bool, T>::value && !std::is_same<std::complex<float>, T>::value, void> {

            switch(entropy){
                case bit_entropy::_0_000 : {
                    std::mt19937 generator(seed);
                    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
                    T randomValue = random_to_item_t<T>(min, max)(distribution(generator));
                    thrust::fill(buff.data(), buff.data() + elements, randomValue);
                }
                case bit_entropy::_1_000 : {
                    const double* unf_dist = device_generator_t{}.new_uniform_distribution(seed, elements);
                    thrust::transform(unf_dist, unf_dist + elements, buff.data(), random_to_item_t<T>(min, max));
                }
                default: {}
            };
            seed++;
            
        }

    } // namespace detail

    template<typename TypeList>
    class benchmark_executor{
    public:

        template<typename Func>
        void execute(Func func) {
            for(auto& state: executor_states_){
                detail::benchmark_executor_impl<TypeList>::execute(state, func);
            }
        }

        benchmark_executor& set_name(const std::string& name){
            std::cout << "benchmark_executor::set_name::name = "  << name << std::endl;
            return *this;
        }

        benchmark_executor& set_type_axes_names(const std::vector<std::string>& names){
            for(const auto& name: names){
                std::cout << "benchmark_executor::set_type_axes_names::name = "  << name << std::endl;
            }
            return *this;
        }

        benchmark_executor& add_int64_power_of_two_axis(const std::string& name, const std::vector<int64_t>& data_range){
            std::cout << "benchmark_executor::add_int64_power_of_two_axis::name = "  << name << std::endl;
            if (executor_states_.empty() || executor_states_.size() < data_range.size()){
                executor_states_.resize(executor_states_.size() + data_range.size());
            }
            
            for(size_t i = 0; i < data_range.size(); i++){
                executor_states_[i].add_cache<int64_t>(name, 1 << data_range[i]);
            }
            return *this;
        }

        benchmark_executor& add_string_axis(const std::string& name, const std::vector<std::string>& data_range){
            std::cout << "benchmark_executor::add_string_axis::name = "  << name << std::endl;
            if (executor_states_.empty() || executor_states_.size() < data_range.size()){
                executor_states_.resize(executor_states_.size() + data_range.size());
            }
            
            for(size_t i = 0; i < data_range.size(); i++){
                executor_states_[i].add_cache<std::string>(name, data_range[i]);
            }
            return *this;
        }

    private:
        std::vector<rocbench::state> executor_states_;
    };

    std::vector<int64_t> range(int64_t start, int64_t end, int64_t inc = 1) {
        std::vector<int64_t> result;
        for (int64_t i = start; i < end; i += inc) {
            result.push_back(i);
        }
        return result;
    }

    bit_entropy str_to_entropy(const std::string& str) {
        auto it = detail::string_entropy_map.find(str);
        if (it != detail::string_entropy_map.end()) {
            return it->second;
        }

        throw std::runtime_error("Can't convert string to bit entropy");
    }

    double entropy_to_probability(bit_entropy entropy) {
        auto it = detail::entropy_probability_map.find(entropy);
        if (it != detail::entropy_probability_map.end()) {
            return it->second;
        }

        // Default case (for unknown entropy values)
        return 0.0;
    }

    namespace detail {
        template<typename T>
        thrust::device_vector<T> generate_input(
            std::size_t elements,
            bit_entropy entropy = bit_entropy::_1_000,
            T min               = std::numeric_limits<T>::lowest(),
            T max               = std::numeric_limits<T>::max()
        ){

            static seed_t seed{0};
            thrust::device_vector<T> output(elements);

            switch(entropy){
                case bit_entropy::_0_000 : 
                case bit_entropy::_1_000 :{
                    detail::generate_input_impl(seed, output, elements, entropy, min, max);
                }
                default : {
                    if(std::is_same<bool, T>::value){
                        const double* unf_dist = device_generator_t{}.new_uniform_distribution(seed, output.size());
                        thrust::transform(
                            unf_dist, unf_dist + output.size(), output.data(),
                            random_to_probability_t{entropy_to_probability(entropy)}
                        );
                    }else{
                        detail::generate_input_impl(seed, output, elements, bit_entropy::_1_000, min, max);
                        const int epochs = static_cast<int>(entropy);
                        thrust::device_vector<T> epoch_output(elements);

                        for(int i = 0; i < epochs; i++){
                            detail::generate_input_impl(seed, epoch_output, elements, bit_entropy::_1_000, min, max);
                            thrust::transform(output.data(), output.data() + elements, epoch_output.data(), output.data(), and_t{});
                        }
                    }
                }
            }
            return output;
        }

    } // namespace rocbench::detail

    #define ROCBENCH_TYPE_AXES(...) __VA_ARGS__

    #define ROCBENCH_BENCH_TYPES(task, type_axis) \
        auto func = [](rocbench::state& state, auto typeInstance) { \
            using T = decltype(typeInstance); \
            task(state, T{}); \
        }; \
        rocbench::benchmark_executor<type_axis> executor; \
        executor

    #define ROCBENCH_EXECUTOR(macro) \
        int main() { \
            macro \
            executor.execute(func); \
            return 0; \
        } \


}

#define generate(elements) rocbench::detail::generate_input<T>(elements)

#endif //ROCBENCH_HELPER_CUH