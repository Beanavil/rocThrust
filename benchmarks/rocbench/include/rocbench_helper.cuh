#ifndef ROCBENCH_HELPER_CUH
#define ROCBENCH_HELPER_CUH

#include "utils/rocbench_types.h"
#include "utils/rocbench_state.h"
#include "utils/rocbench_executor.h"
#include "utils/rocbench_generator.h"

namespace rocbench {

    std::vector<int64_t> range(int64_t start, int64_t end, int64_t inc = 1) {
        std::vector<int64_t> result;
        for (int64_t i = start; i < end; i += inc) {
            result.push_back(i);
        }
        return result;
    }
}

rocbench::detail::thrust_device_vector_generator generate(
        std::size_t elements, rocbench::bit_entropy entropy = rocbench::bit_entropy::_1_000,
        long double min_bound = std::numeric_limits<long double>::infinity(), 
        long double max_bound = std::numeric_limits<long double>::infinity()
){
    return rocbench::detail::thrust_device_vector_generator(elements, entropy, min_bound, max_bound);
}

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



#endif //ROCBENCH_HELPER_CUH