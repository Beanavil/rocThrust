#include <thrust/adjacent_difference.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include "rocbench_helper.cuh"


template<typename T>
static void basic(rocbench::state& state, rocbench::type_list<T>) {
    // Function implementation
    std::cout << std::endl;
    
    const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));
    std::cout << "basic<" << typeid(T).name() << ">(Elements:" << elements << ")" << std::endl;

    thrust::device_vector<T> input = generate(elements);
    thrust::device_vector<T> output(elements);
    
    state.add_element_count(elements);
    state.add_global_memory_reads<T>(elements);
    state.add_global_memory_writes<T>(elements);

    std::cout << "input.size() = " << input.size() << std::endl;

    if (std::is_same<int8_t, T>::value){
        std::cout << "input = {" << static_cast<int>(input[0]) << ", " << static_cast<int>(input[1]) << ", " << static_cast<int>(input[2]) << " ... ";
        std::cout << static_cast<int>(input[input.size() - 1]) << ", " << static_cast<int>(input[input.size() - 2]) << ", " << static_cast<int>(input[input.size() - 3]);
        std::cout << "} " << std::endl;
    }else{
        std::cout << "input = {" << input[0] << ", " << input[1] << ", " << input[2] << " ... ";
        std::cout << input[input.size() - 1] << ", " << input[input.size() - 2] << ", " << input[input.size() - 3];
        std::cout << "} " << std::endl;
    }

    state.exec(rocbench::exec_tag::sync, [&](rocbench::launch & /* launch */) {
        std::cout << "benchmark_executor::exec -> { ... }" << std::endl;
        thrust::adjacent_difference(input.cbegin(), input.cend(), output.begin());
    });

}

using types = rocbench::type_list<int8_t, int16_t, int32_t, int64_t, float, double>;

ROCBENCH_EXECUTOR(
    ROCBENCH_BENCH_TYPES(basic, ROCBENCH_TYPE_AXES(types))
        .set_name("base")
        .set_type_axes_names({"T{ct}"})
        .add_int64_power_of_two_axis("Elements", rocbench::range(16, 28, 4));
)