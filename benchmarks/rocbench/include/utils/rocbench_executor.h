#ifndef ROCBENCH_EXECUTOR_H
#define ROCBENCH_EXECUTOR_H

#include <iostream>
#include <vector>

#include "rocbench_types.h"
#include "rocbench_state.h"

namespace rocbench {

    namespace detail{

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

    }

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


}

#endif //ROCBENCH_EXECUTOR_H