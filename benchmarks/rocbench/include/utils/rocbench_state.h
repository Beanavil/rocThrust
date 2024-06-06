#ifndef ROCBENCH_STATE_H
#define ROCBENCH_STATE_H

#include <functional>
#include <map>
#include <string>
#include <iostream>

#include "rocbench_types.h"

#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || __cplusplus >= 201703L) //C++ 17+ available
    #include <any>
#else
    #include "rocbench_cpp14_compat.h"
#endif

namespace rocbench{

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


}

#endif //ROCBENCH_STATE_H