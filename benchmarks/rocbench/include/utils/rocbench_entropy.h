#ifndef ROCBENCH_ENTROPY_H
#define ROCBENCH_ENTROPY_H

#include <map>
#include <string>

namespace rocbench{
    
    namespace detail {
        constexpr auto bit_entropy_default = (1 << 12) + (200 - 96);
    }

    enum class bit_entropy {
        _0_000 = detail::bit_entropy_default,
        _0_201 = 4,
        _0_337 = 3,
        _0_544 = 2,
        _0_811 = 1,
        _1_000 = 0
    };

    namespace detail {

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

}


#endif // ROCBENCH_ENTROPY_H