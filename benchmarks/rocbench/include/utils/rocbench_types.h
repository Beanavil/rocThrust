#ifndef ROCBENCH_TYPES_H
#define ROCBENCH_TYPES_H

#include <cinttypes>

namespace rocbench {
    using seed_t = uint64_t;

    enum class exec_tag : uint8_t {
        sync
    };

    class launch {};

    template <typename... Ts>
    struct type_list {};
}

#endif // ROCBENCH_TYPES_H