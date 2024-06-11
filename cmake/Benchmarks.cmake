# ########################################################################
# Copyright 2024 Advanced Micro Devices, Inc.
# ########################################################################

# ###########################
# rocThrust benchmarks
# ###########################

# Common functionality for configuring rocThrust's benchmarks

# Registers a .cu as C++ rocThrust benchmark
function(add_thrust_benchmark BENCHMARK_NAME BENCHMARK_SOURCE USES_GOOGLE_BENCH ROCBENCH_DIR)
  if("${ROCBENCH_DIR}" STREQUAL "")
    set(USING_ROCBENCH_DIR FALSE)
  else()
    set(USING_ROCBENCH_DIR TRUE)
  endif()

  set(BENCHMARK_TARGET "benchmark_thrust_${BENCHMARK_NAME}")
  set_source_files_properties(${BENCHMARK_SOURCE}
      PROPERTIES
          LANGUAGE CXX
  )
  add_executable(${BENCHMARK_TARGET} ${BENCHMARK_SOURCE})
  if(USING_ROCBENCH_DIR)
    target_include_directories(${BENCHMARK_TARGET} PRIVATE ${ROCBENCH_DIR})
  endif()

  target_link_libraries(${BENCHMARK_TARGET}
      PRIVATE
          rocthrust
          roc::rocprim_hip
  )

  if(USING_ROCBENCH_DIR)
    ## Add rocRAND for rocbench_helper
    target_link_libraries(${BENCHMARK_TARGET}
        PRIVATE
            rocthrust
            roc::rocrand
    )
  endif()
  # Internal benchmark does not use Google Benchmark.
  # This can be omited when that benchmark is removed.
  if(USES_GOOGLE_BENCH)
      target_link_libraries(${BENCHMARK_TARGET}
          PRIVATE
              rocthrust
              benchmark::benchmark
  )
  endif()
  foreach(gpu_target ${GPU_TARGETS})
      target_link_libraries(${BENCHMARK_TARGET}
          INTERFACE
              --cuda-gpu-arch=${gpu_target}
      )
  endforeach()
  set_target_properties(${BENCHMARK_TARGET}
      PROPERTIES
          RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/benchmarks/"
  )
  rocm_install(TARGETS ${BENCHMARK_TARGET} COMPONENT benchmarks)
endfunction()
