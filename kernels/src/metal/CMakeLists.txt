set(metal_compare_env "LARCH_METAL_TENSOR=${LARCH_METAL_TENSOR_TEST_VALUE}")
if(DEFINED LARCH_METAL_SIMD_KERNELS_TEST_VALUE)
  list(APPEND metal_compare_env "LARCH_METAL_SIMD_KERNELS=${LARCH_METAL_SIMD_KERNELS_TEST_VALUE}")
endif()
add_test(NAME metal_compare COMMAND "${CMAKE_COMMAND}" -E env ${metal_compare_env}
  "MTL_DEBUG_LAYER=1" "MTL_SHADER_VALIDATION=1" "$<TARGET_FILE:metal_compare>")
set_tests_properties(metal_compare PROPERTIES SKIP_RETURN_CODE 77)  # forced tensor GEMM on a GPU without it
