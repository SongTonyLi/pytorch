#include <torch/csrc/stable/c/shim.h>

#include <sstream>
#include <stdexcept>

#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0

// Users of this macro are expected to include cuda_runtime.h
#define STD_CUDA_CHECK(EXPR)                      \
  do {                                            \
    const cudaError_t __err = EXPR;               \
    char* __error_msg = nullptr;                  \
    torch_c10_cuda_check_msg(                     \
        static_cast<int32_t>(__err),              \
        __FILE__,                                 \
        __func__,                                 \
        static_cast<uint32_t>(__LINE__),          \
        true,                                     \
        &__error_msg);                            \
    if (__error_msg != nullptr) {                 \
      std::string __msg(__error_msg);             \
      torch_c10_cuda_free_error_msg(__error_msg); \
      throw std::runtime_error(__msg);            \
    }                                             \
  } while (0)

// Users of this macro are expected to include cuda_runtime.h
#define STD_CUDA_KERNEL_LAUNCH_CHECK() STD_CUDA_CHECK(cudaGetLastError())

#endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0

#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_12_0

// This macro is similar to the header only macro TORCH_ERROR_CODE_CHECK, this
// one does provide more information in the exception, including the error
// message as retrieved through the c shims from the original error message.
#define STABLE_TORCH_ERROR_CODE_CHECK(call)                                    \
  if ((call) != TORCH_SUCCESS) {                                               \
    std::stringstream ss;                                                      \
    ss << call << " API call failed at " << __FILE__ << ", line " << __LINE__; \
    ss << ", with: " << torch_exception_get_what_without_backtrace();          \
    throw std::runtime_error(ss.str());                                        \
  }

#endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_12_0
