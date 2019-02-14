#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

#include "fbgemm/Fbgemm.h"
#include <ATen/native/quantized/Utils.h>

namespace at { namespace native {

at::Tensor quantized_conv_relu2d(
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups) {
  AT_ASSERTM(
    false, "quantized_conv_relu2d is not implemented yet");
}

}} // at::native
