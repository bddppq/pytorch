#pragma once

#include "fbgemm/Fbgemm.h"
#include "fbgemm/QuantUtils.h"
#include <c10/util/ArrayRef.h>

#include <vector>
#include <tuple>

namespace at { namespace native {

std::tuple<Tensor, fbgemm::TensorQuantizationParams> fbgemm_quantize_tensor(
    const Tensor& f);

Tensor fbgemm_dequantize_tensor(
    const Tensor& q,
    const fbgemm::TensorQuantizationParams& params);

std::vector<int64_t> pool_output_sizes(
    IntArrayRef input_size,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode
);

}} // at::native
