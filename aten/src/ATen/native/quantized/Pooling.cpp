#include <ATen/ATen.h>

#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <c10/util/Exception.h>

#include "fbgemm/Fbgemm.h"
#include <ATen/native/quantized/Utils.h>

#include <tuple>
#include <immintrin.h>
#include <cmath>
#include <cstdint>
#include <algorithm>

#include "caffe2/quantization/server/pool_dnnlowp_op_avx2.h"

namespace at { namespace native {

Tensor quantized_avg_pool1d(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad) {
  AT_ASSERTM(
    false, "quantized_avg_pool1d is not implemented yet");
}

Tensor quantized_avg_pool2d(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad) {

    AT_ASSERTM(
      self.dim() == 4,
      "Input tensor to quantized_conv_relu2d should have 4 dims");
  Tensor self_nhwc = self.permute({0, 2, 3, 1});

  const auto N = self_nhwc.size(0);
  const auto height = self_nhwc.size(1);
  const auto width = self_nhwc.size(2);
  const auto channels = self_nhwc.size(3);

  Tensor q_self{};
  fbgemm::TensorQuantizationParams q_in_params{};
  std::tie(q_self, q_in_params) = fbgemm_quantize_tensor(self_nhwc);

  uint8_t* Xdata = q_self.data<uint8_t>();
  const auto output_sizes = pool_output_sizes(
      self_nhwc.sizes(),
      kernel_size,
      stride,
      padding,
      {1, 1},
      ceil_mode
  );
  Tensor q_output = at::zeros(output_sizes, self_nhwc.options().dtype(kByte));
  uint8_t* Ydata = q_output.data<uint8_t>();

  const auto pooled_height = q_output.size(1);
  const auto pooled_width = q_output.size(2);
  const auto kernel_h = kernel_size[0];
  const auto kernel_w = kernel_size[1];
  const auto stride_h = stride[0];
  const auto stride_w = stride[1];
  const auto pad_t = padding[0];
  const auto pad_l = padding[1];

  int32_t precision = q_in_params.precision;
  int32_t minimum = 0;
  int32_t maximum = (1 << precision) - 1;

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < N; ++i) {
    const uint8_t* Xdata_temp = Xdata + i * height * width * channels;
    uint8_t* Ydata_temp = Ydata + i * pooled_height * pooled_width * channels;
    for (int ph = 0; ph < pooled_height; ++ph) {
      int hstart = ph * stride_h - pad_t;
      int hend = std::min(hstart + kernel_h, height);
      hstart = std::max(hstart, 0);
      for (int pw = 0; pw < pooled_width; ++pw) {
        int wstart = pw * stride_w - pad_l;
        int wend = std::min(wstart + kernel_w, width);
        wstart = std::max(wstart, 0);
        int size = (hend - hstart) * (wend - wstart);
        float multiplier = 1. / size;

        for (int c = 0; c < channels; ++c) {
          const int pool_idx = (ph * pooled_width + pw) * channels + c;
          int32_t Yh = -q_in_params.zero_point * size;
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              const int input_idx = (h * width + w) * channels + c;
              Yh += Xdata_temp[input_idx];
            }
          }
          Ydata_temp[pool_idx] = std::min<int32_t>(
              std::max<int32_t>(
                  std::nearbyint(Yh * multiplier + q_in_params.zero_point),
                  minimum),
              maximum);
        } // channel
      } // width
    } // height
  } // for each image
  Tensor f_output = fbgemm_dequantize_tensor(q_output, q_in_params);
  return f_output.permute({0, 3, 1, 2});
}

Tensor quantized_avg_pool3d(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad) {
  AT_ASSERTM(
    false, "quantized_avg_pool3d is not implemented yet");
}

Tensor quantized_max_pool1d(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  AT_ASSERTM(
    false, "quantized_max_pool1d has not been implemented yet");
}

Tensor quantized_max_pool2d(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  AT_ASSERTM(
      self.dim() == 4,
      "Input tensor to quantized_conv_relu2d should have 4 dims");
  Tensor self_nhwc = self.permute({0, 2, 3, 1});

  const auto N = self_nhwc.size(0);
  const auto H = self_nhwc.size(1);
  const auto W = self_nhwc.size(2);
  const auto C = self_nhwc.size(3);

  Tensor q_self{};
  fbgemm::TensorQuantizationParams q_in_params{};
  std::tie(q_self, q_in_params) = fbgemm_quantize_tensor(self_nhwc);

  const auto output_sizes = pool_output_sizes(
      self_nhwc.sizes(),
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode
  );
  Tensor q_output = at::zeros(output_sizes, self_nhwc.options().dtype(kByte));
  for (int i = 0; i < N; ++i) {
    caffe2::max_pool_avx2(
        q_self.data<uint8_t>(),
        i,
        H,
        W,
        C,
        q_output.size(1),
        q_output.size(2),
        kernel_size[0],
        kernel_size[1],
        stride[1],
        stride[0],
        padding[1],
        padding[0],
        q_output.data<uint8_t>()
    );
  }

  Tensor f_output = fbgemm_dequantize_tensor(q_output, q_in_params);
  return f_output.permute({0, 3, 1, 2});
}

Tensor quantized_max_pool3d(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  AT_ASSERTM(
    false, "quantized_max_pool3d has not been implemented yet");
}
} // namespace native
} // namespace at
