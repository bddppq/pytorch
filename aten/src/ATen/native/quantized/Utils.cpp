#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <THNN/generic/pooling_shape.h>

#ifdef USE_FBGEMM
#include "fbgemm/Fbgemm.h"
#include "fbgemm/QuantUtils.h"
#endif // USE_FBGEMM

namespace at { namespace native {

std::vector<int64_t> pool_output_sizes(
    IntArrayRef input_size,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  std::vector<int64_t> output_size(input_size.size());
  output_size.front() = input_size.front();
  output_size.back() = input_size.back();

  for (int i = 0; i < input_size.size() - 2; ++i) {
    output_size[i + 1] = pooling_output_shape<int64_t>(
        input_size[i + 1],
        kernel_size[i],
        padding[i],
        stride[i],
        dilation[i],
        ceil_mode
    );
  }

  return output_size;
}

std::vector<int64_t> conv_output_sizes(
    IntArrayRef input_size,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation) {
  std::vector<int64_t> output_size(input_size.size());
  output_size.front() = input_size.front();
  output_size.back() = kernel_size.front();

  for (int i = 0; i < input_size.size() - 2; ++i) {
    output_size[i + 1] = pooling_output_shape<int64_t>(
        input_size[i + 1],
        kernel_size[i + 1],
        padding[i],
        stride[i],
        dilation[i],
        false
    );
  }

  return output_size;
}

#ifdef USE_FBGEMM

bool fbgemm_is_cpu_supported() {
  return fbgemm::fbgemmSupportedCPU();
}

std::tuple<Tensor, fbgemm::TensorQuantizationParams> fbgemm_quantize_tensor(
    const Tensor& f) {
  // Input tensor is quantized as 8-bit unsigned values
  static constexpr int precision = 8;
  static constexpr bool is_signed = false;

  Tensor f_contig = f.contiguous();

  float input_min, input_max;
  fbgemm::FindMinMax(
      f_contig.data<float>(),
      &input_min,
      &input_max,
      f_contig.numel());

  auto qparams = fbgemm::ChooseQuantizationParams(
      input_min,
      input_max,
      /*qmin=*/is_signed ? -(1 << (precision - 1)) : 0,
      /*qmax=*/is_signed ? ((1 << (precision - 1)) - 1) : (1 << precision) - 1,
      /*preserve_sparsity=*/false);
  qparams.precision = precision;

  Tensor q = at::zeros_like(f, f.options().dtype(at::kByte));
  fbgemm::Quantize<uint8_t>(
      f_contig.data<float>(), q.data<uint8_t>(), q.numel(), qparams);

  return {q, qparams};
}

Tensor fbgemm_dequantize_tensor(
    const Tensor& q,
    const fbgemm::TensorQuantizationParams& params) {
  Tensor q_contig = q.contiguous();

  Tensor f = at::zeros_like(q, q.options().dtype(at::kFloat));
  fbgemm::Dequantize(
      q_contig.data<uint8_t>(),
      f.data<float>(),
      f.numel(),
      params
  );
  return f;
}

Tensor fbgemm_pack_quantized_matrix(const Tensor& weight) {
  // We make a strong guarantee that models using these operators will have the
  // same numerics across different machines. Therefore, we do not provide a
  // fallback path and rather fail loudly if we cannot run FBGEMM.
  AT_ASSERTM(fbgemm::fbgemmSupportedCPU(), "Your CPU does not support FBGEMM.");

  const auto K = weight.size(1);
  const auto N = weight.size(0);

  auto weight_contig = weight.contiguous();
  auto contiguous_ptr = weight_contig.data<int8_t>();
  auto* ptr = new fbgemm::PackBMatrix<int8_t>(
      /*trans=*/fbgemm::matrix_op_t::Transpose,
      /*nRow=*/K,
      /*nCol=*/N,
      /*smat=*/contiguous_ptr,
      /*ld=*/K,
      /*pmat=*/nullptr, // PackBMatrix manages ownership of pmat
      /*groups=*/1);

  // We store this instance away in a Tensor and register a deleter function
  // so that we do not leak memory. On the other side, we pull out the storage's
  // data_ptr and get the PackBMatrix's pointer.
  at::DataPtr at_ptr(
      ptr,
      ptr,
      [](void* ptr) {
        fbgemm::PackBMatrix<int8_t>* typed_ptr =
            reinterpret_cast<fbgemm::PackBMatrix<int8_t>*>(ptr);
        delete typed_ptr;
      },
      at::kCPU);

  auto retval = at::empty(
      {sizeof(fbgemm::PackBMatrix<int8_t>)}, weight.options().dtype(at::kByte));

  retval.storage().set_data_ptr(std::move(at_ptr));

  return retval;
}

#else  // USE_FBGEMM

bool fbgemm_is_cpu_supported() {
  return false;
}

Tensor fbgemm_pack_quantized_matrix(const Tensor& weight) {
  // We make a strong guarantee that models using these operators will have the
  // same numerics across different machines. Therefore, we do not provide a
  // fallback path and rather fail loudly if we cannot run FBGEMM.
  AT_ASSERTM(
      false, "This PyTorch installation was not built with FBGEMM operators");
}

#endif  // USE_FBGEMM

}} // at::native
