
#include "layer/layer_convolution.hpp"

#include <math.h>
#include <sys/types.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "layer/layer.hpp"
#include "layer/layer_factory.hpp"
#include "runtime/runtime_ir.hpp"
#include "runtime/status_code.hpp"
#include "tensor/tensor.hpp"

namespace free_infer {
ConvolutionLayer::ConvolutionLayer(uint32_t output_channel, uint32_t in_channel,
                                   uint32_t kernel_h, uint32_t kernerl_w,
                                   uint32_t padding_h, uint32_t padding_w,
                                   uint32_t stride_h, uint32_t stride_w,
                                   uint32_t groups, bool use_bias)
    : Layer("Convolution"),
      use_bias_(use_bias),
      groups_(groups),
      padding_h_(padding_h),
      padding_w_(padding_w),
      stride_h_(stride_h),
      stride_w_(stride_w) {
  in_channel /= groups;
  this->InitWeightParam(output_channel, in_channel, kernel_h, kernerl_w);
  if (use_bias_) {
    this->InitBiasParam(output_channel, 1, 1, 1);
  }
}

void ConvolutionLayer::InitWeightParam(const uint32_t kernel_n,
                                       const uint32_t kernel_c,
                                       const uint32_t kernel_h,
                                       const uint32_t kernel_w) {
  this->weights_ = std::vector<sftensor>(kernel_n);
  for (uint32_t i = 0; i < kernel_n; ++i) {
    this->weights_[i] =
        std::make_shared<Tensor<float>>(kernel_c, kernel_h, kernel_w);
  }
}

void ConvolutionLayer::InitBiasParam(const uint32_t bias_n,
                                     const uint32_t bias_c,
                                     const uint32_t bias_h,
                                     const uint32_t bias_w) {
  this->bias_ = std::vector<sftensor>(bias_n);
  for (uint32_t i = 0; i < bias_n; ++i) {
    this->bias_[i] = std::make_shared<Tensor<float>>(bias_c, bias_h, bias_w);
  }
}

const std::vector<sftensor>& ConvolutionLayer::weights() const {
  return this->weights_;
}

const std::vector<sftensor>& ConvolutionLayer::bias() const {
  return this->bias_;
}

void ConvolutionLayer::set_weights(const std::vector<sftensor>& weights) {
  CHECK(weights.size() == weights_.size());
  for (uint32_t i = 0; i < weights.size(); ++i) {
    CHECK(this->weights_.at(i) != nullptr);
    CHECK(this->weights_.at(i)->rows() == weights.at(i)->rows());
    CHECK(this->weights_.at(i)->cols() == weights.at(i)->cols());
    CHECK(this->weights_.at(i)->channels() == weights.at(i)->channels());
  }
  this->weights_ = weights;
}

void ConvolutionLayer::set_weights(const std::vector<float>& weights) {
  const uint32_t elem_size = weights.size();

  uint32_t weight_size = 0;
  const uint32_t batch_size = this->weights_.size();
  for (uint32_t i = 0; i < batch_size; ++i) {
    weight_size += this->weights_.at(i)->size();
  }

  CHECK_EQ(weight_size, elem_size);
  CHECK_EQ(elem_size % batch_size, 0);
  const uint32_t blob_size = elem_size / batch_size;
  for (uint32_t idx = 0; idx < batch_size; ++idx) {
    const uint32_t start_offset = idx * blob_size;
    const uint32_t end_offset = start_offset + blob_size;
    const auto& sub_values = std::vector<float>{weights.begin() + start_offset,
                                                weights.begin() + end_offset};
    this->weights_.at(idx)->Fill(sub_values);
  }
}

void ConvolutionLayer::set_bias(const std::vector<sftensor>& bias) {
  if (!this->bias_.empty()) {
    CHECK(bias.size() == bias_.size());
    for (uint32_t i = 0; i < bias.size(); ++i) {
      CHECK(this->bias_.at(i) != nullptr);
      CHECK(this->bias_.at(i)->rows() == bias.at(i)->rows());
      CHECK(this->bias_.at(i)->cols() == bias.at(i)->cols());
      CHECK(this->bias_.at(i)->channels() == bias.at(i)->channels());
    }
    this->bias_ = bias;
  }
}

void ConvolutionLayer::set_bias(const std::vector<float>& bias) {
  const uint32_t elem_size = bias.size();

  uint32_t bias_size = 0;
  const uint32_t batch_size = this->bias_.size();
  for (uint32_t i = 0; i < batch_size; ++i) {
    bias_size += this->bias_.at(i)->size();
  }

  CHECK_EQ(bias_size, elem_size);
  CHECK_EQ(elem_size % batch_size, 0);

  const uint32_t blob_size = elem_size / batch_size;
  for (uint32_t idx = 0; idx < batch_size; ++idx) {
    const uint32_t start_offset = idx * blob_size;
    const uint32_t end_offset = start_offset + blob_size;
    const auto& sub_values = std::vector<float>{bias.begin() + start_offset,
                                                bias.begin() + end_offset};
    this->bias_.at(idx)->Fill(sub_values);
  }
}

InferStatus ConvolutionLayer::Forward(const std::vector<sftensor>& inputs,
                                      std::vector<sftensor>& outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array in the convolution layer is empty";
    return InferStatus::kInferFailedInputEmpty;
  }

  if (inputs.size() != outputs.size()) {
    LOG(ERROR) << "The input and output tensor array size of the convolution "
                  "layer do not match";
    return InferStatus::kInferFailedInputOutSizeMatchError;
  }

  if (weights_.empty()) {
    LOG(ERROR) << "The number of kernel matrix in the convolution layer should "
                  "be greater than zero";
    return InferStatus::kInferFailedWeightParameterError;
  }

  if (this->use_bias_ && this->bias_.size() != this->weights_.size()) {
    LOG(ERROR) << "The number of kernel matrix and bias matrix do not match";
    return InferStatus::kInferFailedBiasParameterError;
  }

  if (!stride_h_ || !stride_w_) {
    LOG(ERROR) << "The stride parameter is set incorrectly. It must always be "
                  "greater than 0";
    return InferStatus::kInferFailedStrideParameterError;
  }

  const uint32_t kernel_n = this->weights_.size();
  const uint32_t kernel_c = this->weights_[0]->channels();
  const uint32_t kernel_w = this->weights_[0]->cols();
  const uint32_t kernel_h = this->weights_[0]->rows();

  const uint32_t im2col_w = kernel_h * kernel_w;
  CHECK(kernel_c > 0 && kernel_h > 0 && kernel_w > 0);

  for (uint32_t k = 0; k < kernel_n; ++k) {
    const std::shared_ptr<Tensor<float>>& kernel = this->weights_.at(k);
    CHECK(kernel->rows() == kernel_h);
    CHECK(kernel->cols() == kernel_w);
    CHECK(kernel->channels() == kernel_c);
  }

  const uint32_t kernel_n_group = kernel_n / groups_;
  const uint32_t batch_size = inputs.size();

  if (im2col_kernel.empty()) {
    this->InitIm2ColKernel();
  }

  if (!im2col_kernel.empty()) {
    if (groups_ == 1) {
      CHECK(im2col_kernel.size() == kernel_n_group)
          << "The number of kernel matrix and kernel_count_group do not match";
    } else {
      CHECK(im2col_kernel.size() == kernel_n)
          << "The number of kernel matrix and kernel_count do not match";
    }
  }

  for (uint32_t i = 0; i < batch_size; ++i) {
    const sftensor& input = inputs[i];

    CHECK(input != nullptr && !input->empty())
        << "The input tensor array in the convolution layer has an empty  "
           "tensor "
        << i << " batch";

    const uint32_t input_c = input->channels();
    const uint32_t input_padded_h = input->rows() + 2 * padding_h_;
    const uint32_t input_padded_w = input->cols() + 2 * padding_w_;

    const uint32_t output_h =
        std::floor((input_padded_h - kernel_h) / stride_h_ + 1);
    const uint32_t output_w =
        std::floor((input_padded_w - kernel_w) / stride_w_ + 1);

    CHECK(output_h > 0 && output_w > 0)
        << "The size of the output tensor should be greater than zero " << i
        << " batch";

    if (groups_ != 1) {
      CHECK(kernel_n % groups_ == 0);
      CHECK(input_c % groups_ == 0);
    }

    uint32_t im2col_h = output_h * output_w;
    CHECK(im2col_h > 0) << "Output_h x output_w for the convolution layer "
                           "should be greater than zero "
                        << i << " th";

    uint32_t input_c_group = input_c / groups_;
    CHECK(input_c_group == kernel_c) << "The number of channel for the kernel "
                                        "matrix and input tensor do not match";

    for (uint32_t g = 0; g < groups_; ++g) {
      const auto& im2col_input =
          Im2Col(input, kernel_h, kernel_w, input->rows(), input->cols(),
                 input_c_group, g, im2col_w, im2col_h);

      sftensor output = outputs[i];
      if (output == nullptr || output->empty()) {
        output = std::make_shared<Tensor<float>>(kernel_n, output_h, output_w);
        outputs[i] = output;
      }

      CHECK(output->rows() == output_h && output->cols() == output_w &&
            output->channels() == kernel_n)
          << "The output tensor array in the convolution layer has an "
             "incorrectly sized tensor "
          << i << "batch";

      for (uint32_t k = 0; k < kernel_n_group; ++k) {
        arma::frowvec kernel = im2col_kernel[k + kernel_n_group * g];
        ConvGemm(im2col_input, output, g, k, kernel_n_group, kernel, output_w,
                 output_h);
      }
    }
  }
  return InferStatus::kInferSuccess;
}

ParseParameterAttrStatus ConvolutionLayer::GetInstace(
    const std::shared_ptr<RuntimeOperator>& op,
    std::shared_ptr<Layer>& conv_layer) {
  CHECK(op != nullptr);
  const std::map<std::string, std::shared_ptr<RuntimeParameter>>& params =
      op->params;

  if (params.find("dilation") == params.end()) {
    LOG(ERROR) << "Can not find the dilation parameter";
    return ParseParameterAttrStatus::kParameterMissingDilation;
  }

  auto dilation_param = std::dynamic_pointer_cast<RuntimeParameterIntArray>(
      params.at("dilation"));

  if (dilation_param == nullptr || dilation_param->value.size() != 2) {
    LOG(ERROR) << "Can not find the dilation parameter";
    return ParseParameterAttrStatus::kParameterMissingDilation;
  }

  CHECK(dilation_param->value.at(0) != 1 || dilation_param->value.at(1))
      << "Only support dilation value equals to one!";

  if (params.find("in_channels") == params.end()) {
    LOG(ERROR) << "Can not find the in channel parameter";
    return ParseParameterAttrStatus::kParameterMissingInChannel;
  }
  auto in_channel =
      std::dynamic_pointer_cast<RuntimeParameterInt>(params.at("in_channels"));
  if (!in_channel) {
    LOG(ERROR) << "Can not find the in channel parameter";
    return ParseParameterAttrStatus::kParameterMissingInChannel;
  }

  if (params.find("out_channels") == params.end()) {
    LOG(ERROR) << "Can not find the out channel parameter";
    return ParseParameterAttrStatus::kParameterMissingOutChannel;
  }

  auto out_channel =
      std::dynamic_pointer_cast<RuntimeParameterInt>(params.at("out_channels"));
  if (!out_channel) {
    LOG(ERROR) << "Can not find the out channel parameter";
    return ParseParameterAttrStatus::kParameterMissingOutChannel;
  }

  if (params.find("padding") == params.end()) {
    LOG(ERROR) << "Can not find the padding parameter";
    return ParseParameterAttrStatus::kParameterMissingPadding;
  }

  auto padding =
      std::dynamic_pointer_cast<RuntimeParameterIntArray>(params.at("padding"));
  if (!padding) {
    LOG(ERROR) << "Can not find the padding parameter";
    return ParseParameterAttrStatus::kParameterMissingPadding;
  }

  if (params.find("bias") == params.end()) {
    LOG(ERROR) << "Can not find the bias parameter";
    return ParseParameterAttrStatus::kParameterMissingUseBias;
  }
  auto use_bias =
      std::dynamic_pointer_cast<RuntimeParameterBool>(params.at("bias"));
  if (!use_bias) {
    LOG(ERROR) << "Can not find the bias parameter";
    return ParseParameterAttrStatus::kParameterMissingUseBias;
  }

  if (params.find("stride") == params.end()) {
    LOG(ERROR) << "Can not find the stride parameter";
    return ParseParameterAttrStatus::kParameterMissingStride;
  }
  auto stride =
      std::dynamic_pointer_cast<RuntimeParameterIntArray>(params.at("stride"));
  if (!stride) {
    LOG(ERROR) << "Can not find the stride parameter";
    return ParseParameterAttrStatus::kParameterMissingStride;
  }

  if (params.find("kernel_size") == params.end()) {
    LOG(ERROR) << "Can not find the kernel parameter";
    return ParseParameterAttrStatus::kParameterMissingKernel;
  }
  auto kernel = std::dynamic_pointer_cast<RuntimeParameterIntArray>(
      params.at("kernel_size"));
  if (!kernel) {
    LOG(ERROR) << "Can not find the kernel parameter";
    return ParseParameterAttrStatus::kParameterMissingKernel;
  }

  if (params.find("padding_mode") != params.end()) {
    auto padding_mode = std::dynamic_pointer_cast<RuntimeParameterString>(
        params.at("padding_mode"));
    if (padding_mode == nullptr) {
      LOG(ERROR) << "Can not find the padding parameter";
      return ParseParameterAttrStatus::kParameterMissingPaddingMode;
    } else {
      const std::string& padding_mode_str = padding_mode->value;
      if (padding_mode_str != "zeros") {
        LOG(ERROR) << "Padding mode unsupported: " << padding_mode_str;
        return ParseParameterAttrStatus::kParameterMissingPaddingMode;
      }
    }
  } else {
    LOG(ERROR) << "Can not find the padding parameter";
    return ParseParameterAttrStatus::kParameterMissingPaddingMode;
  }

  auto groups =
      std::dynamic_pointer_cast<RuntimeParameterInt>(params.at("groups"));
  if (!groups) {
    LOG(ERROR) << "Can not find the groups parameter";
    return ParseParameterAttrStatus::kParameterMissingGroups;
  }

  const uint32_t dims = 2;
  const std::vector<int>& kernels = kernel->value;
  const std::vector<int>& paddings = padding->value;
  const std::vector<int>& strides = stride->value;
  if (paddings.size() != dims) {
    LOG(ERROR) << "Can not find the right padding parameter";
    return ParseParameterAttrStatus::kParameterMissingPadding;
  }

  if (strides.size() != dims) {
    LOG(ERROR) << "Can not find the right stride parameter";
    return ParseParameterAttrStatus::kParameterMissingStride;
  }

  if (kernels.size() != dims) {
    LOG(ERROR) << "Can not find the right kernel size parameter";
    return ParseParameterAttrStatus::kParameterMissingKernel;
  }

  conv_layer = std::make_shared<ConvolutionLayer>(
      out_channel->value, in_channel->value, kernels.at(0), kernels.at(1),
      paddings.at(0), paddings.at(1), strides.at(0), strides.at(1),
      groups->value, use_bias->value);

  auto conv_layer_derived =
      std::dynamic_pointer_cast<ConvolutionLayer>(conv_layer);

  // load weights
  const std::map<std::string, std::shared_ptr<RuntimeAttribute>>& attrs =
      op->attrs;
  if (use_bias->value) {
    if (attrs.find("bias") == attrs.end()) {
      LOG(ERROR) << "Can not find the bias attribute";
      return ParseParameterAttrStatus::kAttrMissingBias;
    }
    const auto& bias = attrs.at("bias");
    const std::vector<int>& bias_shape = bias->shape;
    if (bias_shape.empty() || bias_shape.at(0) != out_channel->value) {
      LOG(ERROR) << "The attribute of bias shape is wrong";
      return ParseParameterAttrStatus::kAttrMissingBias;
    }

    const std::vector<float>& bias_values = bias->get<float>();
    conv_layer_derived->set_bias(bias_values);
  }

  if (attrs.find("weight") == attrs.end()) {
    LOG(ERROR) << "Can not find the weight attribute";
    return ParseParameterAttrStatus::kAttrMissingWeight;
  }

  const auto& weight = attrs.at("weight");
  const std::vector<int>& weight_shape = weight->shape;
  if (weight_shape.empty()) {
    LOG(ERROR) << "The attribute of weight shape is wrong";
    return ParseParameterAttrStatus::kAttrMissingWeight;
  }

  const std::vector<float>& weight_values = weight->get<float>();
  conv_layer_derived->set_weights(weight_values);

  // auto conv_layer_derived =
  //     std::dynamic_pointer_cast<ConvolutionLayer>(conv_layer);
  CHECK(conv_layer_derived != nullptr);
  conv_layer_derived->InitIm2ColKernel();
  return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

void ConvolutionLayer::InitIm2ColKernel() {
  const uint32_t kernel_n = this->weights_.size();
  const uint32_t kernel_c = this->weights_[0]->channels();
  const uint32_t kernel_w = this->weights_[0]->cols();
  const uint32_t kernel_h = this->weights_[0]->rows();
  const uint32_t im2col_w = kernel_h * kernel_w;
  CHECK(kernel_n > 0 && kernel_c > 0 && kernel_h > 0 && kernel_w > 0);

  for (uint32_t k = 0; k < kernel_n; ++k) {
    const std::shared_ptr<Tensor<float>>& kernel = this->weights_.at(k);
    CHECK(kernel->rows() == kernel_h);
    CHECK(kernel->cols() == kernel_w);
    CHECK(kernel->channels() == kernel_c);
  }

  const uint32_t kernel_n_group = kernel_n / groups_;
  std::vector<arma::frowvec> im2col_kernel;
  for (uint32_t g = 0; g < groups_; ++g) {
    // (img2col_w * kernel_c, 1) == (1, kernel_w * kernel_h * kernel_c, 1)
    arma::frowvec im2col_kernel_c(im2col_w * kernel_c);
    for (uint32_t n = 0; n < kernel_n_group; ++n) {
      // (kernel_c, kernel_w, kernel_h)
      const sftensor& kernel = this->weights_[n + g * kernel_n_group];
      for (uint32_t c = 0; c < kernel->channels(); ++c) {
        // copy kernel weights to im2col_kernel
        std::memcpy(im2col_kernel_c.memptr() + im2col_w * c,
                    kernel->matrix_raw_ptr(c), im2col_w * sizeof(float));
      }
      im2col_kernel.emplace_back(im2col_kernel_c);
    }
  }
  CHECK(im2col_kernel.size() == kernel_n);
  this->im2col_kernel = std::move(im2col_kernel);
}

arma::fmat ConvolutionLayer::Im2Col(sftensor input, uint32_t kernel_h,
                                    uint32_t kernel_w, uint32_t input_h,
                                    uint32_t input_w, uint32_t input_c_group,
                                    uint32_t group_i, uint32_t im2col_w,
                                    uint32_t im2col_h) {
  // (input_c_group * im2col_w, im2col_h)
  arma::fmat im2col_input(input_c_group * im2col_w, im2col_h);
  const uint32_t input_padded_h = input_h + 2 * padding_h_;
  const uint32_t input_padded_w = input_w + 2 * padding_w_;
  for (uint32_t ic = 0; ic < input_c_group; ++ic) {
    // input channel fmat
    float* input_channel_ptr =
        input->matrix_raw_ptr(ic + group_i * input_c_group);
    uint32_t current_col = 0;
    uint32_t channel_i = ic * im2col_w;
    for (uint32_t w = 0; w < input_padded_w - kernel_w + 1; w += stride_w_) {
      for (uint32_t h = 0; h < input_padded_h - kernel_h + 1; h += stride_h_) {
        float* im2col_input_ptr = im2col_input.colptr(current_col) + channel_i;
        current_col += 1;
        for (uint32_t kw = 0; kw < kernel_w; ++kw) {
          const uint32_t region_w = input_h * (w + kw - padding_w_);
          for (uint32_t kh = 0; kh < kernel_w; ++kh) {
            if ((kh + h >= padding_h_ && kw + w >= padding_w_) &&
                (kh + h < input_h + padding_h_ &&
                 kw + w < input_w + padding_w_)) {
              float* region_ptr =
                  input_channel_ptr + region_w + (h + kh - padding_h_);
              *im2col_input_ptr = *region_ptr;
            } else {
              *im2col_input_ptr = padding_value;
            }
            im2col_input_ptr++;
          }
        }
      }
    }
  }
  return im2col_input;
}

void ConvolutionLayer::ConvGemm(const arma::fmat& im2col_input, sftensor output,
                                uint32_t group, uint32_t kernel_i,
                                uint32_t kernel_n_group,
                                const arma::frowvec& kernel, uint32_t output_w,
                                uint32_t output_h) {
  arma::fmat output_result(
      output->matrix_raw_ptr(kernel_i + group * kernel_n_group), output_h,
      output_w, false, true);

  CHECK(output_result.size() == output_h * output_w)
      << "Output_h x output_w for the convolution layer "
         "should be output tensor size";

  if (!this->bias_.empty() && this->use_bias_) {
    sftensor bias;
    bias = this->bias_[kernel_i];
    if (bias != nullptr && !bias->empty()) {
      float bias_value = bias->index(0);
      output_result = kernel * im2col_input + bias_value;
    } else {
      LOG(FATAL) << "Bias tensor is empty or nullptr";
    }
  } else {
    output_result = kernel * im2col_input;
  }
}

LayerReigister kConvGetInstace("nn.Conv2d", ConvolutionLayer::GetInstace);
}  // namespace free_infer