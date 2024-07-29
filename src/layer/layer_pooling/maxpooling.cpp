#include "maxpooling.hpp"

#include <sys/types.h>

#include <cstdint>
#include <limits>

#include "layer_factory.hpp"
#include "status_code.hpp"
#include "tensor.hpp"
namespace free_infer {
InferStatus MaxPoolingLayer::Forward(const std::vector<sftensor>& inputs,
                                     std::vector<sftensor>& outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array in the max pooling layer is empty";
    return InferStatus::kInferFailedInputEmpty;
  }

  if (inputs.size() != outputs.size()) {
    LOG(ERROR)
        << "The input and output tensor array size of the max pooling layer "
           "do not match";
    return InferStatus::kInferFailedInputOutSizeMatchError;
  }

  const uint32_t batch = inputs.size();
  const uint32_t pooling_h = pooling_size_h_;
  const uint32_t pooling_w = pooling_size_w_;
  if (!stride_h_ || !stride_w_) {
    LOG(ERROR) << "The stride parameter is set incorrectly. It must always be "
                  "greater than 0";
    return InferStatus::kInferFailedStrideParameterError;
  }

  for (uint32_t i = 0; i < batch; ++i) {
    const std::shared_ptr<Tensor<float>>& input_data = inputs.at(i);
    if (input_data == nullptr || input_data->empty()) {
      LOG(ERROR) << "The input tensor array in the max pooling layer has an "
                    "empty tensor "
                 << i << "batch";
      return InferStatus::kInferFailedInputEmpty;
    } else {
      uint32_t input_h = input_data->rows();
      uint32_t input_w = input_data->cols();
      uint32_t output_h = uint32_t(std::floor(
          (int(input_h) - int(pooling_h) + 2 * padding_h_) / stride_h_ + 1));
      uint32_t output_w = uint32_t(std::floor(
          (int(input_w) - int(pooling_w) + 2 * padding_w_) / stride_w_ + 1));
      if (!output_w || !output_h) {
        LOG(ERROR) << "The output size of tensor " << i << "batch"
                   << " in the max pooling layer is less than zero";
        return InferStatus::kInferFailedOutputSizeError;
      } else {
        const std::shared_ptr<Tensor<float>>& output_data = outputs.at(i);
        if (output_data != nullptr && !output_data->empty()) {
          if (output_data->rows() != output_h ||
              output_data->cols() != output_w) {
            LOG(ERROR) << "The output tensor array in the max pooling layer "
                          "has an incorrectly sized tensor "
                       << i << "batch";
            return InferStatus::kInferFailedOutputSizeError;
          }
        }
      }
    }
  }

  for (uint32_t i = 0; i < batch; ++i) {
    const sftensor& input = inputs.at(i);

    const uint32_t input_h = input->rows();
    const uint32_t input_w = input->cols();
    const uint32_t input_c = input->channels();
    const uint32_t input_padded_h = input_h + 2 * padding_h_;
    const uint32_t input_padded_w = input_w + 2 * padding_w_;

    const uint32_t output_h = uint32_t(std::floor(
        (int(input_h) - int(pooling_h) + 2 * padding_h_) / stride_h_ + 1));
    const uint32_t output_w = uint32_t(std::floor(
        (int(input_w) - int(pooling_w) + 2 * padding_w_) / stride_w_ + 1));

    sftensor output = outputs.at(i);
    if (output == nullptr || output->empty()) {
      output =
          std::make_shared<Tensor<float>>(input_c, output_h, output_w);
      outputs.at(i) = output;
    }

    CHECK(output->rows() == output_h && output->cols() == output_w &&
          output->channels() == input_c)
        << "The output tensor array in the max pooling layer "
           "has an incorrectly sized tensor "
        << i << "batch";

    for (uint32_t ic = 0; ic < input_c; ++ic) {
      const arma::fmat& input_ic = input->slice(ic);
      arma::fmat& output_ic = output->slice(ic);
      for (uint32_t w = 0; w < input_padded_w - pooling_w + 1; w += stride_w_) {
        int output_w = int(w / stride_w_);
        for (uint32_t h = 0; h < input_padded_h - pooling_h + 1;
             h += stride_h_) {
          int output_ic_h = int(h / stride_h_);
          float* output_ic_ptr = output_ic.colptr(output_w);
          // max_value = -inf
          float max_value = std::numeric_limits<float>::lowest();
          for (uint32_t pw = 0; pw < pooling_w; ++pw) {
            const float* input_ic_ptr = input_ic.colptr(w + pw - padding_w_);
            for (uint32_t ph = 0; ph < pooling_h; ++ph) {
              float current_value = 0.f;
              if ((w + pw >= padding_w_ && h + ph >= padding_h_) &&
                  (w + pw < input_w + padding_w_ &
                   h + ph < input_h + padding_h_)) {
                current_value = *(input_ic_ptr + h + ph - padding_h_);
              } else {
                current_value = std::numeric_limits<float>::lowest();
              }
              max_value = max_value > current_value ? max_value : current_value;
            }
          }
          *(output_ic_ptr + output_ic_h) = max_value;
        }
      }
    }
  }
  return InferStatus::kInferSuccess;
}

ParseParameterAttrStatus MaxPoolingLayer::GetInstace(
    const std::shared_ptr<RuntimeOperator>& op,
    std::shared_ptr<Layer>& maxpooling_layer) {

  CHECK(op != nullptr) << "MaxPooling get instance failed, operator is nullptr";
  const std::map<std::string, std::shared_ptr<RuntimeParameter>>& params =
      op->params;

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

  if (params.find("kernel_size") == params.end()) {
    LOG(ERROR) << "Can not find the kernel size parameter";
    return ParseParameterAttrStatus::kParameterMissingKernel;
  }

  auto kernel_size = std::dynamic_pointer_cast<RuntimeParameterIntArray>(
      params.at("kernel_size"));
  if (!kernel_size) {
    LOG(ERROR) << "Can not find the kernel size parameter";
    return ParseParameterAttrStatus::kParameterMissingKernel;
  }
  const auto& padding_values = padding->value;
  const auto& stride_values = stride->value;
  const auto& kernel_values = kernel_size->value;

  const uint32_t dims = 2;
  if (padding_values.size() != dims) {
    LOG(ERROR) << "Can not find the right padding parameter";
    return ParseParameterAttrStatus::kParameterMissingPadding;
  }

  if (stride_values.size() != dims) {
    LOG(ERROR) << "Can not find the right stride parameter";
    return ParseParameterAttrStatus::kParameterMissingStride;
  }

  if (kernel_values.size() != dims) {
    LOG(ERROR) << "Can not find the right kernel size parameter";
    return ParseParameterAttrStatus::kParameterMissingKernel;
  }

  maxpooling_layer = std::make_shared<MaxPoolingLayer>(
      kernel_values.at(0), kernel_values.at(1), padding_values.at(0),
      padding_values.at(1), stride_values.at(0), stride_values.at(1));

  return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

LayerReigister MaxPoolingGetInstace("nn.MaxPool2d", MaxPoolingLayer::GetInstace);
}  // namespace free_infer