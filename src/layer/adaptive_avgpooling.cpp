#include "layer/adaptive_avgpooling.hpp"

#include <cmath>
#include <cstdint>
#include <memory>

#include "layer/layer_factory.hpp"
#include "runtime/runtime_ir.hpp"
#include "runtime/status_code.hpp"
#include "tensor/tensor.hpp"

namespace free_infer {
InferStatus AdaptiveAvgPoolingLayer::Forward(
    const std::vector<sftensor>& inputs, std::vector<sftensor>& outputs) {
  if (inputs.empty()) {
    LOG(ERROR)
        << "The input tensor array in the adaptive avgpooling layer is empty";
    return InferStatus::kInferFailedInputEmpty;
  }

  if (inputs.size() != outputs.size()) {
    LOG(ERROR) << "The input and output tensor array size of the adaptive "
                  "avgpooling layer "
                  "do not match";
    return InferStatus::kInferFailedInputOutSizeMatchError;
  }

  const uint32_t batch = inputs.size();
  for (uint32_t i = 0; i < batch; ++i) {
    const sftensor input_data = inputs.at(i);
    const sftensor output_data = outputs.at(i);
    if (input_data == nullptr || input_data->empty()) {
      LOG(ERROR)
          << "The input tensor array in the adaptive avgpooling layer has an "
             "empty tensor "
          << i << "batch";
      return InferStatus::kInferFailedInputEmpty;
    }
    if (output_data != nullptr && !output_data->empty()) {
      if (output_data->rows() != output_h_ &&
          output_data->cols() != output_w_) {
        LOG(ERROR) << "The output tensor array in the adaptive avgpooling "
                      "layer has an "
                      "incorrectly sized tensor "
                   << i << "batch";
        return InferStatus::kInferFailedOutputSizeError;
      }
    }
  }

  for (uint32_t i = 0; i < batch; ++i) {
    const sftensor& input = inputs.at(i);
    const uint32_t input_c = input->channels();
    const uint32_t input_h = input->rows();
    const uint32_t input_w = input->cols();
    const uint32_t stride_h = uint32_t(std::floor(input_h / output_h_));
    const uint32_t stride_w = uint32_t(std::floor(input_w / output_w_));
    const uint32_t pooling_h = input_h - (output_h_ - 1) * stride_h;
    const uint32_t pooling_w = input_w - (output_w_ - 1) * stride_w;
    const uint32_t pooling_size = pooling_h * pooling_w;

    sftensor output = outputs.at(i);
    if (output == nullptr || output->empty()) {
      output = std::make_shared<Tensor<float>>(input_c, output_h_, output_w_);
      outputs.at(i) = output;
    }

    CHECK(output->rows() == output_h_ && output->cols() == output_w_ &&
          output->channels() == input_c)
        << "The output tensor array in the adaptive avgpooling layer "
           "has an incorrectly sized tensor "
        << i << "batch";

    for (uint32_t ic = 0; ic < input_c; ++ic) {
      const arma::fmat& input_ic = input->slice(ic);
      arma::fmat& output_ic = output->slice(ic);
      for (uint32_t w = 0; w < input_w; w += stride_w) {
        int output_ic_w = int(w / stride_w);
        for (uint32_t h = 0; h < input_h; h += stride_h) {
          int output_ic_h = int(h / stride_h);
          float* output_ic_ptr = output_ic.colptr(output_ic_w);
          float avg_value = 0.f;
          for(uint32_t pw = 0; pw < pooling_w; ++pw){
            const float* input_ic_ptr = input_ic.colptr(w + pw);
            for(uint32_t ph = 0; ph < pooling_h; ++ph){
              float current_value = *(input_ic_ptr + h + ph);
              avg_value += current_value;
            }
          }
        *(output_ic_ptr + output_ic_h) = avg_value / float(pooling_size);
        }
      }
    }
  }

  return InferStatus::kInferSuccess;
}
ParseParameterAttrStatus AdaptiveAvgPoolingLayer::GetInstace(
    const std::shared_ptr<RuntimeOperator>& op,
    std::shared_ptr<Layer>& adaptive_avgpooling_layer) {
  CHECK(op != nullptr)
      << "AdaptiveAvgPooling get instance failed, operator is nullptr";
  const std::map<std::string, std::shared_ptr<RuntimeParameter>>& params =
      op->params;

  if (params.find("output_size") == params.end()) {
    LOG(ERROR) << "Can not find the output_size parameter";
    return ParseParameterAttrStatus::kParameterMissingOutHW;
  }

  auto output_size = std::dynamic_pointer_cast<RuntimeParameterIntArray>(
      params.at("output_size"));

  const auto& output_hw = output_size->value;
  if (output_hw.size() != 2) {
    LOG(ERROR) << "Can not find the output_size parameter";
    return ParseParameterAttrStatus::kParameterMissingOutHW;
  }

  adaptive_avgpooling_layer = std::make_shared<AdaptiveAvgPoolingLayer>(
      output_hw.at(0), output_hw.at(1));

  return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

LayerReigister AdaptiveAvgPoolingGetInstace(
    "nn.AdaptiveAvgPool2d", AdaptiveAvgPoolingLayer::GetInstace);

}  // namespace free_infer