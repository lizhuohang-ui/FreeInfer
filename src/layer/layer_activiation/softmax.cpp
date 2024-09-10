

#include "softmax.hpp"
#include <math.h>

#include <cstdint>
#include <memory>

#include "layer_factory.hpp"
#include "status_code.hpp"
#include "tensor.hpp"

namespace free_infer {

InferStatus SoftmaxLayer::Forward(const std::vector<sftensor>& inputs,
                                  std::vector<sftensor>& outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array in the " << layer_name_
               << " layer is empty";
    return InferStatus::kInferFailedInputEmpty;
  }
  if (inputs.size() != outputs.size()) {
    LOG(ERROR)
        << "The input and output tensor array size of the " << layer_name_ << " layer do "
           "not match";
    return InferStatus::kInferFailedInputOutSizeMatchError;
  }

  const uint32_t batch_size = inputs.size();
  for (uint32_t i = 0; i < batch_size; ++i) {
    const sftensor& input = inputs.at(i);
    sftensor& output = outputs.at(i);
    if (input == nullptr || input->empty()) {
      LOG(ERROR)
          << "The input tensor array in the softmax layer has an empty tensor "
          << i << " th";
      return InferStatus::kInferFailedInputEmpty;
    }
    if (output != nullptr && !output->empty()) {
      if (input->shapes() != output->shapes()) {
        LOG(ERROR) << "The input and output tensor shapes of the softmax "
                      "layer do not match "
                   << i << " th";
        return InferStatus::kInferFailedInputOutSizeMatchError;
      }
    }

    if (output == nullptr || output->empty()) {
      output = std::make_shared<Tensor<float>>(input->shapes());
      outputs.at(i) = output;
    }
    
    auto input_exp = arma::exp(input->data());

    float input_exp_sum = arma::accu(input_exp);

    for (uint32_t j = 0; j < input->size(); ++j) {
      float value = input->index(j);
      output->index(j) = exp(value) / input_exp_sum;
    }
  }
  return InferStatus::kInferSuccess;
}
ParseParameterAttrStatus SoftmaxLayer::GetInstace(
    const std::shared_ptr<RuntimeOperator>& op,
    std::shared_ptr<Layer>& softmax_layer) {
  CHECK(op != nullptr);
  softmax_layer = std::make_shared<SoftmaxLayer>();
  return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

LayerReigister SoftmaxReigister("nn.Softmax", SoftmaxLayer::GetInstace);
}  // namespace free_infer
