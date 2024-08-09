#include "sigmoid.hpp"

#include <cmath>
#include <memory>

#include "status_code.hpp"

namespace free_infer {
InferStatus SigmoidLayer::Forward(const std::vector<sftensor>& inputs, std::vector<sftensor>& outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array in the relu layer is empty";
    return InferStatus::kInferFailedInputEmpty;
  }
  if (inputs.size() != outputs.size()) {
    LOG(ERROR) << "The input and output tensor array size of the relu layer do "
                  "not match";
    return InferStatus::kInferFailedInputOutSizeMatchError;
  }

  const uint32_t batch_size = inputs.size();
  for (uint32_t i = 0; i < batch_size; ++i) {
    const sftensor& input = inputs.at(i);
    sftensor& output = outputs.at(i);
    if (input == nullptr || input->empty()) {
      LOG(ERROR) << "The input tensor array in the relu layer has an empty tensor " << i << " th";
      return InferStatus::kInferFailedInputEmpty;
    }
    if (output != nullptr && !output->empty()) {
      if (input->shapes() != output->shapes()) {
        LOG(ERROR) << "The input and output tensor shapes of the relu "
                      "layer do not match "
                   << i << " th";
        return InferStatus::kInferFailedInputOutSizeMatchError;
      }
    }

    if (output == nullptr || output->empty()) {
      output = std::make_shared<Tensor<float>>(input->shapes());
      outputs.at(i) = output;
    }

    for (uint32_t j = 0; j < input->size(); ++j) {
      float value = input->index(j);
      output->index(j) = 1.f / (1.f + std::exp(-value));
    }
  }
  return InferStatus::kInferSuccess;
}

ParseParameterAttrStatus SigmoidLayer::GetInstace(const std::shared_ptr<RuntimeOperator>& op,
                                                  std::shared_ptr<Layer>& sigmoid_layer) {
  sigmoid_layer = std::make_shared<SigmoidLayer>();
  return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

LayerReigister nnSigmoidReigister("nn.Sigmoid", SigmoidLayer::GetInstace);
LayerReigister FSigmoidReigister("F.sigmoid", SigmoidLayer::GetInstace);

}  // namespace free_infer