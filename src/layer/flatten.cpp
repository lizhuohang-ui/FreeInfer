#include "layer/flatten.hpp"

#include <sys/types.h>

#include <cstdint>

#include "layer/layer.hpp"
#include "layer/layer_factory.hpp"
#include "runtime/status_code.hpp"
#include "tensor/tensor.hpp"
#include "tensor/tensor_util.hpp"

namespace free_infer {
FlattenLayer::FlattenLayer(uint32_t start_dim, uint32_t end_dim)
    : Layer("Flatten"), start_dim_(start_dim), end_dim_(end_dim) {}

InferStatus FlattenLayer::Forward(const std::vector<sftensor>& inputs,
                                  std::vector<sftensor>& outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array in the flatten layer is empty";
    return InferStatus::kInferFailedInputEmpty;
  }

  if (inputs.size() != outputs.size()) {
    LOG(ERROR) << "The input and output tensor array size of the flatten "
                  "layer do not match";
    return InferStatus::kInferFailedInputOutSizeMatchError;
  }

  const uint32_t batch_size = inputs.size();

  int start_dim = start_dim_;
  int end_dim = end_dim_;
  int total_dim = 4;

  if (start_dim < 0) {
    start_dim += total_dim;
  }

  if (end_dim < 0) {
    end_dim += total_dim;
  }

  CHECK(end_dim >= start_dim);
  CHECK(start_dim >= 1 && end_dim <= 3);

  for (uint32_t i = 0; i < batch_size; ++i) {
    const sftensor& input = inputs.at(i);

    const uint32_t input_c = input->channels();
    CHECK(input != nullptr && !input->empty())
        << "The input tensor array in the flatten layer has an empty  "
           "tensor "
        << i << " batch";

    sftensor output = outputs.at(i);

    auto shapes = input->shapes();
    shapes.insert(shapes.begin(), batch_size);
    uint32_t elements_size =
        std::accumulate(shapes.begin() + start_dim,
                        shapes.begin() + end_dim + 1, 1, std::multiplies());

    output = TensorClone(input);

    CHECK(input->size() == output->size())
        << "The output and input shapes of the flatten layer do "
           "not match "
        << i << " th";
    outputs.at(i) = output;

    if (start_dim == 1 && end_dim == 3) {
      output->Reshape({elements_size}, true);
    } else if (start_dim == 2 && end_dim == 3) {
      uint32_t channels = input->channels();
      output->Reshape({channels, elements_size}, true);
    } else if (start_dim == 1 && end_dim == 2) {
      uint32_t cols = input->cols();
      output->Reshape({elements_size, cols}, true);
    } else {
      LOG(FATAL) << "Wrong flatten dim: "
                 << "start dim: " << start_dim << " end dim: " << end_dim;
    }
  }

  return InferStatus::kInferSuccess;
}

ParseParameterAttrStatus FlattenLayer::GetInstace(
    const std::shared_ptr<RuntimeOperator>& op,
    std::shared_ptr<Layer>& flatten_layer) {
  CHECK(op != nullptr);
  const std::map<std::string, std::shared_ptr<RuntimeParameter>>& params =
      op->params;

  if (params.find("start_dim") == params.end()) {
    LOG(ERROR) << "Can not find the start_dim parameter";
    return ParseParameterAttrStatus::kAttrMissingInFeatures;
  }
  auto start_dim =
      std::dynamic_pointer_cast<RuntimeParameterInt>(params.at("start_dim"));
  if (!start_dim) {
    LOG(ERROR) << "Can not find the start_dim parameter";
    return ParseParameterAttrStatus::kAttrMissingInFeatures;
  }

  if (params.find("end_dim") == params.end()) {
    LOG(ERROR) << "Can not find the end_dim parameter";
    return ParseParameterAttrStatus::kAttrMissingInFeatures;
  }

  auto end_dim =
      std::dynamic_pointer_cast<RuntimeParameterInt>(params.at("end_dim"));
  if (!start_dim) {
    LOG(ERROR) << "Can not find the end_dim parameter";
    return ParseParameterAttrStatus::kAttrMissingInFeatures;
  }

  flatten_layer =
      std::make_shared<FlattenLayer>(start_dim->value, end_dim->value);

  auto flatten_layer_derived =
      std::dynamic_pointer_cast<FlattenLayer>(flatten_layer);

  return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

LayerReigister FlattenGetInstace("torch.flatten", FlattenLayer::GetInstace);
}  // namespace free_infer