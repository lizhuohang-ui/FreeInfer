#include "layer.hpp"
#include "tensor.hpp"

namespace free_infer {

InferStatus Layer::Forward(
    const std::vector<sftensor>& inputs,
    std::vector<sftensor>& outputs) {
  LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
}

void Layer::set_runtime_operator(const std::shared_ptr<RuntimeOperator>& runtime_operator) {
  CHECK(runtime_operator != nullptr);
  this->runtime_operator_ = runtime_operator;
}
}  // namespace free_infer