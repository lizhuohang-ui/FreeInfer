#include "layer.hpp"

#include <memory>
#include <vector>

#include "runtime_ir.hpp"
#include "status_code.hpp"
#include "tensor.hpp"

namespace free_infer {

InferStatus Layer::Forward(const std::vector<sftensor>& inputs,
                           std::vector<sftensor>& outputs) {
  LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
}

InferStatus Layer::Forward() {
  LOG_IF(FATAL, this->runtime_operator_.expired())
      << "Runtime operator is expired or nullptr";
  const auto& runtime_operator = this->runtime_operator_.lock();
  const std::vector<std::shared_ptr<RuntimeOperand>>& input_operand_datas =
      runtime_operator->input_operands;
  std::vector<sftensor> layer_input_datas;
  for (const auto& input_operand_data : input_operand_datas) {
    for (const auto& input_data : input_operand_data->datas) {
      layer_input_datas.push_back(input_data);
    }
  }

  const std::shared_ptr<RuntimeOperand>& output_operand_datas =
      runtime_operator->output_operands;
  CHECK(!layer_input_datas.empty())
      << runtime_operator->name << " layer input data is empty";
  CHECK(output_operand_datas != nullptr && !output_operand_datas->datas.empty())
      << runtime_operator->name << " layer output data is empty";

  InferStatus status = runtime_operator->layer->Forward(
      layer_input_datas, output_operand_datas->datas);
  return status;
}

void Layer::set_runtime_operator(
    const std::shared_ptr<RuntimeOperator>& runtime_operator) {
  CHECK(runtime_operator != nullptr);
  this->runtime_operator_ = runtime_operator;
}
}  // namespace free_infer