#include "layer/layer_expression.hpp"

#include <sys/types.h>

#include <cstdint>
#include <memory>
#include <stack>
#include <vector>

#include "layer/layer.hpp"
#include "layer/layer_factory.hpp"
#include "layer/parse_expression.hpp"
#include "runtime/runtime_ir.hpp"
#include "tensor/tensor.hpp"
#include "tensor/tensor_util.hpp"

namespace free_infer {
ExpressionLayer::ExpressionLayer(std::string statement)
    : Layer("Expression"), statement_(std::move(statement)) {
  parser_ = std::make_unique<ExpressionParser>(statement_);
}

InferStatus ExpressionLayer::Forward(const std::vector<sftensor>& inputs,
                                     std::vector<sftensor>& outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array in the expression layer is empty";
    return InferStatus::kInferFailedInputEmpty;
  }

  if (outputs.empty()) {
    LOG(ERROR) << "The output tensor array in the expression layer is empty";
    return InferStatus::kInferFailedOutputEmpty;
  }

  CHECK(this->parser_ != nullptr)
      << "The parser in the expression layer is null!";
  this->parser_->Tokenizer();
  const auto& expressions = this->parser_->tokens();
  CHECK(!expressions.empty())
      << "The expression parser failed to parse " << statement_;

  for (uint32_t i = 0; i < inputs.size(); ++i) {
    const sftensor& input_data = inputs.at(i);
    if (input_data == nullptr || input_data->empty()) {
      LOG(ERROR) << "The input tensor array in the expression layer has an "
                    "empty tensor "
                 << i << "th";
      return InferStatus::kInferFailedInputEmpty;
    }
  }

  const uint32_t batch_size = outputs.size();
  for (uint32_t i = 0; i < batch_size; ++i) {
    if (outputs.at(i) == nullptr || outputs.at(i)->empty()) {
      DLOG(ERROR) << "The output tensor array in the expression layer has an "
                     "empty tensor "
                  << i << "th";
      return InferStatus::kInferFailedOutputEmpty;
    }
    outputs.at(i)->Fill(0.f);
  }
  std::stack<std::vector<sftensor>> op_stack;
  const std::vector<std::shared_ptr<TokenNode>>& token_nodes =
      this->parser_->Generate();
  for (const auto& token_node : token_nodes) {
    if (token_node->num_index >= 0) {
      uint32_t start_pos = token_node->num_index * batch_size;
      std::vector<sftensor> input_token_nodes;
      for (uint32_t i = 0; i < batch_size; ++i) {
        CHECK(i + start_pos < inputs.size())
            << "The " << i
            << "th operand doesn't have appropriate number of tensors";
        //
        input_token_nodes.push_back(inputs.at(i + start_pos));
      }
      op_stack.push(input_token_nodes);
    } else if (token_node->num_index == int(TokenType::TokenSin)) {
      CHECK(op_stack.size() == 1) << "The number of operand is not one";

      std::vector<sftensor> input_node = op_stack.top();
      CHECK(input_node.size() == batch_size)
          << "The first operand doesn't have appropriate number of tensors, "
             "which need "
          << batch_size;
      op_stack.pop();

      std::vector<sftensor> output_token_nodes(batch_size);

      for (uint32_t i = 0; i < batch_size; ++i) {
        output_token_nodes.at(i) = TensorElementSin(input_node.at(i));
      }
      op_stack.push(output_token_nodes);

    } else {
      const int32_t op_type = token_node->num_index;

      if (op_type != int(TokenType::TokenAdd) &&
          op_type != int(TokenType::TokenMul)) {
        LOG(FATAL) << "Unknown operator type: " << op_type;
      }
      CHECK(op_stack.size() >= 2) << "The number of operand is less than two";

      std::vector<sftensor> input_node1 = op_stack.top();
      CHECK(input_node1.size() == batch_size)
          << "The first operand doesn't have appropriate number of tensors, "
             "which need "
          << batch_size;
      op_stack.pop();

      std::vector<sftensor> input_node2 = op_stack.top();
      CHECK(input_node2.size() == batch_size)
          << "The first operand doesn't have appropriate number of tensors, "
             "which need "
          << batch_size;
      op_stack.pop();

      std::vector<sftensor> output_token_nodes(batch_size);
      for (uint32_t i = 0; i < batch_size; ++i) {
        if (op_type == int(TokenType::TokenAdd)) {
          output_token_nodes.at(i) =
              TensorElementAdd(input_node1.at(i), input_node2.at(i));
        } else if (op_type == int(TokenType::TokenMul)) {
          output_token_nodes.at(i) =
              TensorElementMultiply(input_node1.at(i), input_node2.at(i));
        } else {
          LOG(FATAL) << "Unknown operator type: " << op_type;
        }
      }
      op_stack.push(output_token_nodes);
    }
  }

  CHECK(op_stack.size() == 1)
      << "The expression has more than one output operand!";

  std::vector<sftensor> output_node = op_stack.top();
  op_stack.pop();
  for (int i = 0; i < batch_size; ++i) {
    CHECK(outputs.at(i) != nullptr && !outputs.at(i)->empty());
    CHECK(outputs.at(i)->shapes() == output_node.at(i)->shapes());
    outputs.at(i) = output_node.at(i);
  }
  return InferStatus::kInferSuccess;
}

ParseParameterAttrStatus ExpressionLayer::GetInstace(
    const std::shared_ptr<RuntimeOperator>& op,
    std::shared_ptr<Layer>& expression_layer) {
  CHECK(op != nullptr) << "Expression operator is nullptr";
  const auto& params = op->params;
  if (params.find("expr") == params.end()) {
    return ParseParameterAttrStatus::kParameterMissingExpr;
  }

  std::shared_ptr<RuntimeParameterString> statement_param =
      std::dynamic_pointer_cast<RuntimeParameterString>(params.at("expr"));
  if (statement_param == nullptr) {
    LOG(ERROR) << "Can not find the expression parameter";
    return ParseParameterAttrStatus::kParameterMissingExpr;
  }
  if (statement_param->type != RuntimeParameterType::kParameterString) {
    LOG(ERROR) << "Can not find the expression parameter";
    return ParseParameterAttrStatus::kParameterMissingExpr;
  }

  expression_layer = std::make_shared<ExpressionLayer>(statement_param->value);
  return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

LayerReigister kExpressionGetIntance("pnnx.Expression",
                                     ExpressionLayer::GetInstace);

}  // namespace free_infer
