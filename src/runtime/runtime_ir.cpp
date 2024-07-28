#include "runtime_ir.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <map>
#include <memory>
#include <stack>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "ir.h"
#include "status_code.hpp"
#include "tensor.hpp"

namespace free_infer {
RuntimeGraph::RuntimeGraph(std::string param_path, std::string bin_path)
    : param_path_(std::move(param_path)), bin_path_(std::move(bin_path)) {}

void RuntimeGraph::set_bin_path(const std::string& bin_path) {
  this->bin_path_ = bin_path;
}

void RuntimeGraph::set_param_path(const std::string& param_path) {
  this->param_path_ = param_path;
}

const std::string& RuntimeGraph::bin_path() const { return this->bin_path_; }

const std::string& RuntimeGraph::param_path() const {
  return this->param_path_;
}

bool RuntimeGraph::Init() {
  if (this->bin_path_.empty() || this->param_path_.empty()) {
    LOG(ERROR) << "The bin path or param path is empty";
    return false;
  }

  this->graph_ = std::make_unique<pnnx::Graph>();
  int load_result = this->graph_->load(param_path_, bin_path_);
  if (load_result != 0) {
    LOG(ERROR) << "Can not find the param path or bin path: " << param_path_
               << " " << bin_path_;
    return false;
  }

  std::vector<pnnx::Operator*> operators = this->graph_->ops;
  if (operators.empty()) {
    LOG(ERROR) << "Can not read the layers' define";
    return false;
  }

  this->operators_.clear();
  this->operators_maps_.clear();

  for (const pnnx::Operator* op : operators) {
    if (!op) {
      LOG(ERROR) << "Meet the empty node";
      continue;
    } else {
      std::shared_ptr<RuntimeOperator> runtime_operator =
          std::make_shared<RuntimeOperator>();
      runtime_operator->name = op->name;
      runtime_operator->type = op->type;

      const std::vector<pnnx::Operand*>& inputs = op->inputs;
      if (!inputs.empty()) {
        InitGraphOperatorsInput(inputs, runtime_operator);
      }

      const std::vector<pnnx::Operand*>& outputs = op->outputs;
      if (!outputs.empty()) {
        InitGraphOperatorsOutput(outputs, runtime_operator);
      }

      const std::map<std::string, pnnx::Attribute>& attrs = op->attrs;
      if (!attrs.empty()) {
        InitGraphOperatorsAttr(attrs, runtime_operator);
      }

      const std::map<std::string, pnnx::Parameter>& params = op->params;
      if (!params.empty()) {
        InitGraphOperatorsParam(params, runtime_operator);
      }

      this->operators_.push_back(runtime_operator);
      this->operators_maps_.insert({runtime_operator->name, runtime_operator});
    }
  }

  this->graph_state_ = GraphState::NeedBuild;
  return true;
}

void RuntimeGraph::InitGraphOperatorsInput(
    const std::vector<pnnx::Operand*>& inputs,
    const std::shared_ptr<RuntimeOperator>& runtime_operator) {
  for (const pnnx::Operand* input : inputs) {
    if (!input) {
      continue;
    }
    const pnnx::Operator* producer = input->producer;
    std::shared_ptr<RuntimeOperand> runtime_operand =
        std::make_shared<RuntimeOperand>();
    runtime_operand->name = producer->name;
    runtime_operand->shapes = input->shape;

    switch (input->type) {
      case 1: {
        runtime_operand->type = RuntimeDataType::kTypeFloat32;
        break;
      }
      case 0: {
        runtime_operand->type = RuntimeDataType::kTypeUnknown;
        break;
      }
      default: {
        LOG(ERROR) << "Unknown input operand type: " << input->type;
      }
    }
    runtime_operator->input_operands_maps.insert(
        {producer->name, runtime_operand});
    runtime_operator->input_operands.push_back(runtime_operand);
  }
}

void RuntimeGraph::InitGraphOperatorsOutput(
    const std::vector<pnnx::Operand*>& outputs,
    const std::shared_ptr<RuntimeOperator>& runtime_operator) {
  for (const pnnx::Operand* output : outputs) {
    if (!output) {
      continue;
    }
    const auto& consumers = output->consumers;
    for (const auto& c : consumers) {
      runtime_operator->output_names.push_back(c->name);
    }
  }
}

void RuntimeGraph::InitGraphOperatorsParam(
    const std::map<std::string, pnnx::Parameter>& params,
    const std::shared_ptr<RuntimeOperator>& runtime_operator) {
  for (const auto& [name, param] : params) {
    const int type =
        param.type;  // 0=null 1=b 2=i 3=f 4=s 5=ai 6=af 7=as 8=others
    switch (type) {
        //   case 0: {
        //     RuntimeParameter* runtime_parameter = new RuntimeParameter;
        //     runtime_operator->params.insert({name, runtime_parameter});
        //     break;
        //   }

      case 1: {
        RuntimeParameterBool* runtime_parameter =
            new RuntimeParameterBool(RuntimeParameterType::kParameterBool);
        runtime_parameter->value = param.b;
        runtime_operator->params.emplace(name, runtime_parameter);
        break;
      }

      case 2: {
        RuntimeParameterInt* runtime_parameter =
            new RuntimeParameterInt(RuntimeParameterType::kParameterInt);
        runtime_parameter->value = param.i;
        runtime_operator->params.emplace(name, runtime_parameter);
        break;
      }

      case 3: {
        RuntimeParameterFloat* runtime_parameter =
            new RuntimeParameterFloat(RuntimeParameterType::kParameterFloat);
        runtime_parameter->value = param.f;
        runtime_operator->params.emplace(name, runtime_parameter);
        break;
      }
      case 4: {
        RuntimeParameterString* runtime_parameter =
            new RuntimeParameterString(RuntimeParameterType::kParameterString);
        runtime_parameter->value = param.s;
        runtime_operator->params.emplace(name, runtime_parameter);
        break;
      }
      case 5: {
        RuntimeParameterIntArray* runtime_parameter =
            new RuntimeParameterIntArray(
                RuntimeParameterType::kParameterIntArray);
        runtime_parameter->value = param.ai;
        runtime_operator->params.emplace(name, runtime_parameter);
        break;
      }
      case 6: {
        RuntimeParameterFloatArray* runtime_parameter =
            new RuntimeParameterFloatArray(
                RuntimeParameterType::kParameterFloatArray);
        runtime_parameter->value = param.af;
        runtime_operator->params.emplace(name, runtime_parameter);
        break;
      }
      case 7: {
        RuntimeParameterStringArray* runtime_parameter =
            new RuntimeParameterStringArray(
                RuntimeParameterType::kParameterStringArray);
        runtime_parameter->value = param.as;
        runtime_operator->params.emplace(name, runtime_parameter);
        break;
      }
      default: {
        LOG(ERROR) << "Unknown parameter type: " << type;
      }
    }
  }
}

void RuntimeGraph::InitGraphOperatorsAttr(
    const std::map<std::string, pnnx::Attribute>& attrs,
    const std::shared_ptr<RuntimeOperator>& runtime_operator) {
  for (const auto& [name, attr] : attrs) {
    // 0=null 1=f32 2=f64 3=f16 4=i32 5=i64 6=i16 7=i8 8=u8 9=bool
    const int type = attr.type;
    switch (type) {
      case 1: {
        std::shared_ptr<RuntimeAttribute> runtime_attribute =
            std::make_shared<RuntimeAttribute>();
        runtime_attribute->type = RuntimeDataType::kTypeFloat32;
        runtime_attribute->weight_data = attr.data;
        runtime_attribute->shape = attr.shape;
        runtime_operator->attrs.insert({name, runtime_attribute});
        break;
      }
      default: {
        LOG(ERROR) << "Unknown attribute type: " << attr.type;
      }
    }
  }
}

void RuntimeGraph::dfs(std::shared_ptr<RuntimeOperator> op) {
  op->has_forward = true;
  for (const auto& [_, output_op] : op->output_operators_maps) {
    if (!output_op->has_forward) {
      dfs(output_op);
    }
  }
  for (const auto& [_, output_op] : op->output_operators_maps) {
    CHECK(op->has_forward);
  }
  this->operators_topo_.push_back(op);
}

void RuntimeGraph::Topo(void) {
  // build graph
  for (const auto& current_op : this->operators_) {
    const std::vector<std::string>& output_names = current_op->output_names;
    for (const auto& output_name : output_names) {
      if (const auto& output_op = this->operators_maps_.find(output_name);
          output_op != this->operators_maps_.end()) {
        current_op->output_operators_maps.insert(
            {output_name, output_op->second});
      }
    }
  }

  for (const auto& op : this->operators_) {
    CHECK(op != nullptr) << "current operator is nullptr";
    if (!op->has_forward) {
      dfs(op);
    }
  }
  std::reverse(operators_topo_.begin(), operators_topo_.end());
}

const std::vector<std::shared_ptr<RuntimeOperator>>& RuntimeGraph::operators()
    const {
  return this->operators_;
}

const std::vector<std::shared_ptr<RuntimeOperator>>&
RuntimeGraph::get_topo_queues() const {
  return this->operators_topo_;
}

const RuntimeGraph::GraphState RuntimeGraph::graph_state() const {
  return this->graph_state_;
}
void RuntimeGraph::InitOperatorInput(
    const std::vector<std::shared_ptr<RuntimeOperator>>& operators) {
  if (operators.empty()) {
    LOG(ERROR) << "Operators for init input shapes is empty!";
    return;
  }

  for (const auto& op : operators) {
    if (op->input_operands.empty()) {
      continue;
    } else {
      const auto& input_operands_map = op->input_operands_maps;
      for (const auto& [_, input_operand] : input_operands_map) {
        const auto& type = input_operand->type;
        CHECK(type == RuntimeDataType::kTypeFloat32)
            << "The graph only support float32 yet!";
        const auto& input_operand_shape = input_operand->shapes;

        auto& input_datas = input_operand->datas;
        CHECK(!input_operand_shape.empty()) << "Operand shape error";
        const int32_t batch = input_operand_shape.at(0);
        CHECK(batch >= 0) << "Dynaimc batch size is not supported yet!";
        CHECK(input_operand_shape.size() == 2 ||
              input_operand_shape.size() == 3 ||
              input_operand_shape.size() == 4)
            << "Unsupported tensor shape sizes" << input_operand_shape.size();

        if (!input_datas.empty()) {
          CHECK(input_datas.size() == batch);
        } else {
          input_datas.resize(batch);
        }
      }
    }
  }
}

void RuntimeGraph::InitOperatorOutput(
    const std::vector<pnnx::Operator*>& pnnx_operators,
    const std::vector<std::shared_ptr<RuntimeOperator>>& operators) {
  CHECK(!pnnx_operators.empty() && !operators.empty());
  CHECK(pnnx_operators.size() == operators.size());

  for (uint32_t i = 0; i < pnnx_operators.size(); ++i) {
    const auto operands = pnnx_operators.at(i)->outputs;
    CHECK(operands.size() <= 1) << "Only support one node one output yet!";
    if (operands.empty()) {
      continue;
    }
    CHECK(operands.size() == 1)
        << "Only support one output in the FreeInfer yet!";

    pnnx::Operand* operand = operands.front();
    const auto& runtime_op = operators.at(i);
    CHECK(operand != nullptr);
    const std::vector<int32_t>& operand_shapes = operand->shape;
    const auto& output_tensors = runtime_op->output_operands;
    CHECK(operand != nullptr) << "Operand output is null";
    const int32_t batch = operand_shapes.at(0);
    CHECK(batch >= 0) << "Dynamic batch size is not supported!";
    CHECK(operand_shapes.size() == 2 || operand_shapes.size() == 3 ||
          operand_shapes.size() == 4)
        << "Unsupported tensor shape sizes" << operand_shapes.size();

    if (!output_tensors) {
      std::shared_ptr<RuntimeOperand> output_operand =
          std::make_shared<RuntimeOperand>();
      output_operand->name = operand->name + "_output";
      output_operand->type = RuntimeDataType::kTypeFloat32;
      output_operand->shapes = operand_shapes;
      for (uint32_t j = 0; j < batch; ++j) {
        if (operand_shapes.size() == 4) {
          sftensor output_tensor = std::make_shared<Tensor<float>>(
              operand_shapes.at(1), operand_shapes.at(2), operand_shapes.at(3));
          output_operand->datas.push_back(output_tensor);
        } else if (operand_shapes.size() == 2) {
          sftensor output_tensor =
              std::make_shared<Tensor<float>>(operand_shapes.at(1));
          output_operand->datas.push_back(output_tensor);
        } else {
          sftensor output_tensor = std::make_shared<Tensor<float>>(
              operand_shapes.at(1), operand_shapes.at(2));
          output_operand->datas.push_back(output_tensor);
        }
      }
      runtime_op->output_operands = std::move(output_operand);
    } else {
      CHECK(batch == output_tensors->datas.size());
      CHECK(output_tensors->type == RuntimeDataType::kTypeFloat32)
          << "The graph only support float32 yet!";
      CHECK(output_tensors->shapes == operand_shapes);
      for (uint32_t b = 0; b < batch; ++b) {
        sftensor output_tensor = output_tensors->datas.at(b);
        const auto& tensor_shapes = output_tensor->shapes();
        if (operand_shapes.size() == 4) {
          if (tensor_shapes.at(0) != operand_shapes.at(1) ||
              tensor_shapes.at(1) != operand_shapes.at(2) ||
              tensor_shapes.at(2) != operand_shapes.at(3)) {
            DLOG(WARNING)
                << "The shape of tensor do not adapting with output operand";
            const auto& target_shapes = std::vector<uint32_t>{
                (uint32_t)operand_shapes.at(1), (uint32_t)operand_shapes.at(2),
                (uint32_t)operand_shapes.at(3)};
            output_tensor->Reshape(target_shapes);
          }
        } else if (operand_shapes.size() == 2) {
          if (tensor_shapes.at(0) != 1 ||
              tensor_shapes.at(1) != operand_shapes.at(1) ||
              tensor_shapes.at(2) != 1) {
            DLOG(WARNING)
                << "The shape of tensor do not adapting with output operand";
            const auto& target_shapes =
                std::vector<uint32_t>{(uint32_t)operand_shapes.at(1)};
            output_tensor->Reshape(target_shapes);
          }
        } else {
          if (tensor_shapes.at(0) != 1 ||
              tensor_shapes.at(1) != operand_shapes.at(1) ||
              tensor_shapes.at(2) != operand_shapes.at(2)) {
            DLOG(WARNING)
                << "The shape of tensor do not adapting with output operand";
            const auto& target_shapes = std::vector<uint32_t>{
                (uint32_t)operand_shapes.at(1), (uint32_t)operand_shapes.at(2)};
            output_tensor->Reshape(target_shapes);
          }
        }
      }
    }
  }
}

bool RuntimeGraph::Build(const std::string& input_name,
                         const std::string& output_name) {
  if (graph_state_ == GraphState::Complete) {
    LOG(INFO) << "Model has been built already!";
    return true;
  }

  if (graph_state_ == GraphState::NeedInit) {
    bool init_success = Init();
    CHECK(init_success) << "Init graph failed";
    if (init_success) {
      return false;
    }
  }

  CHECK(graph_state_ >= GraphState::NeedBuild)
      << "Graph status error, current state is " << int(graph_state_);
  LOG_IF(ERROR, operators_.empty())
      << "Graph operators is empty, may be no init";

  InitOperatorInput(operators_);
  InitOperatorOutput(graph_->ops, operators_);

  Topo();

  CHECK(operators_topo_.size() == operators_.size())
      << "Build wrong topo queue";
  graph_state_ = GraphState::Complete;
  input_name_ = input_name;
  output_name_ = output_name;
  if (graph_ != nullptr) {
    graph_.reset();
    graph_ = nullptr;
  }

  return true;
}

void RuntimeAttribute::ClearWeight() {
  if (!this->weight_data.empty()) {
    std::vector<char> tmp = std::vector<char>();
    this->weight_data.swap(tmp);
  }
}

template <typename T>
std::vector<T> RuntimeAttribute::get(bool need_clear_weight) {
  CHECK(!weight_data.empty());
  CHECK(this->type != RuntimeDataType::kTypeUnknown);
  std::vector<T> weights;
  switch (this->type) {
    case RuntimeDataType::kTypeFloat32: {
      const bool is_float = std::is_same<T, float>::value;
      CHECK(is_float);
      const uint32_t float_size = sizeof(float);
      for (uint32_t i = 0; i < weight_data.size() / float_size; i++) {
        float weight = *((float*)weight_data.data() + i);
        weights.push_back(weight);
      }
      break;
    }
    default: {
      LOG(ERROR)
          << "0=null 1=f32 2=f64 3=f16 4=i32 5=i64 6=i16 7=i8 8=u8 9=bool";
      LOG(ERROR) << "Unknown weight data type: " << int(this->type);
    }
  }
  if (need_clear_weight) {
    this->ClearWeight();
  }
  return weights;
}

template <>
std::vector<float> RuntimeAttribute::get(bool need_clear_weight) {
  CHECK(!weight_data.empty());
  CHECK(this->type != RuntimeDataType::kTypeUnknown);
  std::vector<float> weights;
  switch (this->type) {
    case RuntimeDataType::kTypeFloat32: {
      const bool is_float = std::is_same<float, float>::value;
      CHECK(is_float);
      const uint32_t float_size = sizeof(float);
      for (uint32_t i = 0; i < weight_data.size() / float_size; i++) {
        float weight = *((float*)weight_data.data() + i);
        weights.push_back(weight);
      }
      break;
    }
    default: {
      LOG(ERROR)
          << "0=null 1=f32 2=f64 3=f16 4=i32 5=i64 6=i16 7=i8 8=u8 9=bool";
      LOG(ERROR) << "Unknown weight data type: " << int(this->type);
    }
  }
  if (need_clear_weight) {
    this->ClearWeight();
  }
  return weights;
}
}  // namespace free_infer
