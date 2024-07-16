#ifndef __FREE_INFER_RUNTIME_IR_HPP__
#define __FREE_INFER_RUNTIME_IR_HPP__

#include <glog/types.h>

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ir.h"
#include "status_code.hpp"
#include "tensor.hpp"

namespace free_infer {

class RuntimeOperator;
class RuntimeOperand;

class RuntimeParameter;

class RuntimeAttribute;
class Layer;  // No completing

class RuntimeGraph {
 public:
  RuntimeGraph(std::string param_path, std::string bin_path);
  void set_bin_path(const std::string& bin_path);
  void set_param_path(const std::string& param_path);
  const std::string& bin_path() const;
  const std::string& param_path() const;
  const std::vector<std::shared_ptr<RuntimeOperator>>& operators() const;
  bool Init();

 private:
  static void InitGraphOperatorsInput(
      const std::vector<pnnx::Operand*>& inputs,
      const std::shared_ptr<RuntimeOperator>& runtime_operator);

  static void InitGraphOperatorsOutput(
      const std::vector<pnnx::Operand*>& outputs,
      const std::shared_ptr<RuntimeOperator>& runtime_operator);

  static void InitGraphOperatorsParam(
      const std::map<std::string, pnnx::Parameter>& params,
      const std::shared_ptr<RuntimeOperator>& runtime_operator);

  static void InitGraphOperatorsAttr(
      const std::map<std::string, pnnx::Attribute>& attrs,
      const std::shared_ptr<RuntimeOperator>& runtime_operator);

 private:
  std::string input_name_;
  std::string output_name_;
  std::string param_path_;
  std::string bin_path_;
  std::vector<std::shared_ptr<RuntimeOperator>> operators_;
  std::map<std::string, std::shared_ptr<RuntimeOperator>> operators_maps_;

  std::unique_ptr<pnnx::Graph> graph_;  // graph in pnnx
};

class RuntimeOperator {
 public:
  virtual ~RuntimeOperator();

  bool has_forward = false;
  std::string type;
  std::string name;

  std::vector<std::shared_ptr<RuntimeOperand>> input_operands;
  std::map<std::string, std::shared_ptr<RuntimeOperand>> input_operands_maps;

  std::vector<std::string> output_names;
  std::shared_ptr<RuntimeOperand> output_opearands;
  std::map<std::string, std::shared_ptr<RuntimeOperator>> output_operators_maps;

  std::shared_ptr<Layer> layer;

  std::map<std::string, RuntimeParameter*> params;
  std::map<std::string, std::shared_ptr<RuntimeAttribute>> attrs;
};

class RuntimeOperand {
 public:
  std::string name;
  RuntimeDataType type = RuntimeDataType::kTypeUnknown;
  std::vector<int> shapes;
  std::vector<std::shared_ptr<Tensor<float>>> datas;
};

class RuntimeParameter {
 public:
  virtual ~RuntimeParameter() = default;
  explicit RuntimeParameter(
      RuntimeParameterType type = RuntimeParameterType::kParameterUnknown)
      : type(type) {}

 private:
  RuntimeParameterType type = RuntimeParameterType::kParameterUnknown;
};

template <class T>
class RuntimeParameterTyped : public RuntimeParameter {
 public:
  explicit RuntimeParameterTyped(RuntimeParameterType type)
      : RuntimeParameter(type) {}
  T value;
};

class RuntimeAttribute {
 public:
  std::vector<char> weight_data;
  std::vector<int> shape;
  RuntimeDataType type = RuntimeDataType::kTypeUnknown;
  template <class T>
  std::vector<T> get(bool need_clear_weight = true);
  void ClearWeight();
};

using RuntimeParameterInt = RuntimeParameterTyped<int>;
using RuntimeParameterFloat = RuntimeParameterTyped<float>;
using RuntimeParameterString = RuntimeParameterTyped<std::string>;
using RuntimeParameterBool = RuntimeParameterTyped<bool>;
using RuntimeParameterIntArray = RuntimeParameterTyped<std::vector<int>>;
using RuntimeParameterFloatArray = RuntimeParameterTyped<std::vector<float>>;
using RuntimeParameterStringArray =
    RuntimeParameterTyped<std::vector<std::string>>;

}  // namespace free_infer

#endif  // __FREE_INFER_RUNTIME_IR_HPP__