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
 private:
  bool has_forward = false;
  std::string type_;
  std::string name_;

  std::vector<std::shared_ptr<RuntimeOperand>> input_opearands_;
  std::map<std::string, std::shared_ptr<RuntimeOperand>> input_opearands_maps;

  std::vector<std::string> output_names_;
  std::shared_ptr<RuntimeOperand> output_opearands_;
  std::map<std::string, std::shared_ptr<RuntimeOperator>> output_operators_;

  std::shared_ptr<Layer> layer_;

  std::map<std::string, RuntimeParameter*> params_;
  std::map<std::string, std::shared_ptr<RuntimeAttribute>> attrs_;
};

class RuntimeOperand {
 private:
  std::string name;
  int type;
  std::vector<int> shapes;
  std::vector<std::shared_ptr<Tensor<float>>> datas;
};

class RuntimeParameter {
 public:
  virtual ~RuntimeParameter() = default;
  explicit RuntimeParameter(RuntimeParameterType type) : type(type) {}

 private:
  RuntimeParameterType type;
};

template <class T>
class RuntimeParameterTyped : public RuntimeParameter {
 public:
  explicit RuntimeParameterTyped(RuntimeParameterType type)
      : RuntimeParameter(type) {}
  T value;
};

class RuntimeAttribute {
 private:
  std::vector<char> weight_data;
  std::vector<int> shape;
  RuntimeDataType type = RuntimeDataType::kTypeUnknown;

 public:
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