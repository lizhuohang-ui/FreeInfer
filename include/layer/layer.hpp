
#ifndef __FREE_INFER_LAYER_HPP__
#define __FREE_INFER_LAYER_HPP__
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "runtime/runtime_ir.hpp"
#include "runtime/status_code.hpp"
#include "tensor/tensor.hpp"
namespace free_infer {
class Layer {
 public:
  explicit Layer(std::string layer_name) : layer_name_(std::move(layer_name)) {}
  virtual ~Layer() = default;

  virtual InferStatus Forward(const std::vector<sftensor>& inputs, std::vector<sftensor>& outputs);
  virtual InferStatus Forward();

  virtual const std::string& layer_name() const { return this->layer_name_; }
  void set_runtime_operator(const std::shared_ptr<RuntimeOperator>& runtime_operator);

 protected:
  std::weak_ptr<RuntimeOperator> runtime_operator_;
  std::string layer_name_;
};
}  // namespace free_infer
#endif  // __FREE_INFER_LAYER_HPP__