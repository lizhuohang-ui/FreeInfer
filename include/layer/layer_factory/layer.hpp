
#ifndef __FREE_INFER_LAYER_HPP__
#define __FREE_INFER_LAYER_HPP__
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "runtime_ir.hpp"
#include "status_code.hpp"
#include "tensor.hpp"
namespace free_infer {
class Layer {
 public:
  explicit Layer(std::string layer_name) : layer_name_(std::move(layer_name)) {}
  virtual ~Layer() = default;

  virtual InferStatus Forward(const std::vector<sftensor>& inputs,
                              std::vector<sftensor>& outputs);

 protected:
  std::weak_ptr<RuntimeOperator> runtime_operator_;
  std::string layer_name_;
};
}  // namespace free_infer
#endif  // __FREE_INFER_LAYER_HPP__