
#ifndef __FREE_INFER_SOFTMAX_LAYER_HPP__
#define __FREE_INFER_SOFTMAX_LAYER_HPP__

#include <memory>
#include <string>
#include <vector>

#include "layer_activiation.hpp"
#include "runtime/runtime_ir.hpp"
#include "runtime/status_code.hpp"
#include "tensor/tensor.hpp"

namespace free_infer {
class SoftmaxLayer : public ActiviationLayer {
 public:
  explicit SoftmaxLayer() : ActiviationLayer("softmax") {}

  InferStatus Forward(const std::vector<sftensor>& inputs,
                      std::vector<sftensor>& outputs) override;

  static ParseParameterAttrStatus GetInstace(
      const std::shared_ptr<RuntimeOperator>& op,
      std::shared_ptr<Layer>& softmax_layer);

 private:
};
}  // namespace free_infer

#endif  //  __FREE_INFER_SOFTMAX_LAYER_HPP__
