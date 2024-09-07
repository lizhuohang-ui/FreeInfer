
#ifndef __FREE_INFER_LAYER_FLATTEN_HPP__
#define __FREE_INFER_LAYER_FLATTEN_HPP__

#include <cstdint>

#include "layer.hpp"
#include "runtime_ir.hpp"
namespace free_infer {
class FlattenLayer : public Layer {
 public:
  explicit FlattenLayer(uint32_t start_dim = 1, uint32_t end_dim = -1);

  InferStatus Forward(const std::vector<sftensor>& inputs,
                      std::vector<sftensor>& outputs) override;

  static ParseParameterAttrStatus GetInstace(
      const std::shared_ptr<RuntimeOperator>& op,
      std::shared_ptr<Layer>& linear_layer);

 private:
  uint32_t start_dim_;
  uint32_t end_dim_;
};

}  // namespace free_infer

#endif  //__FREE_INFER_LAYER_FLATTEN_HPP__