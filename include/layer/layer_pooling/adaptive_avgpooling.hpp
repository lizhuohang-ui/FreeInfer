
#ifndef __FREE_INFER_LAYER_ADAPTIVE_AVGPOOLING_HPP__
#define __FREE_INFER_LAYER_ADAPTIVE_AVGPOOLING_HPP__
#include <cstdint>

#include "layer.hpp"
#include "layer_pooling.hpp"
namespace free_infer {
class AdaptiveAvgPoolingLayer : public Layer {
 public:
  explicit AdaptiveAvgPoolingLayer(uint32_t ouput_h, uint32_t output_w)
      : Layer("AdaptiveAvgPooling"), output_h_(ouput_h), output_w_(output_w) {}

  InferStatus Forward(const std::vector<sftensor>& inputs,
                      std::vector<sftensor>& outputs) override;

  static ParseParameterAttrStatus GetInstace(
      const std::shared_ptr<RuntimeOperator>& op,
      std::shared_ptr<Layer>& adaptive_avgpooling_layer);

 private:
  uint32_t output_h_;
  uint32_t output_w_;
};
}  // namespace free_infer
#endif  // __FREE_INFER_LAYER_ADAPTIVE_AVGPOOLING_HPP__