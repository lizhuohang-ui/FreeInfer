
#ifndef __FREE_INFER_LAYER_MAXPOOLING_HPP__
#define __FREE_INFER_LAYER_MAXPOOLING_HPP__

#include <cstdint>

#include "layer_pooling.hpp"
namespace free_infer {
class MaxPoolingLayer : public PoolingLayer {
 public:
  explicit MaxPoolingLayer(uint32_t pooling_size_h, uint32_t pooling_size_w,
                           uint32_t padding_h, uint32_t padding_w,
                           uint32_t stride_h, uint32_t stride_w)
      : PoolingLayer("MaxPooling", pooling_size_h, pooling_size_w, padding_h,
                     padding_w, stride_h, stride_w) {}

  InferStatus Forward(const std::vector<sftensor>& inputs,
                      std::vector<sftensor>& outputs) override;

  static ParseParameterAttrStatus GetInstace(
      const std::shared_ptr<RuntimeOperator>& op,
      std::shared_ptr<Layer>& maxpooling_layer);
};
}  // namespace free_infer

#endif  //__FREE_INFER_LAYER_MAXPOOLING_HPP__