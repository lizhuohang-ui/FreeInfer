
#ifndef __FREE_INFER_LAYER_POOLING_HPP__
#define __FREE_INFER_LAYER_POOLING_HPP__

#include <cstdint>
#include <string>

#include "layer.hpp"
namespace free_infer {
class PoolingLayer : public Layer {
 public:
  explicit PoolingLayer(std::string layer_name, uint32_t pooling_size_h,
                        uint32_t pooling_size_w, uint32_t padding_h,
                        uint32_t padding_w, uint32_t stride_h,
                        uint32_t stride_w)
      : Layer(layer_name),
        padding_h_(padding_h),
        padding_w_(padding_w),
        pooling_size_h_(pooling_size_h),
        pooling_size_w_(pooling_size_w),
        stride_h_(stride_h),
        stride_w_(stride_w) {}

 protected:
  uint32_t padding_h_ = 0;
  uint32_t padding_w_ = 0;
  uint32_t pooling_size_h_ = 0;
  uint32_t pooling_size_w_ = 0;
  uint32_t stride_h_ = 1;
  uint32_t stride_w_ = 1;
};
}  // namespace free_infer

#endif  //__FREE_INFER_LAYER_POOLING_HPP__