#ifndef __FREE_INFER_LAYER_CONVOLUTION_HPP__
#define __FREE_INFER_LAYER_CONVOLUTION_HPP__

#include <string>
#include "layer.hpp"
namespace free_infer {
class ConvolutionLayer : public Layer {
 public:
  explicit ConvolutionLayer (std::string layer_name) : Layer(layer_name) {}

};
}  // namespace free_infer

#endif  // __FREE_INFER_LAYER_CONVOLUTION_HPP__
