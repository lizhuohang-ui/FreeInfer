
#ifndef __FREE_INFER_LAYER_ACTIVIATION_HPP__
#define __FREE_INFER_LAYER_ACTIVIATION_HPP__

#include <string>
#include "layer.hpp"
namespace free_infer {
class ActiviationLayer : public Layer {
 public:
  explicit ActiviationLayer(std::string layer_name) : Layer(layer_name) {}
  std::string test2 = "test2";
};
}  // namespace free_infer

#endif  //__FREE_INFER_LAYER_ACTIVIATION_HPP__