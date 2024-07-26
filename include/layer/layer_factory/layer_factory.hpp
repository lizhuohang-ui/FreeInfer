
#ifndef __FREE_INFER_LAYER_FACTORY_HPP__
#define __FREE_INFER_LAYER_FACTORY_HPP__

#include <map>
#include <memory>
#include <string>

#include "gtest/gtest.h"
#include "runtime_ir.hpp"
#include "status_code.hpp"

namespace free_infer {
class LayerFactory {
 public:
  typedef ParseParameterAttrStatus (*Creator)(const std::shared_ptr<RuntimeOperator>& op,
                                              std::shared_ptr<Layer>& layer);
  typedef std::map<std::string, Creator> LayerRegistry;

 public:
  LayerFactory(){}
  static LayerFactory& Registry();
  void RegisterCreator(const std::string& layer_type, Creator creator);
  static std::shared_ptr<Layer> CreateLayer(const std::shared_ptr<RuntimeOperator>& op);

  LayerFactory(const LayerFactory&) = delete;
  LayerFactory& operator=(const LayerFactory&) = delete;

 private:
  LayerRegistry layer_registry_;
};

class LayerReigister {
 public:
  LayerReigister(const std::string& layer_type, LayerFactory::Creator creator) {
    LayerFactory::Registry().RegisterCreator(layer_type, creator);
  }
};

}  // namespace free_infer

#endif  //__FREE_INFER_LAYER_FACTORY_IR_HPP__