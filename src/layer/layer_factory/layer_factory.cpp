#include "layer_factory.hpp"

#include <memory>
#include <string>

#include "runtime_ir.hpp"

namespace free_infer {

LayerFactory& LayerFactory::Registry() {
  static LayerFactory layer_factory;
  return layer_factory;
}

void LayerFactory::RegisterCreator(const std::string& layer_type,
                                   Creator creator) {
  layer_registry_[layer_type] = creator;
}

std::shared_ptr<Layer> LayerFactory::CreateLayer(
    const std::shared_ptr<RuntimeOperator>& op) {
  const std::string& layer_type = op->type;
  LOG_IF(ERROR, layer_registry_.count(layer_type) <= 0)
      << "Can not find the layer type: " << layer_type;
  const auto& creator = layer_registry_.find(layer_type)->second;
  std::shared_ptr<Layer> layer;
  const auto& status = creator(op, layer);
  LOG_IF(FATAL, status != ParseParameterAttrStatus::kParameterAttrParseSuccess)
      << "Create the layer: " << layer_type
      << " failed, error code: " << int(status);
  return layer;
}
}  // namespace free_infer
