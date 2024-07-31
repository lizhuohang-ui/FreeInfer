
#ifndef __FREE_INFER_LAYER_EXPRESSION_HPP__
#define __FREE_INFER_LAYER_EXPRESSION_HPP__

#include <memory>
#include <string>

#include "layer.hpp"
#include "parse_expression.hpp"

namespace free_infer {
class ExpressionLayer : public Layer {
 public:
  explicit ExpressionLayer(std::string statement);
  InferStatus Forward(const std::vector<sftensor>& inputs,
                      std::vector<sftensor>& outputs) override;

  static ParseParameterAttrStatus GetInstace(
      const std::shared_ptr<RuntimeOperator>& op,
      std::shared_ptr<Layer>& expression_layer);
  private:
  std::string statement_;
  std::unique_ptr<ExpressionParser> parser_;
};

}  // namespace free_infer

#endif  // __FREE_INFER_LAYER_EXPRESSION_HPP__