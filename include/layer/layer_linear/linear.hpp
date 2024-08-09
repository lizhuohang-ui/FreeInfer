
#ifndef __FREE_INFER_LAYER_LINEAR_HPP__
#define __FREE_INFER_LAYER_LINEAR_HPP__

#include <cstdint>
#include "layer.hpp"
namespace free_infer {
class LinearLayer : public Layer {
 public:
  explicit LinearLayer(uint32_t in_features, uint32_t out_features, bool use_bias);
  const std::vector<sftensor>& weights() const;
  const std::vector<sftensor>& bias() const;

  void set_weights(const std::vector<sftensor>& weights);
  void set_weights(const std::vector<float>& weights);
  void set_bias(const std::vector<sftensor>& bias);
  void set_bias(const std::vector<float>& bias);

  InferStatus Forward(const std::vector<sftensor>& inputs,
                      std::vector<sftensor>& outputs) override;

  static ParseParameterAttrStatus GetInstace(
      const std::shared_ptr<RuntimeOperator>& op,
      std::shared_ptr<Layer>& linear_layer);

 private:
  void InitWeightParam(const uint32_t in_features, const uint32_t out_features);
  void InitBiasParam(const uint32_t out_features);

 private:
  bool use_bias_ = false;
  uint32_t in_features_;
  uint32_t out_features_;
  std::vector<sftensor> weights_;
  std::vector<sftensor> bias_;
};

}  // namespace free_infer

#endif