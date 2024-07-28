#ifndef __FREE_INFER_LAYER_CONVOLUTION_HPP__
#define __FREE_INFER_LAYER_CONVOLUTION_HPP__

#include <sys/types.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "layer.hpp"
#include "runtime_ir.hpp"
#include "status_code.hpp"
#include "tensor.hpp"
namespace free_infer {
class ConvolutionLayer : public Layer {
 public:
  explicit ConvolutionLayer(uint32_t output_channel, uint32_t in_channel,
                            uint32_t kernel_h, uint32_t kernerl_w,
                            uint32_t padding_h, uint32_t padding_w,
                            uint32_t stride_h, uint32_t stride_w,
                            uint32_t groups, bool use_bias = true);

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
      std::shared_ptr<Layer>& conv_layer);

 private:
  void InitWeightParam(const uint32_t kernel_n, const uint32_t kernel_c,
                       const uint32_t kernel_h, const uint32_t kernel_w);
  void InitBiasParam(const uint32_t bias_n, const uint32_t bias_c,
                     const uint32_t bias_h, const uint32_t bias_w);

  void InitIm2ColKernel();
  void CheckWeighsDim();
  arma::fmat Im2Col(sftensor input, uint32_t kernel_h, uint32_t kernel_w,
                    uint32_t input_h, uint32_t input_w, uint32_t input_c_group,
                    uint32_t group_i, uint32_t im2col_w, uint32_t im2col_h);
  void ConvGemm(const arma::fmat& im2col_input, sftensor output, uint32_t group,
                uint32_t kernel_i, uint32_t kernel_n_group,
                const arma::frowvec& kernel, uint32_t output_w,
                uint32_t output_h);

 private:
  bool use_bias_ = false;
  uint32_t groups_ = 1;
  uint32_t padding_h_ = 1;
  uint32_t padding_w_ = 1;
  float padding_value = 0.f;
  uint32_t stride_h_ = 1;
  uint32_t stride_w_ = 1;
  std::vector<arma::frowvec> im2col_kernel;

 protected:
  std::vector<sftensor> weights_;
  std::vector<sftensor> bias_;
};

}  // namespace free_infer

#endif  // __FREE_INFER_LAYER_CONVOLUTION_HPP__
