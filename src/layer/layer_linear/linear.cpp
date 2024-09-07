#include "linear.hpp"

#include <sys/types.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "layer.hpp"
#include "layer_factory.hpp"
#include "status_code.hpp"
#include "tensor.hpp"
namespace free_infer {
LinearLayer::LinearLayer(uint32_t in_features, uint32_t out_features,
                         bool use_bias)
    : Layer("Linear"),
      in_features_(in_features),
      out_features_(out_features),
      use_bias_(use_bias) {
  this->InitWeightParam(in_features, out_features);
  this->InitBiasParam(out_features);
}

const std::vector<sftensor>& LinearLayer::weights() const {
  return this->weights_;
}

const std::vector<sftensor>& LinearLayer::bias() const { return this->bias_; }

void LinearLayer::set_weights(const std::vector<sftensor>& weights) {
  CHECK(weights.size() == weights_.size());
  for (uint32_t i = 0; i < weights.size(); ++i) {
    CHECK(this->weights_.at(i) != nullptr);
    CHECK(this->weights_.at(i)->rows() == weights.at(i)->rows());
    CHECK(this->weights_.at(i)->cols() == weights.at(i)->cols());
    CHECK(this->weights_.at(i)->channels() == weights.at(i)->channels());
  }
  this->weights_ = weights;
}

void LinearLayer::set_weights(const std::vector<float>& weights) {
  const uint32_t elem_size = weights.size();

  uint32_t weight_size = 0;
  const uint32_t batch_size = this->weights_.size();
  for (uint32_t i = 0; i < batch_size; ++i) {
    weight_size += this->weights_.at(i)->size();
  }

  CHECK_EQ(weight_size, elem_size);
  CHECK_EQ(elem_size % batch_size, 0);
  const uint32_t blob_size = elem_size / batch_size;
  for (uint32_t idx = 0; idx < batch_size; ++idx) {
    const uint32_t start_offset = idx * blob_size;
    const uint32_t end_offset = start_offset + blob_size;
    const auto& sub_values = std::vector<float>{weights.begin() + start_offset,
                                                weights.begin() + end_offset};
    this->weights_.at(idx)->Fill(sub_values);
  }
}

void LinearLayer::set_bias(const std::vector<sftensor>& bias) {
  if (!this->bias_.empty()) {
    CHECK(bias.size() == bias_.size());
    for (uint32_t i = 0; i < bias.size(); ++i) {
      CHECK(this->bias_.at(i) != nullptr);
      CHECK(this->bias_.at(i)->rows() == bias.at(i)->rows());
      CHECK(this->bias_.at(i)->cols() == bias.at(i)->cols());
      CHECK(this->bias_.at(i)->channels() == bias.at(i)->channels());
    }
    this->bias_ = bias;
  }
}

void LinearLayer::set_bias(const std::vector<float>& bias) {
  const uint32_t elem_size = bias.size();

  uint32_t bias_size = 0;
  const uint32_t batch_size = this->bias_.size();
  for (uint32_t i = 0; i < batch_size; ++i) {
    bias_size += this->bias_.at(i)->size();
  }

  CHECK_EQ(bias_size, elem_size);
  CHECK_EQ(elem_size % batch_size, 0);

  const uint32_t blob_size = elem_size / batch_size;
  for (uint32_t idx = 0; idx < batch_size; ++idx) {
    const uint32_t start_offset = idx * blob_size;
    const uint32_t end_offset = start_offset + blob_size;
    const auto& sub_values = std::vector<float>{bias.begin() + start_offset,
                                                bias.begin() + end_offset};
    this->bias_.at(idx)->Fill(sub_values);
  }
}

InferStatus LinearLayer::Forward(const std::vector<sftensor>& inputs,
                                 std::vector<sftensor>& outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array in the linear layer is empty";
    return InferStatus::kInferFailedInputEmpty;
  }

  if (inputs.size() != outputs.size()) {
    LOG(ERROR) << "The input and output tensor array size of the linear "
                  "layer do not match";
    return InferStatus::kInferFailedInputOutSizeMatchError;
  }

  if (weights_.empty()) {
    LOG(ERROR) << "The number of kernel matrix in the linear layer should "
                  "be greater than zero";
    return InferStatus::kInferFailedWeightParameterError;
  }

  if (this->use_bias_ && this->bias_.size() != this->weights_.size()) {
    LOG(ERROR) << "The number of kernel matrix and bias matrix do not match";
    return InferStatus::kInferFailedBiasParameterError;
  }

  const uint32_t batch_size = inputs.size();
  const sftensor& weight_data = weights_.front();
  const arma::fmat weight(weight_data->raw_ptr(), out_features_, in_features_,
                          false, true);
  const arma::fmat& weight_t = weight.t();

  for (uint32_t i = 0; i < batch_size; ++i) {
    const sftensor& input = inputs.at(i);
    sftensor output = outputs.at(i);

    CHECK(input != nullptr && !input->empty())
        << "The input tensor array in the linear layer has an empty  "
           "tensor "
        << i << " batch";

    const uint32_t input_c = input->channels();
    const uint32_t input_h = input->rows();
    const uint32_t input_w = input->cols();

    if (output == nullptr && output->empty()) {
      output = std::make_shared<Tensor<float>>(1, input_h, out_features_);
      outputs[i] = output;
    }

    CHECK(input_c == 1 && input_w == in_features_);

    arma::fmat input_vec((float*)input->raw_ptr(), input_h, in_features_, false,
                         true);
    arma::fmat& result = output->slice(0);
    result = input_vec * weight_t;
    if (use_bias_) {
      const auto& bias_data = bias_.front()->data();
      const auto& bias = bias_data.slice(0);
      for (uint32_t r = 0; r < result.n_rows; ++r) {
        result.row(r) += bias;
      }
    }
  }
  return InferStatus::kInferSuccess;
}

ParseParameterAttrStatus LinearLayer::GetInstace(
    const std::shared_ptr<RuntimeOperator>& op,
    std::shared_ptr<Layer>& linear_layer) {
  CHECK(op != nullptr);
  const std::map<std::string, std::shared_ptr<RuntimeParameter>>& params =
      op->params;

  if (params.find("in_features") == params.end()) {
    LOG(ERROR) << "Can not find the in_features parameter";
    return ParseParameterAttrStatus::kAttrMissingInFeatures;
  }
  auto in_features =
      std::dynamic_pointer_cast<RuntimeParameterInt>(params.at("in_features"));
  if (!in_features) {
    LOG(ERROR) << "Can not find the in_features parameter";
    return ParseParameterAttrStatus::kAttrMissingInFeatures;
  }

  if (params.find("out_features") == params.end()) {
    LOG(ERROR) << "Can not find the out_features parameter";
    return ParseParameterAttrStatus::kAttrMissingOutFeatures;
  }
  auto out_features =
      std::dynamic_pointer_cast<RuntimeParameterInt>(params.at("out_features"));
  if (!out_features) {
    LOG(ERROR) << "Can not find the out_features parameter";
    return ParseParameterAttrStatus::kParameterMissingOutChannel;
  }

  if (params.find("bias") == params.end()) {
    LOG(ERROR) << "Can not find the bias parameter";
    return ParseParameterAttrStatus::kParameterMissingUseBias;
  }
  auto use_bias =
      std::dynamic_pointer_cast<RuntimeParameterBool>(params.at("bias"));
  if (!use_bias) {
    LOG(ERROR) << "Can not find the bias parameter";
    return ParseParameterAttrStatus::kParameterMissingUseBias;
  }

  linear_layer = std::make_shared<LinearLayer>(
      in_features->value, out_features->value, use_bias->value);

  auto linear_layer_derived =
      std::dynamic_pointer_cast<LinearLayer>(linear_layer);

  // load weights

  const std::map<std::string, std::shared_ptr<RuntimeAttribute>>& attrs =
      op->attrs;
  if (use_bias->value) {
    if (attrs.find("bias") == attrs.end()) {
      LOG(ERROR) << "Can not find the bias attribute";
      return ParseParameterAttrStatus::kAttrMissingBias;
    }
    const auto& bias = attrs.at("bias");
    const std::vector<int>& bias_shape = bias->shape;
    if (bias_shape.empty() || bias_shape.at(0) != out_features->value) {
      LOG(ERROR) << "The attribute of bias shape is wrong";
      return ParseParameterAttrStatus::kAttrMissingBias;
    }

    const std::vector<float>& bias_values = bias->get<float>();
    linear_layer_derived->set_bias(bias_values);
  }

  if (attrs.find("weight") == attrs.end()) {
    LOG(ERROR) << "Can not find the weight attribute";
    return ParseParameterAttrStatus::kAttrMissingWeight;
  }

  const auto& weight = attrs.at("weight");
  const std::vector<int>& weight_shape = weight->shape;
  if (weight_shape.empty()) {
    LOG(ERROR) << "The attribute of weight shape is wrong";
    return ParseParameterAttrStatus::kAttrMissingWeight;
  }

  const std::vector<float>& weight_values = weight->get<float>();
  linear_layer_derived->set_weights(weight_values);
  return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

void LinearLayer::InitWeightParam(const uint32_t in_features,
                                  const uint32_t out_features) {
  this->weights_ = std::vector<sftensor>(1);
  this->weights_.at(0) =
      std::make_shared<Tensor<float>>(1, out_features, in_features);
}

void LinearLayer::InitBiasParam(const uint32_t out_features) {
  this->bias_ = std::vector<sftensor>(1);
  this->bias_.at(0) = std::make_shared<Tensor<float>>(1, 1, out_features);
}

LayerReigister kLinearGetInstace("nn.Linear", LinearLayer::GetInstace);

}  // namespace free_infer