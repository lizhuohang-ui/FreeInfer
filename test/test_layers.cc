
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "layer_factory.hpp"
#include "relu.hpp"
#include "sigmoid.hpp"
#include "tensor.hpp"
#include "layer_convolution.hpp"

TEST(TestLayer, ReLUForward) {
  using namespace free_infer;
  LOG(INFO) << "============================ReLUForward===============================";

  sftensor input_tensor = std::make_shared<Tensor<float>>(3, 4, 4);
  input_tensor->Rand();
  input_tensor->data() -= 0.5f;

  LOG(INFO) << input_tensor->data();

  std::vector<sftensor> inputs(1);
  std::vector<sftensor> outputs(1);
  inputs.at(0) = input_tensor;

  std::shared_ptr<RuntimeOperator> op = std::make_shared<RuntimeOperator>();
  op->type = "nn.ReLU";
  std::shared_ptr<Layer> layer;

  ASSERT_EQ(layer, nullptr);
  layer = LayerFactory::CreateLayer(op);
  ASSERT_NE(layer, nullptr);

  LOG(INFO) << layer->layer_name();

  layer->Forward(inputs, outputs);
  LOG(INFO) << "===========================After Forward===============================";

  for (const auto &output : outputs) {
    output->Show();
  }
}

TEST(TestLayer, SigmoidForward) {
  using namespace free_infer;
  LOG(INFO) << "============================SigmoidForward===============================";

  sftensor input_tensor = std::make_shared<Tensor<float>>(3, 4, 4);
  input_tensor->Rand();
  input_tensor->data() -= 0.5f;

  LOG(INFO) << input_tensor->data();

  std::vector<sftensor> inputs(1);
  std::vector<sftensor> outputs(1);
  inputs.at(0) = input_tensor;

  std::shared_ptr<RuntimeOperator> op = std::make_shared<RuntimeOperator>();
  op->type = "nn.Sigmoid";
  std::shared_ptr<Layer> layer;

  ASSERT_EQ(layer, nullptr);
  layer = LayerFactory::CreateLayer(op);
  ASSERT_NE(layer, nullptr);

  LOG(INFO) << layer->layer_name();

  layer->Forward(inputs, outputs);
  LOG(INFO) << "===========================After Forward===============================";

  for (const auto &output : outputs) {
    output->Show();
  }
}

TEST(TestLayer, ConvForward) {
  using namespace free_infer;
  const uint32_t batch_size = 1;
  std::vector<sftensor> inputs(batch_size);
  std::vector<sftensor> outputs(batch_size);

  const uint32_t in_channel = 2;
  for (uint32_t i = 0; i < batch_size; ++i) {
    sftensor input = std::make_shared<Tensor<float>>(in_channel, 4, 4);
    input->data().slice(0) = "1,2,3,4;"
                             "5,6,7,8;"
                             "9,10,11,12;"
                             "13,14,15,16;";

    input->data().slice(1) = "1,2,3,4;"
                             "5,6,7,8;"
                             "9,10,11,12;"
                             "13,14,15,16;";
    inputs.at(i) = input;
  }
  const uint32_t kernel_h = 3;
  const uint32_t kernel_w = 3;
  const uint32_t stride_h = 1;
  const uint32_t stride_w = 1;
  const uint32_t kernel_count = 2;
  std::vector<sftensor> weights;
  for (uint32_t i = 0; i < kernel_count; ++i) {
    sftensor kernel = std::make_shared<Tensor<float>>(in_channel, kernel_h, kernel_w);
    kernel->data().slice(0) = arma::fmat("1,2,3;"
                                         "3,2,1;"
                                         "1,2,3;");
    kernel->data().slice(1) = arma::fmat("1,2,3;"
                                         "3,2,1;"
                                         "1,2,3;");
    weights.push_back(kernel);
  }
  ConvolutionLayer conv_layer(kernel_count, in_channel, kernel_h, kernel_w, 0,
                              0, stride_h, stride_w, 1, false);
  conv_layer.set_weights(weights);
  conv_layer.Forward(inputs, outputs);
  outputs.at(0)->Show();
}

TEST(TestLayer, MaxPoolingForward) {
  using namespace free_infer;
  std::shared_ptr<RuntimeOperator> op = std::make_shared<RuntimeOperator>();
  op->type = "nn.MaxPool2d";
  std::vector<int> strides{2, 2};

  std::shared_ptr<RuntimeParameter> stride_param = std::make_shared<RuntimeParameterIntArray>(strides);

  op->params.insert({"stride", stride_param});

  std::vector<int> kernel{2, 2};
  std::shared_ptr<RuntimeParameter> kernel_param = std::make_shared<RuntimeParameterIntArray>(strides);
  op->params.insert({"kernel_size", kernel_param});

  std::vector<int> paddings{0, 0};
  std::shared_ptr<RuntimeParameter> padding_param = std::make_shared<RuntimeParameterIntArray>(paddings);
  op->params.insert({"padding", padding_param});

  std::shared_ptr<Layer> layer;
  layer = LayerFactory::CreateLayer(op);
  ASSERT_NE(layer, nullptr);

  sftensor tensor = std::make_shared<Tensor<float>>(1, 4, 4);
  arma::fmat input = arma::fmat("1,2,3,4;"
                                "2,3,4,5;"
                                "3,4,5,6;"
                                "4,5,6,7");
  tensor->data().slice(0) = input;
  std::vector<sftensor> inputs(1);
  inputs.at(0) = tensor;
  std::vector<sftensor> outputs(1);
  layer->Forward(inputs, outputs);

  ASSERT_EQ(outputs.size(), 1);
  outputs.front()->Show();
}