
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "layer_factory.hpp"
#include "relu.hpp"
#include "sigmoid.hpp"
#include "tensor.hpp"

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