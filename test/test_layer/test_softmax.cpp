#include <glog/logging.h>
#include <gtest/gtest.h>

#include <layer/layer.hpp>
#include <layer/softmax.hpp>
#include <layer/layer_factory.hpp>
TEST(TestLayer, SoftmaxForward) {
  using namespace free_infer;
  SoftmaxLayer softmax_layer;
  LOG(INFO) << "============================SoftmaxForward====================="
               "==========";

  sftensor input_tensor = std::make_shared<Tensor<float>>(1, 1, 1000);
  input_tensor->Ones();

  LOG(INFO) << input_tensor->data();

  std::vector<sftensor> inputs(1);
  std::vector<sftensor> outputs(1);
  inputs.at(0) = input_tensor;

  std::shared_ptr<RuntimeOperator> op = std::make_shared<RuntimeOperator>();
  op->type = "nn.Softmax";
  std::shared_ptr<Layer> layer;

  ASSERT_EQ(layer, nullptr);
  layer = LayerFactory::CreateLayer(op);
  ASSERT_NE(layer, nullptr);

  LOG(INFO) << layer->layer_name();

  layer->Forward(inputs, outputs);
  LOG(INFO) << "===========================After "
               "Forward===============================";

  for (const auto &output : outputs) {
    output->Show();
  }
}