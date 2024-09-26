
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <layer/layer.hpp>
#include <layer/layer_factory.hpp>
#include <layer/adaptive_avgpooling.hpp>

TEST(TestLayer, AdaptiveAvgPoolingForward) {
  using namespace free_infer;
  AdaptiveAvgPoolingLayer adaptive_avgpooling_layer(1, 1);
  std::shared_ptr<RuntimeOperator> op = std::make_shared<RuntimeOperator>();
  op->type = "nn.AdaptiveAvgPool2d";

  std::vector<int> output_size{1, 1};
  std::shared_ptr<RuntimeParameter> output_size_param =
      std::make_shared<RuntimeParameterIntArray>(output_size);
  op->params.insert({"output_size", output_size_param});

  std::shared_ptr<Layer> layer;
  layer = LayerFactory::CreateLayer(op);
  ASSERT_NE(layer, nullptr);

  sftensor tensor = std::make_shared<Tensor<float>>(1, 4, 4);
  arma::fmat input = arma::fmat(
      "1,2,3,4;"
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