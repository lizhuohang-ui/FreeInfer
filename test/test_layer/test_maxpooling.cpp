#include <glog/logging.h>
#include <gtest/gtest.h>

#include <layer/layer.hpp>
#include <layer/maxpooling.hpp>
#include <layer/layer_factory.hpp>

TEST(TestLayer, MaxPoolingForward) {
  using namespace free_infer;
  MaxPoolingLayer maxpooling_layer(2, 2, 0, 0, 2, 2);

  std::shared_ptr<RuntimeOperator> op = std::make_shared<RuntimeOperator>();
  op->type = "nn.MaxPool2d";
  std::vector<int> strides{2, 2};

  std::shared_ptr<RuntimeParameter> stride_param =
      std::make_shared<RuntimeParameterIntArray>(strides);

  op->params.insert({"stride", stride_param});

  std::vector<int> kernel{2, 2};
  std::shared_ptr<RuntimeParameter> kernel_param =
      std::make_shared<RuntimeParameterIntArray>(strides);
  op->params.insert({"kernel_size", kernel_param});

  std::vector<int> paddings{0, 0};
  std::shared_ptr<RuntimeParameter> padding_param =
      std::make_shared<RuntimeParameterIntArray>(paddings);
  op->params.insert({"padding", padding_param});

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