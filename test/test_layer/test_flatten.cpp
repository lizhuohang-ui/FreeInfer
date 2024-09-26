
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <layer/layer.hpp>
#include <layer/layer_factory.hpp>

#include <layer/flatten.hpp>

TEST(TestLayer, FlattenForward) {
  using namespace free_infer;
  FlattenLayer flatten_layer;
  LOG(INFO)
      << "============================FlattenForward========================";

  std::shared_ptr<RuntimeOperator> op = std::make_shared<RuntimeOperator>();
  op->type = "torch.flatten";

  int start_dim = 1;
  int end_dim = -1;

  std::shared_ptr<RuntimeParameter> start_dim_param =
      std::make_shared<RuntimeParameterInt>(start_dim);
  std::shared_ptr<RuntimeParameter> end_dim_param =
      std::make_shared<RuntimeParameterInt>(end_dim);

  op->params.insert({"start_dim", start_dim_param});
  op->params.insert({"end_dim", end_dim_param});

  std::shared_ptr<Layer> layer;
  layer = LayerFactory::CreateLayer(op);
  ASSERT_NE(layer, nullptr);

  sftensor input = std::make_shared<Tensor<float>>(512, 1, 1);
  input->Fill(1);
  std::vector<sftensor> inputs;
  inputs.push_back(input);

  std::vector<sftensor> outputs(1);
  layer->Forward(inputs, outputs);

  for (uint32_t i = 0; i < 3; ++i) {
    LOG(INFO) << outputs.front()->shapes()[i];
  }
}