
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <layer/layer.hpp>
#include <layer/relu.hpp>

TEST(TestLayer, ReluForward) {
  using namespace free_infer;
  sftensor input = std::make_shared<Tensor<float>>(1, 1, 32);
  input->Fill(1);
  std::vector<sftensor> inputs;
  inputs.push_back(input);

  ReluLayer relu_layer;

  std::vector<sftensor> outputs(1);
  relu_layer.Forward(inputs, outputs);
  sftensor output = outputs.front();
  for (uint32_t i = 0; i < 3; ++i) {
    LOG(INFO) << output->shapes().at(i);
  }
  for (int i = 0; i < output->shapes().size(); ++i) {
    LOG(INFO) << output->shapes().at(i);
  }
  output->Show();
}