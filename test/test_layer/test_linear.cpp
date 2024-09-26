
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <layer/layer.hpp>
#include <layer/linear.hpp>

TEST(TestLayer, LinearForward) {
  using namespace free_infer;
  sftensor input = std::make_shared<Tensor<float>>(1, 1, 32);
  input->Fill(1);
  std::vector<sftensor> inputs;
  inputs.push_back(input);

  std::string bin_path("../../model_file/test_linear.pnnx.bin");
  std::string param_path("../../model_file/test_linear.pnnx.param");
  RuntimeGraph graph(param_path, bin_path);
  const bool init_success = graph.Init();
  ASSERT_EQ(init_success, true);
  graph.Build("pnnx_input_0", "pnnx_output_0");

  std::vector<sftensor> outputs = graph.Forward(inputs);
  sftensor output = outputs.front();
  for (uint32_t i = 0; i < 3; ++i) {
    LOG(INFO) << output->shapes().at(i);
  }
  for (int i = 0; i < output->shapes().size(); ++i) {
    LOG(INFO) << output->shapes().at(i);
  }
  output->Show();
}

TEST(TestLayer, LinearForward2) {
  using namespace free_infer;
  const uint32_t in_features = 32;
  const uint32_t out_features = 64;
  const uint32_t in_dims = 1280;

  LinearLayer linear_layer(in_features, out_features, false);
  std::vector<float> weights(in_features * out_features, 1.f);
  linear_layer.set_weights(weights);

  std::shared_ptr<Tensor<float>> input =
      std::make_shared<Tensor<float>>(1, in_dims, in_features);
  input->Fill(1.f);

  std::shared_ptr<Tensor<float>> output =
      std::make_shared<Tensor<float>>(1, in_dims, out_features);
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);

  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  outputs.push_back(output);

  const auto status = linear_layer.Forward(inputs, outputs);
  // outputs.front()->Show();
  ASSERT_EQ(status, InferStatus::kInferSuccess);
  ASSERT_EQ(outputs.size(), 1);
  for (int i = 0; i < outputs.size(); ++i) {
    const auto &output_tensor = outputs.at(i);
    for (int j = 0; j < output_tensor->size(); ++j) {
      ASSERT_EQ(output_tensor->index(j), in_features);
    }
  }
}