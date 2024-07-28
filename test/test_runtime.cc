#include <glog/logging.h>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <vector>

#include "ir.h"
#include "runtime_ir.hpp"

TEST(TestRuntime, RuntimeParams) {
  using namespace free_infer;
  std::string bin_path("../model_file/test_linear.pnnx.bin");
  std::string param_path("../model_file/test_linear.pnnx.param");
  RuntimeGraph graph(param_path, bin_path);
  const bool init_success = graph.Init();
  ASSERT_EQ(init_success, true);
  const auto &operators = graph.operators();
  for (const auto &operator_ : operators) {
    if (operator_->name == "linear") {
      const auto &params = operator_->params;
      LOG(INFO) << params.size();
      ASSERT_EQ(params.size(), 3);
      /////////////////////////////////
      ASSERT_EQ(params.count("bias"), 1);
      std::shared_ptr<RuntimeParameter> parameter_bool = params.at("bias");
      ASSERT_NE(parameter_bool, nullptr);
      ASSERT_EQ((std::dynamic_pointer_cast<RuntimeParameterBool>(parameter_bool)->value),
                true);
      /////////////////////////////////
      ASSERT_EQ(params.count("in_features"), 1);
      std::shared_ptr<RuntimeParameter> parameter_in_features = params.at("in_features");
      ASSERT_NE(parameter_in_features, nullptr);
      ASSERT_EQ(
          (std::dynamic_pointer_cast<RuntimeParameterInt>(parameter_in_features)->value),
          32);

      /////////////////////////////////
      ASSERT_EQ(params.count("out_features"), 1);
      std::shared_ptr<RuntimeParameter> parameter_out_features = params.at("out_features");
      ASSERT_NE(parameter_out_features, nullptr);
      ASSERT_EQ(
          (std::dynamic_pointer_cast<RuntimeParameterInt>(parameter_out_features)->value),
          128);
    }
  }
}

TEST(TestRuntime, TopoSort) {
  using namespace free_infer;
  std::string bin_path("../model_file/resnet18_batch1.pnnx.bin");
  std::string param_path("../model_file/resnet18_batch1.param");
  RuntimeGraph graph(param_path, bin_path);
  const bool init_success = graph.Init();
  ASSERT_EQ(init_success, true);
  graph.Topo();
  const auto &topo_queues = graph.get_topo_queues();

  int index = 0;
  for (const auto &operator_ : topo_queues) {
    LOG(INFO) << "Index: " << index << " Type: " << operator_->type
              << " Name: " << operator_->name;
    index += 1;
  }
}


TEST(test_ir, build1_output_tensors) {
  using namespace free_infer;
  std::string bin_path("../model_file/resnet18_batch1.pnnx.bin");
  std::string param_path("../model_file/resnet18_batch1.param");
  RuntimeGraph graph(param_path, bin_path);
  ASSERT_EQ(int(graph.graph_state()), -2);
  const bool init_success = graph.Init();
  ASSERT_EQ(init_success, true);
  ASSERT_EQ(int(graph.graph_state()), -1);
  graph.Build("pnnx_input_0", "pnnx_output_0");
  ASSERT_EQ(int(graph.graph_state()), 0);

  const auto &ops = graph.operators();
  for (const auto &op : ops) {
    LOG(INFO) << op->name;
    // 打印op输出空间的张量
    const auto &operand = op->output_operands;
    if (!operand || operand->datas.empty()) {
      continue;
    }
    const uint32_t batch_size = operand->datas.size();
    LOG(INFO) << "batch: " << batch_size;

    for (uint32_t i = 0; i < batch_size; ++i) {
      const auto &data = operand->datas.at(i);
      LOG(INFO) << "channel: " << data->channels()
                << " height: " << data->rows() << " cols: " << data->cols();
    }
  }
}


