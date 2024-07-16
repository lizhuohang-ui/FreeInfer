#include <glog/logging.h>
#include <gtest/gtest.h>
#include <iostream>

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
      RuntimeParameter *parameter_bool = params.at("bias");
      ASSERT_NE(parameter_bool, nullptr);
      ASSERT_EQ((dynamic_cast<RuntimeParameterBool *>(parameter_bool)->value),
                true);
      /////////////////////////////////
      ASSERT_EQ(params.count("in_features"), 1);
      RuntimeParameter *parameter_in_features = params.at("in_features");
      ASSERT_NE(parameter_in_features, nullptr);
      ASSERT_EQ(
          (dynamic_cast<RuntimeParameterInt *>(parameter_in_features)->value),
          32);

      /////////////////////////////////
      ASSERT_EQ(params.count("out_features"), 1);
      RuntimeParameter *parameter_out_features = params.at("out_features");
      ASSERT_NE(parameter_out_features, nullptr);
      ASSERT_EQ(
          (dynamic_cast<RuntimeParameterInt *>(parameter_out_features)->value),
          128);
    }
  }
}