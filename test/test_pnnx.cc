#include <glog/logging.h>
#include <gtest/gtest.h>

#include "ir.h"

TEST(TestPnnx, PnnxGraphOps) {
  std::string bin_path("../model_file/test_linear.pnnx.bin");
  std::string param_path("../model_file/test_linear.pnnx.param");
  std::unique_ptr<pnnx::Graph> graph = std::make_unique<pnnx::Graph>();
  int load_result = graph->load(param_path, bin_path);
  ASSERT_EQ(load_result, 0);
  const auto &ops = graph->ops;
  for (int i = 0; i < ops.size(); ++i) {
    LOG(INFO) << ops.at(i)->name;
  }
}