#include <glog/logging.h>
#include <gtest/gtest.h>

#include <pnnx/ir.h>
#include <runtime/runtime_ir.hpp>

static std::string ShapeStr(const std::vector<int> &shapes) {
  std::ostringstream ss;
  for (int i = 0; i < shapes.size(); ++i) {
    ss << shapes.at(i);
    if (i != shapes.size() - 1) {
      ss << " x ";
    }
  }
  return ss.str();
}

TEST(TestPNNX, PNNXGraphOps) {
  std::string bin_path("../../model_file/test_linear.pnnx.bin");
  std::string param_path("../../model_file/test_linear.pnnx.param");
  std::unique_ptr<pnnx::Graph> graph = std::make_unique<pnnx::Graph>();
  int load_result = graph->load(param_path, bin_path);
  ASSERT_EQ(load_result, 0);
  const auto &ops = graph->ops;
  for (int i = 0; i < ops.size(); ++i) {
    LOG(INFO) << ops.at(i)->name;
  }
}

TEST(TestPNNX, PnnxGraphOperands) {
  std::string bin_path("../../model_file/test_linear.pnnx.bin");
  std::string param_path("../../model_file/test_linear.pnnx.param");
  std::unique_ptr<pnnx::Graph> graph = std::make_unique<pnnx::Graph>();
  int load_result = graph->load(param_path, bin_path);
  ASSERT_EQ(load_result, 0);
  const auto &ops = graph->ops;
  for (int i = 0; i < ops.size(); ++i) {
    const auto &op = ops.at(i);
    LOG(INFO) << "OP Name: " << op->name;
    LOG(INFO) << "OP Inputs";
    for (int j = 0; j < op->inputs.size(); ++j) {
      LOG(INFO) << "Input name: " << op->inputs.at(j)->name
                << " shape: " << ShapeStr(op->inputs.at(j)->shape);
    }

    LOG(INFO) << "OP Output";
    for (int j = 0; j < op->outputs.size(); ++j) {
      LOG(INFO) << "Output name: " << op->outputs.at(j)->name
                << " shape: " << ShapeStr(op->outputs.at(j)->shape);
    }
    LOG(INFO) << "---------------------------------------------";
  }
}

TEST(TestPNNX, PnnxGraphOperandsAndParams) {
  std::string bin_path("../../model_file/test_linear.pnnx.bin");
  std::string param_path("../../model_file/test_linear.pnnx.param");
  std::unique_ptr<pnnx::Graph> graph = std::make_unique<pnnx::Graph>();
  int load_result = graph->load(param_path, bin_path);
  ASSERT_EQ(load_result, 0);
  const auto &ops = graph->ops;
  for (int i = 0; i < ops.size(); ++i) {
    const auto &op = ops.at(i);
    if (op->name != "linear") {
      continue;
    }
    LOG(INFO) << "OP Name: " << op->name;
    LOG(INFO) << "OP Inputs";
    for (int j = 0; j < op->inputs.size(); ++j) {
      LOG(INFO) << "Input name: " << op->inputs.at(j)->name
                << " shape: " << ShapeStr(op->inputs.at(j)->shape);
    }

    LOG(INFO) << "OP Output";
    for (int j = 0; j < op->outputs.size(); ++j) {
      LOG(INFO) << "Output name: " << op->outputs.at(j)->name
                << " shape: " << ShapeStr(op->outputs.at(j)->shape);
    }

    LOG(INFO) << "Params";
    for (const auto &attr : op->params) {
      LOG(INFO) << attr.first << " type " << attr.second.type;
    }

    LOG(INFO) << "Weight: ";
    for (const auto &weight : op->attrs) {
      LOG(INFO) << weight.first << " : " << ShapeStr(weight.second.shape)
                << " type " << weight.second.type;
    }
    LOG(INFO) << "---------------------------------------------";
  }
}

TEST(TEST_PNNX, PnnxGraphOperandsConsumerProducer) {
  std::string bin_path("../../model_file/test_linear.pnnx.bin");
  std::string param_path("../../model_file/test_linear.pnnx.param");
  std::unique_ptr<pnnx::Graph> graph = std::make_unique<pnnx::Graph>();
  int load_result = graph->load(param_path, bin_path);
  ASSERT_EQ(load_result, 0);
  const auto &operands = graph->operands;
  for (int i = 0; i < operands.size(); ++i) {
    const auto &operand = operands.at(i);
    LOG(INFO) << "Operand Name: #" << operand->name;
    LOG(INFO) << "Consumer: ";
    for (const auto &consumer : operand->consumers) {
      LOG(INFO) << consumer->name;
    }

    LOG(INFO) << "Producer: " << operand->producer->name;
  }
}

TEST(test_ir, pnnx_graph_all) {
  std::string bin_path("../../model_file/test_linear.pnnx.bin");
  std::string param_path("../../model_file/test_linear.pnnx.param");
  free_infer::RuntimeGraph graph(param_path, bin_path);
  const bool init_success = graph.Init();
  ASSERT_EQ(init_success, true);
  const auto &operators = graph.operators();
  for (const auto &operator_ : operators) {
    LOG(INFO) << "op name: " << operator_->name << " type: " << operator_->type;
    LOG(INFO) << "attribute:";
    for (const auto &[name, attribute_] : operator_->attrs) {
      LOG(INFO) << name << " type: " << int(attribute_->type)
                << " shape: " << ShapeStr(attribute_->shape);
      const auto &weight_data = attribute_->weight_data;
      ASSERT_EQ(weight_data.empty(), false); 
    }
    LOG(INFO) << "inputs: ";
    for (const auto &input : operator_->input_operands_maps) {
      LOG(INFO) << "name: " << input.first
                << " shape: " << ShapeStr(input.second->shapes);
    }

    LOG(INFO) << "outputs: ";
    for (const auto &output : operator_->output_names) {
      LOG(INFO) << "name: " << output;
    }
    LOG(INFO) << "--------------------------------------";
  }
}