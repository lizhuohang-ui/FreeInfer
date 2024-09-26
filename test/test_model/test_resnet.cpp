#include <glog/logging.h>
#include <gtest/gtest.h>
#include <math.h>
#include <opencv2/core/hal/interface.h>
#include <pnnx/ir.h>
#include <sys/types.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <layer/softmax.hpp>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <runtime/runtime_ir.hpp>
#include <string>
#include <tensor/tensor.hpp>
#include <vector>

free_infer::sftensor PreProcessImage(const cv::Mat& image) {
  using namespace free_infer;
  assert(!image.empty());
  cv::Mat resize_image;
  cv::resize(image, resize_image, cv::Size(224, 224));

  cv::Mat rgb_image;
  cv::cvtColor(resize_image, rgb_image, cv::COLOR_BGR2RGB);

  rgb_image.convertTo(rgb_image, CV_32FC3);
  std::vector<cv::Mat> split_images;
  cv::split(rgb_image, split_images);
  uint32_t input_h = 224;
  uint32_t input_w = 224;
  uint32_t input_c = 3;
  sftensor input = std::make_shared<Tensor<float>>(input_c, input_h, input_w);

  uint32_t index = 0;
  for (const auto& split_image : split_images) {
    assert(split_image.total() == input_h * input_w);
    const cv::Mat& split_image_t = split_image.t();
    memcpy(input->slice(index).memptr(), split_image_t.data,
           sizeof(float) * split_image.total());
    index += 1;
  }

  float mean_r = 0.485f;
  float mean_g = 0.456f;
  float mean_b = 0.406f;

  float var_r = 0.229f;
  float var_g = 0.224f;
  float var_b = 0.225f;
  assert(input->channels() == 3);
  input->data() = input->data() / 255.f;
  input->slice(0) = (input->slice(0) - mean_r) / var_r;
  input->slice(1) = (input->slice(1) - mean_g) / var_g;
  input->slice(2) = (input->slice(2) - mean_b) / var_b;
  return input;
}

TEST(TestResNet, ResNet) {
  using namespace free_infer;
  //   const std::string& image_path = "../imgs/car.jpg";
  const std::string& image_path = "../imgs/person.jpg";
  const uint32_t batch_size = 1;
  std::vector<sftensor> inputs;
  for (uint32_t i = 0; i < batch_size; ++i) {
    cv::Mat image = cv::imread(image_path);
    sftensor input = PreProcessImage(image);
    inputs.push_back(input);
  }

  const std::string bin_path("../model_file/resnet18_batch1.pnnx.bin");
  const std::string param_path("../model_file/resnet18_batch1.param");
  RuntimeGraph graph(param_path, bin_path);
  graph.Build("pnnx_input_0", "pnnx_output_0");
  const std::vector<sftensor> outputs = graph.Forward(inputs);
  //   outputs.front()->Show();
  std::vector<sftensor> outputs_softmax(batch_size);
  SoftmaxLayer softmax_layer;
  //   outputs.front()->Show();
  softmax_layer.Forward(outputs, outputs_softmax);
  float sum = arma::accu(outputs_softmax.front()->data());
  LOG(INFO) << sum;
  outputs_softmax.front()->Show();

  for (int i = 0; i < outputs_softmax.size(); ++i) {
    const sftensor& output_tensor = outputs_softmax.at(i);
    assert(output_tensor->size() == 1 * 1000);
    // 找到类别概率最大的种类
    float max_prob = -1;
    int max_index = -1;
    for (int j = 0; j < output_tensor->size(); ++j) {
      float prob = output_tensor->index(j);
      if (max_prob <= prob) {
        max_prob = prob;
        max_index = j;
      }
    }
    printf("class with max prob is %f index %d\n", max_prob, max_index);
  }
}