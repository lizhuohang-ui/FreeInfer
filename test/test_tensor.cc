#include <glog/logging.h>
#include <gtest/gtest.h>

#include "tensor.hpp"

TEST(TensorTest, TensorInit1D) {
  using namespace free_infer;
  Tensor<float> f1(4);
  f1.Fill(1.f);
  const auto &raw_shapes = f1.raw_shapes();
  LOG(INFO) << "-----------------------Tensor1D-----------------------";
  LOG(INFO) << "raw shapes size: " << raw_shapes.size();
  const uint32_t size = raw_shapes.at(0);
  LOG(INFO) << "data numbers: " << size;
  f1.Show();
}

TEST(TensorTest, TensorInit2D) {
  using namespace free_infer;
  Tensor<float> f1(4, 4);
  f1.Fill(1.f);

  const auto &raw_shapes = f1.raw_shapes();
  LOG(INFO) << "-----------------------Tensor2D-----------------------";
  LOG(INFO) << "raw shapes size: " << raw_shapes.size();
  const uint32_t rows = raw_shapes.at(0);
  const uint32_t cols = raw_shapes.at(1);

  LOG(INFO) << "data rows: " << rows;
  LOG(INFO) << "data cols: " << cols;
  f1.Show();
}

TEST(TensorTest, TensorInit3D) {
  using namespace free_infer;
  Tensor<float> f1(2, 3, 4);
  f1.Fill(1.f);

  const auto &raw_shapes = f1.raw_shapes();
  LOG(INFO) << "-----------------------Tensor3D 3Dim-----------------------";
  LOG(INFO) << "raw shapes size: " << raw_shapes.size();
  const uint32_t channels = raw_shapes.at(0);
  const uint32_t rows = raw_shapes.at(1);
  const uint32_t cols = raw_shapes.at(2);

  LOG(INFO) << "data channels: " << channels;
  LOG(INFO) << "data rows: " << rows;
  LOG(INFO) << "data cols: " << cols;
  f1.Show();
}

TEST(TensorTest, TensorTnit3D_2) {
  using namespace free_infer;
  Tensor<float> f1(1, 2, 3);
  f1.Fill(1.f);

  const auto &raw_shapes = f1.raw_shapes();
  LOG(INFO) << "-----------------------Tensor3D 2Dim-----------------------";
  LOG(INFO) << "raw shapes size: " << raw_shapes.size();
  const uint32_t rows = raw_shapes.at(0);
  const uint32_t cols = raw_shapes.at(1);

  LOG(INFO) << "data rows: " << rows;
  LOG(INFO) << "data cols: " << cols;
  f1.Show();
}

TEST(TensorTest, TensorTnit3D_1) {
  using namespace free_infer;
  Tensor<float> f1(1, 1, 3);
  f1.Fill(1.f);

  const auto &raw_shapes = f1.raw_shapes();
  LOG(INFO) << "-----------------------Tensor3D 1Dim-----------------------";
  LOG(INFO) << "raw shapes size: " << raw_shapes.size();
  const uint32_t size = raw_shapes.at(0);

  LOG(INFO) << "data numbers: " << size;
  f1.Show();
}

TEST(TensorTest, TensorSize) {
  using namespace free_infer;
  Tensor<float> f1(2, 3, 4);
  LOG(INFO) << "-----------------------Tensor Get Size-----------------------";
  LOG(INFO) << "channels: " << f1.channels();
  LOG(INFO) << "rows: " << f1.rows();
  LOG(INFO) << "cols: " << f1.cols();
  ASSERT_EQ(f1.channels(), 2);
  ASSERT_EQ(f1.rows(), 3);
  ASSERT_EQ(f1.cols(), 4);
}

TEST(TensorTest, TensorValues) {
  using namespace free_infer;
  Tensor<float> f1(2, 3, 4);
  f1.Rand();
  f1.Show();

  LOG(INFO) << "Data in the first channel: \n" << f1.slice(0);
  LOG(INFO) << "Data in the (0, 0, 0): " << f1.at(0, 0, 0);

  std::vector<float> value = f1.values();
  LOG(INFO) << "Show Data in the first channel by values: \n";
  for (uint32_t i = 0; i < f1.rows(); ++i) {
    for (uint32_t j = 0; j < f1.cols(); ++j) {
      std::cout << value.at(i * f1.cols() + j) << " ";
    }
    std::cout << "\n";
  }
}

TEST(TensorTest, TensorFill) {
  using namespace free_infer;
  LOG(INFO) << "-------------------Fill values-------------------";
  Tensor<float> f1(2, 3, 4);
  std::vector<float> values(2 * 3 * 4);
  for (int i = 0; i < 24; ++i) {
    values.at(i) = float(i + 1);
  }
  f1.Fill(values);
  f1.Show();
  LOG(INFO) << "-------------------Fill value-------------------";
  f1.Fill(3.14f);
  f1.Show();
}

TEST(TensorTest, TensorReshape) {
  using namespace free_infer;
  LOG(INFO) << "-------------------Reshape-------------------";
  Tensor<float> f1(2, 3, 4);
  std::vector<float> values(2 * 3 * 4);
  for (int i = 0; i < 24; ++i) {
    values.at(i) = float(i + 1);
  }
  f1.Fill(values);
  f1.Show();
  /// 将大小调整为(4, 3, 2)
  f1.Reshape({4, 3, 2}, true);
  LOG(INFO) << "-------------------After Reshape-------------------";
  f1.Show();
}

float MinusOne(float value) { return value - 1.f; }
float MulTwoPlusOne(float value) { return value * 2.f + 1.f; }

TEST(TensorTest, TensorTransform) {
  using namespace free_infer;
  LOG(INFO) << "-------------------Transform-------------------";
  Tensor<float> f1(2, 3, 4);
  f1.Rand();
  f1.Show();
  LOG(INFO) << "-------------------Transform: MinusOne-------------------";
  f1.Transform(MinusOne);
  f1.Show();
  LOG(INFO) << "-------------------Transform MulTwoPlusOne-------------------";
  f1.Transform(MulTwoPlusOne);
  f1.Show();
}

TEST(TensorTest, TensorFlatten) {
  using namespace free_infer;
  LOG(INFO) << "-------------------Flatten-------------------";
  Tensor<float> f1(2, 3, 4);
  f1.Flatten(true);
  ASSERT_EQ(f1.raw_shapes().size(), 1);
  ASSERT_EQ(f1.raw_shapes().at(0), 24);

  Tensor<float> f2(12, 24);
  f2.Flatten(true);
  ASSERT_EQ(f2.raw_shapes().size(), 1);
  ASSERT_EQ(f2.raw_shapes().at(0), 24 * 12);
}


TEST(TensorTest, TensorPadding) {
  using namespace free_infer;
  LOG(INFO) << "-------------------Padding-------------------";
  Tensor<float> tensor(3, 4, 5);
  ASSERT_EQ(tensor.channels(), 3);
  ASSERT_EQ(tensor.rows(), 4);
  ASSERT_EQ(tensor.cols(), 5);

  tensor.Fill(1.f);
  tensor.Padding({1, 2, 3, 4}, 0);
  ASSERT_EQ(tensor.rows(), 7);
  ASSERT_EQ(tensor.cols(), 12);

  int index = 0;
  for (int c = 0; c < tensor.channels(); ++c) {
    for (int r = 0; r < tensor.rows(); ++r) {
      for (int c_ = 0; c_ < tensor.cols(); ++c_) {
        if ((r >= 2 && r <= 4) && (c_ >= 3 && c_ <= 7)) {
          ASSERT_EQ(tensor.at(c, r, c_), 1.f) << c << " "
                                              << " " << r << " " << c_;
        }
        index += 1;
      }
    }
  }
}