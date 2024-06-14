#ifndef __TENSOR_HPP__
#define __TENSOR_HPP__

#include <glog/logging.h>

#include <armadillo>
#include <iostream>
#include <vector>

namespace free_infer {

template <typename T = float>
class Tensor {};

template <>
class Tensor<float> {
 public:
  // get dims info of tensor
  uint32_t rows() const;
  uint32_t cols() const;
  uint32_t channels() const;
  uint32_t size() const;

  void set_data(const arma::fcube& data);
  float at(uint32_t row, uint32_t col, uint32_t channels);
  void Fill(const std::vector<float>& values, bool row_major);

  explicit Tensor(uint32_t size);
  explicit Tensor(uint32_t rows, uint32_t cols);
  explicit Tensor(uint32_t rows, uint32_t cols, uint32_t channels);

 private:
  std::vector<uint32_t> raw_shapes_;
  arma::fcube data_;
};
}  // namespace free_infer

#endif  //__TENSOR_HPP__
