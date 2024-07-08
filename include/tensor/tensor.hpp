#ifndef __FREE_INFER_TENSOR_HPP__
#define __FREE_INFER_TENSOR_HPP__

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
  explicit Tensor() = default;

  /**
   * @brief Create a 1dim tensor
   * @param size    size of the 1dim tensor
   */
  explicit Tensor(uint32_t size);

  /**
   * @brief Create a 2dim tensor
   * @param rows    height of the 2dim tensor
   * @param cols    width of the 2dim tensor
   */
  explicit Tensor(uint32_t rows, uint32_t cols);

  /**
   * @brief Create a 3dim tensor
   * @param channels  channels of the 3dim tensor
   * @param rows      rows of the 3dim tensor
   * @param cols      cols of the 3dim tensor
   */
  explicit Tensor(uint32_t channels, uint32_t rows, uint32_t cols);

  /**
   * @brief Create tensor by the shapes
   * @param shapes    the shapes of tensor
   */
  explicit Tensor(const std::vector<uint32_t>& shapes);

  Tensor(const Tensor& tensor);
  Tensor(Tensor&& tensor) noexcept;
  Tensor<float>& operator=(Tensor&& tensor) noexcept;
  Tensor<float>& operator=(const Tensor& tensor);

  // get dims info of tensor
  /**
   * @brief get the rows of the tensor
   * @return the rows of the tensor
   */
  uint32_t rows() const;

  /**
   * @brief get the cols of the tensor
   * @return the cols of the tensor
   */
  uint32_t cols() const;

  /**
   * @brief get the channels of the tensor
   * @return the channels of the tensor
   */
  uint32_t channels() const;

  /**
   * @brief retutn the size of the tensor
   * @return the size of the tensor
   */
  uint32_t size() const;

  /**
   * @brief set data of tensor
   * @param data data of tensor
   */
  void set_data(const arma::fcube& data);

  /**
   * @brief Check if the tensor is empty
   * @return true if tensor is empty not false
   */
  bool empty() const;

  /**
   * @brief get the element in the offset position of the tensor
   * @param offset the position of getting element
   * @return float  element in the offset position of the tensor
   */
  float index(uint32_t offset) const;
  float& index(uint32_t offset);

  /**
   * @brief get the shapes of the tensor
   * @return the shapes pf the tensor
   */
  std::vector<uint32_t> shapes() const;

  /**
   * @brief get the raw shapes of the tensor
   * @return the raw shapes pf the tensor
   */
  const std::vector<uint32_t>& raw_shapes() const;

  /**
   * @brief get the data of the tensor
   * @return the data of the tensor
   */
  arma::fcube& data();
  const arma::fcube& data() const;

  /**
   * @brief get the data of the channel
   * @param channel the channel need
   * @return the data of the channel
   */
  arma::fmat& slice(uint32_t channel);
  const arma::fmat& slice(uint32_t channel) const;

  /**
   * @brief return the element at specific position of the tensor
   * @param channel   channel the specific position
   * @param row       row of the specific position
   * @param col       col of the specific position
   * @return the element at specific position of the tensor
   */
  float at(uint32_t channel, uint32_t row, uint32_t col) const;
  float& at(uint32_t channel, uint32_t row, uint32_t col);

  /**
   * @brief fill the tensor with the data in values
   * @param values the data you want fill
   * @param row_major is row major order = 1: row major order 0: col major order
   */
  void Fill(const std::vector<float>& values, bool row_major = true);
  /**
   * @brief fill the tensor with value
   * @param value the data you want fill
   */
  void Fill(float value);

  /**
   * @brief get the values(element list) of the tensor by row or col
   * @param row_major row or col
   * @return the values(element list) of the tensor
   */
  std::vector<float> values(bool row_major = true);

  /**
   * @brief initialize the tensor to 1
   */
  void Ones();

  /**
   * @brief initialize the tensor to random number
   */
  void Rand();

  /**
   * @brief print the tensor
   */
  void Show();

  /**
   * @brief reshape the tensor by shapes
   * @param shapes the shapes need to reshape
   * @param row_major according to the row major order or the col major order to
   * reshape
   */
  void Reshape(const std::vector<uint32_t>& shapes, bool row_major = false);

  /**
   * @brief flatten the tensor
   * @param row_major according to the row major order or the col major order to
   * flatten
   */
  void Flatten(bool row_major = false);

  /**
   * @brief padding the tenosr by pads(up, bottom, left, right)
   * @param pads the size of padding
   * @param padding_value the value of padding
   */
  void Padding(const std::vector<uint32_t>& pads, float padding_value);

  /**
   * @brief transform the tensor by the filter
   * @param filter the operation to perform the data
   */
  void Transform(const std::function<float(float)>& filter);

  /**
   * @brief get the raw pointer of the data
   * @return the raw pointer of the data
   */
  float* raw_ptr();

  /**
   * @brief get the raw pointer of the data
   * @param offset the offset of the pointer
   * @return the raw pointer of the data
   */
  float* raw_ptr(uint32_t offset);

  /**
   * @brief get the raw pointer of the index matrix
   * @param index the index matrix
   * @return the raw pointer of the index matrix
   */
  float* matrix_raw_ptr(uint32_t index);

 private:
  std::vector<uint32_t> raw_shapes_;  // raw shapes of the tensor
  arma::fcube data_;                  // data of the tensor
                                      //
};
}  // namespace free_infer

#endif  //__TENSOR_HPP__
