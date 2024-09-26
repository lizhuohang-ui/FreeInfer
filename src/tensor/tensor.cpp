#include "tensor/tensor.hpp"
#include <cstdint>
namespace free_infer {

uint32_t Tensor<float>::rows() const {
  CHECK(!this->data_.empty());
  return this->data_.n_rows;
}

uint32_t Tensor<float>::cols() const {
  CHECK(!this->data_.empty());
  return this->data_.n_cols;
}

uint32_t Tensor<float>::channels() const {
  CHECK(!this->data_.empty());
  return this->data_.n_slices;
}

uint32_t Tensor<float>::size() const {
  CHECK(!this->data_.empty());
  return this->data_.size();
}

void Tensor<float>::set_data(const arma::fcube& data) { this->data_ = data; }

bool Tensor<float>::empty() const { return this->data_.empty(); }

float Tensor<float>::index(uint32_t offset) const {
  CHECK_LT(offset, this->data_.size());
  return this->data_.at(offset);
}

float& Tensor<float>::index(uint32_t offset) {
  CHECK_LT(offset, this->data_.size());
  return this->data_.at(offset);
}

std::vector<uint32_t> Tensor<float>::shapes() const {
  CHECK(!this->data_.empty());
  return {this->channels(), this->rows(), this->cols()};
}

const std::vector<uint32_t>& Tensor<float>::raw_shapes() const {
  CHECK(!this->raw_shapes_.empty());
  CHECK_GE(this->raw_shapes_.size(), 1);
  CHECK_LE(this->raw_shapes_.size(), 3);
  return this->raw_shapes_;
}

arma::fcube& Tensor<float>::data() { return this->data_; }

const arma::fcube& Tensor<float>::data() const { return this->data_; }

arma::fmat& Tensor<float>::slice(uint32_t channel) {
  CHECK_LT(channel, this->channels());
  return this->data_.slice(channel);
}

const arma::fmat& Tensor<float>::slice(uint32_t channel) const {
  CHECK_LT(channel, this->channels());
  return this->data_.slice(channel);
}

float Tensor<float>::at(uint32_t channel, uint32_t row, uint32_t col) const {
  CHECK_LT(row, this->rows());
  CHECK_LT(col, this->cols());
  CHECK_LT(channel, this->channels());
  return this->data_.at(row, col, channel);
}

float& Tensor<float>::at(uint32_t channel, uint32_t row, uint32_t col) {
  CHECK_LT(row, this->rows());
  CHECK_LT(col, this->cols());
  CHECK_LT(channel, this->channels());
  return this->data_.at(row, col, channel);
}

void Tensor<float>::Fill(const std::vector<float>& values, bool row_major) {
  CHECK(!this->data_.empty());
  const uint32_t total_elems = this->data_.size();
  CHECK_EQ(values.size(), total_elems);

  // fill by row
  if (row_major) {
    const uint32_t rows = this->rows();
    const uint32_t cols = this->cols();
    const uint32_t channels = this->channels();
    const uint32_t planes = rows * cols;

    for (uint32_t i = 0; i < channels; ++i) {
      auto& channel_data = this->data_.slice(i);
      const arma::fmat& channel_data_t =
          arma::fmat(values.data() + i * planes, this->cols(), this->rows());
      channel_data = channel_data_t.t();
    }
  } else {  // fill by col
    std::copy(values.begin(), values.end(), this->data_.memptr());
  }
}

void Tensor<float>::Fill(float value) { this->data_.fill(value); }

std::vector<float> Tensor<float>::values(bool row_major) {
  CHECK(!this->data_.empty());
  std::vector<float> values(this->data_.size());
  if (!row_major) {
    std::copy(this->data_.mem, this->data_.mem + this->data_.size(),
              values.begin());
  } else {
    uint32_t offset = 0;
    for (uint32_t i = 0; i < this->channels(); ++i) {
      const arma::fmat data = this->data_.slice(i).t();
      std::copy(data.begin(), data.end(), values.begin() + offset);
      offset += data.size();
    }
  }
  return values;
}

Tensor<float>::Tensor(uint32_t size) {
  this->data_ = arma::fcube(1, size, 1);
  this->raw_shapes_ = std::vector<uint32_t>{size};
}

Tensor<float>::Tensor(uint32_t rows, uint32_t cols) {
  this->data_ = arma::fcube(rows, cols, 1);
  this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
}

Tensor<float>::Tensor(uint32_t channels, uint32_t rows, uint32_t cols) {
  this->data_ = arma::fcube(rows, cols, channels);
  if (channels == 1 && rows == 1) {
    this->raw_shapes_ = std::vector<uint32_t>{cols};
  } else if (channels == 1) {
    this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
  } else {
    this->raw_shapes_ = std::vector<uint32_t>{rows, cols, channels};
  }
}

Tensor<float>::Tensor(const std::vector<uint32_t>& shapes) {
  CHECK_LE(shapes.size(), 3);

  if (shapes.size() == 3) {
    uint32_t channels = shapes.at(0);
    uint32_t rows = shapes.at(1);
    uint32_t cols = shapes.at(2);
    this->data_ = arma::fcube(rows, cols, channels);
    if (channels == 1 & rows == 1) {
      this->raw_shapes_ = {cols};
    } else if (channels == 1) {
      this->raw_shapes_ = {rows, cols};
    } else {
      this->raw_shapes_ = shapes;
    }
  } else if (shapes.size() == 2) {
    this->data_ = arma::fcube(1, shapes.at(0), shapes.at(1));
    this->raw_shapes_ = shapes;
  } else {
    this->data_ = arma::fcube(1, shapes.at(0), 1);
    this->raw_shapes_ = shapes;
  }
}

Tensor<float>::Tensor(const Tensor& tensor) {
  if(this != &tensor){
    this->data_ = tensor.data_;
    this->raw_shapes_ = tensor.raw_shapes_;
  }
}

Tensor<float>::Tensor(Tensor&& tensor) noexcept {
  if(this != &tensor){
    this->data_ = std::move(tensor.data_);
    this->raw_shapes_ = std::move(tensor.raw_shapes_);
  }
}

Tensor<float>& Tensor<float>::operator=(Tensor&& tensor) noexcept {
  if(this != &tensor){
    this->data_ = std::move(tensor.data_);
    this->raw_shapes_ = std::move(tensor.raw_shapes_);
  }
  return *this;
}

Tensor<float>& Tensor<float>::operator=(const Tensor& tensor) {
  if(this != &tensor){
    this->data_ = tensor.data_;
    this->raw_shapes_ = tensor.raw_shapes_;
  }
  return *this;
}

void Tensor<float>::Ones() {
  CHECK(!this->data_.empty());
  this->data_.fill(1.f);
}

void Tensor<float>::Rand() {
  CHECK(!this->data_.empty());
  this->data_.randn();
}

void Tensor<float>::Show() {
  uint32_t channels = data_.n_slices;
  for (uint32_t i = 0; i < channels; ++i) {
    LOG(INFO) << "Channels: " << i;
    LOG(INFO) << std::endl << this->data_.slice(i);
  }
}

void Tensor<float>::Reshape(const std::vector<uint32_t>& shapes,
                            bool row_major) {
  CHECK(!this->data_.empty());
  CHECK_LE(shapes.size(), 3);
  uint32_t tensor_size = this->data_.size();
  uint32_t shapes_size =
      std::accumulate(shapes.begin(), shapes.end(), 1, std::multiplies<uint32_t>());
  CHECK_EQ(tensor_size, shapes_size);
  std::vector<float> values;
  if (row_major) {
    values = this->values(true);
  }
  if (shapes.size() == 3) {
    this->data_.reshape(shapes.at(1), shapes.at(2), shapes.at(0));
    raw_shapes_ = {shapes.at(1), shapes.at(2), shapes.at(0)};
  } else if (shapes.size() == 2) {
    this->data_.reshape(shapes.at(1), shapes.at(2), 1);
    raw_shapes_ = {shapes.at(1), shapes.at(2)};
  } else {
    this->data_.reshape(1, shapes.at(0), 1);
    raw_shapes_ = {shapes.at(0)};
  }
  if (row_major) {
    this->Fill(values, true);
  }
}

void Tensor<float>::Flatten(bool row_major) {
  CHECK(!this->data_.empty());
  uint32_t tensor_size = this->data_.size();
  if (row_major) {
    this->data_.resize(tensor_size, 1, 1);
    std::vector<uint32_t> new_raw_shapes = {0};
    new_raw_shapes.at(0) = tensor_size;
    this->raw_shapes_ = std::move(new_raw_shapes);
  } else {
    this->data_.resize(1, tensor_size, 1);
    std::vector<uint32_t> new_raw_shapes = {1, 1};
    new_raw_shapes.at(1) = tensor_size;
    this->raw_shapes_ = std::move(new_raw_shapes);
  }
}

void Tensor<float>::Padding(const std::vector<uint32_t>& pads,
                            float padding_value) {
  CHECK(!this->data_.empty());
  CHECK_EQ(pads.size(), 4);
  uint32_t pad_rows1 = pads.at(0);  // up
  uint32_t pad_rows2 = pads.at(1);  // bottom
  uint32_t pad_cols1 = pads.at(2);  // left
  uint32_t pad_cols2 = pads.at(3);  // right

  uint32_t new_rows = data_.n_rows + pad_rows1 + pad_rows2;
  uint32_t new_cols = data_.n_cols + pad_cols1 + pad_cols2;

  arma::fcube new_data(new_rows, new_cols, data_.n_slices);
  for (uint32_t i = 0; i < data_.n_slices; ++i) {
    arma::fmat data_i = data_.slice(i);
    arma::fmat pad_row1_matrix =
        arma::zeros<arma::fmat>(pad_rows1, data_.n_cols);
    arma::fmat pad_row2_matrix =
        arma::zeros<arma::fmat>(pad_rows2, data_.n_cols);
    arma::fmat pad_col1_matrix = arma::zeros<arma::fmat>(
        data_.n_rows + pad_rows1 + pad_rows2, pad_cols1);
    arma::fmat pad_col2_matrix = arma::zeros<arma::fmat>(
        data_.n_rows + pad_rows1 + pad_rows2, pad_cols2);

    pad_row1_matrix.fill(padding_value);
    pad_row2_matrix.fill(padding_value);
    pad_col1_matrix.fill(padding_value);
    pad_col2_matrix.fill(padding_value);

    data_i = arma::join_cols(pad_row1_matrix, data_i);
    data_i = arma::join_cols(data_i, pad_row2_matrix);
    data_i = arma::join_rows(pad_col1_matrix, data_i);
    data_i = arma::join_rows(data_i, pad_col2_matrix);
    new_data.slice(i) = data_i;
  }
  this->data_ = std::move(new_data);
  this->raw_shapes_.at(0) += pad_rows1 + pad_rows2;
  this->raw_shapes_.at(1) += pad_cols1 + pad_cols2;
}

void Tensor<float>::Transform(const std::function<float(float)>& filter) {
  CHECK(!this->data_.empty());
  this->data_.transform(filter);
}

float* Tensor<float>::raw_ptr() {
  CHECK(!this->data_.empty());
  return this->data_.memptr();
}

float* Tensor<float>::raw_ptr(uint32_t offset) {
  CHECK(!this->data_.empty());
  CHECK_LT(offset, this->data_.size());
  return this->data_.memptr() + offset;
}

float* Tensor<float>::matrix_raw_ptr(uint32_t index) {
  CHECK(!this->data_.empty());
  CHECK_LT(index, this->data_.n_slices);
  uint32_t offset = index * this->data_.n_rows * this->data_.n_cols;
  CHECK_LT(offset, this->data_.size());
  return this->data_.memptr() + offset;
}

}  // namespace free_infer