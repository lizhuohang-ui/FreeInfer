#include "tensor_util.hpp"

#include <memory>

#include "tensor.hpp"

namespace free_infer {
std::tuple<sftensor, sftensor> TensorBroadcast(const sftensor& tensor1,
                                               const sftensor& tensor2) {
  CHECK(tensor1 != nullptr && tensor2 != nullptr);
  if (tensor1->shapes() == tensor2->shapes()) {
    return {tensor1, tensor2};
  } else {
    CHECK(tensor1->channels() == tensor2->channels());
    if (tensor2->rows() == 1 && tensor2->cols() == 1) {
      sftensor new_tensor = std::make_shared<Tensor<float>>(
          tensor2->channels(), tensor1->rows(), tensor1->cols());
      CHECK(tensor2->size() == tensor2->channels());
      for (uint32_t c = 0; c < tensor2->channels(); ++c) {
        new_tensor->slice(c).fill(tensor2->index(c));
      }
      return {tensor1, new_tensor};
    } else if (tensor1->rows() == 1 && tensor1->cols() == 1) {
      sftensor new_tensor = std::make_shared<Tensor<float>>(
          tensor1->channels(), tensor2->rows(), tensor2->cols());
      CHECK(tensor1->size() == tensor1->channels());
      for (uint32_t c = 0; c < tensor1->channels(); ++c) {
        new_tensor->slice(c).fill(tensor1->index(c));
      }
      return {new_tensor, tensor2};
    } else {
      LOG(FATAL) << "Broadcast shape is not adapting!";
      return {tensor1, tensor2};
    }
  }
}
sftensor TensorElementAdd(const sftensor& tensor1, const sftensor& tensor2) {
  if (tensor1->shapes() == tensor2->shapes()) {
    sftensor output_tensor = std::make_shared<Tensor<float>>(tensor1->shapes());
    output_tensor->set_data(tensor1->data() + tensor2->data());
    return output_tensor;
  } else {
    // broadcast
    CHECK(tensor1->channels() == tensor2->channels())
        << "Tensors shape are not adapting";
    const auto& [input_tensor1, input_tensor2] =
        TensorBroadcast(tensor1, tensor2);
    CHECK(input_tensor1->shapes() == input_tensor2->shapes());
    sftensor output_tensor =
        std::make_shared<Tensor<float>>(input_tensor1->shapes());
    output_tensor->set_data(input_tensor1->data() + input_tensor2->data());
    return output_tensor;
  }
}

sftensor TensorElementSin(const sftensor& tensor)
{
  sftensor output_tensor = std::make_shared<Tensor<float>>(tensor->shapes());
  output_tensor->set_data(arma::sin(tensor->data()));
  return output_tensor;
}

sftensor TensorElementMultiply(
    const std::shared_ptr<Tensor<float>>& tensor1,
    const std::shared_ptr<Tensor<float>>& tensor2) {
  CHECK(tensor1 != nullptr && tensor2 != nullptr);
  if (tensor1->shapes() == tensor2->shapes()) {
    sftensor output_tensor = std::make_shared<Tensor<float>>(tensor1->shapes());
    output_tensor->set_data(tensor1->data() % tensor2->data());
    return output_tensor;
  } else {
    // broadcast
    CHECK(tensor1->channels() == tensor2->channels())
        << "Tensors shape are not adapting";
    const auto& [input_tensor1, input_tensor2] =
        TensorBroadcast(tensor1, tensor2);
    CHECK(input_tensor1->shapes() == input_tensor2->shapes());
    sftensor output_tensor = std::make_shared<Tensor<float>>(input_tensor1->shapes());
    output_tensor->set_data(input_tensor1->data() % input_tensor2->data());
    return output_tensor;
  }
}
}  // namespace free_infer