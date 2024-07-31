#ifndef __FREE_INFER_TENSOR_UTIL_HPP__
#define __FREE_INFER_TENSOR_UTIL_HPP__

#include <memory>

#include "tensor.hpp"
namespace free_infer {
std::tuple<sftensor, sftensor> TensorBroadcast(const sftensor& tensor1,
                                               const sftensor& tensor2);
sftensor TensorElementAdd(const sftensor& tensor1, const sftensor& tensor2);
sftensor TensorElementSin(const sftensor& tensor);
sftensor TensorElementMultiply(const std::shared_ptr<Tensor<float>>& tensor1,
                               const std::shared_ptr<Tensor<float>>& tensor2);
}  // namespace free_infer

#endif  //__TENSOR_UTIL_HPP__