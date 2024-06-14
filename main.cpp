#include <glog/logging.h>
#include <gtest/gtest.h>

#include <armadillo>
#include <iostream>
#include <vector>

#include "tensor.hpp"

int main(int argc, char* argv[]) {
  std::cout << "Hello, from free-infer!\n";
  std::cout << ARMA_VERSION_NAME << std::endl;
  testing::InitGoogleTest(&argc, argv);
  google::InitGoogleLogging("free");
  FLAGS_log_dir = "./log";
  FLAGS_alsologtostderr = true;

  LOG(INFO) << "Start test...\n";
  
  free_infer::Tensor<float> tensor(3, 3, 4);
  std::cout << "Tensor rows: " << tensor.rows() << std::endl;
  std::cout << "Tensor cols: " << tensor.cols() << std::endl;
  std::cout << "Tensor channels: " << tensor.channels() << std::endl;
  std::cout << "Tensor size: " << tensor.size() << std::endl;
  std::cout << "Tensor at: " << tensor.at(1, 1, 1) << std::endl;
  
  std::vector<float> test{1, 2, 3, 4};
  std::cout << "vector.data():\n" << test.data() << std::endl;
  const arma::fmat& test_fmat = arma::fmat(test.data() + 2, 2, 3);
  std::cout << "test_fmat.rows:" << test_fmat.n_rows << std::endl;
  std::cout << "test_fmat.cols:" << test_fmat.n_cols << std::endl;
  for(int i = 0; i < 2; ++i){
    for(int j = 0; j < 3; ++j){
      std::cout << test_fmat.at(i, j) << " ";
    }
    std::cout << std::endl;
  }
  
  
  return RUN_ALL_TESTS();
}
