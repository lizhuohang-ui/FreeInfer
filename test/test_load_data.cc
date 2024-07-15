#include <cstdlib>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <iostream>
#include <string>

#include "load_data.hpp"

TEST(LoadData, CSVDataLoader)
{
    free_infer::CSVDataLoader csv_data_loader;
    LOG(INFO) << "-----------------------CSVDataLoader-----------------------";
    std::string file_path = "../test/test_data/test_load_data.csv";
    arma::fmat data = csv_data_loader.LoadData(file_path);
    LOG(INFO) << std::endl << data;
}