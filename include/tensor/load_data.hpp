#ifndef __LOAD_DATA_HPP__
#define __LOAD_DATA_HPP__

#include <armadillo>
#include <cstddef>
#include <fstream>
#include <string>

namespace free_infer {

class CSVDataLoader {
public:
    /**
     * @brief load data from the csv file
     * 
     * @param file_path  the path of the csv file
     * @param split_char split char
     * @return arma::fmat data from the csv file
     */
    static arma::fmat LoadData(const std::string& file_path, const char split_char = ',');

private:
    /**
     * @brief Get the Matrix Size 
     * 
     * @param file the path of the csv file
     * @param split_char split char
     * @return std::pair<size_t, size_t> size of the data from the csv file
     */
    static std::pair<size_t, size_t> GetMatrixSize(std::ifstream &file, const char split_char);
};

}

#endif // __LOAD_DATA_HPP__