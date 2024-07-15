#include "runtime_ir.hpp"

#include <utility>

namespace free_infer {
RuntimeGraph::RuntimeGraph(std::string param_path, std::string bin_path)
    : param_path_(std::move(param_path)), bin_path_(std::move(bin_path)) {}

void RuntimeGraph::set_bin_path(const std::string& bin_path) {
    this->bin_path_ = std::move(bin_path);
}

}  // namespace free_infer
