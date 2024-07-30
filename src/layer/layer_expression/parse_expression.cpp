#include "parse_expression.hpp"

#include <glog/logging.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <memory>
#include <stack>
#include <string>
#include <vector>

namespace free_infer {

void ExpressionParser::Tokenizer(void) {
  CHECK(!statement_.empty()) << "The input statement is empty!";
  statement_.erase(std::remove_if(statement_.begin(), statement_.end(),
                                  [](char c) { return std::isspace(c); }),
                   statement_.end());
  CHECK(!statement_.empty()) << "The input statement is empty!";
  for (int32_t i = 0; i < statement_.size();) {
    char c = statement_.at(i);
    switch (c) {
      case 'a': {
        CHECK(i + 1 < statement_.size() && statement_.at(i + 1) == 'd')
            << "Parse add token failed, illegal character: "
            << statement_.at(i + 1);
        CHECK(i + 2 < statement_.size() && statement_.at(i + 2) == 'd')
            << "Parse add token failed, illegal character: "
            << statement_.at(i + 2);
        Token token(TokenType::TokenAdd, i, i + 3);
        tokens_.push_back(token);
        std::string token_string(statement_.begin() + i,
                                 statement_.begin() + i + 3);
        token_strs_.push_back(token_string);
        i += 3;
        break;
      }
      case 'm': {
        CHECK(i + 1 < statement_.size() && statement_.at(i + 1) == 'u')
            << "Parse add token failed, illegal character: "
            << statement_.at(i + 1);
        CHECK(i + 2 < statement_.size() && statement_.at(i + 2) == 'l')
            << "Parse add token failed, illegal character: "
            << statement_.at(i + 2);
        Token token(TokenType::TokenMul, i, i + 3);
        tokens_.push_back(token);
        std::string token_string(statement_.begin() + i,
                                 statement_.begin() + i + 3);
        token_strs_.push_back(token_string);
        i += 3;
        break;
      }
      case '@': {
        CHECK(i + 1 < statement_.size() && std::isdigit(statement_.at(i + 1)))
            << "Parse number token failed, illegal character: "
            << statement_.at(i + 1);
        int32_t j = i + 1;
        while (j < statement_.size() && std::isdigit(statement_.at(j))) {
          ++j;
        }
        Token token(TokenType::TokenInputNumber, i, j);
        CHECK(token.start_pos < token.end_pos);
        tokens_.push_back(token);
        std::string token_string(statement_.begin() + i,
                                 statement_.begin() + j);
        token_strs_.push_back(token_string);
        i = j;
        break;
      }
      case ',': {
        Token token(TokenType::TokenComma, i, i + 1);
        tokens_.push_back(token);
        std::string token_string(statement_.begin() + i,
                                 statement_.begin() + i + 1);
        token_strs_.push_back(token_string);
        i += 1;
        break;
      }
      case '(': {
        Token token(TokenType::TokenLeftBracket, i, i + 1);
        tokens_.push_back(token);
        std::string token_string(statement_.begin() + i,
                                 statement_.begin() + i + 1);
        token_strs_.push_back(token_string);
        i += 1;
        break;
      }
      case ')': {
        Token token(TokenType::TokenRightBracket, i, i + 1);
        tokens_.push_back(token);
        std::string token_string(statement_.begin() + i,
                                 statement_.begin() + i + 1);
        token_strs_.push_back(token_string);
        i += 1;
        break;
      }
      default: {
        LOG(FATAL) << "Unknown illegal character: " << c;
      }
    }
  }
}

std::shared_ptr<TokenNode> ExpressionParser::Generate_(int32_t& index) {
  CHECK(index < this->tokens_.size());
  const auto current_token = this->tokens_.at(index);

  CHECK(current_token.token_type == TokenType::TokenInputNumber ||
        current_token.token_type == TokenType::TokenAdd ||
        current_token.token_type == TokenType::TokenMul);
  if (current_token.token_type == TokenType::TokenInputNumber) {
    int32_t start_pos = current_token.start_pos + 1;
    int32_t end_pos = current_token.end_pos;

    CHECK(end_pos > start_pos || end_pos <= this->statement_.length())
        << "Current token has a wrong length";

    const std::string& str_number =
        std::string(this->statement_.begin() + start_pos,
                    this->statement_.begin() + end_pos);
    return std::make_shared<TokenNode>(std::stoi(str_number), nullptr, nullptr);
  } else if (current_token.token_type == TokenType::TokenAdd ||
             current_token.token_type == TokenType::TokenMul) {
    auto current_node = std::make_shared<TokenNode>();
    current_node->num_index = int(current_token.token_type);

    index += 1;
    CHECK(index < this->tokens_.size()) << "Missing left bracket!";
    CHECK(this->tokens_.at(index).token_type == TokenType::TokenLeftBracket);

    index += 1;
    CHECK(index < this->tokens_.size()) << "Missing correspond left token!";
    const auto left_token = this->tokens_.at(index);

    if (left_token.token_type == TokenType::TokenInputNumber ||
        left_token.token_type == TokenType::TokenAdd ||
        left_token.token_type == TokenType::TokenMul) {
      current_node->left = Generate_(index);
    } else {
      LOG(FATAL) << "Unknown token type: " << int(left_token.token_type);
    }

    index += 1;
    CHECK(index < this->tokens_.size()) << "Missing comma";
    CHECK(this->tokens_.at(index).token_type == TokenType::TokenComma);

    index += 1;
    CHECK(index < this->tokens_.size()) << "Missing correspond right token!";
    const auto right_token = this->tokens_.at(index);

    if (right_token.token_type == TokenType::TokenInputNumber ||
        right_token.token_type == TokenType::TokenAdd ||
        right_token.token_type == TokenType::TokenMul) {
      current_node->right = Generate_(index);
    } else {
      LOG(FATAL) << "Unknown token type: " << int(right_token.token_type);
    }

    index += 1;
    CHECK(index < this->tokens_.size()) << "Missing right bracket!";
    CHECK(this->tokens_.at(index).token_type == TokenType::TokenRightBracket);
    return current_node;
  } else {
    LOG(FATAL) << "Unknown token type: " << int(current_token.token_type);
  }
}

void ReversePolish(const std::shared_ptr<TokenNode>& root_node,
                   std::vector<std::shared_ptr<TokenNode>>& reverse_polish) {
  if (root_node != nullptr) {
    ReversePolish(root_node->left, reverse_polish);
    ReversePolish(root_node->right, reverse_polish);
    reverse_polish.push_back(root_node);
  }
}

std::vector<std::shared_ptr<TokenNode>> ExpressionParser::Generate() {
  if (this->tokens_.empty()) {
    this->Tokenizer();
  }
  int index = 0;
  std::shared_ptr<TokenNode> root = Generate_(index);
  CHECK(root != nullptr);
  CHECK(index == tokens_.size() - 1);

  std::vector<std::shared_ptr<TokenNode>> reverse_polish;
  ReversePolish(root, reverse_polish);
  return reverse_polish;
}

const std::vector<Token>& ExpressionParser::tokens() const {
  return this->tokens_;
}
const std::vector<std::string>& ExpressionParser::token_strs() const {
  return this->token_strs_;
}
}  // namespace free_infer