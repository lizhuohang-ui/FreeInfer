
#ifndef __FREE_INFER_PARSE_EXPRESSION_HPP__
#define __FREE_INFER_PARSE_EXPRESSION_HPP__

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace free_infer {

enum class TokenType {
  TokenUnknown = -9,
  TokenInputNumber = -8,
  TokenComma = -7,
  TokenAdd = -6,
  TokenMul = -5,
  TokenLeftBracket = -4,
  TokenRightBracket = -3,
  TokenSin = -2,
};

struct Token {
  TokenType token_type = TokenType::TokenUnknown;
  int32_t start_pos = 0;
  int32_t end_pos = 0;
  Token(TokenType token_type, int32_t start_pos, int32_t end_pos)
      : token_type(token_type), start_pos(start_pos), end_pos(end_pos) {}
};

struct TokenNode {
  int32_t num_index = -1;
  std::shared_ptr<TokenNode> left = nullptr;
  std::shared_ptr<TokenNode> right = nullptr;

  TokenNode(int32_t num_index, std::shared_ptr<TokenNode> left,
            std::shared_ptr<TokenNode> right)
      : num_index(num_index), left(left), right(right) {}
  TokenNode() = default;
};

class ExpressionParser {
 public:
  ExpressionParser(std::string statement) : statement_(std::move(statement)) {}

  void Tokenizer(void);
  std::vector<std::shared_ptr<TokenNode>> Generate();
  const std::vector<Token>& tokens() const;
  const std::vector<std::string>& token_strs() const;
  
  std::shared_ptr<TokenNode> Generate_(int32_t& index);

 private:
  std::vector<Token> tokens_;
  std::vector<std::string> token_strs_;
  std::string statement_;
};


}  // namespace free_infer

#endif  // __FREE_INFER_PARSE_EXPRESSION_HPP__