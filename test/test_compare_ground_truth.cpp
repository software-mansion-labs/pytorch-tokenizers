/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// @lint-ignore-every LICENSELINT

#include <gtest/gtest.h>
#include <pytorch/tokenizers/hf_tokenizer.h>

#include <nlohmann/json.hpp>
#include <fstream>
#include <string>
#include <vector>

namespace tokenizers {

using json = nlohmann::json;

namespace {
static inline std::string _get_resource_path(const std::string& name) {
  return std::getenv("RESOURCES_PATH") + std::string("/") + name;
}

json load_ground_truth(const std::string& model_name) {
  auto path = _get_resource_path(model_name + "_ground_truth.json");
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open ground truth file: " + path);
  }
  json ground_truth;
  file >> ground_truth;
  return ground_truth;
}

} // namespace

class GroundTruthComparisonTest : public ::testing::TestWithParam<std::string> {
 protected:
  void SetUp() override {
    model_name_ = GetParam();
    tokenizer_dir_ = _get_resource_path(model_name_ + "_tokenizer_dir/");

    auto error = tokenizer_.load(tokenizer_dir_);
    ASSERT_EQ(error, Error::Ok)
        << "Failed to load tokenizer for " << model_name_;

    try {
      ground_truth_ = load_ground_truth(model_name_);
    } catch (const std::exception& e) {
      FAIL() << "Failed to load ground truth for " << model_name_ << ": "
             << e.what();
    }
  }

  std::string model_name_;
  std::string tokenizer_dir_;
  HFTokenizer tokenizer_;
  json ground_truth_;
};

INSTANTIATE_TEST_SUITE_P(
    AllModels,
    GroundTruthComparisonTest,
    ::testing::Values(
        "bielik",
        "phi",
        "minilm",
        "hammer",
        "qwen2_5",
        "qwen3",
        "llama",
        "smollm",
        "lfm2",
        "gemma4",
        "glmOCR"));

TEST_P(GroundTruthComparisonTest, VocabSizeMatches) {
  ASSERT_TRUE(ground_truth_.contains("vocab_size"))
      << "Ground truth missing vocab_size for " << model_name_;

  size_t expected_vocab_size = ground_truth_["vocab_size"];
  size_t actual_vocab_size = tokenizer_.vocab_size();

  EXPECT_EQ(actual_vocab_size, expected_vocab_size)
      << "Vocab size mismatch for " << model_name_;
}

TEST_P(GroundTruthComparisonTest, AllTestCasesMatch) {
  ASSERT_TRUE(ground_truth_.contains("test_cases"))
      << "Ground truth missing test_cases for " << model_name_;

  auto test_cases = ground_truth_["test_cases"];
  ASSERT_TRUE(test_cases.is_array())
      << "test_cases should be an array in ground truth";

  int case_num = 0;
  for (const auto& test_case : test_cases) {
    ASSERT_TRUE(test_case.contains("input"))
        << "Test case " << case_num << " missing input";
    ASSERT_TRUE(test_case.contains("tokens"))
        << "Test case " << case_num << " missing tokens";

    std::string input = test_case["input"];
    std::vector<int32_t> expected_tokens = test_case["tokens"];

    auto encode_result = tokenizer_.encode(input, 0, 0);  // no BOS/EOS
    ASSERT_TRUE(encode_result.ok())
        << "Encoding failed for test case " << case_num << " (input: \""
        << input << "\") in " << model_name_;

    auto actual_tokens = encode_result.get();

    EXPECT_EQ(actual_tokens.size(), expected_tokens.size())
        << "Token count mismatch for test case " << case_num << " in "
        << model_name_ << "\n"
        << "Input: \"" << input << "\"\n"
        << "Expected " << expected_tokens.size() << " tokens, got "
        << actual_tokens.size();

    size_t min_size = std::min(actual_tokens.size(), expected_tokens.size());
    for (size_t i = 0; i < min_size; i++) {
      EXPECT_EQ(actual_tokens[i], static_cast<uint64_t>(expected_tokens[i]))
          << "Token mismatch at position " << i << " for test case " << case_num
          << " in " << model_name_ << "\n"
          << "Input: \"" << input << "\"\n"
          << "Expected token ID: " << expected_tokens[i]
          << ", got: " << actual_tokens[i];
    }

    case_num++;
  }

  std::cout << "✓ All " << case_num << " test cases passed for " << model_name_
            << std::endl;
}

TEST_P(GroundTruthComparisonTest, SpecialTokensMatch) {
  ASSERT_TRUE(ground_truth_.contains("test_cases"))
      << "Ground truth missing test_cases for " << model_name_;

  auto test_cases = ground_truth_["test_cases"];
  int case_num = 0;

  for (const auto& test_case : test_cases) {
    if (!test_case.contains("tokens_with_special")) {
      continue;
    }

    std::string input = test_case["input"];
    std::vector<int32_t> expected_tokens = test_case["tokens_with_special"];

    auto encode_result = tokenizer_.encode(input, 1, 1);
    ASSERT_TRUE(encode_result.ok())
        << "Encoding with special tokens failed for test case " << case_num
        << " in " << model_name_;

    auto actual_tokens = encode_result.get();

    EXPECT_EQ(actual_tokens.size(), expected_tokens.size())
        << "Token count mismatch (with special tokens) for test case "
        << case_num << " in " << model_name_ << "\n"
        << "Input: \"" << input << "\"\n"
        << "Expected " << expected_tokens.size() << " tokens, got "
        << actual_tokens.size();

    size_t min_size = std::min(actual_tokens.size(), expected_tokens.size());
    for (size_t i = 0; i < min_size; i++) {
      EXPECT_EQ(actual_tokens[i], static_cast<uint64_t>(expected_tokens[i]))
          << "Token mismatch at position " << i << " (with special tokens) "
          << "for test case " << case_num << " in " << model_name_ << "\n"
          << "Input: \"" << input << "\"\n"
          << "Expected token ID: " << expected_tokens[i]
          << ", got: " << actual_tokens[i];
    }

    case_num++;
  }

  std::cout << "✓ Special tokens test passed for " << model_name_ << std::endl;
}

TEST_P(GroundTruthComparisonTest, DecodingMatches) {
  ASSERT_TRUE(ground_truth_.contains("test_cases"))
      << "Ground truth missing test_cases for " << model_name_;

  auto test_cases = ground_truth_["test_cases"];
  int case_num = 0;

  for (const auto& test_case : test_cases) {
    if (!test_case.contains("decoded")) {
      continue;
    }

    std::string input = test_case["input"];
    std::vector<int32_t> tokens = test_case["tokens"];
    std::string expected_decoded = test_case["decoded"];

    std::vector<uint64_t> tokens_uint64;
    for (auto token : tokens) {
      tokens_uint64.push_back(static_cast<uint64_t>(token));
    }

    auto decode_result = tokenizer_.decode(tokens_uint64, false);
    ASSERT_TRUE(decode_result.ok()) << "Decoding failed for test case "
                                    << case_num << " in " << model_name_;

    std::string actual_decoded = decode_result.get();

    // Compare decoded strings
    EXPECT_EQ(actual_decoded, expected_decoded)
        << "Decoded string mismatch for test case " << case_num << " in "
        << model_name_ << "\n"
        << "Input: \"" << input << "\"\n"
        << "Expected: \"" << expected_decoded << "\"\n"
        << "Got: \"" << actual_decoded << "\"";

    case_num++;
  }
}

TEST_P(GroundTruthComparisonTest, DecodingSkipSpecialTokens) {
  ASSERT_TRUE(ground_truth_.contains("test_cases"))
      << "Ground truth missing test_cases for " << model_name_;

  auto test_cases = ground_truth_["test_cases"];
  int case_num = 0;

  for (const auto& test_case : test_cases) {
    if (!test_case.contains("decoded_skip_special") ||
        !test_case.contains("tokens_with_special")) {
      continue;
    }

    std::string input = test_case["input"];
    std::vector<int32_t> tokens = test_case["tokens_with_special"];
    std::string expected_decoded = test_case["decoded_skip_special"];

    std::vector<uint64_t> tokens_uint64;
    for (auto token : tokens) {
      tokens_uint64.push_back(static_cast<uint64_t>(token));
    }

    auto decode_result = tokenizer_.decode(tokens_uint64, true);
    ASSERT_TRUE(decode_result.ok())
        << "Decoding (skip special) failed for test case " << case_num << " in "
        << model_name_;

    std::string actual_decoded = decode_result.get();

    // Compare decoded strings
    EXPECT_EQ(actual_decoded, expected_decoded)
        << "Decoded string mismatch (skip special tokens) for test case "
        << case_num << " in " << model_name_ << "\n"
        << "Input: \"" << input << "\"\n"
        << "Expected: \"" << expected_decoded << "\"\n"
        << "Got: \"" << actual_decoded << "\"";

    case_num++;
  }
}

TEST_P(GroundTruthComparisonTest, IdToPieceMatches) {
  ASSERT_TRUE(ground_truth_.contains("test_cases"))
      << "Ground truth missing test_cases for " << model_name_;

  auto test_cases = ground_truth_["test_cases"];
  int case_num = 0;

  for (const auto& test_case : test_cases) {
    if (!test_case.contains("token_strings") || !test_case.contains("tokens")) {
      continue;
    }

    std::string input = test_case["input"];
    std::vector<int32_t> tokens = test_case["tokens"];
    std::vector<std::string> expected_token_strings =
        test_case["token_strings"];

    ASSERT_EQ(tokens.size(), expected_token_strings.size())
        << "Token and token_strings size mismatch in ground truth for test case "
        << case_num;

    for (size_t i = 0; i < tokens.size(); i++) {
      auto piece_result =
          tokenizer_.id_to_piece(static_cast<uint64_t>(tokens[i]));
      ASSERT_TRUE(piece_result.ok())
          << "id_to_piece failed for token " << tokens[i] << " in test case "
          << case_num << " for " << model_name_;

      std::string actual_piece = piece_result.get();
      std::string expected_piece = expected_token_strings[i];

      EXPECT_EQ(actual_piece, expected_piece)
          << "Token string mismatch for token ID " << tokens[i] << " (position "
          << i << ") in test case " << case_num << " for " << model_name_
          << "\n"
          << "Input: \"" << input << "\"\n"
          << "Expected: \"" << expected_piece << "\"\n"
          << "Got: \"" << actual_piece << "\"";
    }

    case_num++;
  }
}

TEST_P(GroundTruthComparisonTest, PieceToIdMatches) {
  if (!ground_truth_.contains("piece_to_id")) {
    GTEST_SKIP() << "Ground truth missing piece_to_id for " << model_name_
                 << " (regenerate ground truth)";
  }

  auto piece_to_id_map = ground_truth_["piece_to_id"];
  ASSERT_TRUE(piece_to_id_map.is_object())
      << "piece_to_id should be an object in ground truth";

  int checked = 0;
  for (auto it = piece_to_id_map.begin(); it != piece_to_id_map.end(); ++it) {
    std::string piece = it.key();
    int32_t expected_id = it.value();

    auto id_result = tokenizer_.piece_to_id(piece);
    ASSERT_TRUE(id_result.ok())
        << "piece_to_id failed for \"" << piece << "\" in " << model_name_;

    EXPECT_EQ(id_result.get(), static_cast<uint64_t>(expected_id))
        << "piece_to_id mismatch for \"" << piece << "\" in " << model_name_
        << "\nExpected ID: " << expected_id << ", got: " << id_result.get();

    // Also verify round-trip: piece_to_id -> id_to_piece -> same piece
    auto piece_result = tokenizer_.id_to_piece(id_result.get());
    ASSERT_TRUE(piece_result.ok()) << "id_to_piece failed for ID "
                                   << id_result.get() << " in " << model_name_;
    EXPECT_EQ(piece_result.get(), piece)
        << "piece_to_id -> id_to_piece round-trip mismatch for \"" << piece
        << "\" in " << model_name_;

    checked++;
  }

  std::cout << "✓ piece_to_id verified for " << checked << " tokens in "
            << model_name_ << std::endl;
}

// Stronger end-to-end check: decode(encode(input)) must match Rust reference's
// decode(rust_tokens), catching cases where tokens are correct but decoding diverges.
TEST_P(GroundTruthComparisonTest, DecodeOwnTokensMatches) {
  ASSERT_TRUE(ground_truth_.contains("test_cases"))
      << "Ground truth missing test_cases for " << model_name_;

  auto test_cases = ground_truth_["test_cases"];
  int case_num = 0;

  for (const auto& test_case : test_cases) {
    if (!test_case.contains("decoded")) {
      continue;
    }

    std::string input = test_case["input"];
    std::string expected_decoded = test_case["decoded"];

    auto encode_result = tokenizer_.encode(input, 0, 0);
    ASSERT_TRUE(encode_result.ok()) << "Encoding failed for test case "
                                    << case_num << " in " << model_name_;

    auto our_tokens = encode_result.get();

    auto decode_result = tokenizer_.decode(our_tokens, false);
    ASSERT_TRUE(decode_result.ok())
        << "Decoding own tokens failed for test case " << case_num << " in "
        << model_name_;

    EXPECT_EQ(decode_result.get(), expected_decoded)
        << "Decode-own-tokens mismatch for test case " << case_num << " in "
        << model_name_ << "\n"
        << "Input: \"" << input << "\"\n"
        << "Expected: \"" << expected_decoded << "\"\n"
        << "Got: \"" << decode_result.get() << "\"";

    case_num++;
  }

  std::cout << "✓ Decode-own-tokens passed for " << case_num
            << " test cases in " << model_name_ << std::endl;
}

} // namespace tokenizers
