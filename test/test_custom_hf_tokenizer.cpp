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
#include <set>

namespace tokenizers {

using json = nlohmann::json;

namespace {
static inline std::string _get_resource_path(const std::string& name) {
  return std::getenv("RESOURCES_PATH") + std::string("/") + name;
}

json load_json_file(const std::string& path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    return json::object();
  }
  json j;
  file >> j;
  return j;
}

// Handles both plain string and {"content": "..."} object formats
std::string extract_token_string(const json& value) {
  if (value.is_string()) {
    return value.get<std::string>();
  }
  if (value.is_object() && value.contains("content")) {
    return value["content"].get<std::string>();
  }
  return "";
}

// Categorize normalizer behavior for round-trip testing:
// - Lossless: decode(encode(text)) == text
// - PrependSpace: decode(encode(text)) == " " + text (for non-empty)
// - LossyIdempotent: not idempotent on first pass, but second pass is stable
//   (e.g. BertNormalizer lowercases and strips whitespace)
enum class NormalizerBehavior { Lossless, PrependSpace, LossyIdempotent };

NormalizerBehavior detect_normalizer_behavior(const json& normalizer) {
  if (normalizer.is_null()) {
    return NormalizerBehavior::Lossless;
  }
  std::string type = normalizer.value("type", "");
  if (type == "Prepend") {
    return NormalizerBehavior::PrependSpace;
  }
  if (type == "BertNormalizer" && normalizer.value("lowercase", false)) {
    return NormalizerBehavior::LossyIdempotent;
  }
  if (type == "Precompiled") {
    // Precompiled normalizer does NFKC-like normalization which is lossy
    // (e.g. full-width chars → ASCII) but idempotent on second pass.
    return NormalizerBehavior::LossyIdempotent;
  }
  if (type == "Sequence" && normalizer.contains("normalizers")) {
    for (const auto& n : normalizer["normalizers"]) {
      auto sub = detect_normalizer_behavior(n);
      if (sub != NormalizerBehavior::Lossless) {
        return sub;
      }
    }
  }
  return NormalizerBehavior::Lossless;
}

class CustomHFTokenizerTest : public ::testing::TestWithParam<std::string> {
 protected:
  void SetUp() override {
    model_name_ = GetParam();
    tokenizer_dir_ = _get_resource_path(model_name_ + "_tokenizer_dir/");

    auto error = tokenizer_.load(tokenizer_dir_);
    ASSERT_EQ(error, Error::Ok)
        << "Failed to load tokenizer for " << model_name_;

    config_ = load_json_file(tokenizer_dir_ + "tokenizer_config.json");

    auto tokenizer_json = load_json_file(tokenizer_dir_ + "tokenizer.json");
    has_post_processor_ = tokenizer_json.contains("post_processor") &&
        !tokenizer_json["post_processor"].is_null();

    if (tokenizer_json.contains("normalizer")) {
      normalizer_ = detect_normalizer_behavior(tokenizer_json["normalizer"]);
    }

    if (config_.contains("bos_token")) {
      bos_token_str_ = extract_token_string(config_["bos_token"]);
    }
    if (config_.contains("eos_token")) {
      eos_token_str_ = extract_token_string(config_["eos_token"]);
    }
  }

  void assert_round_trip(const std::string& text) {
    auto encode_result = tokenizer_.encode(text, 0, 0);
    ASSERT_TRUE(encode_result.ok())
        << "Encoding failed for " << model_name_ << ": \"" << text << "\"";

    auto tokens = encode_result.get();

    auto decode_result = tokenizer_.decode(tokens);
    ASSERT_TRUE(decode_result.ok())
        << "Decoding failed for " << model_name_ << ": \"" << text << "\"";

    std::string decoded = decode_result.get();

    switch (normalizer_) {
      case NormalizerBehavior::Lossless:
        EXPECT_GT(tokens.size(), 0)
            << "Empty token list for " << model_name_ << ": \"" << text << "\"";
        EXPECT_EQ(decoded, text)
            << "Round-trip mismatch for " << model_name_ << "\n"
            << "Input:   \"" << text << "\"\n"
            << "Decoded: \"" << decoded << "\"";
        break;

      case NormalizerBehavior::PrependSpace: {
        std::string expected = text.empty() ? "" : " " + text;
        EXPECT_EQ(decoded, expected)
            << "Prepend round-trip mismatch for " << model_name_ << "\n"
            << "Input:    \"" << text << "\"\n"
            << "Expected: \"" << expected << "\"\n"
            << "Decoded:  \"" << decoded << "\"";
        break;
      }

      case NormalizerBehavior::LossyIdempotent: {
        // Normalization is lossy (e.g. lowercasing, whitespace stripping)
        // but a second round-trip must produce identical results
        if (decoded.empty()) {
          break; // Whitespace-only inputs may normalize to empty; nothing to
                 // check
        }
        auto second_encode = tokenizer_.encode(decoded, 0, 0);
        ASSERT_TRUE(second_encode.ok())
            << "Second encoding failed for " << model_name_;

        auto second_decode = tokenizer_.decode(second_encode.get());
        ASSERT_TRUE(second_decode.ok())
            << "Second decoding failed for " << model_name_;

        EXPECT_EQ(second_decode.get(), decoded)
            << "Idempotency failed for " << model_name_ << "\n"
            << "Input:          \"" << text << "\"\n"
            << "First decoded:  \"" << decoded << "\"\n"
            << "Second decoded: \"" << second_decode.get() << "\"";

        EXPECT_EQ(second_encode.get(), tokens)
            << "Token sequences differ on second encode for " << model_name_;
        break;
      }
    }
  }

  std::string model_name_;
  std::string tokenizer_dir_;
  HFTokenizer tokenizer_;
  json config_;
  bool has_post_processor_ = false;
  NormalizerBehavior normalizer_ = NormalizerBehavior::Lossless;
  std::string bos_token_str_;
  std::string eos_token_str_;
};

INSTANTIATE_TEST_SUITE_P(
    AllTokenizers,
    CustomHFTokenizerTest,
    ::testing::Values(
        "bielik",
        "phi",
        "minilm",
        "minilm_l12",
        "hammer",
        "qwen2_5",
        "qwen3",
        "llama",
        "smollm",
        "lfm2vl",
        "lfm2",
        "gemma3",
        "gemma4",
        "glmOCR",
        "gpt2",
        "camemBERT",
        "t5small",
        "roberta"));

TEST_P(CustomHFTokenizerTest, SimpleEncodeDecode) {
  assert_round_trip("Hello world!");
}

TEST_P(CustomHFTokenizerTest, LongEncodeDecode) {
  assert_round_trip(
      "This is a longer sentence, with punctuation and numbers like 1, 2, 3."
      "Let's see how the tokenizer handles it!");
}

TEST_P(CustomHFTokenizerTest, EmptyAndWhitespaceStrings) {
  auto result_empty = tokenizer_.encode("", 0, 0);
  ASSERT_TRUE(result_empty.ok());

  std::vector<std::string> whitespace_inputs = {"   ", "\t\n  \n\t"};
  for (const auto& ws : whitespace_inputs) {
    auto encode_result = tokenizer_.encode(ws, 0, 0);
    ASSERT_TRUE(encode_result.ok())
        << "Encoding whitespace failed for " << model_name_;

    auto tokens = encode_result.get();
    if (!tokens.empty()) {
      auto decode_result = tokenizer_.decode(tokens);
      ASSERT_TRUE(decode_result.ok())
          << "Decoding whitespace failed for " << model_name_;
    }
  }
}

TEST_P(CustomHFTokenizerTest, SpecialCharacters) {
  assert_round_trip("!@#$%^&*()_+-={}[]|\\:;\"'<>,.?/~`");
}

TEST_P(CustomHFTokenizerTest, CodeLikeContent) {
  assert_round_trip("def hello_world():\n    print(\"Hello, World!\")");
}

TEST_P(CustomHFTokenizerTest, IdToPieceOutOfRange) {
  auto vocab_size = tokenizer_.vocab_size();

  auto result_valid = tokenizer_.id_to_piece(0);
  EXPECT_TRUE(result_valid.ok());

  auto result_invalid = tokenizer_.id_to_piece(vocab_size + 1000);
  EXPECT_FALSE(result_invalid.ok());
  EXPECT_EQ(result_invalid.error(), Error::OutOfRange);
}

TEST_P(CustomHFTokenizerTest, PieceToIdAndBack) {
  std::vector<std::string> test_pieces = {" ", "a", "the", ".", ","};

  for (const auto& piece : test_pieces) {
    auto id_result = tokenizer_.piece_to_id(piece);
    if (id_result.ok()) {
      auto id = id_result.get();
      auto piece_result = tokenizer_.id_to_piece(id);
      ASSERT_TRUE(piece_result.ok())
          << "id_to_piece failed for ID " << id << " (piece: \"" << piece
          << "\") in " << model_name_;
      EXPECT_EQ(piece_result.get(), piece)
          << "piece_to_id -> id_to_piece round-trip failed for \"" << piece
          << "\" in " << model_name_;
    }
  }
}

TEST_P(CustomHFTokenizerTest, BosEosTokenIdsMatchConfig) {
  if (!bos_token_str_.empty()) {
    auto id_result = tokenizer_.piece_to_id(bos_token_str_);
    if (id_result.ok()) {
      EXPECT_EQ(tokenizer_.bos_tok(), id_result.get())
          << "BOS token ID mismatch for " << model_name_ << "\n"
          << "Config bos_token: \"" << bos_token_str_ << "\"\n"
          << "piece_to_id result: " << id_result.get() << "\n"
          << "bos_tok() result: " << tokenizer_.bos_tok();
    }
  }

  if (!eos_token_str_.empty()) {
    auto id_result = tokenizer_.piece_to_id(eos_token_str_);
    if (id_result.ok()) {
      EXPECT_EQ(tokenizer_.eos_tok(), id_result.get())
          << "EOS token ID mismatch for " << model_name_ << "\n"
          << "Config eos_token: \"" << eos_token_str_ << "\"\n"
          << "piece_to_id result: " << id_result.get() << "\n"
          << "eos_tok() result: " << tokenizer_.eos_tok();
    }
  }

  auto vocab_size = tokenizer_.vocab_size();
  EXPECT_LT(tokenizer_.bos_tok(), static_cast<uint64_t>(vocab_size))
      << "BOS token ID out of vocab range for " << model_name_;
  EXPECT_LT(tokenizer_.eos_tok(), static_cast<uint64_t>(vocab_size))
      << "EOS token ID out of vocab range for " << model_name_;
}

TEST_P(CustomHFTokenizerTest, SpecialTokensWithPostProcessor) {
  std::string text = "Test";

  auto result_without_special = tokenizer_.encode(text, 0, 0);
  ASSERT_TRUE(result_without_special.ok());
  auto tokens_without = result_without_special.get();

  auto result_with_special = tokenizer_.encode(text, 1, 1);
  ASSERT_TRUE(result_with_special.ok());
  auto tokens_with = result_with_special.get();

  if (!has_post_processor_) {
    // No post-processor: special token insertion is the post-processor's
    // responsibility (matches Rust HF tokenizers behavior), so encode(1,1)
    // must equal encode(0,0).
    EXPECT_EQ(tokens_with, tokens_without)
        << "Model " << model_name_ << " has no post-processor but "
        << "encode(text, 1, 1) differs from encode(text, 0, 0)\n"
        << "Without special: " << tokens_without.size() << " tokens\n"
        << "With special: " << tokens_with.size() << " tokens";
  } else {
    EXPECT_GE(tokens_with.size(), tokens_without.size())
        << "Post-processor removed tokens for " << model_name_;
  }
}

TEST_P(CustomHFTokenizerTest, BosEosDoesNotAffectLengthWithoutPostProcessor) {
  if (has_post_processor_) {
    GTEST_SKIP() << "Skipping for models with post-processor";
  }

  std::string text = "Test sentence";

  auto result_base = tokenizer_.encode(text, 0, 0);
  ASSERT_TRUE(result_base.ok());
  auto tokens_base = result_base.get();

  auto result_bos = tokenizer_.encode(text, 1, 0);
  ASSERT_TRUE(result_bos.ok());
  EXPECT_EQ(result_bos.get(), tokens_base)
      << "encode(text, 1, 0) differs from encode(text, 0, 0) without post-processor";

  auto result_eos = tokenizer_.encode(text, 0, 1);
  ASSERT_TRUE(result_eos.ok());
  EXPECT_EQ(result_eos.get(), tokens_base)
      << "encode(text, 0, 1) differs from encode(text, 0, 0) without post-processor";

  auto result_both = tokenizer_.encode(text, 1, 1);
  ASSERT_TRUE(result_both.ok());
  EXPECT_EQ(result_both.get(), tokens_base)
      << "encode(text, 1, 1) differs from encode(text, 0, 0) without post-processor";
}

TEST_P(CustomHFTokenizerTest, EncodeDecodeWithBosEosSkip) {
  std::string text = "Hello world";

  auto encode_result = tokenizer_.encode(text, 1, 1);
  ASSERT_TRUE(encode_result.ok());
  auto tokens = encode_result.get();

  auto decode_with_special = tokenizer_.decode(tokens, false);
  ASSERT_TRUE(decode_with_special.ok());

  auto decode_skip_special = tokenizer_.decode(tokens, true);
  ASSERT_TRUE(decode_skip_special.ok());

  // skip_special_tokens=true can only shorten or equal, never lengthen
  EXPECT_GE(decode_with_special.get().size(), decode_skip_special.get().size())
      << "skip_special_tokens=true produced longer text than false for "
      << model_name_;
}

TEST_P(CustomHFTokenizerTest, UnicodeEmojis) {
  assert_round_trip("Hello 😀 World 🎉 Test ✨");
}

TEST_P(CustomHFTokenizerTest, UnicodeMixedScripts) {
  assert_round_trip("Hello Привет Γειά σας مرحبا");
}

TEST_P(CustomHFTokenizerTest, UnicodeCombiningDiacritics) {
  assert_round_trip("café naïve résumé");
}

class BielikSpecificTest : public ::testing::Test {
 protected:
  void SetUp() override {
    tokenizer_dir_ = _get_resource_path("bielik_tokenizer_dir/");
    auto error = tokenizer_.load(tokenizer_dir_);
    ASSERT_EQ(error, Error::Ok) << "Failed to load Bielik tokenizer";
  }

  // Bielik has a Prepend("▁") normalizer: decode(encode(x)) == " " + x
  void assert_round_trip(const std::string& text) {
    auto encode_result = tokenizer_.encode(text, 0, 0);
    ASSERT_TRUE(encode_result.ok()) << "Encoding failed: \"" << text << "\"";

    auto tokens = encode_result.get();
    EXPECT_GT(tokens.size(), 0) << "Empty tokens for: \"" << text << "\"";

    auto decode_result = tokenizer_.decode(tokens);
    ASSERT_TRUE(decode_result.ok()) << "Decoding failed: \"" << text << "\"";

    std::string expected = text.empty() ? "" : " " + text;
    EXPECT_EQ(decode_result.get(), expected)
        << "Prepend round-trip mismatch for Bielik\n"
        << "Input:    \"" << text << "\"\n"
        << "Expected: \"" << expected << "\"\n"
        << "Decoded:  \"" << decode_result.get() << "\"";
  }

  std::string tokenizer_dir_;
  HFTokenizer tokenizer_;
};

TEST_F(BielikSpecificTest, PolishCharacters) {
  assert_round_trip("ąćęłńóśźż ĄĆĘŁŃÓŚŹŻ");
}

TEST_F(BielikSpecificTest, PolishSentences) {
  std::vector<std::string> polish_texts = {
      "Dzień dobry, jak się masz?",
      "Żółć gęsi jaźń",
      "Pchnąć w tę łódź jeża lub ośm skrzyń fig",
      "Zażółć gęślą jaźń",
      "Chrząszcz brzmi w trzcinie w Szczebrzeszynie"};

  for (const auto& text : polish_texts) {
    assert_round_trip(text);
  }
}

TEST_F(BielikSpecificTest, PolishTechnicalTerms) {
  std::vector<std::string> technical_terms = {
      "komputer",
      "oprogramowanie",
      "przetwarzanie",
      "sztuczna inteligencja",
      "uczenie maszynowe",
      "sieć neuronowa"};

  for (const auto& term : technical_terms) {
    assert_round_trip(term);
  }
}

TEST_F(BielikSpecificTest, PolishAndEnglishMixed) {
  assert_round_trip(
      "I am learning Polish: Witam, jestem programistą. "
      "Machine learning po polsku to uczenie maszynowe.");
}

TEST_F(BielikSpecificTest, PolishPoetry) {
  assert_round_trip(
      "Litwo! Ojczyzno moja! ty jesteś jak zdrowie;\n"
      "Ile cię trzeba cenić, ten tylko się dowie,\n"
      "Kto cię stracił.");
}

TEST_P(CustomHFTokenizerTest, VeryLongString) {
  std::string long_text;
  for (int i = 0; i < 200; i++) {
    long_text += "This is sentence number " + std::to_string(i) + ". ";
  }

  auto encode_result = tokenizer_.encode(long_text, 0, 0);
  ASSERT_TRUE(encode_result.ok()) << "Failed to encode very long string";

  auto tokens = encode_result.get();
  EXPECT_GT(tokens.size(), 100) << "Suspiciously few tokens for long string";

  auto decode_result = tokenizer_.decode(tokens);
  ASSERT_TRUE(decode_result.ok()) << "Failed to decode very long string";

  std::string decoded = decode_result.get();
  switch (normalizer_) {
    case NormalizerBehavior::Lossless:
      EXPECT_EQ(decoded, long_text)
          << "Round-trip mismatch for very long string in " << model_name_;
      break;
    case NormalizerBehavior::PrependSpace:
      EXPECT_EQ(decoded, " " + long_text)
          << "Prepend round-trip mismatch for very long string in "
          << model_name_;
      break;
    case NormalizerBehavior::LossyIdempotent: {
      auto second_encode = tokenizer_.encode(decoded, 0, 0);
      ASSERT_TRUE(second_encode.ok());
      EXPECT_EQ(second_encode.get(), tokens)
          << "Token sequences differ on second encode for very long string in "
          << model_name_;
      break;
    }
  }
}

TEST_P(CustomHFTokenizerTest, RepeatedCharacters) {
  assert_round_trip("aaaaaaaaaa bbbbbbbbbb cccccccccc");
}

TEST_P(CustomHFTokenizerTest, NumbersAndDigits) {
  std::vector<std::string> number_tests = {
      "0123456789", "123.456", "1,000,000", "3.14159", "0x1A2B3C", "1e-10"};

  for (const auto& text : number_tests) {
    assert_round_trip(text);
  }
}

TEST_P(CustomHFTokenizerTest, MixedAlphanumeric) {
  assert_round_trip("abc123def456 test_123 CamelCase123Test");
}

TEST_P(CustomHFTokenizerTest, EncodeDecodeIdempotent) {
  // Prepend normalizer adds a space each round-trip, so idempotency doesn't
  // hold for it; tested separately via PrependSpace check.
  if (normalizer_ == NormalizerBehavior::PrependSpace) {
    GTEST_SKIP()
        << "Skipping idempotency test for Prepend normalizer (tested via PrependSpace check)";
  }

  std::vector<std::string> test_texts = {
      "Simple test",
      "Test with numbers 123",
      "Test with punctuation!!!",
      "new\nline\tand\ttabs"};

  for (const auto& original : test_texts) {
    auto encode_result = tokenizer_.encode(original, 0, 0);
    ASSERT_TRUE(encode_result.ok());
    auto tokens = encode_result.get();

    auto decode_result = tokenizer_.decode(tokens);
    ASSERT_TRUE(decode_result.ok());
    std::string decoded = decode_result.get();

    if (decoded.empty()) {
      continue;
    }

    auto second_encode = tokenizer_.encode(decoded, 0, 0);
    ASSERT_TRUE(second_encode.ok());
    auto second_tokens = second_encode.get();

    EXPECT_EQ(tokens, second_tokens)
        << "Encode is not idempotent after round-trip for " << model_name_
        << "\nInput: \"" << original << "\"\n"
        << "Decoded: \"" << decoded << "\"\n"
        << "First encode:  " << tokens.size() << " tokens\n"
        << "Second encode: " << second_tokens.size() << " tokens";
  }
}

} // namespace
} // namespace tokenizers
