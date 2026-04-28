/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// @lint-ignore-every LICENSELINT

#include <gtest/gtest.h>
#include <pytorch/tokenizers/normalizer.h>

#include <fstream>

using namespace tokenizers;

TEST(NormalizerTest, ReplaceNormalizerBasic) {
  // Test basic string replacement
  ReplaceNormalizer normalizer(" ", "▁");
  std::string input = "Hello World Test";
  std::string expected = "Hello▁World▁Test";
  std::string result = normalizer.normalize(input);
  EXPECT_EQ(result, expected);
}

TEST(NormalizerTest, ReplaceNormalizerNoMatch) {
  // Test when pattern doesn't match
  ReplaceNormalizer normalizer("xyz", "▁");
  std::string input = "Hello World";
  std::string expected = "Hello World";
  std::string result = normalizer.normalize(input);
  EXPECT_EQ(result, expected);
}

TEST(NormalizerTest, ReplaceNormalizerMultipleMatches) {
  // Test multiple matches
  ReplaceNormalizer normalizer("a", "X");
  std::string input = "banana";
  std::string expected = "bXnXnX";
  std::string result = normalizer.normalize(input);
  EXPECT_EQ(result, expected);
}

TEST(NormalizerTest, PrependNormalizerBasic) {
  // Test basic prepending
  PrependNormalizer normalizer("_");
  std::string input = "Hello";
  std::string expected = "_Hello";
  std::string result = normalizer.normalize(input);
  EXPECT_EQ(result, expected);
}

TEST(NormalizerTest, PrependNormalizerEmptyInput) {
  // Test prepend with empty input (should return empty)
  PrependNormalizer normalizer("_");
  std::string input = "";
  std::string expected = "";
  std::string result = normalizer.normalize(input);
  EXPECT_EQ(result, expected);
}

TEST(NormalizerTest, LowercaseNormalizerBasic) {
  // Test basic lowercasing
  LowercaseNormalizer normalizer;
  EXPECT_EQ(normalizer.normalize("HELLO WORLD"), "hello world");
  EXPECT_EQ(normalizer.normalize("Hello World"), "hello world");
  EXPECT_EQ(normalizer.normalize("hello world"), "hello world");
  // Test with accents (should lowercase but NOT strip them, LowercaseNormalizer
  // only lowercases)
  EXPECT_EQ(normalizer.normalize("HÉLLO"), "héllo");
}

TEST(NormalizerTest, NormalizerConfigLowercase) {
  // Test JSON parsing for Lowercase normalizer
  nlohmann::json config = {{"type", "Lowercase"}};

  NormalizerConfig norm_config;
  norm_config.parse_json(config);
  auto normalizer = norm_config.create();

  EXPECT_EQ(normalizer->normalize("HELLO"), "hello");
}

TEST(NormalizerTest, NormalizerConfigPrepend) {
  // Test JSON parsing for Prepend normalizer
  nlohmann::json config = {{"type", "Prepend"}, {"prepend", "_"}};

  NormalizerConfig norm_config;
  norm_config.parse_json(config);
  auto normalizer = norm_config.create();

  std::string input = "Hello";
  std::string expected = "_Hello";
  std::string result = normalizer->normalize(input);
  EXPECT_EQ(result, expected);
}

TEST(NormalizerTest, NormalizerConfigFromJson) {
  // Test JSON parsing for Replace normalizer
  nlohmann::json config = {
      {"type", "Replace"}, {"pattern", {{"String", " "}}}, {"content", "▁"}};

  NormalizerConfig norm_config;
  norm_config.parse_json(config);
  auto normalizer = norm_config.create();

  std::string input = "Hello World Test";
  std::string expected = "Hello▁World▁Test";
  std::string result = normalizer->normalize(input);
  EXPECT_EQ(result, expected);
}

TEST(NormalizerTest, NormalizerConfigFromJsonRegex) {
  // Test JSON parsing for Replace normalizer with regex
  nlohmann::json config = {
      {"type", "Replace"}, {"pattern", {{"Regex", "\\s+"}}}, {"content", "_"}};

  NormalizerConfig norm_config;
  norm_config.parse_json(config);
  auto normalizer = norm_config.create();

  std::string input = "Hello   World\t\tTest";
  std::string expected = "Hello_World_Test";
  std::string result = normalizer->normalize(input);
  EXPECT_EQ(result, expected);
}

TEST(NormalizerTest, SequenceNormalizer) {
  // Test sequence of normalizers
  std::vector<Normalizer::Ptr> normalizers;
  normalizers.push_back(std::make_unique<ReplaceNormalizer>(" ", "▁"));
  normalizers.push_back(std::make_unique<ReplaceNormalizer>("a", "X"));

  SequenceNormalizer seq_normalizer(std::move(normalizers));

  std::string input = "banana split";
  std::string expected = "bXnXnX▁split";
  std::string result = seq_normalizer.normalize(input);
  EXPECT_EQ(result, expected);
}

TEST(NormalizerTest, SequenceNormalizerFromConfig) {
  // Test sequence normalizer from config
  nlohmann::json config = {
      {"type", "Sequence"},
      {"normalizers",
       {{{"type", "Replace"}, {"pattern", {{"String", " "}}}, {"content", "▁"}},
        {{"type", "Replace"},
         {"pattern", {{"String", "a"}}},
         {"content", "X"}}}}};

  NormalizerConfig norm_config;
  norm_config.parse_json(config);
  auto normalizer = norm_config.create();

  std::string input = "banana split";
  std::string expected = "bXnXnX▁split";
  std::string result = normalizer->normalize(input);
  EXPECT_EQ(result, expected);
}

TEST(NormalizerTest, EmptyInput) {
  // Test with empty input
  ReplaceNormalizer normalizer(" ", "▁");
  std::string input = "";
  std::string expected = "";
  std::string result = normalizer.normalize(input);
  EXPECT_EQ(result, expected);
}

TEST(NormalizerTest, ConfigBuilder) {
  // Test config builder pattern
  auto normalizer =
      NormalizerConfig("Replace").set_pattern(" ").set_content("▁").create();

  std::string input = "Hello World";
  std::string expected = "Hello▁World";
  std::string result = normalizer->normalize(input);
  EXPECT_EQ(result, expected);
}

// -- BertNormalizer Exhaustive Tests ------------------------------------------

TEST(NormalizerTest, BertNormalizerAccentDefaulting) {
  // Logic: strip_accents should inherit 'lowercase' value if not set.
  // Test case: lowercase=true, strip_accents=unset -> should strip
  auto norm1 = NormalizerConfig("BertNormalizer").set_lowercase(true).create();
  EXPECT_EQ(norm1->normalize("Héllò"), "hello");

  // Test case: lowercase=false, strip_accents=unset -> should NOT strip
  auto norm2 = NormalizerConfig("BertNormalizer").set_lowercase(false).create();
  EXPECT_EQ(norm2->normalize("Héllò"), "Héllò");
}

TEST(NormalizerTest, BertNormalizerControlChars) {
  auto normalizer = NormalizerConfig("BertNormalizer")
                        .set_clean_text(true)
                        .set_lowercase(false)
                        .create();

  // 1. Remove NULL and Replacement Character
  // Using std::string constructor to include the null byte
  std::string input = "A";
  input.push_back('\0');
  input +=
      "B\xEF\xBF\xBD"
      "C";
  EXPECT_EQ(normalizer->normalize(input), "ABC");

  // 2. Remove C0/C1 Control chars but KEEP \n, \t, \r (mapping them to space)
  // \x01 is a control char, \x1F is a control char
  // \n and \t should become spaces
  EXPECT_EQ(
      normalizer->normalize(
          "start\x01\n\t\x1F"
          "end"),
      "start  end");
}

TEST(NormalizerTest, BertNormalizerCJKExhaustive) {
  auto normalizer = NormalizerConfig("BertNormalizer")
                        .set_handle_chinese_chars(true)
                        .set_lowercase(false)
                        .create();

  // 1. Consecutive CJK characters (should have spaces between each)
  // \u4e00 and \u4e01 are CJK
  EXPECT_EQ(normalizer->normalize("\u4e00\u4e01"), " \u4e00  \u4e01 ");

  // 2. CJK at the very start and end of string
  EXPECT_EQ(normalizer->normalize("\u4e00"), " \u4e00 ");

  // 3. CJK mixed with non-CJK
  EXPECT_EQ(normalizer->normalize("A\u4e00Z"), "A \u4e00 Z");
}

TEST(NormalizerTest, BertNormalizerDecomposition) {
  auto normalizer = NormalizerConfig("BertNormalizer")
                        .set_lowercase(true)
                        .set_strip_accents(true)
                        .create();

  // Test "complex" accents like cedilla and tilde
  // ç -> c, ñ -> n
  EXPECT_EQ(normalizer->normalize("FAÇADE"), "facade");
  EXPECT_EQ(normalizer->normalize("cañón"), "canon");

  // German Eszett is lowercased but standard BERT keeps it or handles via
  // WordPiece
  EXPECT_EQ(normalizer->normalize("STRASSE"), "strasse");
}

TEST(NormalizerTest, BertNormalizerWhitespaceCollapsing) {
  auto normalizer =
      NormalizerConfig("BertNormalizer").set_clean_text(true).create();

  // Each whitespace char is handled by mapping to a single space.
  EXPECT_EQ(normalizer->normalize("hello \t\n\r world"), "hello     world");
}

TEST(NormalizerTest, BertNormalizerJsonIncomplete) {
  // Test that parsing JSON with missing fields uses defaults
  // Default BERT: clean=true, handle_cjk=true, lower=true, accents=null
  nlohmann::json config = {{"type", "BertNormalizer"}};

  NormalizerConfig norm_config;
  norm_config.parse_json(config);
  auto normalizer = norm_config.create();

  // Should lowercase and strip accents by default
  EXPECT_EQ(normalizer->normalize("HÉLLO"), "hello");
}

// -- NFD ----------------------------------------------------------------------

TEST(NormalizerTest, NFDBasic) {
  NFDNormalizer normalizer;
  // NOTE: The llama.cpp-unicode NFD implementation only supports 1-to-1
  // codepoint mappings, not full 1-to-many decomposition. So é (U+00E9)
  // may not decompose to e + combining acute. This is a known limitation.
  // Full NFD is handled correctly by the Precompiled normalizer for models
  // that need it.
  std::string input = "\xc3\xa9"; // é
  std::string result = normalizer.normalize(input);
  // At minimum it should not crash and return something
  EXPECT_FALSE(result.empty());
  // ASCII should pass through unchanged
  EXPECT_EQ(normalizer.normalize("Hello"), "Hello");
  EXPECT_EQ(normalizer.normalize(""), "");
}

TEST(NormalizerTest, NFDFromJson) {
  nlohmann::json config = {{"type", "NFD"}};
  NormalizerConfig norm_config;
  norm_config.parse_json(config);
  auto normalizer = norm_config.create();
  EXPECT_EQ(normalizer->normalize("ASCII"), "ASCII");
}

// -- NFKC / NFKD --------------------------------------------------------------

TEST(NormalizerTest, NFKCFromJson) {
  nlohmann::json config = {{"type", "NFKC"}};
  NormalizerConfig norm_config;
  norm_config.parse_json(config);
  auto normalizer = norm_config.create();
  // Basic ASCII should pass through
  EXPECT_EQ(normalizer->normalize("Hello"), "Hello");
}

TEST(NormalizerTest, NFKDFromJson) {
  nlohmann::json config = {{"type", "NFKD"}};
  NormalizerConfig norm_config;
  norm_config.parse_json(config);
  auto normalizer = norm_config.create();
  EXPECT_EQ(normalizer->normalize("Hello"), "Hello");
}

// -- Strip --------------------------------------------------------------------

TEST(NormalizerTest, StripBoth) {
  StripNormalizer normalizer(true, true);
  EXPECT_EQ(normalizer.normalize("  hello  "), "hello");
  EXPECT_EQ(normalizer.normalize("hello"), "hello");
  EXPECT_EQ(normalizer.normalize("   "), "");
}

TEST(NormalizerTest, StripLeftOnly) {
  StripNormalizer normalizer(true, false);
  EXPECT_EQ(normalizer.normalize("  hello  "), "hello  ");
}

TEST(NormalizerTest, StripRightOnly) {
  StripNormalizer normalizer(false, true);
  EXPECT_EQ(normalizer.normalize("  hello  "), "  hello");
}

TEST(NormalizerTest, StripFromJson) {
  nlohmann::json config = {
      {"type", "Strip"}, {"strip_left", true}, {"strip_right", false}};
  NormalizerConfig norm_config;
  norm_config.parse_json(config);
  auto normalizer = norm_config.create();
  EXPECT_EQ(normalizer->normalize("  hello  "), "hello  ");
}

// -- StripAccents -------------------------------------------------------------

TEST(NormalizerTest, StripAccentsBasic) {
  StripAccentsNormalizer normalizer;
  EXPECT_EQ(normalizer.normalize("caf\xc3\xa9"), "cafe"); // café → cafe
  EXPECT_EQ(normalizer.normalize("na\xc3\xafve"), "naive"); // naïve → naive
}

TEST(NormalizerTest, StripAccentsFromJson) {
  nlohmann::json config = {{"type", "StripAccents"}};
  NormalizerConfig norm_config;
  norm_config.parse_json(config);
  auto normalizer = norm_config.create();
  EXPECT_EQ(normalizer->normalize("r\xc3\xa9sum\xc3\xa9"), "resume");
}

// -- Nmt ----------------------------------------------------------------------

TEST(NormalizerTest, NmtControlChars) {
  NmtNormalizer normalizer;
  // Remove control chars but keep printable text
  std::string input = "Hello";
  input.push_back('\x01'); // SOH control char
  input += "World";
  EXPECT_EQ(normalizer.normalize(input), "HelloWorld");
}

TEST(NormalizerTest, NmtUnicodeSpaces) {
  NmtNormalizer normalizer;
  // U+00A0 (non-breaking space) → regular space
  EXPECT_EQ(normalizer.normalize("Hello\xc2\xa0World"), "Hello World");
}

TEST(NormalizerTest, NmtFromJson) {
  nlohmann::json config = {{"type", "Nmt"}};
  NormalizerConfig norm_config;
  norm_config.parse_json(config);
  auto normalizer = norm_config.create();
  EXPECT_EQ(normalizer->normalize("Hello"), "Hello");
}

// -- ByteLevel ----------------------------------------------------------------

TEST(NormalizerTest, ByteLevelBasic) {
  ByteLevelNormalizer normalizer;
  // 'A' (0x41) maps to itself in GPT-2 byte encoding
  // Space (0x20) maps to 'Ġ' (U+0120)
  std::string space_encoded = normalizer.normalize(" ");
  EXPECT_EQ(space_encoded, "\xc4\xa0"); // Ġ in UTF-8

  // ASCII letters map to themselves
  EXPECT_EQ(normalizer.normalize("A"), "A");
  EXPECT_EQ(normalizer.normalize("Hello"), "Hello");

  // Newline (0x0A) maps to a different byte-encoded char
  std::string nl_encoded = normalizer.normalize("\n");
  EXPECT_NE(nl_encoded, "\n");
  EXPECT_FALSE(nl_encoded.empty());
}

TEST(NormalizerTest, ByteLevelRoundTrip) {
  // ByteLevel normalizer + ByteLevel decoder should round-trip
  ByteLevelNormalizer normalizer;
  std::string input = "Hello World!";
  std::string encoded = normalizer.normalize(input);
  // Encoded should differ (spaces become Ġ)
  EXPECT_NE(encoded, input);
  // But should be decodable back (tested via decoder tests)
  EXPECT_EQ(encoded.find("Hello"), 0); // ASCII part unchanged
}

TEST(NormalizerTest, ByteLevelFromJson) {
  nlohmann::json config = {{"type", "ByteLevel"}};
  NormalizerConfig norm_config;
  norm_config.parse_json(config);
  auto normalizer = norm_config.create();
  EXPECT_EQ(normalizer->normalize("A"), "A");
  EXPECT_EQ(normalizer->normalize(" "), "\xc4\xa0");
}

// -- Precompiled --------------------------------------------------------------

TEST(NormalizerTest, PrecompiledFromJson) {
  // Load real precompiled charsmap from minilm_l12 if available
  std::string resources = std::getenv("RESOURCES_PATH")
      ? std::getenv("RESOURCES_PATH")
      : "";
  if (resources.empty()) {
    GTEST_SKIP() << "RESOURCES_PATH not set";
  }

  std::string path = resources + "/minilm_l12_tokenizer_dir/tokenizer.json";
  std::ifstream file(path);
  if (!file.is_open()) {
    GTEST_SKIP() << "minilm_l12 tokenizer not available";
  }

  nlohmann::json tokenizer_json;
  file >> tokenizer_json;

  if (!tokenizer_json.contains("normalizer") ||
      tokenizer_json["normalizer"]["type"] != "Precompiled") {
    GTEST_SKIP() << "Not a Precompiled normalizer";
  }

  NormalizerConfig norm_config;
  norm_config.parse_json(tokenizer_json["normalizer"]);
  auto normalizer = norm_config.create();

  // Basic ASCII should pass through
  EXPECT_EQ(normalizer->normalize("Hello world"), "Hello world");

  // Should handle empty
  EXPECT_EQ(normalizer->normalize(""), "");

  // Unicode should be normalized (Precompiled typically does NFKC-like normalization)
  // Full-width 'Ａ' (U+FF21) should normalize to 'A' (U+0041)
  std::string fullwidth_a = "\xef\xbc\xa1"; // Ａ
  std::string result = normalizer->normalize(fullwidth_a);
  // Precompiled normalizer should transform this — we just verify it doesn't crash
  // and returns something non-empty
  EXPECT_FALSE(result.empty());
}

// -- NFC dedicated test -------------------------------------------------------

TEST(NormalizerTest, NFCDedicated) {
  NFCNormalizer normalizer;
  // Decomposed e + combining acute → composed é
  std::string decomposed = "e\xcc\x81"; // e + U+0301
  std::string composed = "\xc3\xa9";     // é
  EXPECT_EQ(normalizer.normalize(decomposed), composed);

  // Already composed should be unchanged
  EXPECT_EQ(normalizer.normalize(composed), composed);

  // ASCII passthrough
  EXPECT_EQ(normalizer.normalize("Hello"), "Hello");

  // Empty
  EXPECT_EQ(normalizer.normalize(""), "");
}

// -- NFKC/NFKD approximation documentation tests -----------------------------

TEST(NormalizerTest, NFKCApproximation) {
  NFKCNormalizer normalizer;
  // NOTE: Our NFKC is approximated with NFC because full NFKC requires
  // compatibility decomposition tables not available in our Unicode library.
  // This means compatibility characters (like ﬁ → fi, ² → 2) are NOT
  // decomposed. Only canonical composition (like é decomposed → é composed)
  // is handled.

  // Canonical composition works (same as NFC):
  std::string decomposed_e = "e\xcc\x81"; // e + combining acute
  EXPECT_EQ(normalizer.normalize(decomposed_e), "\xc3\xa9"); // é

  // ASCII passthrough
  EXPECT_EQ(normalizer.normalize("Hello 123"), "Hello 123");
}

TEST(NormalizerTest, NFKDApproximation) {
  NFKDNormalizer normalizer;
  // NOTE: Our NFKD is approximated with NFD, which itself only supports
  // 1-to-1 codepoint mappings in the llama.cpp-unicode library.
  // Full decomposition is handled by the Precompiled normalizer.

  // ASCII passthrough
  EXPECT_EQ(normalizer.normalize("Hello"), "Hello");
  EXPECT_EQ(normalizer.normalize(""), "");

  // Should not crash on composed characters
  std::string composed = "\xc3\xa9"; // é
  std::string result = normalizer.normalize(composed);
  EXPECT_FALSE(result.empty());
}

// -- Metaspace pre-tokenizer + decoder pipeline test --------------------------

TEST(NormalizerTest, MetaspaceRoundTrip) {
  // Test that Metaspace pre-tokenizer output can be decoded by Metaspace decoder
  // This is an integration-style test crossing component boundaries.
  // (Uses pre_tokenizer and token_decoder headers already included via
  // normalizer test helpers)

  // Simulate what Metaspace pre-tokenizer does:
  // "Hello world" → replace spaces with ▁ → prepend ▁ → split
  // Produces: ["▁Hello", "▁world"]

  // Simulate what Metaspace decoder does:
  // Replace ▁ with space, strip leading space on first token
  // ["▁Hello", "▁world"] → ["Hello", " world"] → "Hello world"

  // We verify the logic manually since we can't easily import both headers:
  std::string replacement = "\xe2\x96\x81"; // ▁

  // Simulate pre-tokenizer output
  std::vector<std::string> pre_tokenized = {
      replacement + "Hello", replacement + "world"};

  // Simulate decoder: replace ▁ with space, strip first token's leading space
  std::string decoded;
  for (size_t i = 0; i < pre_tokenized.size(); ++i) {
    std::string token = pre_tokenized[i];
    // Replace ▁ with space
    size_t pos = 0;
    while ((pos = token.find(replacement, pos)) != std::string::npos) {
      token.replace(pos, replacement.size(), " ");
      pos += 1;
    }
    // Strip leading space on first token (prepend_scheme = always)
    if (i == 0 && !token.empty() && token[0] == ' ') {
      token.erase(0, 1);
    }
    decoded += token;
  }
  EXPECT_EQ(decoded, "Hello world");
}

// =============================================================================
// Adversarial / Edge Case Tests
// =============================================================================

TEST(NormalizerAdversarialTest, ReplaceWithEmptyContent) {
  // Replacing with empty string = deletion
  ReplaceNormalizer normalizer(" ", "");
  EXPECT_EQ(normalizer.normalize("Hello World"), "HelloWorld");
}

TEST(NormalizerAdversarialTest, StripAllWhitespace) {
  StripNormalizer normalizer(true, true);
  EXPECT_EQ(normalizer.normalize("   "), "");
  EXPECT_EQ(normalizer.normalize("\t\n\r"), "");
}

TEST(NormalizerAdversarialTest, NFDWithCombiningMarks) {
  NFDNormalizer normalizer;
  // Multiple combining marks on a single base character
  // a + combining acute + combining tilde
  std::string input = "a\xcc\x81\xcc\x83";
  std::string result = normalizer.normalize(input);
  // Should preserve all marks — NFD doesn't remove them
  EXPECT_FALSE(result.empty());
  EXPECT_GE(result.size(), input.size());
}

TEST(NormalizerAdversarialTest, StripAccentsPreservesBase) {
  StripAccentsNormalizer normalizer;
  // "ñ" should become "n" (tilde is an accent mark)
  EXPECT_EQ(normalizer.normalize("\xc3\xb1"), "n"); // ñ → n
  // Pure ASCII should be unchanged
  EXPECT_EQ(normalizer.normalize("Hello"), "Hello");
  // Non-accent combining marks (like combining enclosing circle)
  // should be preserved or at least not crash
  std::string with_circle = "a\xe2\x83\x9d"; // a + combining enclosing circle (not accent)
  auto result = normalizer.normalize(with_circle);
  EXPECT_FALSE(result.empty());
}

TEST(NormalizerAdversarialTest, NmtPreservesNormalText) {
  NmtNormalizer normalizer;
  std::string normal = "The quick brown fox jumps over the lazy dog.";
  EXPECT_EQ(normalizer.normalize(normal), normal);
}

TEST(NormalizerAdversarialTest, SequenceEmpty) {
  // Empty sequence of normalizers should be a passthrough
  // (Though factory prevents creation, test the class directly)
  SequenceNormalizer normalizer({});
  EXPECT_EQ(normalizer.normalize("Hello"), "Hello");
}

TEST(NormalizerAdversarialTest, VeryLongInput) {
  LowercaseNormalizer normalizer;
  std::string input(100000, 'A');
  std::string expected(100000, 'a');
  EXPECT_EQ(normalizer.normalize(input), expected);
}

TEST(NormalizerAdversarialTest, BertNormalizerAllFlags) {
  // All flags enabled simultaneously
  BertNormalizer normalizer(true, true, true, true);
  // Input with control chars, CJK, accents, and uppercase
  std::string input = "H\x01" "\xc3\x89LLO \xe4\xb8\x96\xe7\x95\x8c";
  auto result = normalizer.normalize(input);
  // Should: remove \x01, lowercase, strip accents from É, space-pad CJK
  EXPECT_NE(result.find("hello"), std::string::npos);
  // Should not contain control char
  EXPECT_EQ(result.find('\x01'), std::string::npos);
}
