/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// @lint-ignore-every LICENSELINT

#include <gtest/gtest.h>
#include <pytorch/tokenizers/token_decoder.h>
#include <unicode.h>

namespace tokenizers {

// Test ReplaceTokenDecoder
TEST(ReplaceTokenDecoderTest, TestBasicReplace) {
  ReplaceTokenDecoder decoder("_", " ");

  EXPECT_EQ(decoder.decode(std::vector<std::string>{"_Hello"})[0], " Hello");
  EXPECT_EQ(decoder.decode(std::vector<std::string>{"_world!"})[0], " world!");
  EXPECT_EQ(
      decoder.decode(std::vector<std::string>{"Hello_world"})[0],
      "Hello world");
  EXPECT_EQ(
      decoder.decode(std::vector<std::string>{"no_replacement"})[0],
      "no replacement");
}

TEST(ReplaceTokenDecoderTest, TestMultipleReplacements) {
  ReplaceTokenDecoder decoder("_", " ");

  EXPECT_EQ(
      decoder.decode(std::vector<std::string>{"_Hello_world_!"})[0],
      " Hello world !");
}

TEST(ReplaceTokenDecoderTest, TestEmptyPattern) {
  ReplaceTokenDecoder decoder("", "X");

  // Empty pattern should not cause infinite loop
  EXPECT_EQ(decoder.decode(std::vector<std::string>{"test"})[0], "test");
}

// Test ByteFallbackTokenDecoder
TEST(ByteFallbackTokenDecoderTest, TestValidHexTokens) {
  ByteFallbackTokenDecoder decoder;

  EXPECT_EQ(
      decoder.decode(std::vector<std::string>{"<0x41>"})[0],
      "A"); // 0x41 = 65 = 'A'
  EXPECT_EQ(
      decoder.decode(std::vector<std::string>{"<0x42>"})[0],
      "B"); // 0x42 = 66 = 'B'
  EXPECT_EQ(
      decoder.decode(std::vector<std::string>{"<0x20>"})[0],
      " "); // 0x20 = 32 = space
  EXPECT_EQ(
      decoder.decode(std::vector<std::string>{"<0x00>"})[0],
      std::string(1, '\0')); // null byte
}

TEST(ByteFallbackTokenDecoderTest, TestInvalidHexTokens) {
  ByteFallbackTokenDecoder decoder;

  // Invalid format - should return original token
  EXPECT_EQ(decoder.decode(std::vector<std::string>{"<0xGG>"})[0], "<0xGG>");
  EXPECT_EQ(decoder.decode(std::vector<std::string>{"<0x>"})[0], "<0x>");
  EXPECT_EQ(decoder.decode(std::vector<std::string>{"0x41>"})[0], "0x41>");
  EXPECT_EQ(decoder.decode(std::vector<std::string>{"<0x41"})[0], "<0x41");
  EXPECT_EQ(
      decoder.decode(std::vector<std::string>{"regular_token"})[0],
      "regular_token");
}

TEST(ByteFallbackTokenDecoderTest, TestOutOfRangeValues) {
  ByteFallbackTokenDecoder decoder;

  // Values > 255 should return original token
  EXPECT_EQ(decoder.decode(std::vector<std::string>{"<0x100>"})[0], "<0x100>");
  EXPECT_EQ(
      decoder.decode(std::vector<std::string>{"<0xFFFF>"})[0], "<0xFFFF>");
}

// Test FuseTokenDecoder
TEST(FuseTokenDecoderTest, TestPassthrough) {
  FuseTokenDecoder decoder;
  EXPECT_EQ(decoder.decode(std::vector<std::string>{"test"})[0], "test");
}

TEST(FuseTokenDecoderTest, TestFuseMultiple) {
  FuseTokenDecoder decoder;
  auto result = decoder.decode({"Hel", "lo", " ", "world"});
  ASSERT_EQ(result.size(), 1);
  EXPECT_EQ(result[0], "Hello world");
}

TEST(FuseTokenDecoderTest, TestFuseEmpty) {
  FuseTokenDecoder decoder;
  auto result = decoder.decode({});
  ASSERT_EQ(result.size(), 1);
  EXPECT_EQ(result[0], "");
}

TEST(FuseTokenDecoderTest, TestFuseWithSpecialChars) {
  FuseTokenDecoder decoder;
  auto result = decoder.decode({"<s>", "Hello", "</s>"});
  ASSERT_EQ(result.size(), 1);
  EXPECT_EQ(result[0], "<s>Hello</s>");
}

// Test SequenceTokenDecoder
TEST(SequenceTokenDecoderTest, TestEmptySequence) {
  std::vector<TokenDecoder::Ptr> decoders;
  SequenceTokenDecoder sequence_decoder(std::move(decoders));

  EXPECT_EQ(
      sequence_decoder.decode(std::vector<std::string>{"test"})[0], "test");
}

TEST(SequenceTokenDecoderTest, TestSingleDecoder) {
  std::vector<TokenDecoder::Ptr> decoders;
  decoders.push_back(TokenDecoder::Ptr(new ReplaceTokenDecoder("_", " ")));

  SequenceTokenDecoder sequence_decoder(std::move(decoders));

  EXPECT_EQ(
      sequence_decoder.decode(std::vector<std::string>{"_Hello"})[0], " Hello");
}

TEST(SequenceTokenDecoderTest, TestMultipleDecoders) {
  std::vector<TokenDecoder::Ptr> decoders;

  // Add Replace decoder to replace _ with space
  decoders.push_back(TokenDecoder::Ptr(new ReplaceTokenDecoder("_", " ")));

  // Add ByteFallback decoder
  decoders.push_back(TokenDecoder::Ptr(new ByteFallbackTokenDecoder()));

  // Add Fuse decoder
  decoders.push_back(TokenDecoder::Ptr(new FuseTokenDecoder()));

  SequenceTokenDecoder sequence_decoder(std::move(decoders));

  // Test cases
  EXPECT_EQ(
      sequence_decoder.decode(std::vector<std::string>{"_Hello"})[0], " Hello");
  EXPECT_EQ(
      sequence_decoder.decode(std::vector<std::string>{"_world!"})[0],
      " world!");
  EXPECT_EQ(
      sequence_decoder.decode(std::vector<std::string>{"<0x41>"})[0], "A");
  EXPECT_EQ(
      sequence_decoder.decode(std::vector<std::string>{"normal_token"})[0],
      "normal token");
}

TEST(SequenceTokenDecoderTest, TestComplexSequence) {
  std::vector<TokenDecoder::Ptr> decoders;

  // First replace underscores with spaces
  decoders.push_back(TokenDecoder::Ptr(new ReplaceTokenDecoder("_", " ")));

  // Then replace _ with spaces
  decoders.push_back(TokenDecoder::Ptr(new ReplaceTokenDecoder("_", " ")));

  // Then handle byte fallback
  decoders.push_back(TokenDecoder::Ptr(new ByteFallbackTokenDecoder()));

  SequenceTokenDecoder sequence_decoder(std::move(decoders));

  EXPECT_EQ(
      sequence_decoder.decode(std::vector<std::string>{"Hello_world"})[0],
      "Hello world");
  EXPECT_EQ(
      sequence_decoder.decode(std::vector<std::string>{"_test_token"})[0],
      " test token");
}

// Test TokenDecoderConfig parsing and creation
TEST(TokenDecoderConfigTest, TestReplaceConfig) {
  nlohmann::json config = {
      {"type", "Replace"}, {"pattern", {{"String", "_"}}}, {"content", " "}};

  TokenDecoderConfig decoder_config;
  decoder_config.parse_json(config);

  EXPECT_EQ(decoder_config.type, "Replace");
  EXPECT_EQ(decoder_config.replace_pattern, "_");
  EXPECT_EQ(decoder_config.replace_content, " ");

  auto decoder = decoder_config.create();
  EXPECT_EQ(decoder->decode(std::vector<std::string>{"_Hello"})[0], " Hello");
}

TEST(TokenDecoderConfigTest, TestSequenceConfig) {
  nlohmann::json config = {
      {"type", "Sequence"},
      {"decoders",
       {{{"type", "Replace"}, {"pattern", {{"String", "_"}}}, {"content", " "}},
        {{"type", "ByteFallback"}},
        {{"type", "Fuse"}}}}};

  TokenDecoderConfig decoder_config;
  decoder_config.parse_json(config);

  EXPECT_EQ(decoder_config.type, "Sequence");
  EXPECT_EQ(decoder_config.sequence_decoders->size(), 3);

  auto decoder = decoder_config.create();
  EXPECT_EQ(decoder->decode(std::vector<std::string>{"_Hello"})[0], " Hello");
  EXPECT_EQ(decoder->decode(std::vector<std::string>{"<0x41>"})[0], "A");
}

// Test StripTokenDecoder
TEST(StripTokenDecoderTest, TestBasicStrip) {
  StripTokenDecoder decoder("H", 1, 0); // Strip 'H' from start, 1 char max
  std::vector<std::string> tokens = {"Hey", " friend!", "HHH"};
  std::vector<std::string> expected = {"ey", " friend!", "HH"};
  EXPECT_EQ(decoder.decode(tokens), expected);

  StripTokenDecoder decoder2("y", 0, 1); // Strip 'y' from end, 1 char max
  tokens = {"Hey", " friend!"};
  expected = {"He", " friend!"};
  EXPECT_EQ(decoder2.decode(tokens), expected);
}

TEST(StripTokenDecoderTest, TestNoStrip) {
  StripTokenDecoder decoder("X", 1, 1); // Try to strip 'X'
  std::vector<std::string> tokens = {"Hello", "World"};
  std::vector<std::string> expected = {"Hello", "World"};
  EXPECT_EQ(decoder.decode(tokens), expected);

  StripTokenDecoder decoder2("H", 0, 0); // No stripping
  tokens = {"Hello", "World"};
  expected = {"Hello", "World"};
  EXPECT_EQ(decoder2.decode(tokens), expected);
}

TEST(StripTokenDecoderTest, TestFullStrip) {
  StripTokenDecoder decoder("H", 3, 0); // Strip 'H' from start, 3 chars max
  std::vector<std::string> tokens = {"HHH"};
  std::vector<std::string> expected = {""};
  EXPECT_EQ(decoder.decode(tokens), expected);

  StripTokenDecoder decoder2("H", 1, 1); // Strip 'H' from start and end
  tokens = {"HHH"};
  expected = {"H"};
  EXPECT_EQ(decoder2.decode(tokens), expected);
}

TEST(StripTokenDecoderTest, TestMixedContent) {
  StripTokenDecoder decoder(" ", 1, 1); // Strip spaces from start and end
  std::vector<std::string> tokens = {" Hello World ", "  TrimMe  "};
  std::vector<std::string> expected = {"Hello World", " TrimMe "};
  EXPECT_EQ(decoder.decode(tokens), expected);

  StripTokenDecoder decoder2("H", 5, 5); // Strip 'H's from start and end
  tokens = {"HHHHelloHHH", "HWorldH"};
  expected = {"ello", "World"};
  EXPECT_EQ(decoder2.decode(tokens), expected);
}

TEST(StripTokenDecoderTest, TestUnicodeContent) {
  // Test with a multi-byte character for content
  StripTokenDecoder decoder("_", 1, 0); // Strip '_' from start, 1 char max
  std::vector<std::string> tokens = {"_Hello", "__World"};
  std::vector<std::string> expected = {"Hello", "_World"};
  EXPECT_EQ(decoder.decode(tokens), expected);

  // Test stripping from end with Unicode
  StripTokenDecoder decoder2("∎", 0, 1); // Strip '∎' from end, 1 char max
  tokens = {"Hello∎", "World∎∎"};
  expected = {"Hello", "World∎"};
  EXPECT_EQ(decoder2.decode(tokens), expected);
}

TEST(TokenDecoderConfigTest, TestStripConfig) {
  nlohmann::json config = {
      {"type", "Strip"}, {"content", "_"}, {"start", 1}, {"stop", 0}};

  TokenDecoderConfig decoder_config;
  decoder_config.parse_json(config);

  EXPECT_EQ(decoder_config.type, "Strip");
  EXPECT_EQ(decoder_config.strip_content, "_");
  EXPECT_EQ(decoder_config.strip_start, 1);
  EXPECT_EQ(decoder_config.strip_stop, 0);

  auto decoder = decoder_config.create();
  std::vector<std::string> tokens = {"_Hello", "__World"};
  std::vector<std::string> expected = {"Hello", "_World"};
  EXPECT_EQ(decoder->decode(tokens), expected);
}

TEST(WordPieceTokenDecoderTest, BasicDecodingNoCleanup) {
  WordPieceTokenDecoder decoder("##", false);
  std::vector<std::string> tokens = {"##uelo", "Ara", "##új", "##o", "No", "##guera"};
  std::vector<std::string> expected = {"##uelo", " Ara", "új", "o", " No", "guera"};
  EXPECT_EQ(decoder.decode(tokens), expected);
}

TEST(WordPieceTokenDecoderTest, DefaultCleanup) {
  WordPieceTokenDecoder decoder;
  std::vector<std::string> tokens = {"hello", ",", "##world", "."};
  std::vector<std::string> expected = {"hello", ",", "world", "."};
  EXPECT_EQ(decoder.decode(tokens), expected);
}

TEST(TokenDecoderConfigTest, TestWordPieceConfig) {
  nlohmann::json config = {
      {"type", "WordPiece"}, {"prefix", "@@"}, {"cleanup", true}};

  TokenDecoderConfig decoder_config;
  decoder_config.parse_json(config);

  EXPECT_EQ(decoder_config.type, "WordPiece");
  EXPECT_EQ(decoder_config.wordpiece_prefix, "@@");
  EXPECT_EQ(decoder_config.wordpiece_cleanup, true);

  auto decoder = decoder_config.create();
  std::vector<std::string> tokens = {"@@hello", "world"};
  std::vector<std::string> expected = {"@@hello", " world"};
  EXPECT_EQ(decoder->decode(tokens), expected);
}

TEST(WordPieceTokenDecoderTest, EnglishContractionCleanup) {
  WordPieceTokenDecoder decoder("##", true);
  std::vector<std::string> tokens = {"it", "##'", "##s", "i", "##'", "##m"};
  std::vector<std::string> result = decoder.decode(tokens);

  std::string joined;
  for (const auto& s : result) {
    joined += s;
  }
  EXPECT_TRUE(joined.find("it's") != std::string::npos || joined.find("it 's") != std::string::npos);
  EXPECT_TRUE(joined.find("i'm") != std::string::npos || joined.find("i 'm") != std::string::npos);
}

TEST(WordPieceTokenDecoderTest, MisplacedPrefixes) {
  WordPieceTokenDecoder decoder("##", true);
  std::vector<std::string> tokens = {"##hello", "world", "##!", "##!"};
  std::vector<std::string> expected = {"##hello", " world", "!", "!"};
  EXPECT_EQ(decoder.decode(tokens), expected);
}

TEST(WordPieceTokenDecoderTest, PunctuationBoundary) {
  WordPieceTokenDecoder decoder("##", true);
  std::vector<std::string> tokens = {"hello", ",", "##world"};
  std::vector<std::string> result = decoder.decode(tokens);

  std::vector<std::string> expected = {"hello", ",", "world"};
  EXPECT_EQ(result, expected);
}

// =============================================================================
// ByteLevelTokenDecoder
// =============================================================================

TEST(ByteLevelTokenDecoderTest, BasicASCII) {
  ByteLevelTokenDecoder decoder;
  // GPT-2 byte encoding: 'H' (0x48) maps to 'H', space (0x20) maps to 'Ġ'
  // The decoder reverses this mapping.
  // For simple ASCII that maps to itself, input == output
  auto result = decoder.decode({"Hello"});
  ASSERT_EQ(result.size(), 1);
  EXPECT_EQ(result[0], "Hello");
}

TEST(ByteLevelTokenDecoderTest, ByteEncodedSpace) {
  ByteLevelTokenDecoder decoder;
  // unicode_byte_to_utf8(0x20) = "Ġ" — the byte-level space encoding
  std::string encoded_space = unicode_byte_to_utf8(0x20);
  auto result = decoder.decode({encoded_space + "Hello"});
  ASSERT_EQ(result.size(), 1);
  EXPECT_EQ(result[0], " Hello");
}

TEST(ByteLevelTokenDecoderTest, MultipleTokens) {
  ByteLevelTokenDecoder decoder;
  std::string sp = unicode_byte_to_utf8(0x20);
  auto result = decoder.decode({sp + "Hello", sp + "world"});
  // ByteLevel decoder fuses all tokens into one string
  ASSERT_EQ(result.size(), 1);
  EXPECT_EQ(result[0], " Hello world");
}

// =============================================================================
// MetaspaceTokenDecoder
// =============================================================================

TEST(MetaspaceTokenDecoderTest, Basic) {
  MetaspaceTokenDecoder decoder;
  std::vector<std::string> tokens = {
      "\xe2\x96\x81Hello", "\xe2\x96\x81world", "!"};
  auto result = decoder.decode(tokens);
  ASSERT_EQ(result.size(), 3);
  EXPECT_EQ(result[0], "Hello"); // Leading ▁ stripped on first token
  EXPECT_EQ(result[1], " world");
  EXPECT_EQ(result[2], "!");
}

TEST(MetaspaceTokenDecoderTest, NeverPrepend) {
  MetaspaceTokenDecoder decoder("\xe2\x96\x81", "never");
  std::vector<std::string> tokens = {
      "\xe2\x96\x81Hello", "\xe2\x96\x81world"};
  auto result = decoder.decode(tokens);
  ASSERT_EQ(result.size(), 2);
  // With "never" prepend_scheme, don't strip leading space on first token
  EXPECT_EQ(result[0], " Hello");
  EXPECT_EQ(result[1], " world");
}

TEST(MetaspaceTokenDecoderTest, Empty) {
  MetaspaceTokenDecoder decoder;
  auto result = decoder.decode({});
  EXPECT_TRUE(result.empty());
}

TEST(MetaspaceTokenDecoderTest, FromJson) {
  nlohmann::json config = {
      {"type", "Metaspace"},
      {"replacement", "\xe2\x96\x81"},
      {"prepend_scheme", "always"}};
  TokenDecoderConfig dec_config;
  dec_config.parse_json(config);
  auto decoder = dec_config.create();
  auto result =
      decoder->decode({"\xe2\x96\x81Hello", "\xe2\x96\x81world"});
  ASSERT_EQ(result.size(), 2);
  EXPECT_EQ(result[0], "Hello");
  EXPECT_EQ(result[1], " world");
}

// =============================================================================
// BPETokenDecoder
// =============================================================================

TEST(BPETokenDecoderTest, Basic) {
  BPETokenDecoder decoder("</w>");
  std::vector<std::string> tokens = {"Hel", "lo</w>", "world</w>"};
  auto result = decoder.decode(tokens);
  ASSERT_EQ(result.size(), 3);
  EXPECT_EQ(result[0], "Hel");
  EXPECT_EQ(result[1], "lo "); // suffix removed, space added
  EXPECT_EQ(result[2], "world"); // last token, no trailing space
}

TEST(BPETokenDecoderTest, NoSuffix) {
  BPETokenDecoder decoder("</w>");
  std::vector<std::string> tokens = {"hello", "world"};
  auto result = decoder.decode(tokens);
  EXPECT_EQ(result[0], "hello");
  EXPECT_EQ(result[1], "world");
}

TEST(BPETokenDecoderTest, FromJson) {
  nlohmann::json config = {{"type", "BPEDecoder"}, {"suffix", "</w>"}};
  TokenDecoderConfig dec_config;
  dec_config.parse_json(config);
  auto decoder = dec_config.create();
  auto result = decoder->decode({"He", "llo</w>", "world</w>"});
  ASSERT_EQ(result.size(), 3);
  EXPECT_EQ(result[0], "He");
  EXPECT_EQ(result[1], "llo ");
  EXPECT_EQ(result[2], "world");
}

// =============================================================================
// CTCTokenDecoder
// =============================================================================

TEST(CTCTokenDecoderTest, Basic) {
  CTCTokenDecoder decoder("<pad>", "|", true);
  std::vector<std::string> tokens = {
      "<pad>", "H", "H", "E", "<pad>", "L", "L", "O", "|", "W", "O", "R",
      "L", "D"};
  auto result = decoder.decode(tokens);
  // After dedup + pad removal + | → space + cleanup
  ASSERT_EQ(result.size(), 1);
  EXPECT_EQ(result[0], "HELO WORLD");
}

TEST(CTCTokenDecoderTest, NoCleanup) {
  CTCTokenDecoder decoder("<pad>", "|", false);
  std::vector<std::string> tokens = {"H", "E", "|", "L", "O"};
  auto result = decoder.decode(tokens);
  // Without cleanup, returns individual pieces
  ASSERT_EQ(result.size(), 5);
  EXPECT_EQ(result[0], "H");
  EXPECT_EQ(result[1], "E");
  EXPECT_EQ(result[2], " ");
  EXPECT_EQ(result[3], "L");
  EXPECT_EQ(result[4], "O");
}

TEST(CTCTokenDecoderTest, FromJson) {
  nlohmann::json config = {
      {"type", "CTC"},
      {"pad_token", "<pad>"},
      {"word_delimiter_token", "|"},
      {"cleanup", true}};
  TokenDecoderConfig dec_config;
  dec_config.parse_json(config);
  auto decoder = dec_config.create();
  auto result = decoder->decode({"H", "E", "L", "L", "O"});
  ASSERT_EQ(result.size(), 1);
  EXPECT_EQ(result[0], "HELO");
}

// =============================================================================
// Adversarial / Edge Case Tests
// =============================================================================

TEST(DecoderAdversarialTest, MetaspaceEmptyTokens) {
  MetaspaceTokenDecoder decoder;
  auto result = decoder.decode({"", "", ""});
  ASSERT_EQ(result.size(), 3);
  // Empty tokens should stay empty (first one won't have leading space to strip)
  EXPECT_EQ(result[0], "");
}

TEST(DecoderAdversarialTest, MetaspaceSingleToken) {
  MetaspaceTokenDecoder decoder;
  auto result = decoder.decode({"\xe2\x96\x81"});
  ASSERT_EQ(result.size(), 1);
  // Single ▁ → decoded to space → stripped on first token → empty
  EXPECT_EQ(result[0], "");
}

TEST(DecoderAdversarialTest, BPEDecoderNoSuffixAnywhere) {
  BPETokenDecoder decoder("</w>");
  auto result = decoder.decode({"hel", "lo", "wor", "ld"});
  ASSERT_EQ(result.size(), 4);
  // No suffix found — tokens pass through unchanged
  EXPECT_EQ(result[0], "hel");
  EXPECT_EQ(result[1], "lo");
  EXPECT_EQ(result[2], "wor");
  EXPECT_EQ(result[3], "ld");
}

TEST(DecoderAdversarialTest, CTCAllPadTokens) {
  CTCTokenDecoder decoder("<pad>", "|", true);
  auto result = decoder.decode({"<pad>", "<pad>", "<pad>"});
  ASSERT_EQ(result.size(), 1);
  EXPECT_EQ(result[0], "");
}

TEST(DecoderAdversarialTest, CTCAllDuplicates) {
  CTCTokenDecoder decoder("<pad>", "|", true);
  auto result = decoder.decode({"A", "A", "A", "A"});
  ASSERT_EQ(result.size(), 1);
  EXPECT_EQ(result[0], "A");
}

TEST(DecoderAdversarialTest, WordPieceAllContinuations) {
  WordPieceTokenDecoder decoder("##", true);
  auto result = decoder.decode({"##a", "##b", "##c"});
  // First token keeps ##, rest get prefix stripped
  ASSERT_EQ(result.size(), 3);
  EXPECT_EQ(result[0], "##a");
  EXPECT_EQ(result[1], "b");
  EXPECT_EQ(result[2], "c");
}

TEST(DecoderAdversarialTest, SequenceDecoderOrder) {
  // Replace ▁ with space, THEN fuse — order matters
  auto replace = std::make_shared<ReplaceTokenDecoder>(
      "\xe2\x96\x81", " ");
  auto fuse = std::make_shared<FuseTokenDecoder>();
  SequenceTokenDecoder decoder({replace, fuse});

  auto result = decoder.decode(
      {"\xe2\x96\x81Hello", "\xe2\x96\x81world"});
  ASSERT_EQ(result.size(), 1);
  EXPECT_EQ(result[0], " Hello world");
}

TEST(DecoderAdversarialTest, StripDecoderEmptyToken) {
  StripTokenDecoder decoder(" ", 1, 1);
  auto result = decoder.decode({""});
  ASSERT_EQ(result.size(), 1);
  EXPECT_EQ(result[0], "");
}

TEST(DecoderAdversarialTest, ByteFallbackPartialHex) {
  ByteFallbackTokenDecoder decoder;
  // Partial hex patterns should pass through unchanged
  auto result = decoder.decode({"<0x", "<0x4", "<0xZZ>"});
  ASSERT_EQ(result.size(), 3);
  EXPECT_EQ(result[0], "<0x");
  EXPECT_EQ(result[1], "<0x4");
  EXPECT_EQ(result[2], "<0xZZ>");
}

} // namespace tokenizers
