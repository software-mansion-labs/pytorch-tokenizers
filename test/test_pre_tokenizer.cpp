/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Third Party
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <re2/re2.h>

// Local
#include <pytorch/tokenizers/pre_tokenizer.h>

using json = nlohmann::json;
using namespace tokenizers;

// Helpers /////////////////////////////////////////////////////////////////////

static void assert_split_match(
    const PreTokenizer& ptok,
    const std::string& prompt,
    const std::vector<std::string>& expected) {
  const auto& got = ptok.pre_tokenize(prompt);
  EXPECT_EQ(expected.size(), got.size());
  for (auto i = 0; i < got.size(); ++i) {
    EXPECT_EQ(expected[i], got[i]);
  }
}

// RegexPreTokenizer ///////////////////////////////////////////////////////////
class RegexPreTokenizerTest : public ::testing::Test {};

// Test the basic construction
TEST_F(RegexPreTokenizerTest, Construct) {
  RegexPreTokenizer ptok("[0-9]+");
  // Verify the regex actually works after construction
  auto result = ptok.pre_tokenize("abc123def456");
  ASSERT_EQ(result.size(), 2);
  EXPECT_EQ(result[0], "123");
  EXPECT_EQ(result[1], "456");
}

// Test basic splitting using the expression for Tiktoken
TEST_F(RegexPreTokenizerTest, TiktokenExpr) {
  RegexPreTokenizer ptok(
      R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+)");
  assert_split_match(
      ptok, "How are you doing?", {"How", " are", " you", " doing", "?"});
}

// DigitsPreTokenizer //////////////////////////////////////////////////////////
class DigitsPreTokenizerTest : public ::testing::Test {};

// Test digit splitting with individual digits
TEST_F(DigitsPreTokenizerTest, IndividualDigits) {
  DigitsPreTokenizer ptok(true);
  assert_split_match(
      ptok,
      "The number 1 then 234 then 5.",
      {"The number ", "1", " then ", "2", "3", "4", " then ", "5", "."});
}

// Test digit splitting with contiguous digits
TEST_F(DigitsPreTokenizerTest, ContiguousDigits) {
  DigitsPreTokenizer ptok(false);
  assert_split_match(
      ptok,
      "The number 1 then 234 then 5.",
      {"The number ", "1", " then ", "234", " then ", "5", "."});
}

// ByteLevelPreTokenizer ///////////////////////////////////////////////////////
class ByteLevelPreTokenizerTest : public ::testing::Test {};

TEST_F(ByteLevelPreTokenizerTest, PreTokenizeDefault) {
  ByteLevelPreTokenizer ptok;
  assert_split_match(ptok, "Hello World", {"ĠHello", "ĠWorld"});
  assert_split_match(
      ptok,
      "The number 1 then 234 then 5.",
      {"ĠThe", "Ġnumber", "Ġ1", "Ġthen", "Ġ234", "Ġthen", "Ġ5", "."});
}

TEST_F(ByteLevelPreTokenizerTest, PreTokenizeNoPrefix) {
  ByteLevelPreTokenizer ptok(false);
  assert_split_match(ptok, "Hello World", {"Hello", "ĠWorld"});
}

TEST_F(ByteLevelPreTokenizerTest, PreTokenizeCustomRegex) {
  ByteLevelPreTokenizer ptok(false, R"(o)");
  assert_split_match(ptok, "Hello World", {"Hell", "o", "ĠW", "o", "rld"});
}

// SequencePreTokenizer ////////////////////////////////////////////////////////
class SequencePreTokenizerTest : public ::testing::Test {};

TEST_F(SequencePreTokenizerTest, PreTokenizeDigitAndByteLevel) {
  PreTokenizer::Ptr dptok(new DigitsPreTokenizer(true));
  PreTokenizer::Ptr bptok(new ByteLevelPreTokenizer(false));
  SequencePreTokenizer ptok({dptok, bptok});
  assert_split_match(
      ptok,
      "The number 1 then 234 then 5.",
      {"The",
       "Ġnumber",
       "Ġ",
       "1",
       "Ġthen",
       "Ġ",
       "2",
       "3",
       "4",
       "Ġthen",
       "Ġ",
       "5",
       "."});
}

// BertPreTokenizer ////////////////////////////////////////////////////////////
class BertPreTokenizerTest : public ::testing::Test {};

TEST_F(BertPreTokenizerTest, Basic) {
  BertPreTokenizer ptok;
  assert_split_match(
      ptok,
      "Hey friend!     How are you?!?",
      {"Hey", "friend", "!", "How", "are", "you", "?", "!", "?"});
}

TEST_F(BertPreTokenizerTest, ChineseChars) {
  BertPreTokenizer ptok;
  // Note: BertNormalizer handles adding spaces around Chinese characters.
  // We simulate the normalized string here.
  assert_split_match(
      ptok,
      " 野  口  里  佳  Noguchi Rika",
      {"野", "口", "里", "佳", "Noguchi", "Rika"});
}

// -- Extended BertPreTokenizer Tests ------------------------------------------

TEST_F(BertPreTokenizerTest, PunctuationSplitting) {
  BertPreTokenizer ptok;
  // BERT should isolate every punctuation mark into its own token
  assert_split_match(ptok, "hello,world!", {"hello", ",", "world", "!"});

  assert_split_match(ptok, "one(two)three", {"one", "(", "two", ")", "three"});
}

TEST_F(BertPreTokenizerTest, MultipleWhitespaces) {
  BertPreTokenizer ptok;
  // BERT should collapse/ignore multiple spaces and tabs
  assert_split_match(ptok, "hello \t \n world", {"hello", "world"});
}

TEST_F(BertPreTokenizerTest, CJKAndPunctuationMix) {
  BertPreTokenizer ptok;

  // BERT treats CJK characters as individual units (handled by Normalizer
  // usually)

  // but the pre-tokenizer must ensure they aren't clumped with punctuation.

  // Note: BertNormalizer adds spaces around CJK characters.

  assert_split_match(ptok, "你好,world.", {"你", "好", ",", "world", "."});
}

TEST_F(BertPreTokenizerTest, Contractions) {
  BertPreTokenizer ptok;
  // BERT pre-tokenization splits on the apostrophe as punctuation
  assert_split_match(ptok, "don't stop", {"don", "'", "t", "stop"});
}

TEST_F(BertPreTokenizerTest, NumbersAndSymbols) {
  BertPreTokenizer ptok;
  // Verify symbols like $ or % are treated as punctuation tokens
  assert_split_match(
      ptok, "Cost: $100.00", {"Cost", ":", "$", "100", ".", "00"});
}

// PreTokenizerConfig //////////////////////////////////////////////////////////
//
// NOTE: When adding a new pre-tokenizer or changing arguments, add it to these
//  tests!
class PreTokenizerConfigTest : public ::testing::Test {};

TEST_F(PreTokenizerConfigTest, AllTypesSuccess) {
  // Regex
  PreTokenizerConfig("Split").set_pattern(R"(o)").create();

  // Digits
  PreTokenizerConfig("Digits").create();
  PreTokenizerConfig("Digits").set_individual_digits(true).create();
  PreTokenizerConfig("Digits").set_individual_digits(false).create();

  // ByteLevel
  PreTokenizerConfig("ByteLevel").create();
  PreTokenizerConfig("ByteLevel").set_pattern(R"(o)").create();
  PreTokenizerConfig("ByteLevel").set_add_prefix_space(true).create();
  PreTokenizerConfig("ByteLevel")
      .set_add_prefix_space(false)
      .set_pattern(R"(o)")
      .create();

  // Sequence
  PreTokenizerConfig("Sequence")
      .set_pretokenizers(
          {PreTokenizerConfig("Digits"), PreTokenizerConfig("ByteLevel")})
      .create();

  // Bert
  PreTokenizerConfig("BertPreTokenizer").create();
}

TEST_F(PreTokenizerConfigTest, AllTypesFailureCases) {
  // Regex
  EXPECT_THROW(PreTokenizerConfig("Split").create(), std::runtime_error);

  // Sequence
  EXPECT_THROW(PreTokenizerConfig("Sequence").create(), std::runtime_error);
  EXPECT_THROW(
      PreTokenizerConfig("Sequence").set_pretokenizers({}).create(),
      std::runtime_error);
  EXPECT_THROW(
      PreTokenizerConfig("Sequence")
          .set_pretokenizers({PreTokenizerConfig("Split")})
          .create(),
      std::runtime_error);

  // Unsupported
  EXPECT_THROW(PreTokenizerConfig("Unsupported").create(), std::runtime_error);
}

TEST_F(PreTokenizerConfigTest, ParseJson) {
  PreTokenizerConfig config;
  const auto ptok = config
                        .parse_json(
                            json{
                                {"type", "Sequence"},
                                {"pretokenizers",
                                 json{
                                     json{
                                         {"type", "Digits"},
                                         {"individual_digits", true},
                                     },
                                     json{
                                         {"type", "ByteLevel"},
                                         {"add_prefix_space", false},
                                     },
                                 }},
                            })
                        .create();
  assert_split_match(
      *ptok,
      "The number 1 then 234 then 5.",
      {"The",
       "Ġnumber",
       "Ġ",
       "1",
       "Ġthen",
       "Ġ",
       "2",
       "3",
       "4",
       "Ġthen",
       "Ġ",
       "5",
       "."});
}

TEST_F(PreTokenizerConfigTest, ParseJsonOptionalKey) {
  PreTokenizerConfig config;
  const auto ptok = config
                        .parse_json(
                            json{
                                {"type", "Digits"},
                            })
                        .create();
  assert_split_match(
      *ptok,
      "The number 1 then 234 then 5.",
      {"The number ", "1", " then ", "234", " then ", "5", "."});
}

TEST_F(PreTokenizerConfigTest, Split) {
  PreTokenizerConfig config;
  const auto ptok =
      config
          .parse_json(
              json{
                  {"type", "Split"},
                  {"pattern",
                   {{"Regex",
                     R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+)"}}},
              })
          .create();
  assert_split_match(*ptok, "Hello World", {"Hello", " World"});
}

TEST_F(PreTokenizerConfigTest, SplitWithStringPattern) {
  PreTokenizerConfig config;
  const auto ptok = config
                        .parse_json(
                            json{
                                {"type", "Split"},
                                {"pattern", {{"String", " "}}},
                            })
                        .create();
  assert_split_match(*ptok, "Hello world!", {"Hello", "world!"});
}

TEST_F(PreTokenizerConfigTest, SplitWithStringPatternSpecialChars) {
  PreTokenizerConfig config;
  const auto ptok = config
                        .parse_json(
                            json{
                                {"type", "Split"},
                                {"pattern", {{"String", "."}}},
                            })
                        .create();
  assert_split_match(*ptok, "Hello.world.test", {"Hello", "world", "test"});
}

TEST_F(PreTokenizerConfigTest, SplitWithStringPatternNoMatches) {
  PreTokenizerConfig config;
  const auto ptok = config
                        .parse_json(
                            json{
                                {"type", "Split"},
                                {"pattern", {{"String", "xyz"}}},
                            })
                        .create();
  assert_split_match(*ptok, "Hello world", {"Hello world"});
}

TEST_F(PreTokenizerConfigTest, SplitWithRegexMetaCharacters) {
  PreTokenizerConfig config;
  const auto ptok = config
                        .parse_json(
                            json{
                                {"type", "Split"},
                                {"pattern", {{"String", "+"}}},
                            })
                        .create();
  assert_split_match(*ptok, "a+b+c", {"a", "b", "c"});
}

TEST_F(PreTokenizerConfigTest, SplitWithRegexBrackets) {
  PreTokenizerConfig config;
  const auto ptok = config
                        .parse_json(
                            json{
                                {"type", "Split"},
                                {"pattern", {{"String", "["}}},
                            })
                        .create();
  assert_split_match(*ptok, "a[b[c", {"a", "b", "c"});
}

TEST_F(PreTokenizerConfigTest, SplitEmptyInput) {
  PreTokenizerConfig config;
  const auto ptok = config
                        .parse_json(
                            json{
                                {"type", "Split"},
                                {"pattern", {{"String", " "}}},
                            })
                        .create();
  assert_split_match(*ptok, "", {""});
}

TEST_F(PreTokenizerConfigTest, SplitSingleCharacterInput) {
  PreTokenizerConfig config;
  const auto ptok = config
                        .parse_json(
                            json{
                                {"type", "Split"},
                                {"pattern", {{"String", " "}}},
                            })
                        .create();
  assert_split_match(*ptok, "a", {"a"});
}

TEST_F(PreTokenizerConfigTest, SplitWithMergedWithPrevious) {
  PreTokenizerConfig config;
  const auto ptok = config
                        .parse_json(
                            json{
                                {"type", "Split"},
                                {"pattern", {{"String", "-"}}},
                                {"behavior", "MergedWithPrevious"},
                                {"invert", false},
                            })
                        .create();
  // Example from docstring: "the-final--countdown" with delimiter "-"
  // -> ["the-", "final-", "-", "countdown"]
  assert_split_match(
      *ptok, "the-final--countdown", {"the-", "final-", "-", "countdown"});
}

TEST_F(PreTokenizerConfigTest, SplitWithMergedWithPreviousSpaces) {
  PreTokenizerConfig config;
  const auto ptok = config
                        .parse_json(
                            json{
                                {"type", "Split"},
                                {"pattern", {{"String", " "}}},
                                {"behavior", "MergedWithPrevious"},
                                {"invert", false},
                            })
                        .create();
  assert_split_match(*ptok, "Hello world test", {"Hello ", "world ", "test"});
}

TEST_F(PreTokenizerConfigTest, SplitWithMergedWithPreviousStartingDelimiter) {
  PreTokenizerConfig config;
  const auto ptok = config
                        .parse_json(
                            json{
                                {"type", "Split"},
                                {"pattern", {{"String", "-"}}},
                                {"behavior", "MergedWithPrevious"},
                                {"invert", false},
                            })
                        .create();
  assert_split_match(*ptok, "-hello-world", {"-", "hello-", "world"});
}

TEST_F(PreTokenizerConfigTest, SplitWithMergedWithPreviousEndingDelimiter) {
  PreTokenizerConfig config;
  const auto ptok = config
                        .parse_json(
                            json{
                                {"type", "Split"},
                                {"pattern", {{"String", "-"}}},
                                {"behavior", "MergedWithPrevious"},
                                {"invert", false},
                            })
                        .create();
  assert_split_match(*ptok, "hello-world-", {"hello-", "world-"});
}

TEST_F(PreTokenizerConfigTest, SplitWithMergedWithNext) {
  PreTokenizerConfig config;
  auto pre_tokenizer = config
                           .parse_json(
                               json{
                                   {"type", "Split"},
                                   {"pattern", {{"String", "-"}}},
                                   {"behavior", "MergedWithNext"},
                                   {"invert", false},
                               })
                           .create();
  auto result = pre_tokenizer->pre_tokenize("the-final--countdown");
  ASSERT_EQ(result.size(), 4);
  EXPECT_EQ(result[0], "the");
  EXPECT_EQ(result[1], "-final");
  EXPECT_EQ(result[2], "-");
  EXPECT_EQ(result[3], "-countdown");
}

TEST_F(PreTokenizerConfigTest, SplitWithInvertTrueString) {
  // invert=true on a String pattern flips from delimiter mode to keep mode
  // The string "-" is normally a delimiter; with invert, we keep the matches
  PreTokenizerConfig config;
  auto pre_tokenizer = config
                           .parse_json(
                               json{
                                   {"type", "Split"},
                                   {"pattern", {{"String", "-"}}},
                                   {"behavior", "Removed"},
                                   {"invert", true},
                               })
                           .create();
  auto result = pre_tokenizer->pre_tokenize("the-final--countdown");
  // With invert=true on string, the "-" matches are kept as tokens
  ASSERT_EQ(result.size(), 3);
  EXPECT_EQ(result[0], "-");
  EXPECT_EQ(result[1], "-");
  EXPECT_EQ(result[2], "-");
}

// =============================================================================
// Metaspace
// =============================================================================

TEST(MetaspacePreTokenizerTest, Basic) {
  MetaspacePreTokenizer pt;
  auto result = pt.pre_tokenize("Hello world");
  ASSERT_EQ(result.size(), 2);
  EXPECT_EQ(result[0], "\xe2\x96\x81Hello");  // ▁Hello
  EXPECT_EQ(result[1], "\xe2\x96\x81world");  // ▁world
}

TEST(MetaspacePreTokenizerTest, AlreadyHasReplacement) {
  MetaspacePreTokenizer pt;
  auto result = pt.pre_tokenize("\xe2\x96\x81Hello");
  // Already starts with ▁, should not double-prepend
  ASSERT_EQ(result.size(), 1);
  EXPECT_EQ(result[0], "\xe2\x96\x81Hello");
}

TEST(MetaspacePreTokenizerTest, NeverPrepend) {
  MetaspacePreTokenizer pt(
      "\xe2\x96\x81", MetaspacePreTokenizer::PrependScheme::Never, true);
  auto result = pt.pre_tokenize("Hello world");
  ASSERT_EQ(result.size(), 2);
  EXPECT_EQ(result[0], "Hello");
  EXPECT_EQ(result[1], "\xe2\x96\x81world");
}

TEST(MetaspacePreTokenizerTest, NoSplit) {
  MetaspacePreTokenizer pt(
      "\xe2\x96\x81", MetaspacePreTokenizer::PrependScheme::Always, false);
  auto result = pt.pre_tokenize("Hello world");
  ASSERT_EQ(result.size(), 1);
  EXPECT_EQ(result[0], "\xe2\x96\x81Hello\xe2\x96\x81world");
}

TEST(MetaspacePreTokenizerTest, Empty) {
  MetaspacePreTokenizer pt;
  auto result = pt.pre_tokenize("");
  EXPECT_TRUE(result.empty());
}

TEST(MetaspacePreTokenizerTest, FromJson) {
  PreTokenizerConfig config;
  config.parse_json(json{
      {"type", "Metaspace"},
      {"replacement", "\xe2\x96\x81"},
      {"prepend_scheme", "always"},
      {"split", true}});
  auto pt = config.create();
  auto result = pt->pre_tokenize("Hello world");
  ASSERT_EQ(result.size(), 2);
  EXPECT_EQ(result[0], "\xe2\x96\x81Hello");
  EXPECT_EQ(result[1], "\xe2\x96\x81world");
}

TEST(MetaspacePreTokenizerTest, LegacyAddPrefixSpace) {
  PreTokenizerConfig config;
  config.parse_json(json{
      {"type", "Metaspace"},
      {"replacement", "\xe2\x96\x81"},
      {"add_prefix_space", false}});
  auto pt = config.create();
  auto result = pt->pre_tokenize("Hello world");
  // add_prefix_space=false maps to prepend_scheme="never"
  EXPECT_EQ(result[0], "Hello");
}

// =============================================================================
// Whitespace
// =============================================================================

TEST(WhitespacePreTokenizerTest, Basic) {
  WhitespacePreTokenizer pt;
  auto result = pt.pre_tokenize("Hey man!");
  ASSERT_EQ(result.size(), 3);
  EXPECT_EQ(result[0], "Hey");
  EXPECT_EQ(result[1], "man");
  EXPECT_EQ(result[2], "!");
}

TEST(WhitespacePreTokenizerTest, MultipleSpaces) {
  WhitespacePreTokenizer pt;
  auto result = pt.pre_tokenize("Hello   world");
  ASSERT_EQ(result.size(), 2);
  EXPECT_EQ(result[0], "Hello");
  EXPECT_EQ(result[1], "world");
}

TEST(WhitespacePreTokenizerTest, FromJson) {
  PreTokenizerConfig config;
  config.parse_json(json{{"type", "Whitespace"}});
  auto pt = config.create();
  auto result = pt->pre_tokenize("test 123");
  EXPECT_GE(result.size(), 2);
}

// =============================================================================
// WhitespaceSplit
// =============================================================================

TEST(WhitespaceSplitTest, Basic) {
  WhitespaceSplitPreTokenizer pt;
  auto result = pt.pre_tokenize("Hey, man, Good?");
  ASSERT_EQ(result.size(), 3);
  EXPECT_EQ(result[0], "Hey,");
  EXPECT_EQ(result[1], "man,");
  EXPECT_EQ(result[2], "Good?");
}

TEST(WhitespaceSplitTest, TabsAndNewlines) {
  WhitespaceSplitPreTokenizer pt;
  auto result = pt.pre_tokenize("a\tb\nc");
  ASSERT_EQ(result.size(), 3);
  EXPECT_EQ(result[0], "a");
  EXPECT_EQ(result[1], "b");
  EXPECT_EQ(result[2], "c");
}

TEST(WhitespaceSplitTest, Empty) {
  WhitespaceSplitPreTokenizer pt;
  auto result = pt.pre_tokenize("");
  EXPECT_TRUE(result.empty());
}

TEST(WhitespaceSplitTest, FromJson) {
  PreTokenizerConfig config;
  config.parse_json(json{{"type", "WhitespaceSplit"}});
  auto pt = config.create();
  auto result = pt->pre_tokenize("a b c");
  ASSERT_EQ(result.size(), 3);
}

// =============================================================================
// Punctuation
// =============================================================================

TEST(PunctuationPreTokenizerTest, Isolated) {
  PunctuationPreTokenizer pt("Isolated");
  auto result = pt.pre_tokenize("Hello!");
  ASSERT_EQ(result.size(), 2);
  EXPECT_EQ(result[0], "Hello");
  EXPECT_EQ(result[1], "!");
}

TEST(PunctuationPreTokenizerTest, MultiplePunc) {
  PunctuationPreTokenizer pt("Isolated");
  auto result = pt.pre_tokenize("Wait?!");
  ASSERT_EQ(result.size(), 3);
  EXPECT_EQ(result[0], "Wait");
  EXPECT_EQ(result[1], "?");
  EXPECT_EQ(result[2], "!");
}

TEST(PunctuationPreTokenizerTest, Contiguous) {
  PunctuationPreTokenizer pt("Contiguous");
  auto result = pt.pre_tokenize("Wait?!");
  ASSERT_EQ(result.size(), 2);
  EXPECT_EQ(result[0], "Wait");
  EXPECT_EQ(result[1], "?!");
}

TEST(PunctuationPreTokenizerTest, NoPunc) {
  PunctuationPreTokenizer pt("Isolated");
  auto result = pt.pre_tokenize("Hello");
  ASSERT_EQ(result.size(), 1);
  EXPECT_EQ(result[0], "Hello");
}

TEST(PunctuationPreTokenizerTest, FromJson) {
  PreTokenizerConfig config;
  config.parse_json(json{{"type", "Punctuation"}, {"behavior", "Isolated"}});
  auto pt = config.create();
  auto result = pt->pre_tokenize("Hi!");
  ASSERT_EQ(result.size(), 2);
}

// =============================================================================
// CharDelimiterSplit
// =============================================================================

TEST(CharDelimiterSplitTest, Basic) {
  CharDelimiterSplitPreTokenizer pt(",");
  auto result = pt.pre_tokenize("a,b,c");
  ASSERT_EQ(result.size(), 3);
  EXPECT_EQ(result[0], "a");
  EXPECT_EQ(result[1], "b");
  EXPECT_EQ(result[2], "c");
}

TEST(CharDelimiterSplitTest, NoDelimiter) {
  CharDelimiterSplitPreTokenizer pt(",");
  auto result = pt.pre_tokenize("hello");
  ASSERT_EQ(result.size(), 1);
  EXPECT_EQ(result[0], "hello");
}

TEST(CharDelimiterSplitTest, ConsecutiveDelimiters) {
  CharDelimiterSplitPreTokenizer pt(",");
  auto result = pt.pre_tokenize("a,,b");
  // Empty parts between consecutive delimiters are dropped
  ASSERT_EQ(result.size(), 2);
  EXPECT_EQ(result[0], "a");
  EXPECT_EQ(result[1], "b");
}

TEST(CharDelimiterSplitTest, FromJson) {
  PreTokenizerConfig config;
  config.parse_json(json{{"type", "CharDelimiterSplit"}, {"delimiter", "-"}});
  auto pt = config.create();
  auto result = pt->pre_tokenize("a-b-c");
  ASSERT_EQ(result.size(), 3);
}

// =============================================================================
// UnicodeScripts
// =============================================================================

TEST(UnicodeScriptsTest, LatinOnly) {
  UnicodeScriptsPreTokenizer pt;
  auto result = pt.pre_tokenize("Hello world");
  // All Latin + space (Any) — no script boundary
  ASSERT_EQ(result.size(), 1);
  EXPECT_EQ(result[0], "Hello world");
}

TEST(UnicodeScriptsTest, LatinAndCJK) {
  UnicodeScriptsPreTokenizer pt;
  auto result = pt.pre_tokenize("Hello\xe4\xb8\x96\xe7\x95\x8c"); // Hello世界
  ASSERT_EQ(result.size(), 2);
  EXPECT_EQ(result[0], "Hello");
  EXPECT_EQ(result[1], "\xe4\xb8\x96\xe7\x95\x8c");
}

TEST(UnicodeScriptsTest, LatinAndCyrillic) {
  UnicodeScriptsPreTokenizer pt;
  // "Hello" (Latin) + "Мир" (Cyrillic)
  auto result = pt.pre_tokenize("Hello\xd0\x9c\xd0\xb8\xd1\x80");
  ASSERT_EQ(result.size(), 2);
  EXPECT_EQ(result[0], "Hello");
  EXPECT_EQ(result[1], "\xd0\x9c\xd0\xb8\xd1\x80");
}

TEST(UnicodeScriptsTest, Empty) {
  UnicodeScriptsPreTokenizer pt;
  auto result = pt.pre_tokenize("");
  EXPECT_TRUE(result.empty());
}

TEST(UnicodeScriptsTest, FromJson) {
  PreTokenizerConfig config;
  config.parse_json(json{{"type", "UnicodeScripts"}});
  auto pt = config.create();
  auto result = pt->pre_tokenize("test");
  ASSERT_EQ(result.size(), 1);
}

// =============================================================================
// FixedLength
// =============================================================================

TEST(FixedLengthTest, Basic) {
  FixedLengthPreTokenizer pt(5);
  auto result = pt.pre_tokenize("Hello world");
  ASSERT_EQ(result.size(), 3);
  EXPECT_EQ(result[0], "Hello");
  EXPECT_EQ(result[1], " worl");
  EXPECT_EQ(result[2], "d");
}

TEST(FixedLengthTest, ExactMultiple) {
  FixedLengthPreTokenizer pt(3);
  auto result = pt.pre_tokenize("abcdef");
  ASSERT_EQ(result.size(), 2);
  EXPECT_EQ(result[0], "abc");
  EXPECT_EQ(result[1], "def");
}

TEST(FixedLengthTest, UTF8Chars) {
  FixedLengthPreTokenizer pt(2);
  // "aé" is 3 bytes but 2 characters
  auto result = pt.pre_tokenize("a\xc3\xa9" "bc");
  ASSERT_EQ(result.size(), 2);
  EXPECT_EQ(result[0], "a\xc3\xa9"); // "aé" = 2 chars
  EXPECT_EQ(result[1], "bc");
}

TEST(FixedLengthTest, Empty) {
  FixedLengthPreTokenizer pt(5);
  auto result = pt.pre_tokenize("");
  EXPECT_TRUE(result.empty());
}

TEST(FixedLengthTest, FromJson) {
  PreTokenizerConfig config;
  config.parse_json(json{{"type", "FixedLength"}, {"length", 3}});
  auto pt = config.create();
  auto result = pt->pre_tokenize("abcde");
  ASSERT_EQ(result.size(), 2);
  EXPECT_EQ(result[0], "abc");
  EXPECT_EQ(result[1], "de");
}

// =============================================================================
// Adversarial / Edge Case Tests
// =============================================================================

TEST(PreTokenizerAdversarialTest, MalformedUTF8_Whitespace) {
  // NOTE: WhitespaceSplitPreTokenizer uses unicode_cpts_from_utf8 which may
  // not handle malformed UTF-8 gracefully. This test documents the limitation.
  // For now, we only test valid UTF-8 edge cases.
  WhitespaceSplitPreTokenizer pt;
  // Valid but unusual: zero-width space (U+200B) between words
  auto result = pt.pre_tokenize("hello\xe2\x80\x8bworld");
  EXPECT_GE(result.size(), 1);
}

TEST(PreTokenizerAdversarialTest, OnlyWhitespace) {
  WhitespaceSplitPreTokenizer pt;
  auto result = pt.pre_tokenize("   \t\n  ");
  EXPECT_TRUE(result.empty());
}

TEST(PreTokenizerAdversarialTest, MetaspaceConsecutiveSpaces) {
  MetaspacePreTokenizer pt;
  auto result = pt.pre_tokenize("a  b");
  // "a  b" → "a▁▁b" → prepend → "▁a▁▁b" → split on ▁
  // Should produce ["▁a", "▁", "▁b"]
  ASSERT_EQ(result.size(), 3);
  EXPECT_EQ(result[0], "\xe2\x96\x81" "a");
  EXPECT_EQ(result[1], "\xe2\x96\x81");
  EXPECT_EQ(result[2], "\xe2\x96\x81" "b");
}

TEST(PreTokenizerAdversarialTest, MetaspaceSingleSpace) {
  MetaspacePreTokenizer pt;
  auto result = pt.pre_tokenize(" ");
  // " " → "▁" → already starts with ▁ → no double prepend → ["▁"]
  ASSERT_EQ(result.size(), 1);
  EXPECT_EQ(result[0], "\xe2\x96\x81");
}

TEST(PreTokenizerAdversarialTest, PunctuationOnlyInput) {
  PunctuationPreTokenizer pt("Isolated");
  auto result = pt.pre_tokenize("!@#$");
  // Each punctuation char should be isolated
  ASSERT_EQ(result.size(), 4);
  EXPECT_EQ(result[0], "!");
  EXPECT_EQ(result[1], "@");
  EXPECT_EQ(result[2], "#");
  EXPECT_EQ(result[3], "$");
}

TEST(PreTokenizerAdversarialTest, FixedLengthSingleChar) {
  FixedLengthPreTokenizer pt(1);
  auto result = pt.pre_tokenize("abc");
  ASSERT_EQ(result.size(), 3);
  EXPECT_EQ(result[0], "a");
  EXPECT_EQ(result[1], "b");
  EXPECT_EQ(result[2], "c");
}

TEST(PreTokenizerAdversarialTest, UnicodeScriptsWithEmoji) {
  UnicodeScriptsPreTokenizer pt;
  // Emoji between Latin text — emoji is Script::Any, shouldn't trigger boundary
  auto result = pt.pre_tokenize("Hello\xf0\x9f\x98\x80World"); // Hello😀World
  // Depends on whether emoji is classified as Any or Other
  // At minimum it shouldn't crash and should return the full text
  EXPECT_GE(result.size(), 1);
  // Rejoin should equal original
  std::string joined;
  for (const auto& r : result) joined += r;
  EXPECT_EQ(joined, "Hello\xf0\x9f\x98\x80World");
}

TEST(PreTokenizerAdversarialTest, VeryLongInput) {
  WhitespacePreTokenizer pt;
  // 100k character input
  std::string long_input;
  for (int i = 0; i < 10000; ++i) {
    long_input += "word" + std::to_string(i) + " ";
  }
  auto result = pt.pre_tokenize(long_input);
  EXPECT_EQ(result.size(), 10000);
}
