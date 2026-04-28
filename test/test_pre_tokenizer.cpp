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

#include <atomic>
#include <thread>
#include <vector>

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

TEST_F(ByteLevelPreTokenizerTest, PreTokenizeNoRegexSinglePiece) {
  // use_regex=false: the whole input is byte-encoded as ONE piece, with no
  // GPT2 split. Space byte (0x20) maps to the byte-level char Ġ.
  ByteLevelPreTokenizer ptok(
      /*add_prefix_space=*/false, /*pattern=*/"", /*use_regex=*/false);
  assert_split_match(ptok, "Hello World", {"HelloĠWorld"});
  // Punctuation glued to a following letter run stays in one piece (the
  // divergence this fix targets); the default path would split "(" off.
  assert_split_match(ptok, "foo(bar", {"foo(bar"});
}

TEST_F(ByteLevelPreTokenizerTest, PreTokenizeNoRegexWithPrefix) {
  // The prefix space is added first, then the whole thing is one byte-encoded
  // piece.
  ByteLevelPreTokenizer ptok(
      /*add_prefix_space=*/true, /*pattern=*/"", /*use_regex=*/false);
  assert_split_match(ptok, "Hello World", {"ĠHelloĠWorld"});
}

// SequencePreTokenizer ////////////////////////////////////////////////////////
class SequencePreTokenizerTest : public ::testing::Test {};

TEST_F(SequencePreTokenizerTest, PreTokenizeDigitAndByteLevel) {
  std::vector<PreTokenizer::Ptr> ptoks;
  ptoks.push_back(std::make_unique<DigitsPreTokenizer>(true));
  ptoks.push_back(std::make_unique<ByteLevelPreTokenizer>(false));
  SequencePreTokenizer ptok(std::move(ptoks));
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

TEST_F(SequencePreTokenizerTest, PreTokenizeSplitThenByteLevelNoRegex) {
  // Mirror the production model: Sequence[Split(GPT2-like regex), ByteLevel(
  // use_regex=false)]. Split glues a leading punctuation char to the following
  // letter run ("foo(bar" -> ["foo", "(bar"]); with use_regex=false ByteLevel
  // keeps "(bar" whole. The default ByteLevel would further split it into
  // "(" and "bar", which is the bug this fix corrects.
  PreTokenizer::Ptr split = std::make_shared<RegexPreTokenizer>(
      R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+)");
  PreTokenizer::Ptr byte_no_regex = std::make_shared<ByteLevelPreTokenizer>(
      /*add_prefix_space=*/false, /*pattern=*/"", /*use_regex=*/false);
  SequencePreTokenizer ptok({split, byte_no_regex});
  assert_split_match(ptok, "foo(bar", {"foo", "(bar"});
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
  PreTokenizerConfig("ByteLevel").set_use_regex(false).create();
  PreTokenizerConfig("ByteLevel")
      .set_add_prefix_space(false)
      .set_use_regex(false)
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

TEST_F(PreTokenizerConfigTest, ByteLevelUseRegexParsing) {
  // use_regex=false -> the whole input is one byte-encoded piece.
  const auto no_regex = PreTokenizerConfig()
                            .parse_json(
                                json{
                                    {"type", "ByteLevel"},
                                    {"add_prefix_space", false},
                                    {"use_regex", false},
                                })
                            .create();
  assert_split_match(*no_regex, "Hello World", {"HelloĠWorld"});

  // use_regex=true -> GPT2 split (the historical default behavior).
  const auto with_regex = PreTokenizerConfig()
                              .parse_json(
                                  json{
                                      {"type", "ByteLevel"},
                                      {"add_prefix_space", false},
                                      {"use_regex", true},
                                  })
                              .create();
  assert_split_match(*with_regex, "Hello World", {"Hello", "ĠWorld"});

  // use_regex omitted -> defaults to true, so behavior is unchanged for configs
  // that do not set the field.
  const auto omitted = PreTokenizerConfig()
                           .parse_json(
                               json{
                                   {"type", "ByteLevel"},
                                   {"add_prefix_space", false},
                               })
                           .create();
  assert_split_match(*omitted, "Hello World", {"Hello", "ĠWorld"});
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

// Regex cache (unicode_regex_split) ///////////////////////////////////////////
// The ByteLevel pre-tokenizer drives unicode_regex_split, which caches the
// compiled std::regex/std::wregex per pattern. This test guards that cache:
// (1) results are deterministic / behavior-neutral, including across the full
// Unicode White_Space class and near-miss non-whitespace codepoints, and
// (2) it is thread-safe (the tokenizer pool calls it concurrently).
class RegexCacheTest : public ::testing::Test {};

TEST_F(RegexCacheTest, ByteLevelDeterministicAndThreadSafe) {
  ByteLevelPreTokenizer ptok(/*add_prefix_space=*/false);

  const std::vector<std::string> corpus = {
      "Hello World",
      "  leading and   multiple   spaces  ",
      "tabs\t\tand\nnewlines\r\n",
      "code: { return x; }   // trailing   ",
      // Unicode White_Space codepoints (must be treated as whitespace).
      "a\xC2\x85"
      "b", // U+0085 NEL
      "a\x0B"
      "b\x0C"
      "c", // U+000B VT, U+000C FF
      "a\xE1\x9A\x80"
      "b", // U+1680 OGHAM SPACE MARK
      "a\xE2\x80\x80\xE2\x80\x8A"
      "b", // U+2000 .. U+200A
      "a\xE2\x80\xAF"
      "b\xE2\x81\x9F"
      "c", // U+202F NNBSP, U+205F MMSP
      "a\xE3\x80\x80"
      "b", // U+3000 ideographic space
      // Near-miss NON-whitespace codepoints (must NOT be treated as ws).
      "a\xE2\x80\x8B"
      "b", // U+200B ZERO WIDTH SPACE
      "a\xE1\xA0\x8E"
      "b", // U+180E MONGOLIAN VOWEL SEPARATOR
      "a\xEF\xBB\xBF"
      "b", // U+FEFF BOM / ZWNBSP
      // Unicode letters and a representative SID prompt fragment.
      "caf\xC3\xA9 \xE4\xBD\xA0\xE5\xA5\xBD",
      "history: i0:<1326-617-1617> i1:<197-296-385> next:",
  };

  // Single-threaded reference.
  std::vector<std::vector<std::string>> ref;
  ref.reserve(corpus.size());
  for (const auto& s : corpus) {
    ref.push_back(ptok.pre_tokenize(s));
    EXPECT_FALSE(ref.back().empty());
  }

  // Hammer the shared pre-tokenizer (and the static regex cache inside
  // unicode_regex_split) from many threads; every result must match the
  // reference. A recompile race or a wrong cache lookup would surface as a
  // mismatch (or a crash under TSAN).
  constexpr int kThreads = 16;
  constexpr int kIters = 200;
  std::atomic<int> mismatches{0};
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&]() {
      for (int iter = 0; iter < kIters; ++iter) {
        for (size_t i = 0; i < corpus.size(); ++i) {
          if (ptok.pre_tokenize(corpus[i]) != ref[i]) {
            mismatches.fetch_add(1, std::memory_order_relaxed);
          }
        }
      }
    });
  }
  for (auto& th : threads) {
    th.join();
  }
  EXPECT_EQ(0, mismatches.load());
}

TEST_F(RegexCacheTest, ByteLevelNoRegexSinglePieceEqualsByteEncoding) {
  // With use_regex=false the input is never split: each item yields exactly one
  // piece. That piece must equal the byte-encoding of the whole input, which is
  // the concatenation of the default (GPT2-split) path's byte-encoded pieces
  // (splitting happens on codepoint boundaries, so concatenating the encoded
  // pieces reconstructs the encoding of the whole input). This pins the new
  // behavior without hardcoding the byte-level alphabet.
  ByteLevelPreTokenizer no_regex(
      /*add_prefix_space=*/false, /*pattern=*/"", /*use_regex=*/false);
  ByteLevelPreTokenizer with_regex(/*add_prefix_space=*/false);

  const std::vector<std::string> corpus = {
      "Hello World",
      "  leading and   multiple   spaces  ",
      "tabs\t\tand\nnewlines\r\n",
      "code: { return x; }   // trailing   ",
      "foo(bar",
      "a(b",
      "(the",
      "hi_there",
      "next:\n",
      "a\xC2\x85"
      "b", // U+0085 NEL
      "a\xE2\x80\x80\xE2\x80\x8A"
      "b", // U+2000 .. U+200A
      "a\xE2\x80\x8B"
      "b", // U+200B ZERO WIDTH SPACE (non-ws)
      "caf\xC3\xA9 \xE4\xBD\xA0\xE5\xA5\xBD",
      "history: i0:<1326-617-1617> next:",
  };

  for (const auto& s : corpus) {
    const auto one_piece = no_regex.pre_tokenize(s);
    ASSERT_EQ(one_piece.size(), 1u) << "input: " << s;
    std::string concat;
    for (const auto& p : with_regex.pre_tokenize(s)) {
      concat += p;
    }
    EXPECT_EQ(one_piece[0], concat) << "input: " << s;
  }
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
