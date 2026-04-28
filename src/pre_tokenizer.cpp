/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// @lint-ignore-every LICENSELINT

// Local
#include <pytorch/tokenizers/pre_tokenizer.h>
#include <pytorch/tokenizers/unicode_utils.h>
#include <unicode.h>

// Standard
#include <cassert>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// Third Party
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace tokenizers {

// =============================================================================
// PreTokenizerConfig — Factory
// =============================================================================

PreTokenizerConfig::PreTokenizerConfig(std::string type)
    : type(std::move(type)) {}

PreTokenizer::Ptr PreTokenizerConfig::create() const {
  if (type == "Split") {
    if (!pattern) {
      throw std::runtime_error(
          "Missing pattern for PreTokenizer of type Split");
    }
    std::string behavior_str = behavior.value_or("Removed");
    const bool invert_flag = invert.value_or(false);
    bool delimiter_flag = is_delimiter.value_or(false);

    // For String patterns (is_delimiter=true): the string is a delimiter,
    // behavior controls how delimiters are handled.
    // invert=true on String: flip to non-delimiter (keep matches).
    if (invert_flag && delimiter_flag) {
      delimiter_flag = false;
    }
    // For Regex patterns (is_delimiter=false): regex matches are returned
    // as tokens. invert=true is the default semantic (keep matches).
    // invert=false would mean "use matches as delimiters" but in practice
    // all real tokenizer.json files with Regex+invert=false use behavior=
    // Isolated which works correctly in non-delimiter mode too.

    return std::make_unique<RegexPreTokenizer>(
        *pattern, delimiter_flag, behavior_str);
  }
  if (type == "Digits") {
    return std::make_unique<DigitsPreTokenizer>(
        individual_digits.value_or(false));
  }
  if (type == "ByteLevel") {
    return std::make_unique<ByteLevelPreTokenizer>(
        add_prefix_space.value_or(true),
        pattern.value_or(""),
        use_regex.value_or(true));
  }
  if (type == "Sequence") {
    if (!pretokenizers || pretokenizers->empty()) {
      throw std::runtime_error(
          "Missing pretokenizers for PreTokenizer of type Sequence");
    }
    std::vector<PreTokenizer::Ptr> pretoks;
    for (const auto& cfg : *pretokenizers) {
      pretoks.push_back(cfg.create());
    }
    return std::make_unique<SequencePreTokenizer>(std::move(pretoks));
  }
  if (type == "BertPreTokenizer") {
    return std::make_unique<BertPreTokenizer>();
  }
  if (type == "Metaspace") {
    auto rep = replacement.value_or("\xe2\x96\x81"); // ▁
    auto scheme_str = prepend_scheme.value_or("always");
    auto do_split = split.value_or(true);

    MetaspacePreTokenizer::PrependScheme scheme;
    if (scheme_str == "first") {
      scheme = MetaspacePreTokenizer::PrependScheme::First;
    } else if (scheme_str == "never") {
      scheme = MetaspacePreTokenizer::PrependScheme::Never;
    } else {
      scheme = MetaspacePreTokenizer::PrependScheme::Always;
    }
    return std::make_unique<MetaspacePreTokenizer>(rep, scheme, do_split);
  }
  if (type == "Whitespace") {
    return std::make_unique<WhitespacePreTokenizer>();
  }
  if (type == "WhitespaceSplit") {
    return std::make_unique<WhitespaceSplitPreTokenizer>();
  }
  if (type == "Punctuation") {
    return std::make_unique<PunctuationPreTokenizer>(
        behavior.value_or("Isolated"));
  }
  if (type == "CharDelimiterSplit") {
    if (!delimiter) {
      throw std::runtime_error(
          "Missing delimiter for PreTokenizer of type CharDelimiterSplit");
    }
    return std::make_unique<CharDelimiterSplitPreTokenizer>(*delimiter);
  }
  if (type == "UnicodeScripts") {
    return std::make_unique<UnicodeScriptsPreTokenizer>();
  }
  if (type == "FixedLength") {
    return std::make_unique<FixedLengthPreTokenizer>(length.value_or(5));
  }
  throw std::runtime_error("Unsupported PreTokenizer type: " + type);
}

PreTokenizerConfig& PreTokenizerConfig::parse_json(const json& json_config) {
  type = json_config.at("type");

  if (type == "Split") {
    try {
      set_pattern(json_config.at("pattern").at("Regex"));
      set_is_delimiter(false);
    } catch (json::out_of_range&) {
      std::string delim = json_config.at("pattern").at("String");
      set_pattern(IRegex::escape(delim));
      set_is_delimiter(true);
    }
    if (json_config.contains("behavior")) {
      set_behavior(json_config["behavior"]);
    }
    if (json_config.contains("invert")) {
      set_invert(json_config["invert"]);
    }

  } else if (type == "Digits") {
    if (json_config.contains("individual_digits")) {
      set_individual_digits(json_config["individual_digits"]);
    }

  } else if (type == "ByteLevel") {
    if (json_config.contains("add_prefix_space")) {
      set_add_prefix_space(json_config["add_prefix_space"]);
    }
    try {
      use_regex = json_config.at("use_regex");
    } catch (json::out_of_range&) {
    }

  } else if (type == "Sequence") {
    std::vector<PreTokenizerConfig> cfgs;
    for (const auto& entry : json_config.at("pretokenizers")) {
      cfgs.push_back(PreTokenizerConfig().parse_json(entry));
    }
    set_pretokenizers(std::move(cfgs));

  } else if (type == "BertPreTokenizer") {
    // No config parameters

  } else if (type == "Metaspace") {
    if (json_config.contains("replacement")) {
      set_replacement(json_config["replacement"].get<std::string>());
    }
    if (json_config.contains("prepend_scheme")) {
      set_prepend_scheme(json_config["prepend_scheme"].get<std::string>());
    } else if (json_config.contains("add_prefix_space")) {
      // Legacy compat: add_prefix_space=true → "always", false → "never"
      bool aps = json_config["add_prefix_space"].get<bool>();
      set_prepend_scheme(aps ? "always" : "never");
    }
    if (json_config.contains("split")) {
      set_split(json_config["split"].get<bool>());
    }

  } else if (type == "Whitespace" || type == "WhitespaceSplit") {
    // No config parameters

  } else if (type == "Punctuation") {
    if (json_config.contains("behavior")) {
      set_behavior(json_config["behavior"].get<std::string>());
    }

  } else if (type == "CharDelimiterSplit") {
    // JSON stores delimiter as a single-char string
    set_delimiter(json_config.at("delimiter").get<std::string>());

  } else if (type == "UnicodeScripts") {
    // No config parameters

  } else if (type == "FixedLength") {
    if (json_config.contains("length")) {
      set_length(json_config["length"].get<size_t>());
    }

  } else {
    throw std::runtime_error("Unsupported PreTokenizer type: " + type);
  }
  return *this;
}

// =============================================================================
// RegexPreTokenizer (Split)
// =============================================================================

std::unique_ptr<IRegex> RegexPreTokenizer::create_regex_(
    const std::string& pattern) {
  assert(!pattern.empty());
  auto regex_result = create_regex(pattern);
  if (!regex_result.ok()) {
    throw std::runtime_error(
        "Error: " + std::to_string(static_cast<int>(regex_result.error())));
  }
  return std::move(regex_result.get());
}

std::vector<std::string> RegexPreTokenizer::pre_tokenize(
    const std::string& input) const {
  if (!regex_) {
    return {};
  }

  std::vector<std::string> results;
  auto matches = regex_->find_all(input);

  if (!is_delimiter_) {
    for (const auto& match : matches) {
      results.push_back(input.substr(match.start, match.end - match.start));
    }
    return results;
  }

  // Delimiter mode
  if (matches.empty()) {
    results.push_back(input);
    return results;
  }

  if (behavior_ == "MergedWithPrevious") {
    // Build a sequence of (offset, is_delimiter) segments covering the
    // entire input, then merge each delimiter into the preceding segment.
    // Consecutive delimiters accumulate so BPE sees multi-char runs.
    struct Seg { size_t start; size_t end; bool is_delim; };
    std::vector<Seg> segs;
    size_t pos = 0;
    for (const auto& m : matches) {
      if (m.start > pos) {
        segs.push_back({pos, m.start, false});
      }
      segs.push_back({m.start, m.end, true});
      pos = m.end;
    }
    if (pos < input.size()) {
      segs.push_back({pos, input.size(), false});
    }

    // Merge: each delimiter appends to previous result. After a text
    // segment, the first delimiter merges with the text, then subsequent
    // consecutive delimiters start a new accumulating group.
    bool prev_seg_was_text = false;
    bool merged_first_after_text = false;
    for (const auto& seg : segs) {
      std::string piece = input.substr(seg.start, seg.end - seg.start);
      if (!seg.is_delim) {
        // Text segment
        results.push_back(piece);
        prev_seg_was_text = true;
        merged_first_after_text = false;
      } else if (prev_seg_was_text && !merged_first_after_text) {
        // First delimiter right after text — merge with text
        results.back() += piece;
        merged_first_after_text = true;
      } else if (!results.empty() && !prev_seg_was_text) {
        // Consecutive delimiter in leading/trailing run — accumulate
        results.back() += piece;
      } else {
        // Start new delimiter group (2nd+ delimiter after text)
        results.push_back(piece);
        prev_seg_was_text = false;
      }
    }

  } else if (behavior_ == "MergedWithNext") {
    // Delimiter merges with the following token.
    size_t last_end = 0;
    for (size_t i = 0; i < matches.size(); ++i) {
      const auto& match = matches[i];
      // Text before the delimiter
      if (match.start > last_end) {
        results.push_back(input.substr(last_end, match.start - last_end));
      }
      // Delimiter + text until next delimiter (or end)
      size_t next_start;
      if (i + 1 < matches.size()) {
        next_start = matches[i + 1].start;
      } else {
        next_start = input.length();
      }
      results.push_back(input.substr(match.start, next_start - match.start));
      last_end = next_start;
    }

  } else if (behavior_ == "Isolated") {
    size_t last_end = 0;
    for (const auto& match : matches) {
      if (match.start > last_end) {
        results.push_back(input.substr(last_end, match.start - last_end));
      }
      results.push_back(
          input.substr(match.start, match.end - match.start));
      last_end = match.end;
    }
    if (last_end < input.length()) {
      results.push_back(input.substr(last_end));
    }

  } else if (behavior_ == "Contiguous") {
    // Consecutive matching characters form a single delimiter token.
    // Non-matching text and delimiter groups alternate.
    size_t last_end = 0;
    for (const auto& match : matches) {
      if (match.start > last_end) {
        results.push_back(input.substr(last_end, match.start - last_end));
      }
      results.push_back(
          input.substr(match.start, match.end - match.start));
      last_end = match.end;
    }
    if (last_end < input.length()) {
      results.push_back(input.substr(last_end));
    }

  } else {
    // Removed (default)
    size_t last_end = 0;
    for (const auto& match : matches) {
      if (match.start > last_end) {
        results.push_back(input.substr(last_end, match.start - last_end));
      }
      last_end = match.end;
    }
    if (last_end < input.length()) {
      results.push_back(input.substr(last_end));
    }
  }
  return results;
}

// =============================================================================
// ByteLevelPreTokenizer
// =============================================================================

namespace {
constexpr char GPT2_EXPR[] =
    R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)";

// Byte-encode the whole input as a single piece, matching the byte-level
// alphabet mapping that unicode_regex_split applies via
// unicode_byte_encoding_process. Used when use_regex is false so the input is
// not split by the GPT2 regex.
std::string byte_encode(const std::string& input) {
  std::string result;
  // Each input byte maps to a byte-level codepoint that is at most 2 bytes in
  // UTF-8, so reserve up front to avoid repeated reallocations on the hot path.
  result.reserve(input.size() * 2);
  for (const unsigned char byte : input) {
    result += unicode_byte_to_utf8(byte);
  }
  return result;
}

} // namespace

ByteLevelPreTokenizer::ByteLevelPreTokenizer(
    bool add_prefix_space,
    const std::string& pattern,
    bool use_regex)
    : pattern_(pattern.empty() ? GPT2_EXPR : pattern),
      add_prefix_space_(add_prefix_space),
      use_regex_(use_regex) {}

std::vector<std::string> ByteLevelPreTokenizer::pre_tokenize(
    const std::string& input) const {
  if (!use_regex_) {
    std::string encoded;
    for (unsigned char byte : input) {
      encoded += unicode_byte_to_utf8(byte);
    }
    return {encoded};
  }

  std::string formatted_input = input;
  if (add_prefix_space_ && !formatted_input.empty() &&
      formatted_input[0] != ' ') {
    formatted_input.insert(formatted_input.begin(), ' ');
  }

  // When use_regex is false, do not split with the GPT2 regex; byte-encode the
  // whole input as a single piece (matches HF ByteLevel use_regex=false).
  if (!use_regex_) {
    return {byte_encode(formatted_input)};
  }

  return unicode_regex_split(formatted_input, {pattern_});
}

// =============================================================================
// SequencePreTokenizer
// =============================================================================

SequencePreTokenizer::SequencePreTokenizer(
    std::vector<PreTokenizer::Ptr> pre_tokenizers)
    : pre_tokenizers_(std::move(pre_tokenizers)) {}

std::vector<std::string> SequencePreTokenizer::pre_tokenize(
    const std::string& input) const {
  std::vector<std::string> pieces{std::string(input)};
  for (const auto& pre_tokenizer : pre_tokenizers_) {
    std::vector<std::string> new_pieces;
    for (const auto& piece : pieces) {
      for (const auto& subpiece : pre_tokenizer->pre_tokenize(piece)) {
        new_pieces.push_back(subpiece);
      }
    }
    pieces = std::move(new_pieces);
  }
  return pieces;
}

// =============================================================================
// BertPreTokenizer
// =============================================================================


std::vector<std::string> BertPreTokenizer::pre_tokenize(
    const std::string& input) const {
  std::vector<uint32_t> cpts = unicode_cpts_from_utf8(input);

  // 1. Split by whitespace (Removed)
  std::vector<std::string> words;
  std::string current_word;
  for (uint32_t cp : cpts) {
    if (unicode_cpt_flags(cp).is_whitespace) {
      if (!current_word.empty()) {
        words.push_back(std::move(current_word));
        current_word.clear();
      }
    } else {
      current_word += unicode_cpt_to_utf8(cp);
    }
  }
  if (!current_word.empty()) {
    words.push_back(std::move(current_word));
  }

  // 2. Split by punctuation and CJK (Isolated)
  std::vector<std::string> final_tokens;
  for (const auto& word : words) {
    std::vector<uint32_t> word_cpts = unicode_cpts_from_utf8(word);
    std::string current_token;
    for (uint32_t cp : word_cpts) {
      if (unicode_utils::is_punctuation(cp) || unicode_utils::is_cjk(cp)) {
        if (!current_token.empty()) {
          final_tokens.push_back(std::move(current_token));
          current_token.clear();
        }
        final_tokens.push_back(unicode_cpt_to_utf8(cp));
      } else {
        current_token += unicode_cpt_to_utf8(cp);
      }
    }
    if (!current_token.empty()) {
      final_tokens.push_back(std::move(current_token));
    }
  }

  return final_tokens;
}

// =============================================================================
// MetaspacePreTokenizer
// =============================================================================

std::vector<std::string> MetaspacePreTokenizer::pre_tokenize(
    const std::string& input) const {
  // 1. Replace all spaces with the replacement character.
  std::string transformed;
  transformed.reserve(input.size() + replacement_.size());
  for (size_t i = 0; i < input.size(); ++i) {
    if (input[i] == ' ') {
      transformed += replacement_;
    } else {
      transformed += input[i];
    }
  }

  // 2. Prepend replacement character if scheme requires it.
  if (prepend_scheme_ != PrependScheme::Never && !transformed.empty()) {
    // Check if already starts with replacement
    if (transformed.size() < replacement_.size() ||
        transformed.compare(0, replacement_.size(), replacement_) != 0) {
      transformed = replacement_ + transformed;
    }
  }

  if (transformed.empty()) {
    return {};
  }

  // 3. Split on the replacement character with MergedWithNext behavior.
  if (!split_) {
    return {transformed};
  }

  std::vector<std::string> results;
  size_t pos = 0;
  size_t rep_len = replacement_.size();

  while (pos < transformed.size()) {
    // Find the next occurrence of replacement after current position.
    size_t next = transformed.find(replacement_, pos + rep_len);

    if (next == std::string::npos) {
      // No more delimiters — take the rest.
      results.push_back(transformed.substr(pos));
      break;
    }
    // Take from current position to (but not including) the next delimiter.
    results.push_back(transformed.substr(pos, next - pos));
    pos = next;
  }

  return results;
}

// =============================================================================
// WhitespaceSplitPreTokenizer
// =============================================================================

std::vector<std::string> WhitespaceSplitPreTokenizer::pre_tokenize(
    const std::string& input) const {
  std::vector<std::string> results;
  std::vector<uint32_t> cpts = unicode_cpts_from_utf8(input);
  std::string current;

  for (uint32_t cp : cpts) {
    if (unicode_cpt_flags(cp).is_whitespace) {
      if (!current.empty()) {
        results.push_back(std::move(current));
        current.clear();
      }
    } else {
      current += unicode_cpt_to_utf8(cp);
    }
  }
  if (!current.empty()) {
    results.push_back(std::move(current));
  }
  return results;
}

// =============================================================================
// PunctuationPreTokenizer
// =============================================================================


std::vector<std::string> PunctuationPreTokenizer::pre_tokenize(
    const std::string& input) const {
  std::vector<uint32_t> cpts = unicode_cpts_from_utf8(input);
  std::vector<std::string> results;
  std::string current;
  bool current_is_punc = false;

  for (size_t i = 0; i < cpts.size(); ++i) {
    uint32_t cp = cpts[i];
    bool punc = unicode_utils::is_punctuation(cp);

    if (i == 0) {
      current_is_punc = punc;
      current = unicode_cpt_to_utf8(cp);
      continue;
    }

    if (behavior_ == "Isolated") {
      // Each punctuation char is its own token.
      if (punc) {
        if (!current.empty()) {
          results.push_back(std::move(current));
          current.clear();
        }
        results.push_back(unicode_cpt_to_utf8(cp));
        current_is_punc = true;
      } else {
        if (current_is_punc) {
          // Previous was punctuation (already pushed), start new word.
          current_is_punc = false;
        }
        current += unicode_cpt_to_utf8(cp);
      }
    } else if (behavior_ == "Contiguous") {
      // Consecutive punctuation chars form one token.
      if (punc != current_is_punc) {
        if (!current.empty()) {
          results.push_back(std::move(current));
          current.clear();
        }
        current_is_punc = punc;
      }
      current += unicode_cpt_to_utf8(cp);
    } else if (behavior_ == "MergedWithPrevious") {
      if (punc) {
        current += unicode_cpt_to_utf8(cp);
        results.push_back(std::move(current));
        current.clear();
        current_is_punc = true;
      } else {
        if (current_is_punc && !current.empty()) {
          results.push_back(std::move(current));
          current.clear();
        }
        current_is_punc = false;
        current += unicode_cpt_to_utf8(cp);
      }
    } else if (behavior_ == "MergedWithNext") {
      if (punc && !current_is_punc) {
        if (!current.empty()) {
          results.push_back(std::move(current));
          current.clear();
        }
        current_is_punc = true;
      } else if (!punc && current_is_punc) {
        current_is_punc = false;
      }
      current += unicode_cpt_to_utf8(cp);
    } else {
      // Removed: drop punctuation
      if (punc) {
        if (!current.empty()) {
          results.push_back(std::move(current));
          current.clear();
        }
        current_is_punc = true;
      } else {
        current_is_punc = false;
        current += unicode_cpt_to_utf8(cp);
      }
    }
  }
  if (!current.empty()) {
    results.push_back(std::move(current));
  }
  return results;
}

// =============================================================================
// CharDelimiterSplitPreTokenizer
// =============================================================================

std::vector<std::string> CharDelimiterSplitPreTokenizer::pre_tokenize(
    const std::string& input) const {
  std::vector<std::string> results;
  size_t pos = 0;
  size_t delim_len = delimiter_.size();

  while (pos <= input.size()) {
    size_t found = input.find(delimiter_, pos);
    if (found == std::string::npos) {
      std::string part = input.substr(pos);
      if (!part.empty()) {
        results.push_back(std::move(part));
      }
      break;
    }
    if (found > pos) {
      results.push_back(input.substr(pos, found - pos));
    }
    pos = found + delim_len;
  }
  return results;
}

// =============================================================================
// UnicodeScriptsPreTokenizer
// =============================================================================

namespace {

// Simplified Unicode script categories.
// Groups CJK-related scripts (Han, Hiragana, Katakana) together.
enum class UScript {
  Any,     // Spaces, common symbols — does not trigger boundaries.
  Latin,
  Han,     // Also covers Hiragana, Katakana.
  Cyrillic,
  Arabic,
  Devanagari,
  Bengali,
  Greek,
  Hebrew,
  Thai,
  Georgian,
  Armenian,
  Hangul,
  Ethiopic,
  Tamil,
  Telugu,
  Kannada,
  Malayalam,
  Myanmar,
  Tibetan,
  Other,   // Any script not listed above.
};

UScript get_script(uint32_t cp) {
  // Space → Any (never triggers boundary)
  if (cp == ' ') return UScript::Any;
  // Common punctuation → Any
  if (unicode_cpt_flags(cp).is_punctuation) return UScript::Any;
  // Control, whitespace → Any
  if (unicode_cpt_flags(cp).is_whitespace) return UScript::Any;

  // Latin
  if ((cp >= 0x0041 && cp <= 0x024F) ||
      (cp >= 0x1E00 && cp <= 0x1EFF) ||
      (cp >= 0x2C60 && cp <= 0x2C7F) ||
      (cp >= 0xA720 && cp <= 0xA7FF) ||
      (cp >= 0xFB00 && cp <= 0xFB06))
    return UScript::Latin;

  // Cyrillic
  if ((cp >= 0x0400 && cp <= 0x052F) ||
      (cp >= 0x2DE0 && cp <= 0x2DFF) ||
      (cp >= 0xA640 && cp <= 0xA69F))
    return UScript::Cyrillic;

  // Greek
  if ((cp >= 0x0370 && cp <= 0x03FF) ||
      (cp >= 0x1F00 && cp <= 0x1FFF))
    return UScript::Greek;

  // Arabic
  if ((cp >= 0x0600 && cp <= 0x06FF) ||
      (cp >= 0x0750 && cp <= 0x077F) ||
      (cp >= 0x08A0 && cp <= 0x08FF) ||
      (cp >= 0xFB50 && cp <= 0xFDFF) ||
      (cp >= 0xFE70 && cp <= 0xFEFF))
    return UScript::Arabic;

  // Hebrew
  if ((cp >= 0x0590 && cp <= 0x05FF) ||
      (cp >= 0xFB1D && cp <= 0xFB4F))
    return UScript::Hebrew;

  // Devanagari
  if (cp >= 0x0900 && cp <= 0x097F) return UScript::Devanagari;

  // Bengali
  if (cp >= 0x0980 && cp <= 0x09FF) return UScript::Bengali;

  // Tamil
  if (cp >= 0x0B80 && cp <= 0x0BFF) return UScript::Tamil;

  // Telugu
  if (cp >= 0x0C00 && cp <= 0x0C7F) return UScript::Telugu;

  // Kannada
  if (cp >= 0x0C80 && cp <= 0x0CFF) return UScript::Kannada;

  // Malayalam
  if (cp >= 0x0D00 && cp <= 0x0D7F) return UScript::Malayalam;

  // Thai
  if (cp >= 0x0E00 && cp <= 0x0E7F) return UScript::Thai;

  // Georgian
  if ((cp >= 0x10A0 && cp <= 0x10FF) ||
      (cp >= 0x2D00 && cp <= 0x2D2F))
    return UScript::Georgian;

  // Armenian
  if (cp >= 0x0530 && cp <= 0x058F) return UScript::Armenian;

  // Tibetan
  if (cp >= 0x0F00 && cp <= 0x0FFF) return UScript::Tibetan;

  // Myanmar
  if (cp >= 0x1000 && cp <= 0x109F) return UScript::Myanmar;

  // Ethiopic
  if (cp >= 0x1200 && cp <= 0x137F) return UScript::Ethiopic;

  // Hangul
  if ((cp >= 0x1100 && cp <= 0x11FF) ||
      (cp >= 0x3130 && cp <= 0x318F) ||
      (cp >= 0xAC00 && cp <= 0xD7AF))
    return UScript::Hangul;

  // CJK Unified + Extensions, Hiragana, Katakana → Han
  // Hiragana
  if (cp >= 0x3040 && cp <= 0x309F) return UScript::Han;
  // Katakana
  if (cp >= 0x30A0 && cp <= 0x30FF) return UScript::Han;
  // CJK prolonged sound mark → Han
  if (cp == 0x30FC) return UScript::Han;
  // CJK
  if (unicode_utils::is_cjk(cp)) return UScript::Han;
  // CJK Symbols and Punctuation
  if (cp >= 0x3000 && cp <= 0x303F) return UScript::Han;

  // Numbers and common symbols → Any
  if (cp >= 0x0030 && cp <= 0x0039) return UScript::Any;
  if (unicode_cpt_flags(cp).is_number) return UScript::Any;
  if (unicode_cpt_flags(cp).is_symbol) return UScript::Any;

  return UScript::Other;
}

} // namespace

std::vector<std::string> UnicodeScriptsPreTokenizer::pre_tokenize(
    const std::string& input) const {
  std::vector<uint32_t> cpts = unicode_cpts_from_utf8(input);
  if (cpts.empty()) {
    return {};
  }

  std::vector<std::string> results;
  std::string current;
  UScript prev_script = UScript::Any;

  for (uint32_t cp : cpts) {
    UScript script = get_script(cp);
    // Script::Any never triggers a boundary.
    if (script != UScript::Any && prev_script != UScript::Any &&
        script != prev_script) {
      if (!current.empty()) {
        results.push_back(std::move(current));
        current.clear();
      }
    }
    current += unicode_cpt_to_utf8(cp);
    if (script != UScript::Any) {
      prev_script = script;
    }
  }
  if (!current.empty()) {
    results.push_back(std::move(current));
  }
  return results;
}

// =============================================================================
// FixedLengthPreTokenizer
// =============================================================================

std::vector<std::string> FixedLengthPreTokenizer::pre_tokenize(
    const std::string& input) const {
  // Split by character count (not byte count).
  std::vector<uint32_t> cpts = unicode_cpts_from_utf8(input);
  if (cpts.empty() || length_ == 0) {
    return {};
  }

  std::vector<std::string> results;
  std::string current;
  size_t char_count = 0;

  for (uint32_t cp : cpts) {
    current += unicode_cpt_to_utf8(cp);
    ++char_count;
    if (char_count >= length_) {
      results.push_back(std::move(current));
      current.clear();
      char_count = 0;
    }
  }
  if (!current.empty()) {
    results.push_back(std::move(current));
  }
  return results;
}

} // namespace tokenizers
