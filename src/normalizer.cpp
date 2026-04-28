/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// @lint-ignore-every LICENSELINT

// Local
#include <pytorch/tokenizers/normalizer.h>
#include <pytorch/tokenizers/log.h>
#include <pytorch/tokenizers/regex.h>
#include <pytorch/tokenizers/unicode-nfc.h>
#include <pytorch/tokenizers/unicode_utils.h>

// Third Party
#include <unicode.h>

// Darts double-array trie (from SentencePiece third-party)
#include "third_party/darts_clone/darts.h"

// Standard
#include <cassert>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// Third Party
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace tokenizers {

// =============================================================================
// NormalizerConfig — Factory
// =============================================================================

NormalizerConfig::NormalizerConfig(std::string type) : type(std::move(type)) {}

Normalizer::Ptr NormalizerConfig::create() const {
  if (type == "Replace") {
    if (!pattern || !content) {
      throw std::runtime_error("Missing pattern/content for Replace");
    }
    return std::make_unique<ReplaceNormalizer>(*pattern, *content);
  }
  if (type == "Prepend") {
    if (!prepend) {
      throw std::runtime_error("Missing prepend for Prepend");
    }
    return std::make_unique<PrependNormalizer>(*prepend);
  }
  if (type == "Sequence") {
    if (!normalizers || normalizers->empty()) {
      throw std::runtime_error("Missing normalizers for Sequence");
    }
    std::vector<Normalizer::Ptr> norms;
    for (const auto& cfg : *normalizers) {
      norms.push_back(cfg.create());
    }
    return std::make_unique<SequenceNormalizer>(std::move(norms));
  }
  if (type == "BertNormalizer") {
    return std::make_unique<BertNormalizer>(
        clean_text.value_or(true),
        handle_chinese_chars.value_or(true),
        lowercase.value_or(true),
        strip_accents);
  }
  if (type == "NFC") {
    return std::make_unique<NFCNormalizer>();
  }
  if (type == "NFD") {
    return std::make_unique<NFDNormalizer>();
  }
  if (type == "NFKC") {
    return std::make_unique<NFKCNormalizer>();
  }
  if (type == "NFKD") {
    return std::make_unique<NFKDNormalizer>();
  }
  if (type == "Lowercase") {
    return std::make_unique<LowercaseNormalizer>();
  }
  if (type == "Strip") {
    return std::make_unique<StripNormalizer>(
        strip_left.value_or(true), strip_right.value_or(true));
  }
  if (type == "StripAccents") {
    return std::make_unique<StripAccentsNormalizer>();
  }
  if (type == "Nmt") {
    return std::make_unique<NmtNormalizer>();
  }
  if (type == "ByteLevel") {
    return std::make_unique<ByteLevelNormalizer>();
  }
  if (type == "Precompiled") {
    if (!precompiled_charsmap || precompiled_charsmap->empty()) {
      throw std::runtime_error("Missing precompiled_charsmap for Precompiled");
    }
    return std::make_unique<PrecompiledNormalizer>(*precompiled_charsmap);
  }
  throw std::runtime_error("Unsupported Normalizer type: " + type);
}

NormalizerConfig& NormalizerConfig::parse_json(const json& json_config) {
  type = json_config.at("type");

  if (type == "Replace") {
    try {
      set_pattern(json_config.at("pattern").at("Regex"));
    } catch (json::out_of_range&) {
      std::string literal = json_config.at("pattern").at("String");
      set_pattern(IRegex::escape(literal));
    }
    set_content(json_config.at("content"));

  } else if (type == "Prepend") {
    set_prepend(json_config.at("prepend"));

  } else if (type == "Sequence") {
    std::vector<NormalizerConfig> cfgs;
    for (const auto& entry : json_config.at("normalizers")) {
      cfgs.push_back(NormalizerConfig().parse_json(entry));
    }
    set_normalizers(std::move(cfgs));

  } else if (type == "NFC" || type == "NFD" || type == "NFKC" ||
             type == "NFKD" || type == "Lowercase" || type == "StripAccents" ||
             type == "Nmt" || type == "ByteLevel") {
    // No config parameters

  } else if (type == "BertNormalizer") {
    if (json_config.contains("clean_text") &&
        !json_config.at("clean_text").is_null()) {
      set_clean_text(json_config.at("clean_text"));
    }
    if (json_config.contains("handle_chinese_chars") &&
        !json_config.at("handle_chinese_chars").is_null()) {
      set_handle_chinese_chars(json_config.at("handle_chinese_chars"));
    }
    if (json_config.contains("lowercase") &&
        !json_config.at("lowercase").is_null()) {
      set_lowercase(json_config.at("lowercase"));
    }
    if (json_config.contains("strip_accents") &&
        !json_config.at("strip_accents").is_null()) {
      set_strip_accents(json_config.at("strip_accents"));
    }

  } else if (type == "Strip") {
    if (json_config.contains("strip_left")) {
      set_strip_left(json_config["strip_left"].get<bool>());
    }
    if (json_config.contains("strip_right")) {
      set_strip_right(json_config["strip_right"].get<bool>());
    }

  } else if (type == "Precompiled") {
    if (json_config.contains("precompiled_charsmap")) {
      set_precompiled_charsmap(
          json_config["precompiled_charsmap"].get<std::string>());
    }

  } else {
    throw std::runtime_error("Unsupported Normalizer type: " + type);
  }
  return *this;
}

// =============================================================================
// Replace
// =============================================================================

std::unique_ptr<IRegex> ReplaceNormalizer::create_regex_(
    const std::string& pattern) {
  assert(!pattern.empty());
  auto regex_result = create_regex(pattern);
  if (!regex_result.ok()) {
    throw std::runtime_error(
        "Error: " + std::to_string(static_cast<int>(regex_result.error())));
  }
  return std::move(regex_result.get());
}

std::string ReplaceNormalizer::normalize(const std::string& input) const {
  if (!regex_)
    return input;

  std::string result = input;
  auto matches = regex_->find_all(result);

  for (auto it = matches.rbegin(); it != matches.rend(); ++it) {
    result.replace(it->start, it->end - it->start, content_);
  }
  return result;
}

// =============================================================================
// Prepend
// =============================================================================

std::string PrependNormalizer::normalize(const std::string& input) const {
  if (input.empty()) {
    return "";
  }
  return prepend_ + input;
}

// =============================================================================
// Sequence
// =============================================================================

SequenceNormalizer::SequenceNormalizer(std::vector<Normalizer::Ptr> normalizers)
    : normalizers_(std::move(normalizers)) {}

std::string SequenceNormalizer::normalize(const std::string& input) const {
  std::string result = input;
  for (const auto& normalizer : normalizers_) {
    result = normalizer->normalize(result);
  }
  return result;
}

// =============================================================================
// NFC
// =============================================================================

std::string NFCNormalizer::normalize(const std::string& input) const {
  return unicode_normalize_nfc_utf8(input);
}

// =============================================================================
// NFD
// =============================================================================

std::string NFDNormalizer::normalize(const std::string& input) const {
  auto cpts = unicode_cpts_from_utf8(input);
  auto nfd = unicode_cpts_normalize_nfd(cpts);
  std::string result;
  for (uint32_t cp : nfd) {
    result += unicode_cpt_to_utf8(cp);
  }
  return result;
}

// =============================================================================
// NFKC / NFKD — approximations
// =============================================================================

std::string NFKCNormalizer::normalize(const std::string& input) const {
  static bool warned = false;
  if (!warned) {
    TK_LOG(
        Info,
        "NFKC normalizer is experimental: approximated with NFC. "
        "Compatibility decompositions (e.g. fi-ligature, superscripts) are "
        "NOT applied. Full NFKC requires Unicode compatibility decomposition "
        "tables. For SentencePiece models, use the Precompiled normalizer.");
    warned = true;
  }
  return unicode_normalize_nfc_utf8(input);
}

std::string NFKDNormalizer::normalize(const std::string& input) const {
  static bool warned = false;
  if (!warned) {
    TK_LOG(
        Info,
        "NFKD normalizer is experimental: approximated with NFD. "
        "Compatibility decompositions (e.g. fi-ligature, superscripts) are "
        "NOT applied. Full NFKD requires Unicode compatibility decomposition "
        "tables. For SentencePiece models, use the Precompiled normalizer.");
    warned = true;
  }
  auto cpts = unicode_cpts_from_utf8(input);
  auto nfd = unicode_cpts_normalize_nfd(cpts);
  std::string result;
  for (uint32_t cp : nfd) {
    result += unicode_cpt_to_utf8(cp);
  }
  return result;
}

// =============================================================================
// Lowercase
// =============================================================================

std::string LowercaseNormalizer::normalize(const std::string& input) const {
  auto codepoints = unicode_cpts_from_utf8(input);
  for (auto& cp : codepoints) {
    cp = unicode_tolower(cp);
  }
  std::string result;
  for (uint32_t cpt : codepoints) {
    result += unicode_cpt_to_utf8(cpt);
  }
  return result;
}

// =============================================================================
// BertNormalizer
// =============================================================================

namespace {

bool is_bert_whitespace(uint32_t c) {
  switch (c) {
    case '\t':
    case '\n':
    case '\r':
      return true;
  }
  return unicode_cpt_flags(c).is_whitespace;
}

bool is_bert_control(uint32_t c) {
  switch (c) {
    case '\t':
    case '\n':
    case '\r':
      return false;
  }
  return unicode_cpt_flags(c).is_control;
}

} // namespace

std::string BertNormalizer::normalize(const std::string& input) const {
  std::vector<uint32_t> cpts = unicode_cpts_from_utf8(input);

  if (clean_text_) {
    std::vector<uint32_t> cleaned;
    cleaned.reserve(cpts.size());
    for (uint32_t cp : cpts) {
      if (cp == 0 || cp == 0xfffd || is_bert_control(cp))
        continue;
      cleaned.push_back(is_bert_whitespace(cp) ? ' ' : cp);
    }
    cpts = std::move(cleaned);
  }

  if (handle_chinese_chars_) {
    std::vector<uint32_t> handled;
    handled.reserve(cpts.size() * 3);
    for (uint32_t cp : cpts) {
      if (unicode_utils::is_cjk(cp)) {
        handled.push_back(' ');
        handled.push_back(cp);
        handled.push_back(' ');
      } else {
        handled.push_back(cp);
      }
    }
    cpts = std::move(handled);
  }

  const bool do_strip =
      strip_accents_.has_value() ? *strip_accents_ : lowercase_;
  if (do_strip) {
    cpts = unicode_cpts_normalize_nfd(cpts);
    std::vector<uint32_t> stripped;
    stripped.reserve(cpts.size());
    for (uint32_t cp : cpts) {
      if (!unicode_cpt_flags(cp).is_accent_mark)
        stripped.push_back(cp);
    }
    cpts = std::move(stripped);
  }

  if (lowercase_) {
    for (auto& cp : cpts) {
      cp = unicode_tolower(cp);
    }
  }

  std::string result;
  for (uint32_t cp : cpts) {
    result += unicode_cpt_to_utf8(cp);
  }
  return result;
}

// =============================================================================
// Strip
// =============================================================================

std::string StripNormalizer::normalize(const std::string& input) const {
  auto cpts = unicode_cpts_from_utf8(input);
  if (cpts.empty())
    return input;

  size_t start = 0;
  size_t end = cpts.size();

  if (strip_left_) {
    while (start < end && unicode_cpt_flags(cpts[start]).is_whitespace)
      ++start;
  }
  if (strip_right_) {
    while (end > start && unicode_cpt_flags(cpts[end - 1]).is_whitespace)
      --end;
  }

  std::string result;
  for (size_t i = start; i < end; ++i) {
    result += unicode_cpt_to_utf8(cpts[i]);
  }
  return result;
}

// =============================================================================
// StripAccents
// =============================================================================

std::string StripAccentsNormalizer::normalize(const std::string& input) const {
  auto cpts = unicode_cpts_from_utf8(input);
  auto nfd = unicode_cpts_normalize_nfd(cpts);

  std::string result;
  for (uint32_t cp : nfd) {
    if (!unicode_cpt_flags(cp).is_accent_mark) {
      result += unicode_cpt_to_utf8(cp);
    }
  }
  return result;
}

// =============================================================================
// Nmt
// =============================================================================

std::string NmtNormalizer::normalize(const std::string& input) const {
  auto cpts = unicode_cpts_from_utf8(input);
  std::string result;

  for (uint32_t cp : cpts) {
    // Remove control characters (except common whitespace)
    if (cp == 0x0001 || cp == 0x0002 || cp == 0x0003 || cp == 0x0004 ||
        cp == 0x0005 || cp == 0x0006 || cp == 0x0007 || cp == 0x0008 ||
        cp == 0x000E || cp == 0x000F || (cp >= 0x0010 && cp <= 0x001F) ||
        (cp >= 0x007F && cp <= 0x009F) || cp == 0x00AD) {
      continue; // Remove
    }
    // Normalize various Unicode spaces to regular space
    if (cp == 0x00A0 || cp == 0x1680 ||
        (cp >= 0x2000 && cp <= 0x200B) ||
        cp == 0x202F || cp == 0x205F || cp == 0x3000) {
      result += ' ';
      continue;
    }
    result += unicode_cpt_to_utf8(cp);
  }
  return result;
}

// =============================================================================
// ByteLevel
// =============================================================================

std::string ByteLevelNormalizer::normalize(const std::string& input) const {
  std::string result;
  for (unsigned char byte : input) {
    result += unicode_byte_to_utf8(byte);
  }
  return result;
}

// =============================================================================
// Precompiled — uses SentencePiece's Normalizer internally
// =============================================================================

// ---------------------------------------------------------------------------
// Precompiled — self-contained using darts_clone double-array trie
// ---------------------------------------------------------------------------
//
// The precompiled_charsmap binary format (from SentencePiece):
//   [4 bytes LE] trie_size
//   [trie_size bytes] double-array trie blob
//   [remaining bytes] null-delimited normalized string data
//
// For each input position, we search the trie for the longest matching prefix.
// If found, the trie value is an offset into the normalized string data (a
// null-terminated C string at that offset). The matched input bytes are
// replaced with that normalized string. If no match is found, the character
// passes through unchanged.

namespace {
/// Decode base64 string to raw bytes.
std::string decode_base64(const std::string& input) {
  static const unsigned char kTable[128] = {
      64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
      64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
      64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 62, 64, 64, 64, 63,
      52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 64, 64, 64, 64, 64, 64,
      64,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
      15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 64, 64, 64, 64, 64,
      64, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
      41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 64, 64, 64, 64, 64};

  size_t in_len = input.size();
  size_t out_len = in_len / 4 * 3;
  if (in_len >= 1 && input[in_len - 1] == '=') out_len--;
  if (in_len >= 2 && input[in_len - 2] == '=') out_len--;

  std::string out(out_len, '\0');
  size_t j = 0;
  for (size_t i = 0; i < in_len; i += 4) {
    auto v = [&](size_t idx) -> uint32_t {
      return input[idx] == '=' ? 0u : kTable[static_cast<unsigned char>(input[idx])];
    };
    uint32_t triple = (v(i) << 18) | (v(i+1) << 12) | (v(i+2) << 6) | v(i+3);
    if (j < out_len) out[j++] = static_cast<char>((triple >> 16) & 0xFF);
    if (j < out_len) out[j++] = static_cast<char>((triple >> 8) & 0xFF);
    if (j < out_len) out[j++] = static_cast<char>(triple & 0xFF);
  }
  return out;
}

/// Return the byte-length of a UTF-8 character given its lead byte.
inline size_t utf8_len(unsigned char byte) {
  if ((byte & 0x80) == 0) return 1;
  if ((byte & 0xE0) == 0xC0) return 2;
  if ((byte & 0xF0) == 0xE0) return 3;
  if ((byte & 0xF8) == 0xF0) return 4;
  return 1;
}
} // namespace

struct PrecompiledNormalizer::Impl {
  std::string blob; // raw decoded binary — must outlive trie/normalized
  Darts::DoubleArray trie;
  const char* normalized = nullptr; // pointer into blob
};

PrecompiledNormalizer::PrecompiledNormalizer(
    const std::string& precompiled_charsmap_b64) {
  impl_ = std::make_unique<Impl>();
  impl_->blob = decode_base64(precompiled_charsmap_b64);

  if (impl_->blob.size() < 4) {
    throw std::runtime_error("Precompiled charsmap blob too small");
  }

  // Read trie size (4 bytes, little-endian).
  uint32_t trie_size = 0;
  std::memcpy(&trie_size, impl_->blob.data(), sizeof(trie_size));

  if (sizeof(trie_size) + trie_size > impl_->blob.size()) {
    throw std::runtime_error("Trie size exceeds blob");
  }

  const char* trie_data = impl_->blob.data() + sizeof(trie_size);
  impl_->trie.set_array(
      const_cast<char*>(trie_data),
      trie_size / impl_->trie.unit_size());

  impl_->normalized = trie_data + trie_size;
}

PrecompiledNormalizer::~PrecompiledNormalizer() = default;

std::string PrecompiledNormalizer::normalize(const std::string& input) const {
  if (input.empty()) return {};

  std::string result;
  result.reserve(input.size());

  static constexpr size_t kMaxResults = 64;
  Darts::DoubleArray::result_pair_type trie_results[kMaxResults];

  size_t pos = 0;
  while (pos < input.size()) {
    // Search for the longest matching prefix in the trie.
    size_t num = impl_->trie.commonPrefixSearch(
        input.data() + pos, trie_results, kMaxResults, input.size() - pos);

    size_t best_len = 0;
    int best_value = 0;
    for (size_t k = 0; k < num; ++k) {
      if (trie_results[k].length > static_cast<int>(best_len)) {
        best_len = trie_results[k].length;
        best_value = trie_results[k].value;
      }
    }

    if (best_len > 0) {
      // Found a match — append the normalized replacement.
      const char* replacement = &impl_->normalized[best_value];
      result.append(replacement); // null-terminated C string
      pos += best_len;
    } else {
      // No match — pass through one UTF-8 character.
      size_t char_len = utf8_len(static_cast<unsigned char>(input[pos]));
      if (pos + char_len > input.size()) char_len = input.size() - pos;
      result.append(input, pos, char_len);
      pos += char_len;
    }
  }

  return result;
}

} // namespace tokenizers
