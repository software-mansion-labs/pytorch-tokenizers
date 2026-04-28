/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// @lint-ignore-every LICENSELINT

#include <pytorch/tokenizers/token_decoder.h>

#include <pytorch/tokenizers/log.h>

// Third Party
#include <nlohmann/json.hpp>

// Local
#include <unicode.h>

using json = nlohmann::json;

namespace tokenizers {

// TokenDecoderConfig //////////////////////////////////////////////////////////

TokenDecoderConfig::TokenDecoderConfig(std::string type)
    : type(std::move(type)) {}

TokenDecoder::Ptr TokenDecoderConfig::create() const {
  // NOTE: These types must line up with the type strings found in the
  //  tokenizers library
  //  https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/decoders/mod.rs#L55
  if (type == "ByteLevel") {
    return std::make_unique<ByteLevelTokenDecoder>();
  } else if (type == "Replace") {
    return std::make_unique<ReplaceTokenDecoder>(
        replace_pattern.value_or(""), replace_content.value_or(""));
  } else if (type == "ByteFallback") {
    return std::make_unique<ByteFallbackTokenDecoder>();
  } else if (type == "Fuse") {
    return std::make_unique<FuseTokenDecoder>();
  } else if (type == "Strip") {
    return std::make_unique<StripTokenDecoder>(
        strip_content.value_or(""),
        strip_start.value_or(0),
        strip_stop.value_or(0));
  } else if (type == "WordPiece") {
    return std::make_unique<WordPieceTokenDecoder>(
        wordpiece_prefix.value_or("##"), wordpiece_cleanup.value_or(true));
  } else if (type == "Sequence") {
    std::vector<TokenDecoder::Ptr> decoders;
    if (sequence_decoders) {
      for (const auto& decoder_json : *sequence_decoders) {
        TokenDecoderConfig sub_config;
        sub_config.parse_json(decoder_json);
        decoders.push_back(sub_config.create());
      }
    }
    return std::make_unique<SequenceTokenDecoder>(std::move(decoders));
  } else if (type == "Metaspace") {
    return std::make_unique<MetaspaceTokenDecoder>(
        metaspace_replacement.value_or("\xe2\x96\x81"),
        metaspace_prepend_scheme.value_or("always"));
  } else if (type == "BPEDecoder") {
    return std::make_unique<BPETokenDecoder>(
        bpe_suffix.value_or("</w>"));
  } else if (type == "CTC") {
    return std::make_unique<CTCTokenDecoder>(
        ctc_pad_token.value_or("<pad>"),
        ctc_word_delimiter_token.value_or("|"),
        ctc_cleanup.value_or(true));
  }
  throw std::runtime_error("Unsupported TokenDecoder type: " + type);
}

TokenDecoderConfig& TokenDecoderConfig::parse_json(const json& json_config) {
  type = json_config.at("type");
  if (type == "ByteLevel") {
    // No parameters to parse
  } else if (type == "Replace") {
    // Parse pattern and content for Replace decoder
    if (json_config.contains("pattern") && json_config.contains("content")) {
      if (json_config["pattern"].contains("String")) {
        set_replace_pattern(json_config["pattern"]["String"]);
      }
      set_replace_content(json_config["content"]);
    }
  } else if (type == "WordPiece") {
    set_wordpiece_prefix(json_config.value("prefix", "##"));
    set_wordpiece_cleanup(json_config.value("cleanup", true));
  } else if (type == "ByteFallback") {
    // No parameters to parse
  } else if (type == "Fuse") {
    // No parameters to parse
  } else if (type == "Strip") {
    // Parse content, start, and stop for Strip decoder
    if (json_config.contains("content")) {
      set_strip_content(json_config["content"]);
    } else {
      throw std::runtime_error("Strip decoder 'content' is required.");
    }
    set_strip_start(json_config.value("start", 0));
    set_strip_stop(json_config.value("stop", 0));
  } else if (type == "Sequence") {
    if (json_config.contains("decoders")) {
      std::vector<json> decoders;
      for (const auto& d : json_config["decoders"]) {
        decoders.push_back(d);
      }
      set_sequence_decoders(std::move(decoders));
    }
  } else if (type == "Metaspace") {
    if (json_config.contains("replacement")) {
      set_metaspace_replacement(
          json_config["replacement"].get<std::string>());
    }
    if (json_config.contains("prepend_scheme")) {
      set_metaspace_prepend_scheme(
          json_config["prepend_scheme"].get<std::string>());
    } else if (json_config.contains("add_prefix_space")) {
      bool aps = json_config["add_prefix_space"].get<bool>();
      set_metaspace_prepend_scheme(aps ? "always" : "never");
    }
  } else if (type == "BPEDecoder") {
    if (json_config.contains("suffix")) {
      set_bpe_suffix(json_config["suffix"].get<std::string>());
    }
  } else if (type == "CTC") {
    if (json_config.contains("pad_token")) {
      set_ctc_pad_token(json_config["pad_token"].get<std::string>());
    }
    if (json_config.contains("word_delimiter_token")) {
      set_ctc_word_delimiter_token(
          json_config["word_delimiter_token"].get<std::string>());
    }
    if (json_config.contains("cleanup")) {
      set_ctc_cleanup(json_config["cleanup"].get<bool>());
    }
  } else {
    throw std::runtime_error("Unsupported TokenDecoder type: " + type);
  }
  return *this;
}

// ByteLevel ///////////////////////////////////////////////////////////////////


std::vector<std::string> ByteLevelTokenDecoder::decode(
    const std::vector<std::string>& tokens) const {
  std::vector<uint8_t> all_bytes;

  for (const auto& token : tokens) {
    std::vector<uint8_t> current_token_bytes;
    bool all_chars_are_bytes = true;

    const auto cpts = unicode_cpts_from_utf8(token);
    for (const auto cpt : cpts) {
      const auto utf8 = unicode_cpt_to_utf8(cpt);
      try {
        current_token_bytes.push_back(unicode_utf8_to_byte(utf8));
      } catch (const std::out_of_range& /*e*/) {
        all_chars_are_bytes = false;
        break;
      }
    }

    if (all_chars_are_bytes) {
      all_bytes.insert(
          all_bytes.end(),
          current_token_bytes.begin(),
          current_token_bytes.end());
    } else {
      all_bytes.insert(all_bytes.end(), token.begin(), token.end());
    }
  }

  std::string final_string(all_bytes.begin(), all_bytes.end());
  return {final_string};
}

// ReplaceTokenDecoder ////////////////////////////////////////////////////////

ReplaceTokenDecoder::ReplaceTokenDecoder(
    const std::string& pattern,
    const std::string& content)
    : pattern_(pattern), content_(content) {}

std::vector<std::string> ReplaceTokenDecoder::decode(
    const std::vector<std::string>& tokens) const {
  std::vector<std::string> decoded_tokens;
  decoded_tokens.reserve(tokens.size());
  for (const auto& token : tokens) {
    // Guard against empty pattern to prevent infinite loop
    if (pattern_.empty()) {
      decoded_tokens.push_back(token);
      continue;
    }

    std::string result = token;
    size_t pos = 0;
    while ((pos = result.find(pattern_, pos)) != std::string::npos) {
      result.replace(pos, pattern_.length(), content_);
      pos += content_.length();
    }
    decoded_tokens.push_back(result);
  }
  return decoded_tokens;
}

// ByteFallbackTokenDecoder ///////////////////////////////////////////////////

std::vector<std::string> ByteFallbackTokenDecoder::decode(
    const std::vector<std::string>& tokens) const {
  std::vector<std::string> decoded_tokens;
  decoded_tokens.reserve(tokens.size());
  for (const auto& token : tokens) {
    // ByteFallback handles tokens that represent individual bytes
    // For tokens that start with <0x and end with >, extract the hex value
    if (token.length() >= 5 && token.substr(0, 3) == "<0x" &&
        token.back() == '>') {
      std::string hex_str = token.substr(3, token.length() - 4);
      try {
        unsigned long byte_val = std::stoul(hex_str, nullptr, 16);
        if (byte_val <= 255) {
          decoded_tokens.push_back(std::string(1, static_cast<char>(byte_val)));
          continue;
        }
      } catch (const std::exception&) {
        // Fall through to return original token
      }
    }
    decoded_tokens.push_back(token);
  }
  return decoded_tokens;
}

// StripTokenDecoder //////////////////////////////////////////////////////////

StripTokenDecoder::StripTokenDecoder(
    const std::string& content_str,
    size_t start,
    size_t stop)
    : start_(start), stop_(stop) {
  if (content_str.length() == 0) {
    throw std::runtime_error("Strip decoder 'content' cannot be empty.");
  }
  const auto cpts = unicode_cpts_from_utf8(content_str);
  if (cpts.size() != 1) {
    throw std::runtime_error(
        "Strip decoder 'content' must represent a single Unicode character.");
  }
  content_ = cpts[0];
}

std::vector<std::string> StripTokenDecoder::decode(
    const std::vector<std::string>& tokens) const {
  std::vector<std::string> decoded_tokens;
  decoded_tokens.reserve(tokens.size());

  for (const auto& token : tokens) {
    const auto cpts = unicode_cpts_from_utf8(token); // Convert to code points
    const size_t total_cpts = cpts.size();

    size_t start_cut = 0;
    // Strip from start
    // Iterate over code points up to start_ or the size of cpts
    for (size_t i = 0; i < start_ && i < total_cpts; ++i) {
      if (cpts[i] == content_) {
        start_cut = i + 1;
      } else {
        break; // Stop stripping if a non-content char is found
      }
    }

    size_t stop_cut = total_cpts;
    // Strip from stop
    // Iterate over code points up to stop_ or until stop_cut_cpts reaches
    // start_cut_cpts
    for (size_t i = 0; i < stop_ && stop_cut > start_cut; ++i) {
      size_t index = total_cpts - 1 - i;
      if (cpts[index] == content_) {
        stop_cut = index;
      } else {
        break; // Stop stripping if a non-content char is found
      }
    }

    if (start_cut >= stop_cut) {
      decoded_tokens.push_back("");
    } else {
      std::string new_token_str;
      for (size_t i = start_cut; i < stop_cut; ++i) {
        new_token_str += unicode_cpt_to_utf8(cpts[i]);
      }
      decoded_tokens.push_back(new_token_str);
    }
  }
  return decoded_tokens;
}

//  FuseTokenDecoder ///////////////////////////////////////////////////////
std::vector<std::string> FuseTokenDecoder::decode(
    const std::vector<std::string>& tokens) const {
  size_t total = 0;
  for (const auto& token : tokens) {
    total += token.size();
  }
  std::string fused;
  fused.reserve(total);
  for (const auto& token : tokens) {
    fused += token;
  }
  return {fused};
}

// SequenceTokenDecoder ///////////////////////////////////////////////////////

SequenceTokenDecoder::SequenceTokenDecoder(
    std::vector<TokenDecoder::Ptr> decoders)
    : decoders_(std::move(decoders)) {}

std::vector<std::string> SequenceTokenDecoder::decode(
    const std::vector<std::string>& tokens) const {
  std::vector<std::string> results = tokens;
  for (const auto& decoder : decoders_) {
    results = decoder->decode(results);
  }
  return results;
}

// WordPiece //////////////////////////////////////////////////////////////////

namespace {

std::string wordpiece_cleanup_string(std::string dirty_input) {
  struct Replacement {
    std::string pattern;
    std::string content;
  };
  static const std::vector<Replacement> replacements = {
      {" .", "."},
      {" ?", "?"},
      {" !", "!"},
      {" ,", ","},
      {" ' ", "'"},
      {" n't", "n't"},
      {" 'm", "'m"},
      {" do not", " don't"},
      {" 's", "'s"},
      {" 've", "'ve"},
      {" 're", "'re"}};

  std::string result = std::move(dirty_input);
  for (const auto& r : replacements) {
    size_t pos = 0;
    while ((pos = result.find(r.pattern, pos)) != std::string::npos) {
      result.replace(pos, r.pattern.length(), r.content);
      pos += r.content.length();
    }
  }
  return result;
}

} // namespace

WordPieceTokenDecoder::WordPieceTokenDecoder(std::string prefix, bool cleanup)
    : prefix_(std::move(prefix)), cleanup_(cleanup) {}

std::vector<std::string> WordPieceTokenDecoder::decode(
    const std::vector<std::string>& tokens) const {
  if (tokens.empty()) {
    return {};
  }

  std::vector<std::string> decoded_tokens = tokens;

  for (size_t i = 1; i < decoded_tokens.size(); ++i) {
    std::string& token = decoded_tokens[i];
    if (token.size() >= prefix_.size() &&
        token.compare(0, prefix_.size(), prefix_) == 0) {
      token = token.substr(prefix_.size());
    } else {
      token = " " + token;
    }
  }

  if (cleanup_) {
    for (auto& token : decoded_tokens) {
      token = wordpiece_cleanup_string(std::move(token));
    }
  }

  return decoded_tokens;
}

// MetaspaceTokenDecoder ///////////////////////////////////////////////////////

std::vector<std::string> MetaspaceTokenDecoder::decode(
    const std::vector<std::string>& tokens) const {
  std::vector<std::string> decoded;
  decoded.reserve(tokens.size());

  for (size_t i = 0; i < tokens.size(); ++i) {
    std::string token = tokens[i];

    // Replace all occurrences of the replacement char with space.
    size_t pos = 0;
    while ((pos = token.find(replacement_, pos)) != std::string::npos) {
      token.replace(pos, replacement_.length(), " ");
      pos += 1;
    }

    // For the first token, strip the leading space that was added by prepend.
    if (i == 0 && prepend_scheme_ != "never") {
      if (!token.empty() && token[0] == ' ') {
        token.erase(0, 1);
      }
    }

    decoded.push_back(std::move(token));
  }
  return decoded;
}

// BPETokenDecoder /////////////////////////////////////////////////////////////

std::vector<std::string> BPETokenDecoder::decode(
    const std::vector<std::string>& tokens) const {
  std::vector<std::string> decoded;
  decoded.reserve(tokens.size());

  for (size_t i = 0; i < tokens.size(); ++i) {
    std::string token = tokens[i];

    // Remove the end-of-word suffix and add a space after the word.
    if (token.size() >= suffix_.size() &&
        token.compare(token.size() - suffix_.size(), suffix_.size(), suffix_) ==
            0) {
      token = token.substr(0, token.size() - suffix_.size());
      if (i + 1 < tokens.size()) {
        token += " ";
      }
    }

    decoded.push_back(std::move(token));
  }
  return decoded;
}

// CTCTokenDecoder /////////////////////////////////////////////////////////////

std::vector<std::string> CTCTokenDecoder::decode(
    const std::vector<std::string>& tokens) const {
  std::vector<std::string> decoded;
  decoded.reserve(tokens.size());
  std::string prev;

  for (const auto& token : tokens) {
    // Skip pad tokens.
    if (token == pad_token_) {
      continue;
    }
    // Skip consecutive duplicate tokens (CTC blank collapsing).
    if (token == prev) {
      continue;
    }
    prev = token;

    // Replace word delimiter with space.
    if (token == word_delimiter_token_) {
      decoded.push_back(" ");
      continue;
    }

    decoded.push_back(token);
  }

  // Cleanup: join and clean up spaces.
  if (cleanup_) {
    std::string joined;
    for (const auto& t : decoded) {
      joined += t;
    }
    // Trim leading/trailing spaces.
    size_t start = joined.find_first_not_of(' ');
    if (start == std::string::npos) {
      return {""};
    }
    size_t end = joined.find_last_not_of(' ');
    return {joined.substr(start, end - start + 1)};
  }

  return decoded;
}

} // end  namespace tokenizers
