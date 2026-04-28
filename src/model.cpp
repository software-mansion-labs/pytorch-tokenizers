/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// @lint-ignore-every LICENSELINT

#include <pytorch/tokenizers/model.h>

#include <fstream>
#include <iostream>

#include <pytorch/tokenizers/bpe_model.h>
#include <pytorch/tokenizers/log.h>
#include <pytorch/tokenizers/map_utils.h>
#include <pytorch/tokenizers/unigram_model.h>
#include <pytorch/tokenizers/wordlevel_model.h>
#include <pytorch/tokenizers/wordpiece_model.h>

namespace tokenizers {

namespace {
// Helper to extract token string from either string or object format
std::string extract_token_string(const nlohmann::json& token_json) {
  if (token_json.is_string()) {
    return token_json.get<std::string>();
  } else if (token_json.is_object() && token_json.contains("content")) {
    return token_json["content"].get<std::string>();
  }
  return "";
};

Error parse_special_tokens(
    ModelConfig& config,
    const nlohmann::json& json_config) {
  try {
    if (json_config.contains("added_tokens")) {
      const auto& added_tokens = json_config.at("added_tokens");
      std::vector<std::pair<std::string, uint64_t>> sp_pairs;
      std::vector<std::pair<std::string, uint64_t>> all_pairs;
      for (const auto& entry : added_tokens) {
        std::string content = entry.at("content").get<std::string>();
        uint64_t id = entry.at("id").get<uint64_t>();
        // All added_tokens go into all_pairs (for encoding regex — HF
        // matches every added_token as a never-split unit during encoding).
        all_pairs.emplace_back(content, id);
        // Only tokens with "special": true are true special tokens
        // (used for is_special_token / skip_special_tokens in decoding
        // and for filtering from the model vocab to avoid BPE conflicts).
        // Default to true when field is missing (older HF formats).
        if (entry.value("special", !entry.contains("special"))) {
          sp_pairs.emplace_back(content, id);
        }
        if (entry.value("rstrip", false)) {
          config.rstrip_tokens.insert(content);
        }
        if (entry.value("lstrip", false)) {
          config.lstrip_tokens.insert(content);
        }
      }
      config.set_special_token_pairs(std::move(sp_pairs));
      config.set_all_added_token_pairs(std::move(all_pairs));
    }
  } catch (const std::exception& e) {
    TK_LOG(Info, "Could not parse special tokens: %s", e.what());
    return Error::LoadFailure;
  }
  return Error::Ok;
}

void parse_tokens(ModelConfig& config, const nlohmann::json& model_json) {
  if (model_json.contains("vocab")) {
    const auto& vocab = model_json.at("vocab");
    std::vector<std::pair<std::string, uint64_t>> pairs;
    for (const auto& entry : vocab.items()) {
      pairs.emplace_back(entry.key(), entry.value().get<uint64_t>());
    }
    config.set_token_pairs(std::move(pairs));
  }
}

void parse_merges(ModelConfig& config, const nlohmann::json& model_json) {
  if (model_json.contains("merges")) {
    const auto& merges_json = model_json.at("merges");
    std::vector<std::string> m_vec;
    for (const auto& m : merges_json) {
      if (m.is_string()) {
        m_vec.push_back(m.get<std::string>());
      } else if (m.is_array() && m.size() == 2) {
        std::string first = m[0].get<std::string>();
        std::string second = m[1].get<std::string>();
        m_vec.push_back(first + " " + second);
      }
    }
    config.set_merges(std::move(m_vec));
  }
}

} // anonymous namespace

SequenceTokenIds resolve_sequence_tokens(
    const ModelConfig& config,
    const detail::TokenMap& token_map,
    const detail::TokenMap& special_token_map) {
  SequenceTokenIds ids;
  std::string final_unk = config.unk_token.value_or("");
  std::string final_bos = config.bos_token.value_or("");
  std::string final_eos = config.eos_token.value_or("");

  auto check_extra = [&](const std::string& path) {
    if (path.empty())
      return;
    std::ifstream f(path);
    if (!f)
      return;
    try {
      nlohmann::json j = nlohmann::json::parse(f);
      if (final_bos.empty()) {
        if (j.contains("bos_token"))
          final_bos = extract_token_string(j["bos_token"]);
        else if (j.contains("cls_token"))
          final_bos = extract_token_string(j["cls_token"]);
      }
      if (final_eos.empty()) {
        if (j.contains("eos_token"))
          final_eos = extract_token_string(j["eos_token"]);
        else if (j.contains("sep_token"))
          final_eos = extract_token_string(j["sep_token"]);
      }
      if (final_unk.empty() && j.contains("unk_token")) {
        if (!j["unk_token"].is_null()) {
          final_unk = extract_token_string(j["unk_token"]);
        }
      }
    } catch (...) {
      TK_LOG(Info, "Failed to parse extra config file: %s", path.c_str());
    }
  };

  check_extra(config.special_tokens_map_path.value_or(""));
  check_extra(config.model_config_path.value_or(""));

  // Fallback discovery: search by name
  if (final_bos.empty() || final_eos.empty()) {
    std::vector<std::string_view> bos_c, eos_c;
    // Search special tokens
    for (size_t i = 0; i < special_token_map.size(); ++i) {
      const auto& [token, _] = special_token_map.getElement(i);
      if (final_bos.empty() &&
          (token.find("bos") != std::string::npos ||
           token.find("begin") != std::string::npos ||
           token.find("cls") != std::string::npos))
        bos_c.push_back(token);
      if (final_eos.empty() &&
          (token.find("eos") != std::string::npos ||
           token.find("end") != std::string::npos ||
           token.find("sep") != std::string::npos))
        eos_c.push_back(token);
    }
    if (final_bos.empty() && bos_c.size() == 1) {
      final_bos = std::string(bos_c[0]);
    }
    if (final_eos.empty() && eos_c.size() == 1) {
      final_eos = std::string(eos_c[0]);
    }
  }

  // UNK priority list
  if (final_unk.empty()) {
    for (const auto& name : {"<unk>", "[UNK]", "<|endoftext|>"}) {
      if (special_token_map.tryGetInteger(name) ||
          token_map.tryGetInteger(name)) {
        final_unk = name;
        break;
      }
    }
  }

  // Resolve strings to IDs (check both maps)
  auto resolve = [&](const std::string& s) -> std::optional<uint64_t> {
    if (s.empty())
      return std::nullopt;
    auto id = special_token_map.tryGetInteger(s);
    if (!id) {
      id = token_map.tryGetInteger(s);
    }
    return id;
  };

  ids.unk_token_id = resolve(final_unk);
  ids.bos_token_id = resolve(final_bos);
  ids.eos_token_id = resolve(final_eos);

  return ids;
}

ModelConfig& ModelConfig::parse_json(const nlohmann::json& json_config) {
  if (json_config.contains("model") && !json_config.at("model").is_null()) {
    const auto& model_json = json_config.at("model");

    // Read explicit type if present.
    if (model_json.contains("type")) {
      type = model_json.at("type").get<std::string>();
    }

    // Infer type from JSON structure when "type" is missing.
    // Older HuggingFace tokenizer.json files omit the type field.
    if (type.empty() && model_json.contains("vocab")) {
      const auto& vocab_json = model_json.at("vocab");
      bool has_merges = model_json.contains("merges");
      bool has_unk_id = model_json.contains("unk_id");

      if (has_unk_id && vocab_json.is_array() && !vocab_json.empty() &&
          vocab_json[0].is_array() && vocab_json[0].size() == 2 &&
          vocab_json[0][1].is_number_float()) {
        // Array of [string, float] pairs + unk_id → Unigram
        // The float scores are the distinguishing feature of Unigram vocab.
        type = "Unigram";
      } else if (has_merges) {
        // Has merges → BPE (regardless of vocab format)
        type = "BPE";
      } else if (vocab_json.is_object() &&
                 model_json.contains("continuing_subword_prefix")) {
        // Dict vocab + continuing_subword_prefix → WordPiece
        type = "WordPiece";
      } else if (vocab_json.is_object()) {
        // Dict vocab, no merges, no subword prefix → WordLevel
        type = "WordLevel";
      }

      if (!type.empty()) {
        TK_LOG(
            Info,
            "Model type not specified in tokenizer.json, inferred as: %s",
            type.c_str());
      }
    }

    // Parse vocab for non-Unigram models (dict format: {"token": id}).
    // Unigram uses a different format parsed below.
    if (type != "Unigram") {
      parse_tokens(*this, model_json);
    }

    // Common config parsing
    if (model_json.contains("unk_token") &&
        !model_json.at("unk_token").is_null()) {
      set_unk_token(model_json.at("unk_token").get<std::string>());
    }

    if (type == "BPE") {
      parse_merges(*this, model_json);
      if (model_json.contains("byte_fallback")) {
        set_byte_fallback(model_json.at("byte_fallback").get<bool>());
      }
    } else if (type == "WordPiece") {
      if (model_json.contains("continuing_subword_prefix")) {
        set_continuing_subword_prefix(
            model_json.at("continuing_subword_prefix").get<std::string>());
      }
      if (model_json.contains("max_input_chars_per_word")) {
        set_max_input_chars_per_word(
            model_json.at("max_input_chars_per_word").get<size_t>());
      }
    } else if (type == "WordLevel") {
      // WordLevel uses the same dict vocab format as BPE/WordPiece (already
      // parsed above). No merges, no byte_fallback.
    } else if (type == "Unigram") {
      if (model_json.contains("vocab")) {
        std::vector<std::pair<std::string, double>> vocab;
        for (const auto& entry : model_json.at("vocab")) {
          vocab.emplace_back(
              entry[0].get<std::string>(), entry[1].get<double>());
        }
        set_unigram_vocab(std::move(vocab));
      }
      if (model_json.contains("unk_id") &&
          !model_json.at("unk_id").is_null()) {
        set_unigram_unk_id(model_json.at("unk_id").get<size_t>());
      }
      if (model_json.contains("byte_fallback")) {
        set_byte_fallback(model_json.at("byte_fallback").get<bool>());
      }
    }
  }

  parse_special_tokens(*this, json_config);

  return *this;
}

Model::Ptr ModelConfig::create() const {
  const auto& raw_token_pairs =
      token_pairs.value_or(std::vector<std::pair<std::string, uint64_t>>());
  const auto& raw_special_pairs = special_token_pairs.value_or(
      std::vector<std::pair<std::string, uint64_t>>());
  const auto& raw_all_added_pairs = all_added_token_pairs.value_or(
      raw_special_pairs); // fallback to special_token_pairs if not set

  // Filter out truly-special tokens from vocab (only special=true tokens).
  // Non-special added_tokens (e.g. gemma3's 6406 vocab entries) must remain
  // in the model vocab for BPE merges to work.
  std::unordered_set<uint64_t> special_ids;
  for (const auto& sp : raw_special_pairs) {
    special_ids.insert(sp.second);
  }
  std::unordered_set<uint64_t> model_ids;
  std::vector<std::pair<std::string, uint64_t>> model_token_pairs;
  for (const auto& tp : raw_token_pairs) {
    if (special_ids.find(tp.second) == special_ids.end()) {
      model_token_pairs.push_back(tp);
      model_ids.insert(tp.second);
    }
  }
  // Add non-special added_tokens that aren't already in the model vocab.
  // These tokens (e.g. <|tool|> in phi) need to be in token_map for
  // vocab_size counting and piece_to_id lookups.
  for (const auto& ap : raw_all_added_pairs) {
    if (special_ids.find(ap.second) == special_ids.end() &&
        model_ids.find(ap.second) == model_ids.end()) {
      model_token_pairs.push_back(ap);
    }
  }

  auto token_map_res = detail::build_token_map(std::move(model_token_pairs));
  auto special_token_map_res = detail::build_token_map(raw_special_pairs);

  if (!token_map_res.ok() || !special_token_map_res.ok()) {
    return nullptr;
  }

  auto token_map = std::move(*token_map_res);
  auto special_token_map = std::move(*special_token_map_res);

  // Resolve sequence tokens
  auto ids = resolve_sequence_tokens(*this, token_map, special_token_map);

  // Build special token regex (only special=true tokens, for standalone
  // model.tokenize()). HFTokenizer builds its own regex from ALL added tokens.
  // Must be built before moving special_token_map.
  std::unique_ptr<IRegex> special_regex;
  {
    auto res = detail::build_special_token_regex(special_token_map);
    if (res.ok()) {
      special_regex = std::move(*res);
    }
  }

  if (type == "BPE") {
    return create_bpe(
        std::move(token_map), std::move(special_token_map),
        std::move(special_regex), ids);
  } else if (type == "WordPiece") {
    return create_wordpiece(
        std::move(token_map), std::move(special_token_map), ids);
  } else if (type == "Unigram") {
    return create_unigram(
        std::move(special_token_map), std::move(special_regex), ids);
  } else if (type == "WordLevel") {
    return create_wordlevel(
        std::move(token_map), std::move(special_token_map),
        std::move(special_regex), ids);
  }

  return nullptr;
}

// ---------------------------------------------------------------------------
// Per-type factory helpers
// ---------------------------------------------------------------------------

Model::Ptr ModelConfig::create_bpe(
    detail::TokenMap token_map,
    detail::TokenMap special_token_map,
    std::unique_ptr<IRegex> regex,
    const SequenceTokenIds& ids) const {
  std::optional<detail::TokenMap> merge_ranks;
  detail::MergeMap merge_map;
  const auto& raw_merges = merges.value_or(std::vector<std::string>());
  for (size_t i = 0; i < raw_merges.size(); ++i) {
    const std::string& m = raw_merges[i];
    if (m.rfind("#version", 0) == 0)
      continue;
    auto space_pos = m.find(' ');
    if (space_pos != std::string::npos) {
      std::string first = m.substr(0, space_pos);
      std::string second = m.substr(space_pos + 1);
      auto first_id = token_map.tryGetInteger(first);
      auto second_id = token_map.tryGetInteger(second);
      if (first_id && second_id) {
        std::string merged = first + second;
        auto merged_id = token_map.tryGetInteger(merged);
        if (merged_id) {
          merge_map.emplace(
              std::make_pair(*first_id, *second_id),
              std::make_pair(static_cast<uint32_t>(i), *merged_id));
        }
      }
    }
  }
  auto merge_ranks_res = detail::build_merge_ranks_map(merge_map, token_map);
  if (merge_ranks_res.ok()) {
    merge_ranks.emplace(std::move(*merge_ranks_res));
  }

  return std::make_unique<BPEModel>(
      std::move(token_map),
      std::move(special_token_map),
      std::move(merge_ranks),
      std::move(regex),
      byte_fallback.value_or(false),
      ids.unk_token_id,
      ids.bos_token_id,
      ids.eos_token_id,
      rstrip_tokens,
      lstrip_tokens);
}

Model::Ptr ModelConfig::create_wordpiece(
    detail::TokenMap token_map,
    detail::TokenMap special_token_map,
    const SequenceTokenIds& ids) const {
  return std::make_unique<WordPieceModel>(
      std::move(token_map),
      std::move(special_token_map),
      unk_token.value_or("[UNK]"),
      continuing_subword_prefix.value_or("##"),
      max_input_chars_per_word.value_or(100),
      ids.unk_token_id,
      ids.bos_token_id,
      ids.eos_token_id,
      rstrip_tokens,
      lstrip_tokens);
}

Model::Ptr ModelConfig::create_unigram(
    detail::TokenMap special_token_map,
    std::unique_ptr<IRegex> regex,
    const SequenceTokenIds& ids) const {
  const auto& vocab = unigram_vocab.value_or(UnigramVocab{});

  std::vector<std::string> pieces;
  std::vector<double> scores;
  pieces.reserve(vocab.size());
  scores.reserve(vocab.size());
  for (const auto& [piece, score] : vocab) {
    pieces.push_back(piece);
    scores.push_back(score);
  }

  std::optional<uint64_t> resolved_unk;
  if (unigram_unk_id) {
    resolved_unk = static_cast<uint64_t>(*unigram_unk_id);
  } else if (ids.unk_token_id) {
    resolved_unk = ids.unk_token_id;
  }

  return std::make_unique<UnigramModel>(
      std::move(pieces),
      std::move(scores),
      std::move(special_token_map),
      std::move(regex),
      byte_fallback.value_or(false),
      resolved_unk,
      ids.bos_token_id,
      ids.eos_token_id,
      rstrip_tokens,
      lstrip_tokens);
}

Model::Ptr ModelConfig::create_wordlevel(
    detail::TokenMap token_map,
    detail::TokenMap special_token_map,
    std::unique_ptr<IRegex> regex,
    const SequenceTokenIds& ids) const {
  return std::make_unique<WordLevelModel>(
      std::move(token_map),
      std::move(special_token_map),
      std::move(regex),
      ids.unk_token_id,
      ids.bos_token_id,
      ids.eos_token_id,
      rstrip_tokens,
      lstrip_tokens);
}

} // namespace tokenizers
