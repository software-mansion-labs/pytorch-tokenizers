/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// @lint-ignore-every LICENSELINT

#include <pytorch/tokenizers/bpe_model.h>

#include <limits>
#include <queue>

#include <pytorch/tokenizers/log.h>

namespace tokenizers {

namespace {
// Simple Word structure to mimic Rust's Word behavior, as used in HFTokenizer
struct HFWord {
  std::vector<uint64_t> tokens;
  std::vector<size_t> byte_lengths;

  void add(uint64_t token_id, size_t byte_len) {
    tokens.push_back(token_id);
    byte_lengths.push_back(byte_len);
  }

  size_t size() const {
    return tokens.size();
  }

  // Apply all possible merges using the merge ranks.
  //
  // Mirrors HuggingFace's Word::merge_all: an intrusive doubly linked list over
  // the symbols plus a min-heap keyed on merge rank, giving O(n log n) instead
  // of the naive rescan-all-pairs O(n^2) this used to do. Ported from upstream
  // tokenizers' HFWord::merge_all, adapted because our merge table is keyed by
  // the merged token string rather than by an (id, id) pair.
  void merge_all(
      const detail::TokenMap& merge_ranks,
      const detail::TokenMap& token_map) {
    const size_t n = tokens.size();
    if (n < 2) {
      return;
    }

    std::vector<int64_t> prev(n), next(n);
    std::vector<uint32_t> version(n, 0);
    std::vector<bool> alive(n, true);
    for (size_t i = 0; i < n; ++i) {
      prev[i] = static_cast<int64_t>(i) - 1;
      next[i] = (i + 1 < n) ? static_cast<int64_t>(i + 1) : -1;
    }

    // Rank and resulting id for the pair (tokens[i], tokens[j]).
    auto pair_lookup =
        [&](int64_t i,
            int64_t j) -> std::optional<std::pair<uint64_t, uint64_t>> {
      const auto first_token = token_map.tryGetString(tokens[i]);
      const auto second_token = token_map.tryGetString(tokens[j]);
      if (!first_token || !second_token) {
        return std::nullopt;
      }
      std::string merged_token;
      merged_token.reserve(first_token->size() + second_token->size());
      merged_token += *first_token;
      merged_token += *second_token;

      const auto rank = merge_ranks.tryGetInteger(merged_token);
      if (!rank) {
        return std::nullopt;
      }
      const auto merged_id = token_map.tryGetInteger(merged_token);
      if (!merged_id) {
        return std::nullopt;
      }
      return std::make_pair(*rank, *merged_id);
    };

    // (rank, pos, version): lowest rank wins, ties broken by leftmost position
    // to match left-to-right BPE order. `version` tracks only the left symbol;
    // a candidate whose right neighbor changed is rejected below by recomputing
    // the pair's rank and comparing to `c.rank`. That check relies on merge
    // ranks being unique (they are merge indices), so a changed pair cannot
    // coincidentally carry the same rank.
    struct Candidate {
      uint64_t rank;
      int64_t pos;
      uint32_t version;
    };
    struct Compare {
      bool operator()(const Candidate& a, const Candidate& b) const {
        if (a.rank != b.rank) {
          return a.rank > b.rank;
        }
        return a.pos > b.pos;
      }
    };
    std::priority_queue<Candidate, std::vector<Candidate>, Compare> heap;

    auto push_pair = [&](int64_t i) {
      if (i < 0 || next[i] < 0) {
        return;
      }
      if (const auto found = pair_lookup(i, next[i])) {
        heap.push({found->first, i, version[i]});
      }
    };

    for (size_t i = 0; i + 1 < n; ++i) {
      push_pair(static_cast<int64_t>(i));
    }

    while (!heap.empty()) {
      const Candidate c = heap.top();
      heap.pop();
      const int64_t i = c.pos;
      if (!alive[i] || version[i] != c.version || next[i] < 0) {
        continue; // stale entry
      }
      const int64_t j = next[i];
      const auto found = pair_lookup(i, j);
      if (!found || found->first != c.rank) {
        continue; // superseded by a different merge at this position
      }

      // Merge j into i.
      tokens[i] = found->second;
      byte_lengths[i] += byte_lengths[j];
      alive[j] = false;
      const int64_t jn = next[j];
      next[i] = jn;
      if (jn >= 0) {
        prev[jn] = i;
      }
      ++version[i];

      // Only the two adjacencies touching the merged symbol can change.
      push_pair(prev[i]);
      push_pair(i);
    }

    // Compact surviving symbols in list order (head is always index 0; it is
    // only ever a left operand, never removed).
    std::vector<uint64_t> merged_tokens;
    std::vector<size_t> merged_byte_lengths;
    merged_tokens.reserve(tokens.size());
    merged_byte_lengths.reserve(byte_lengths.size());
    for (int64_t i = 0; i != -1; i = next[i]) {
      merged_tokens.push_back(tokens[i]);
      merged_byte_lengths.push_back(byte_lengths[i]);
    }
    tokens = std::move(merged_tokens);
    byte_lengths = std::move(merged_byte_lengths);
  }
};
} // namespace

BPEModel::BPEModel(
    detail::TokenMap token_map,
    detail::TokenMap special_token_map,
    std::optional<detail::TokenMap> merge_ranks,
    std::unique_ptr<IRegex> special_token_regex,
    bool byte_fallback,
    std::optional<uint64_t> unk_token_id,
    std::optional<uint64_t> bos_token_id,
    std::optional<uint64_t> eos_token_id,
    std::unordered_set<std::string> rstrip_tokens,
    std::unordered_set<std::string> lstrip_tokens)
    : token_map_(std::move(token_map)),
      special_token_map_(std::move(special_token_map)),
      merge_ranks_(std::move(merge_ranks)),
      special_token_regex_(std::move(special_token_regex)),
      byte_fallback_(byte_fallback),
      unk_token_id_(unk_token_id),
      bos_token_id_(bos_token_id),
      eos_token_id_(eos_token_id),
      rstrip_tokens_(std::move(rstrip_tokens)),
      lstrip_tokens_(std::move(lstrip_tokens)) {
  vocab_size_ = token_map_.size() + special_token_map_.size();
  initialized_ = true;
}

Result<std::vector<uint64_t>> BPEModel::tokenize(
    const std::string& piece) const {
  if (!initialized_) {
    return Error::Uninitialized;
  }
  auto encode_result = encode_with_special_token(piece);
  if (!encode_result.ok()) {
    return encode_result.error();
  }
  return (*encode_result).first;
}

Result<std::string> BPEModel::id_to_piece(uint64_t token) const {
  if (!initialized_) {
    return Error::Uninitialized;
  }
  auto result = token_map_.tryGetString(token);
  if (!result) {
    result = special_token_map_.tryGetString(token);
  }
  if (!result) {
    return Error::OutOfRange;
  }
  return std::string(*result);
}

Result<uint64_t> BPEModel::piece_to_id(const std::string& token) const {
  if (!initialized_) {
    return Error::Uninitialized;
  }
  auto result = token_map_.tryGetInteger(token);
  if (!result) {
    result = special_token_map_.tryGetInteger(token);
  }
  if (!result) {
    return Error::OutOfRange;
  }
  return *result;
}

bool BPEModel::is_special_token(uint64_t token) const {
  return special_token_map_.tryGetString(token).has_value();
}

Result<std::pair<std::vector<uint64_t>, uint64_t>>
BPEModel::encode_with_special_token(const std::string& text) const {
  std::vector<uint64_t> tokens;
  uint64_t last_piece_token_len = 0;
  size_t offset = 0;

  while (offset < text.size()) {
    auto [special, sub_input] =
        split_with_allowed_special_token(text, offset);

    if (!sub_input.empty()) {
      auto tokens_result = byte_pair_encode(sub_input);
      if (!tokens_result.ok()) {
        return tokens_result.error();
      }
      auto piece_tokens = std::move(*tokens_result);
      tokens.insert(tokens.end(), piece_tokens.begin(), piece_tokens.end());
      last_piece_token_len = piece_tokens.size();
    }
    offset += sub_input.size();

    if (special) {
      const auto result = special_token_map_.tryGetInteger(*special);
      if (!result) {
        TK_LOG(Error, "unknown special token: %s\n", special->c_str());
        return Error::EncodeFailure;
      }

      tokens.push_back(*result);
      last_piece_token_len = 0;
      offset += special->size();
    } else {
      break;
    }
  }

  return std::make_pair(tokens, last_piece_token_len);
}

std::pair<std::optional<std::string>, std::string>
BPEModel::split_with_allowed_special_token(
    const std::string& input,
    size_t offset) const {
  return detail::split_with_special_token(
      special_token_regex_.get(), special_token_map_, input, offset);
}

Result<std::vector<uint64_t>> BPEModel::byte_pair_encode(
    const std::string& piece) const {
  if (piece.size() == 1) {
    const auto result = token_map_.tryGetInteger(piece);
    if (result) {
      return std::vector<uint64_t>(1, *result);
    }
    if (byte_fallback_) {
      char hex[7];
      snprintf(
          hex, sizeof(hex), "<0x%02X>", static_cast<unsigned char>(piece[0]));
      const auto byte_result = token_map_.tryGetInteger(std::string(hex));
      if (byte_result) {
        return std::vector<uint64_t>(1, *byte_result);
      }
      if (unk_token_id_) {
        return std::vector<uint64_t>(1, *unk_token_id_);
      }
      return Error::EncodeFailure;
    } else {
      if (unk_token_id_) {
        return std::vector<uint64_t>(1, *unk_token_id_);
      }
      return Error::EncodeFailure;
    }
  }

  const detail::TokenMap& merge_ranks =
      merge_ranks_ ? *merge_ranks_ : token_map_;

  return byte_pair_merge(
      piece,
      merge_ranks,
      [this, &piece](uint64_t start, uint64_t stop) {
        std::string key = piece.substr(start, stop - start);
        const auto result = token_map_.tryGetInteger(key);
        if (result) {
          return *result;
        }
        if (byte_fallback_) {
          return std::numeric_limits<uint64_t>::max();
        }
        if (unk_token_id_) {
          return *unk_token_id_;
        }
        return std::numeric_limits<uint64_t>::max() - 1;
      });
}

std::vector<uint64_t> BPEModel::byte_pair_merge(
    const std::string& piece,
    const detail::TokenMap& ranks,
    std::function<uint64_t(uint64_t, uint64_t)> func) const {
  HFWord word;
  size_t i = 0;
  while (i < piece.size()) {
    size_t char_start = i;
    size_t char_len = 1;
    unsigned char byte = static_cast<unsigned char>(piece[i]);
    if ((byte & 0x80) == 0)
      char_len = 1;
    else if ((byte & 0xE0) == 0xC0)
      char_len = 2;
    else if ((byte & 0xF0) == 0xE0)
      char_len = 3;
    else if ((byte & 0xF8) == 0xF0)
      char_len = 4;
    if (char_start + char_len > piece.size())
      char_len = piece.size() - char_start;

    uint64_t token_id = func(char_start, char_start + char_len);
    if (token_id == std::numeric_limits<uint64_t>::max()) { // byte_fallback
      for (size_t j = 0; j < char_len; ++j) {
        char hex[7];
        snprintf(
            hex,
            sizeof(hex),
            "<0x%02X>",
            static_cast<unsigned char>(piece[char_start + j]));
        const auto byte_result = token_map_.tryGetInteger(std::string(hex));
        if (byte_result) {
          word.add(*byte_result, 1);
        } else if (unk_token_id_) {
          word.add(*unk_token_id_, 1);
        } else {
          return {}; // Unhandled byte fallback
        }
      }
    } else if (token_id == (std::numeric_limits<uint64_t>::max() - 1)) { // error
      return {};
    } else {
      word.add(token_id, char_len);
    }
    i += char_len;
  }

  if (merge_ranks_) {
    word.merge_all(*merge_ranks_, token_map_);
  }
  return word.tokens;
}

} // namespace tokenizers
