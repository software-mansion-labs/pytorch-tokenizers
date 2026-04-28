/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// @lint-ignore-every LICENSELINT

#include <pytorch/tokenizers/unigram_model.h>

#include <algorithm>
#include <cstdio>
#include <limits>

#include <pytorch/tokenizers/log.h>

namespace tokenizers {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

UnigramModel::UnigramModel(
    std::vector<std::string> pieces,
    std::vector<double> scores,
    detail::TokenMap special_token_map,
    std::unique_ptr<IRegex> special_token_regex,
    bool byte_fallback,
    std::optional<uint64_t> unk_token_id,
    std::optional<uint64_t> bos_token_id,
    std::optional<uint64_t> eos_token_id,
    std::unordered_set<std::string> rstrip_tokens,
    std::unordered_set<std::string> lstrip_tokens)
    : pieces_(std::move(pieces)),
      scores_(std::move(scores)),
      special_token_map_(std::move(special_token_map)),
      special_token_regex_(std::move(special_token_regex)),
      byte_fallback_(byte_fallback),
      unk_token_id_(unk_token_id),
      bos_token_id_(bos_token_id),
      eos_token_id_(eos_token_id),
      rstrip_tokens_(std::move(rstrip_tokens)),
      lstrip_tokens_(std::move(lstrip_tokens)) {
  // Build piece → id map.
  for (size_t i = 0; i < pieces_.size(); ++i) {
    piece_to_id_map_[pieces_[i]] = static_cast<uint64_t>(i);
  }

  // Find the minimum score across the vocabulary (used for UNK penalty).
  min_score_ = std::numeric_limits<double>::max();
  for (double s : scores_) {
    if (s < min_score_) {
      min_score_ = s;
    }
  }

  build_trie();

  // Vocab size = unique token strings across both vocab and special tokens.
  // Unigram vocabs may contain duplicates (e.g. <unk> at two positions)
  // and special tokens that overlap with the vocab.
  // piece_to_id_map_ already deduplicates vocab entries (last write wins).
  size_t unique_count = piece_to_id_map_.size();
  for (size_t i = 0; i < special_token_map_.size(); ++i) {
    const auto& [token, _] = special_token_map_.getElement(i);
    if (piece_to_id_map_.find(std::string(token)) == piece_to_id_map_.end()) {
      ++unique_count;
    }
  }
  vocab_size_ = static_cast<int32_t>(unique_count);
  initialized_ = true;
}

// ---------------------------------------------------------------------------
// Trie construction
// ---------------------------------------------------------------------------

void UnigramModel::build_trie() {
  // Start with a root node.
  trie_.clear();
  trie_.push_back(TrieNode{}); // index 0 = root

  for (size_t id = 0; id < pieces_.size(); ++id) {
    const std::string& piece = pieces_[id];
    size_t node_idx = 0; // start at root

    for (uint8_t byte : piece) {
      auto it = trie_[node_idx].children.find(byte);
      if (it == trie_[node_idx].children.end()) {
        size_t new_idx = trie_.size();
        trie_.push_back(TrieNode{});
        trie_[node_idx].children[byte] = new_idx;
        node_idx = new_idx;
      } else {
        node_idx = it->second;
      }
    }
    trie_[node_idx].token_id = static_cast<int64_t>(id);
  }
}

// ---------------------------------------------------------------------------
// Tokenize — optimized Viterbi (no explicit lattice)
// ---------------------------------------------------------------------------

namespace {

/// Return the byte-length of a UTF-8 character given its lead byte.
inline size_t utf8_char_len(uint8_t byte) {
  if ((byte & 0x80) == 0)
    return 1;
  if ((byte & 0xE0) == 0xC0)
    return 2;
  if ((byte & 0xF0) == 0xE0)
    return 3;
  if ((byte & 0xF8) == 0xF0)
    return 4;
  return 1; // invalid lead byte — treat as single byte
}

struct BestPathNode {
  double score = 0.0;
  uint64_t token_id = 0;
  size_t starts_at = 0;
  bool reachable = false;
};

} // namespace

Result<std::vector<uint64_t>> UnigramModel::tokenize(
    const std::string& piece) const {
  if (!initialized_) {
    return Error::Uninitialized;
  }
  if (piece.empty()) {
    return std::vector<uint64_t>{};
  }

  const size_t size = piece.size();
  const double unk_score = min_score_ - 10.0;

  // best[i] = best segmentation of piece[0..i).
  std::vector<BestPathNode> best(size + 1);
  best[0].reachable = true;
  best[0].score = 0.0;

  for (size_t pos = 0; pos < size; ++pos) {
    if (!best[pos].reachable) {
      continue;
    }

    const double score_so_far = best[pos].score;
    const size_t mblen = std::min(
        utf8_char_len(static_cast<uint8_t>(piece[pos])), size - pos);
    bool has_single_char_token = false;

    // Walk the trie to find all vocabulary tokens starting at `pos`.
    size_t node_idx = 0; // root
    for (size_t i = pos; i < size; ++i) {
      uint8_t b = static_cast<uint8_t>(piece[i]);
      auto it = trie_[node_idx].children.find(b);
      if (it == trie_[node_idx].children.end()) {
        break;
      }
      node_idx = it->second;

      if (trie_[node_idx].token_id >= 0) {
        auto tid = static_cast<uint64_t>(trie_[node_idx].token_id);
        size_t end_pos = i + 1;
        double candidate = score_so_far + scores_[tid];

        if (end_pos - pos == mblen) {
          has_single_char_token = true;
        }

        if (!best[end_pos].reachable || candidate > best[end_pos].score) {
          best[end_pos].score = candidate;
          best[end_pos].token_id = tid;
          best[end_pos].starts_at = pos;
          best[end_pos].reachable = true;
        }
      }
    }

    // UNK fallback: if no vocabulary token covers the single character at
    // this position, insert an UNK token spanning one UTF-8 character.
    if (!has_single_char_token && unk_token_id_) {
      size_t end_pos = pos + mblen;
      double candidate = score_so_far + unk_score;
      if (!best[end_pos].reachable || candidate > best[end_pos].score) {
        best[end_pos].score = candidate;
        best[end_pos].token_id = *unk_token_id_;
        best[end_pos].starts_at = pos;
        best[end_pos].reachable = true;
      }
    }
  }

  if (!best[size].reachable) {
    TK_LOG(Error, "Viterbi could not reach end of input");
    return Error::EncodeFailure;
  }

  // --- Backtrack to recover token IDs and their byte spans. ---
  struct TokenSpan {
    uint64_t id;
    size_t start;
    size_t end;
  };
  std::vector<TokenSpan> spans;
  {
    size_t pos = size;
    while (pos > 0) {
      spans.push_back({best[pos].token_id, best[pos].starts_at, pos});
      pos = best[pos].starts_at;
    }
    std::reverse(spans.begin(), spans.end());
  }

  // --- Fuse consecutive UNK tokens (standard SentencePiece behavior). ---
  // Also apply byte fallback when enabled.
  std::vector<uint64_t> result;
  result.reserve(spans.size());

  for (size_t i = 0; i < spans.size();) {
    if (unk_token_id_ && spans[i].id == *unk_token_id_) {
      // Collect consecutive UNK spans.
      size_t fused_start = spans[i].start;
      size_t fused_end = spans[i].end;
      while (i + 1 < spans.size() && spans[i + 1].id == *unk_token_id_) {
        ++i;
        fused_end = spans[i].end;
      }
      ++i;

      if (byte_fallback_) {
        // Replace the fused UNK with individual byte tokens <0xHH>.
        bool all_found = true;
        std::vector<uint64_t> byte_tokens;
        for (size_t j = fused_start; j < fused_end; ++j) {
          char hex[7];
          std::snprintf(
              hex,
              sizeof(hex),
              "<0x%02X>",
              static_cast<unsigned char>(piece[j]));
          auto it = piece_to_id_map_.find(std::string(hex));
          if (it != piece_to_id_map_.end()) {
            byte_tokens.push_back(it->second);
          } else {
            all_found = false;
            break;
          }
        }
        if (all_found) {
          result.insert(result.end(), byte_tokens.begin(), byte_tokens.end());
        } else {
          // Byte tokens not in vocab — emit single UNK.
          result.push_back(*unk_token_id_);
        }
      } else {
        // No byte fallback — emit single fused UNK.
        result.push_back(*unk_token_id_);
      }
    } else {
      result.push_back(spans[i].id);
      ++i;
    }
  }

  return result;
}

// ---------------------------------------------------------------------------
// Vocabulary queries
// ---------------------------------------------------------------------------

Result<std::string> UnigramModel::id_to_piece(uint64_t token) const {
  if (!initialized_) {
    return Error::Uninitialized;
  }
  if (token < pieces_.size()) {
    return pieces_[token];
  }
  // Check special tokens.
  auto result = special_token_map_.tryGetString(token);
  if (result) {
    return std::string(*result);
  }
  return Error::OutOfRange;
}

Result<uint64_t> UnigramModel::piece_to_id(const std::string& piece) const {
  if (!initialized_) {
    return Error::Uninitialized;
  }
  auto it = piece_to_id_map_.find(piece);
  if (it != piece_to_id_map_.end()) {
    return it->second;
  }
  // Check special tokens.
  auto result = special_token_map_.tryGetInteger(piece);
  if (result) {
    return *result;
  }
  return Error::OutOfRange;
}

bool UnigramModel::is_special_token(uint64_t token) const {
  // Check by ID first (fast path).
  if (special_token_map_.tryGetString(token).has_value()) {
    return true;
  }
  // For Unigram models, special tokens may exist in both the vocab and
  // added_tokens with different IDs (e.g. <unk> at vocab index 3 and
  // added_token id 4). Check if this token's piece string is a special token.
  if (token < pieces_.size()) {
    if (special_token_map_.tryGetInteger(pieces_[token]).has_value()) {
      return true;
    }
  }
  return false;
}

// ---------------------------------------------------------------------------
// Special-token splitting (shared pattern with BPE / WordPiece)
// ---------------------------------------------------------------------------

std::pair<std::optional<std::string>, std::string>
UnigramModel::split_with_allowed_special_token(
    const std::string& input,
    size_t offset) const {
  return detail::split_with_special_token(
      special_token_regex_.get(), special_token_map_, input, offset);
}

} // namespace tokenizers
