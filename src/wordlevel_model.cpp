/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// @lint-ignore-every LICENSELINT

#include <pytorch/tokenizers/wordlevel_model.h>

#include <pytorch/tokenizers/log.h>

namespace tokenizers {

WordLevelModel::WordLevelModel(
    detail::TokenMap token_map,
    detail::TokenMap special_token_map,
    std::unique_ptr<IRegex> special_token_regex,
    std::optional<uint64_t> unk_token_id,
    std::optional<uint64_t> bos_token_id,
    std::optional<uint64_t> eos_token_id,
    std::unordered_set<std::string> rstrip_tokens,
    std::unordered_set<std::string> lstrip_tokens)
    : token_map_(std::move(token_map)),
      special_token_map_(std::move(special_token_map)),
      special_token_regex_(std::move(special_token_regex)),
      unk_token_id_(unk_token_id),
      bos_token_id_(bos_token_id),
      eos_token_id_(eos_token_id),
      rstrip_tokens_(std::move(rstrip_tokens)),
      lstrip_tokens_(std::move(lstrip_tokens)) {
  vocab_size_ =
      static_cast<int32_t>(token_map_.size() + special_token_map_.size());
  initialized_ = true;
}

Result<std::vector<uint64_t>> WordLevelModel::tokenize(
    const std::string& piece) const {
  if (!initialized_) {
    return Error::Uninitialized;
  }

  // Direct vocabulary lookup — the whole piece is one token.
  auto result = token_map_.tryGetInteger(piece);
  if (result) {
    return std::vector<uint64_t>{*result};
  }

  // Check special tokens.
  auto special_result = special_token_map_.tryGetInteger(piece);
  if (special_result) {
    return std::vector<uint64_t>{*special_result};
  }

  // Unknown token.
  if (unk_token_id_) {
    return std::vector<uint64_t>{*unk_token_id_};
  }

  TK_LOG(Error, "Unknown word and no unk_token: '%s'", piece.c_str());
  return Error::EncodeFailure;
}

Result<std::string> WordLevelModel::id_to_piece(uint64_t token) const {
  if (!initialized_) {
    return Error::Uninitialized;
  }
  auto result = token_map_.tryGetString(token);
  if (result) {
    return std::string(*result);
  }
  auto special = special_token_map_.tryGetString(token);
  if (special) {
    return std::string(*special);
  }
  return Error::OutOfRange;
}

Result<uint64_t> WordLevelModel::piece_to_id(const std::string& piece) const {
  if (!initialized_) {
    return Error::Uninitialized;
  }
  auto result = token_map_.tryGetInteger(piece);
  if (result) {
    return *result;
  }
  auto special = special_token_map_.tryGetInteger(piece);
  if (special) {
    return *special;
  }
  return Error::OutOfRange;
}

bool WordLevelModel::is_special_token(uint64_t token) const {
  return special_token_map_.tryGetString(token).has_value();
}

std::pair<std::optional<std::string>, std::string>
WordLevelModel::split_with_allowed_special_token(
    const std::string& input,
    size_t offset) const {
  if (!special_token_regex_) {
    return {std::nullopt, input.substr(offset)};
  }

  auto matches = special_token_regex_->find_all(input.substr(offset));
  for (const auto& m : matches) {
    std::string matched_text = input.substr(offset + m.start, m.end - m.start);
    if (special_token_map_.tryGetInteger(matched_text).has_value()) {
      return {matched_text, input.substr(offset, m.start)};
    }
  }

  return {std::nullopt, input.substr(offset)};
}

} // namespace tokenizers
