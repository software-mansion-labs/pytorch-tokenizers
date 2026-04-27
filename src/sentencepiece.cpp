/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <pytorch/tokenizers/sentencepiece.h>

#include <cinttypes>
#include <string>

#include <pytorch/tokenizers/log.h>
#include "sentencepiece_processor.h"

namespace tokenizers {

// ▁ (U+2581 LOWER ONE EIGHTH BLOCK) — the SentencePiece space marker.
static const std::string kSpaceSymbol = "\xe2\x96\x81";

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// Replace every occurrence of `from` with `to` in `str`, in-place.
static void replace_all(
    std::string& str,
    const std::string& from,
    const std::string& to) {
  size_t pos = 0;
  while ((pos = str.find(from, pos)) != std::string::npos) {
    str.replace(pos, from.size(), to);
    pos += to.size();
  }
}

/// Decode a single byte-token piece like "<0x0A>" → "\n".
/// Returns true (and overwrites `piece`) when the piece matched the pattern.
static bool try_decode_byte_token(std::string& piece) {
  // Byte tokens are exactly 6 chars: <0xHH>
  if (piece.size() != 6 || piece[0] != '<' || piece[1] != '0' ||
      piece[2] != 'x' || piece[5] != '>') {
    return false;
  }
  unsigned int byte_val = 0;
  if (std::sscanf(piece.c_str(), "<0x%02X>", &byte_val) == 1) {
    piece = std::string(1, static_cast<char>(byte_val));
    return true;
  }
  return false;
}

// ---------------------------------------------------------------------------
// lifetime
// ---------------------------------------------------------------------------

SPTokenizer::SPTokenizer()
    : Tokenizer(),
      processor_(std::make_unique<sentencepiece::SentencePieceProcessor>()) {}

SPTokenizer::~SPTokenizer() = default;

// ---------------------------------------------------------------------------
// load
// ---------------------------------------------------------------------------

Error SPTokenizer::load(const std::string& tokenizer_path) {
  if (initialized_) {
    TK_LOG(Info, "Tokenizer already initialized");
    return Error::Ok;
  }
  const auto status = processor_->Load(tokenizer_path);
  if (!status.ok()) {
    TK_LOG(
        Error,
        "Failed to load %s: %s",
        tokenizer_path.c_str(),
        status.error_message());
    return Error::LoadFailure;
  }
  vocab_size_ = processor_->GetPieceSize();
  bos_tok_ = processor_->bos_id();
  eos_tok_ = processor_->eos_id();
  unk_tok_ = processor_->unk_id();
  initialized_ = true;
  return Error::Ok;
}

// ---------------------------------------------------------------------------
// vocabulary queries
// ---------------------------------------------------------------------------

Result<std::string> SPTokenizer::id_to_piece(uint64_t token) const {
  if (!initialized_) {
    return Error::Uninitialized;
  }
  if (token >= static_cast<uint64_t>(vocab_size_)) {
    return Error::OutOfRange;
  }
  return processor_->IdToPiece(static_cast<int>(token));
}

Result<uint64_t> SPTokenizer::piece_to_id(const std::string& text) const {
  if (!initialized_) {
    return Error::Uninitialized;
  }
  int32_t id = processor_->PieceToId(text);
  if (id == processor_->unk_id()) {
    // PieceToId returns unk_id for unknown pieces, but the unk piece itself
    // (e.g. "<unk>") is a legitimate lookup — disambiguate.
    if (text == processor_->IdToPiece(id)) {
      return static_cast<uint64_t>(id);
    }
    return Error::OutOfRange;
  }
  return static_cast<uint64_t>(id);
}

// ---------------------------------------------------------------------------
// encode
// ---------------------------------------------------------------------------

Result<std::vector<uint64_t>>
SPTokenizer::encode(const std::string& text, int8_t bos, int8_t eos) const {
  if (!initialized_) {
    return Error::Uninitialized;
  }

  std::vector<int> sp_ids;
  auto status = processor_->Encode(text, &sp_ids);
  if (!status.ok()) {
    TK_LOG(Error, "Failed to encode input");
    return Error::EncodeFailure;
  }

  std::vector<uint64_t> tokens;
  tokens.reserve(static_cast<size_t>(bos) + sp_ids.size() + eos);

  for (int8_t i = 0; i < bos; ++i) {
    tokens.push_back(bos_tok_);
  }
  for (int id : sp_ids) {
    tokens.push_back(static_cast<uint64_t>(id));
  }
  for (int8_t i = 0; i < eos; ++i) {
    tokens.push_back(eos_tok_);
  }
  return tokens;
}

// ---------------------------------------------------------------------------
// decode — streaming (single token)
// ---------------------------------------------------------------------------

Result<std::string> SPTokenizer::decode(
    uint64_t prev_token,
    uint64_t token,
    bool skip_special_tokens) const {
  if (!initialized_) {
    return Error::Uninitialized;
  }

  int id = static_cast<int>(token);

  // Control tokens (BOS, EOS, PAD, …) have no text representation.
  if (processor_->IsControl(id)) {
    if (skip_special_tokens) {
      return std::string("");
    }
    // Even when not skipping, control tokens are silent in decoded output.
    return std::string("");
  }

  std::string piece = processor_->IdToPiece(id);

  // Byte tokens: <0x0A> → '\n', etc.
  if (processor_->IsByte(id)) {
    try_decode_byte_token(piece);
    return piece;
  }

  // Replace ▁ with space.
  replace_all(piece, kSpaceSymbol, " ");

  // After BOS, SentencePiece convention is to strip the leading space.
  if (prev_token == bos_tok_ && !piece.empty() && piece[0] == ' ') {
    piece.erase(0, 1);
  }

  return piece;
}

// ---------------------------------------------------------------------------
// decode — batch
// ---------------------------------------------------------------------------

Result<std::string> SPTokenizer::decode(
    const std::vector<uint64_t>& tokens,
    bool skip_special_tokens) const {
  if (!initialized_) {
    return Error::Uninitialized;
  }

  std::vector<int> ids;
  ids.reserve(tokens.size());
  for (uint64_t t : tokens) {
    if (skip_special_tokens &&
        processor_->IsControl(static_cast<int>(t))) {
      continue;
    }
    ids.push_back(static_cast<int>(t));
  }

  std::string result;
  auto status = processor_->Decode(ids, &result);
  if (!status.ok()) {
    TK_LOG(Error, "Failed to decode token sequence");
    return Error::DecodeFailure;
  }
  return result;
}

} // namespace tokenizers
