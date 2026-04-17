/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <set>
#include <string>
#include <vector>

#include <tokenizers_c.h>
#include <tokenizers_cpp.h>

using json = nlohmann::json;

std::vector<std::string> get_special_tokens(const std::string& tokenizer_json_path) {
  std::ifstream f(tokenizer_json_path);
  if (!f.is_open()) return {};
  json j;
  try {
    f >> j;
  } catch (...) {
    return {};
  }
  std::vector<std::string> tokens;
  if (j.contains("added_tokens")) {
    for (const auto& entry : j["added_tokens"]) {
      if (entry.contains("content") && entry.value("special", false)) {
        tokens.push_back(entry["content"].get<std::string>());
      }
    }
  }
  return tokens;
}

std::vector<std::string> get_special_token_inputs(
    const std::vector<std::string>& special_tokens) {
  std::vector<std::string> inputs;
  if (special_tokens.empty()) return inputs;

  for (const auto& tok : special_tokens) {
    inputs.push_back(tok);
    inputs.push_back(tok + "Hello, world!");
    inputs.push_back("Hello, world!" + tok);
    inputs.push_back("Before " + tok + " After");
  }

  if (special_tokens.size() >= 2) {
    inputs.push_back(special_tokens[0] + special_tokens[1]);
    inputs.push_back(special_tokens[0] + " text " + special_tokens[1]);
  }

  {
    std::string all;
    for (const auto& tok : special_tokens) all += tok;
    inputs.push_back(all);
  }

  // Realistic prompt wrapping (bos + content + eos) for common token names
  {
    auto has = [&](const std::string& s) {
      return std::find(special_tokens.begin(), special_tokens.end(), s) !=
             special_tokens.end();
    };
    for (auto bos : {"<bos>", "<s>", "<|startoftext|>"}) {
      for (auto eos : {"<eos>", "</s>", "<|endoftext|>"}) {
        if (has(bos) && has(eos)) {
          inputs.push_back(
              std::string(bos) + "What is the capital of France?" + std::string(eos));
          inputs.push_back(
              std::string(bos) + "Explain quantum entanglement." + std::string(eos));
        }
      }
    }
  }

  return inputs;
}

std::vector<std::string> get_test_inputs() {
  return {
      "Hello world!",
      "The quick brown fox jumps over the lazy dog.",
      "How are you doing today?",

      "",
      " ",
      "   ",
      "\t",
      "\n",
      "\t\n  \n\t",
      "  Leading and trailing spaces  ",

      "!@#$%^&*()_+-={}[]|\\:;\"'<>,.?/~`",
      "Test with punctuation!!!",
      "Question? Answer!",

      "0123456789",
      "123.456",
      "1,000,000",
      "3.14159",
      "0x1A2B3C",
      "1e-10",
      "Test with numbers 123 456",

      "abc123def456",
      "test_123",
      "CamelCase123Test",
      "UPPER_CASE_123",
      "lower_case_456",

      "Hello 😀 World 🎉 Test ✨",
      "café naïve résumé",
      "Emoji: 👍 ❤️ 🌟",

      "Привет мир",
      "Γειά σου κόσμε",
      "مرحبا بالعالم",
      "你好世界",
      "こんにちは世界",

      // Polish (for Bielik)
      "ąćęłńóśźż ĄĆĘŁŃÓŚŹŻ",
      "Dzień dobry, jak się masz?",
      "Żółć gęsi jaźń",
      "Pchnąć w tę łódź jeża lub ośm skrzyń fig",
      "Zażółć gęślą jaźń",
      "Chrząszcz brzmi w trzcinie w Szczebrzeszynie",
      "komputer oprogramowanie przetwarzanie",
      "sztuczna inteligencja uczenie maszynowe",

      "def hello_world():\n    print(\"Hello, World!\")",
      "function test() { return 42; }",
      "#include <iostream>",
      "import numpy as np",

      "if (x > 0) { return true; } else { return false; }",
      "for (int i = 0; i < 10; i++) { printf(\"%d\\n\", i); }",
      "while (ptr != nullptr && *ptr != '\\0') { ++ptr; }",
      "switch (value) { case 0: break; default: return -1; }",

      "int* ptr = &value;",
      "std::vector<int>& ref = container;",
      "result = (a && b) || (c && d);",
      "x = y << 2 | z >> 4;",
      "obj->method();",
      "*ptr++ = value & 0xFF;",

      "array[index] = {1, 2, 3, 4, 5};",
      "map[\"key\"] = {\"nested\": [1, 2, 3]};",
      "((x + y) * (z - w)) / ((a * b) + (c * d))",

      "x = 0; y = 1; z = x + y; return z * 2;",
      "for (;;) { if (cond) break; else continue; }",

      "result = func(arg1, arg2) && validate();",
      "value = &obj.field;",
      "mask = 0xFF & ~(1 << bit);",

      "std::map<std::string, std::vector<int>> data;",
      "template<typename T> class Container { };",
      "List<Map<String, Integer>> nested;",

      "[&](int x) -> int { return x * 2; }",
      "lambda = [=]() { return a + b; };",

      "class Foo { public: void bar() { std::cout << \"test\"; } };",
      "try { throw std::runtime_error(\"error\"); } catch (...) { }",

      "int main() {\n"
      "    int* ptr = new int[10];\n"
      "    for (int i = 0; i < 10; ++i) {\n"
      "        ptr[i] = i * i;\n"
      "    }\n"
      "    delete[] ptr;\n"
      "    return 0;\n"
      "}",

      "aaaaaaaaaa bbbbbbbbbb cccccccccc",
      "!!!!!!!!!!!!!!",
      "          spaces          ",

      "new\nline\tand\ttabs",
      "Line 1\nLine 2\nLine 3",
      "Tab\tseparated\tvalues",

      "This sentence has 123 numbers, punctuation!, and CAPS.",
      "Mixed Ενγλισχη with ελληνικά characters.",
      "Code in text: x = 42; y = x + 1;",

      "This is sentence number 0. This is sentence number 1. This is sentence number 2. "
      "This is sentence number 3. This is sentence number 4. This is sentence number 5. "
      "This is sentence number 6. This is sentence number 7. This is sentence number 8. "
      "This is sentence number 9. This is sentence number 10.",

      "aaaa",
      "abababab",
      "the the the",
      "tion tion tion",
      "pre-processing pre-tokenization",
      "can't won't shouldn't",
      "COVID-19 SARS-CoV-2",
      "http://example.com/path?query=value",
      "user@example.com",
      "1234567890",
      "____",

      "a",
      "Z",
      "0",
      "!",
      " ",
      "\n",

      "\x01\x02\x03",  // SOH, STX, ETX
      "\x7f",          // DEL
      "\r\n",
      "\r",

      "zero\xE2\x80\x8Bwidth",  // U+200B zero-width space
      "zero\xE2\x80\x8Cwidth",  // U+200C zero-width non-joiner
      "zero\xE2\x80\x8Dwidth",  // U+200D zero-width joiner
      "\xEF\xBB\xBFwith BOM",   // U+FEFF BOM
      "left\xE2\x80\x8Fright",  // U+200F right-to-left mark
      "left\xE2\x80\x8Eright",  // U+200E left-to-right mark

      "\xC3\xA9",                                          // é (U+00E9, 2-byte)
      "\xE4\xB8\xAD",                                      // 中 (U+4E2D, 3-byte)
      "\xF0\x9F\x98\x80",                                  // 😀 (U+1F600, 4-byte)
      "\xF0\x9F\x87\xB5\xF0\x9F\x87\xB1",                 // 🇵🇱 flag emoji
      "\xF0\x9F\x91\xA8\xE2\x80\x8D\xF0\x9F\x92\xBB",    // 👨‍💻 ZWJ sequence

      "\xD8\xA7\xD9\x84\xD8\xB3\xD9\x84\xD8\xA7\xD9\x85 hello \xD8\xB9\xD8\xA7\xD9\x84\xD9\x85",  // Arabic-English-Arabic
      "hello \xD7\xA9\xD7\x9C\xD7\x95\xD7\x9D world",  // English-Hebrew-English

      "the",
      "Hello",
      "  ",  // Two spaces (common BPE token)

      std::string(500, 'a'),  // deep merge recursion
      std::string(100, ' '),
      "a b c d e f g h i j k l m n o p q r s t u v w x y z",

      "The Transformer architecture, introduced in the seminal paper \"Attention Is All "
      "You Need\" by Vaswani et al. (2017), revolutionized natural language processing. "
      "It replaced recurrent neural networks (RNNs) and long short-term memory (LSTM) "
      "networks with self-attention mechanisms, enabling parallel computation and "
      "capturing long-range dependencies more effectively. Key innovations include "
      "multi-head attention, positional encoding, and the encoder-decoder structure "
      "that has become the foundation for models like BERT, GPT, T5, and LLaMA.",

      // These look like special tokens but shouldn't be treated as such
      "<s>this is not a real BOS token</s>",
      "[CLS] fake special tokens [SEP]",
      "<|endoftext|> embedded special token markers",
  };
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <model_name>" << std::endl;
    std::cerr << "Example: " << argv[0] << " llama" << std::endl;
    return 1;
  }

  std::string model_name = argv[1];
  std::string resources_path = std::getenv("RESOURCES_PATH")
      ? std::getenv("RESOURCES_PATH")
      : "./test/resources";

  std::string tokenizer_dir = resources_path + "/" + model_name + "_tokenizer_dir/";
  std::string tokenizer_json_path = tokenizer_dir + "tokenizer.json";
  std::string output_path = resources_path + "/" + model_name + "_ground_truth.json";

  std::cout << "Generating ground truth for: " << model_name << std::endl;
  std::cout << "Reading from: " << tokenizer_json_path << std::endl;

  std::ifstream tokenizer_file(tokenizer_json_path);
  if (!tokenizer_file.is_open()) {
    std::cerr << "Error: Could not open " << tokenizer_json_path << std::endl;
    return 1;
  }

  std::string tokenizer_json_str(
      (std::istreambuf_iterator<char>(tokenizer_file)),
      std::istreambuf_iterator<char>());
  tokenizer_file.close();

  std::unique_ptr<tokenizers::Tokenizer> tokenizer;
  TokenizerHandle handle = nullptr;
  try {
    tokenizer = tokenizers::Tokenizer::FromBlobJSON(tokenizer_json_str);
    // C handle needed for add_special_tokens/skip_special_tokens encoding
    handle = tokenizers_new_from_str(
        tokenizer_json_str.data(),
        tokenizer_json_str.length());
  } catch (const std::exception& e) {
    std::cerr << "Error: Failed to load tokenizer: " << e.what() << std::endl;
    return 1;
  }

  size_t vocab_size = tokenizer->GetVocabSize();
  std::cout << "Loaded tokenizer. Vocab size: " << vocab_size << std::endl;

  json ground_truth = json::object();
  ground_truth["model_name"] = model_name;
  ground_truth["vocab_size"] = vocab_size;
  ground_truth["test_cases"] = json::array();

  auto test_inputs = get_test_inputs();

  {
    auto special_tokens = get_special_tokens(tokenizer_json_path);
    std::cout << "Found " << special_tokens.size()
              << " special tokens for this model" << std::endl;
    auto special_inputs = get_special_token_inputs(special_tokens);
    std::cout << "Adding " << special_inputs.size()
              << " special-token test cases" << std::endl;
    test_inputs.insert(
        test_inputs.end(), special_inputs.begin(), special_inputs.end());
  }

  for (const auto& input : test_inputs) {
    json test_case = json::object();
    test_case["input"] = input;

    auto tokens = tokenizer->Encode(input);
    test_case["tokens"] = tokens;

    TokenizerEncodeResult result_with_special;
    tokenizers_encode(handle,
                     input.data(),
                     input.length(),
                     1,  // add_special_tokens
                     &result_with_special);
    std::vector<int32_t> tokens_with_special(
        result_with_special.token_ids,
        result_with_special.token_ids + result_with_special.len);
    tokenizers_free_encode_results(&result_with_special, 1);
    test_case["tokens_with_special"] = tokens_with_special;

    std::string decoded = tokenizer->Decode(tokens);
    test_case["decoded"] = decoded;

    tokenizers_decode(handle,
                     reinterpret_cast<const uint32_t*>(tokens_with_special.data()),
                     tokens_with_special.size(),
                     1);  // skip_special_tokens
    const char* decoded_data;
    size_t decoded_len;
    tokenizers_get_decode_str(handle, &decoded_data, &decoded_len);
    std::string decoded_skip_special(decoded_data, decoded_len);
    test_case["decoded_skip_special"] = decoded_skip_special;

    json token_strings = json::array();
    for (auto token_id : tokens) {
      token_strings.push_back(tokenizer->IdToToken(token_id));
    }
    test_case["token_strings"] = token_strings;

    json token_strings_with_special = json::array();
    for (auto token_id : tokens_with_special) {
      token_strings_with_special.push_back(tokenizer->IdToToken(token_id));
    }
    test_case["token_strings_with_special"] = token_strings_with_special;

    ground_truth["test_cases"].push_back(test_case);
  }

  // Build piece_to_id map from all token IDs seen in test cases
  {
    std::set<int32_t> seen_ids;
    for (const auto& tc : ground_truth["test_cases"]) {
      for (auto id : tc["tokens"]) {
        seen_ids.insert(id.get<int32_t>());
      }
    }

    json piece_to_id_map = json::object();
    for (auto id : seen_ids) {
      std::string piece = tokenizer->IdToToken(id);
      if (!piece.empty()) {
        piece_to_id_map[piece] = id;
      }
    }
    ground_truth["piece_to_id"] = piece_to_id_map;
  }

  std::ofstream output_file(output_path);
  if (!output_file.is_open()) {
    std::cerr << "Error: Could not open output file " << output_path << std::endl;
    return 1;
  }

  output_file << ground_truth.dump(2) << std::endl;
  output_file.close();

  std::cout << "Saved ground truth to: " << output_path << std::endl;
  std::cout << "Processed " << test_inputs.size() << " test inputs" << std::endl;

  if (handle != nullptr) {
    tokenizers_free(handle);
  }

  return 0;
}
