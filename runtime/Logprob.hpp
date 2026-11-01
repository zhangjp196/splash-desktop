#pragma once

#include <array>
#include <cstdint>

namespace splash {

// The most top tokens a logprobs request may ask for (OpenAI's
// top_logprobs ceiling).
inline constexpr uint32_t kMaximumTopLogprobs = 20;

// One token of a logprobs distribution: the token and its natural-log
// probability, after the request's temperature and penalties and over the
// whole vocabulary.
struct LogprobCandidate final {
  uint32_t token = 0;
  float logprob = 0.0F;

  bool operator==(const LogprobCandidate &) const = default;
};

// One generated token's log probability, and the tokens above it the request
// asked for, in logprob order (ties by token id). `top` is filled up to
// `topCount`; the chosen token is not repeated there unless it is among them.
struct TokenLogprob final {
  uint32_t token = 0;
  float logprob = 0.0F;
  uint32_t topCount = 0;
  std::array<LogprobCandidate, kMaximumTopLogprobs> top{};

  bool operator==(const TokenLogprob &) const = default;
};

// The most logit_bias entries a request may name.
inline constexpr uint32_t kMaximumLogitBias = 256;

// One raw-logit adjustment (OpenAI's logit_bias), applied before the
// request's penalties and sampling.
struct LogitBias final {
  uint32_t token = 0;
  float value = 0.0F;

  bool operator==(const LogitBias &) const = default;
};

} // namespace splash
