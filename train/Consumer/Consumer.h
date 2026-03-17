#pragma once

#include <array>
#include <immintrin.h>
#include <random>
#define HOURS 24
#define SOLARPRODUCES 10.f
#define SOLARHOURS 12

#define SOLAR 1
#define DAY 2
#define NIGHT 4
#define SMALL 8
#define MEDIUM 16
#define HIGH 32

#define SMALLMIN 0.4
#define SMALLMAX 0.8
#define MEDIUMMIN 0.8
#define MEDIUMMAX 1.5
#define HIGHMIN 1.25
#define HIGHMAX 2.5

struct LogNormalParams {
  const float mu;
  const float s;
  constexpr LogNormalParams(const float mu, const float s) : mu(mu), s(s) {}
};
struct Consumer {
  Consumer() {}
  Consumer(const std::array<float, HOURS> &in) : data(std::move(in)) {
    sum = calculate_sum();
  }
  Consumer(const Consumer &other) : data(std::move(other.data)) {
    sum = calculate_sum();
  }

  constexpr float mean() const { return sum / HOURS; }
  void setData(std::mt19937_64 &gen,
               const std::array<const LogNormalParams, HOURS> &distParams,
               const std::array<float, SOLARHOURS> &solarCurve);
  void updateData(std::mt19937_64 &gen,
                  const std::array<const LogNormalParams, HOURS> &distParams,
                  const std::array<float, SOLARHOURS> &solarCurve,
                  const std::array<float, HOURS> &price);
  __m256
  generateRandomData(std::mt19937_64 &gen,
                     const std::array<const LogNormalParams, HOURS> &distParams,
                     const unsigned distIdx, const unsigned idx,
                     const float multiplier);

  alignas(32) std::array<float, HOURS> data{0.0f};
  float sum;
  unsigned char flags = 0;

private:
  void adjustByPrice(const std::array<float, HOURS> &price);

  float calculate_sum() const;
};
