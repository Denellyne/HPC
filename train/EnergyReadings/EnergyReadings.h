#pragma once
#include <array>
#include <immintrin.h>
#include <iostream>
#include <random>
#include <vector>
constexpr unsigned short ARRSIZE = 24;

class EnergyReadings {
private:
  struct EnergyReading {
    EnergyReading() {}
    EnergyReading(const std::array<float, ARRSIZE> &in) : data(std::move(in)) {}
    EnergyReading(const EnergyReading &other) : data(std::move(other.data)) {}
    alignas(32) std::array<float, ARRSIZE> data{0.0f};
    float mean() const {
      float result = 0;
      __m256 v1 = _mm256_load_ps(&data[0]);
      v1 = _mm256_add_ps(v1, _mm256_load_ps(&data[8]));
      v1 = _mm256_add_ps(v1, _mm256_load_ps(&data[16]));
      __m128 low = _mm256_castps256_ps128(v1);
      __m128 high = _mm256_extractf128_ps(v1, 1);

      low = _mm_add_ps(low, high);
      __m128 odd = _mm_movehdup_ps(low);
      __m128 sums = _mm_add_ps(low, odd);
      odd = _mm_movehl_ps(odd, sums);
      sums = _mm_add_ss(sums, odd);

      return _mm_cvtss_f32(sums) / ARRSIZE;
    }
  };

public:
  EnergyReadings() = delete;
  EnergyReadings(const unsigned long long size);
  std::vector<EnergyReading> data;
  std::array<float, ARRSIZE> calculate_aggregate();
  long double mean;
  long double variance;

private:
  double calculate_mean();
  double calculate_variance();
};
