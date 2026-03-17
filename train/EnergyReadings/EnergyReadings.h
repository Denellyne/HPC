#pragma once
#include <array>
#include <cmath>
#include <immintrin.h>
#include <iostream>
#include <random>
#include <vector>
constexpr unsigned short HOURS = 24;

enum class ConsumerClass {
  Day = 0,
  Night = 1,
};
enum class HouseholdClass {
  Small,
  Medium,
  High,
};

#define SOLARPRODUCES 10.f
#define SOLARHOURS 12

class EnergyReadings {
private:
  struct Consumer {
    Consumer() {}
    Consumer(const std::array<float, HOURS> &in, ConsumerClass type,
             HouseholdClass house)
        : data(std::move(in)), type(type), house(house) {
      sum = calculate_sum();
    }
    Consumer(const Consumer &other)
        : data(std::move(other.data)), type(other.type), house(other.house) {
      sum = calculate_sum();
    }
    alignas(32) std::array<float, HOURS> data{0.0f};
    ConsumerClass type{ConsumerClass::Day};
    HouseholdClass house{HouseholdClass::Medium};
    bool solar;
    long double sum;
    float mean() const { return sum / HOURS; }
    void setData(std::mt19937 &gen,
                 const std::array<float, SOLARHOURS> &solarCurve) {
      int idx = 0;
      float multiplier = 0.f;
      switch (this->type) {
      case ConsumerClass::Day:
        idx = 0;
        break;
      case ConsumerClass::Night:
        idx = 13;
        break;
      }
      switch (this->house) {

      case HouseholdClass::Small:
        multiplier = 0.5f;
        break;
      case HouseholdClass::Medium:
        multiplier = 1.f;
        break;
      case HouseholdClass::High:
        multiplier = 2.5f;
        break;
      }

      for (int i = 0; i < HOURS; i++) {
        const auto &[mu, s] = distParams[idx % HOURS];

        std::lognormal_distribution<long double> dist(mu, s);

        data[i] = dist(gen) * multiplier;
        idx++;
      }
      constexpr float scalingFactor = -SOLARPRODUCES / 5.6f;
      if (solar) {
        __m256 vec256 = _mm256_set1_ps(scalingFactor);
        __m128 vec128 = _mm_set1_ps(scalingFactor);
        vec256 = _mm256_mul_ps(vec256, _mm256_load_ps(&solarCurve[0]));
        vec128 = _mm_mul_ps(vec128, _mm_load_ps(&solarCurve[8]));
        const __m256 data256 = _mm256_add_ps(vec256, _mm256_load_ps(&data[6]));
        const __m128 data128 = _mm_add_ps(vec128, _mm_load_ps(&data[14]));
        _mm256_storeu_ps(&data[6], data256);
        _mm_storeu_ps(&data[14], data128);
      }

      sum = calculate_sum();
    }
    void updateData(std::mt19937 &gen,
                    const std::array<float, SOLARHOURS> &solarCurve) {
      int idx = 0;
      float multiplier = 0.f;
      switch (this->type) {
      case ConsumerClass::Day:
        idx = 0;
        break;
      case ConsumerClass::Night:
        idx = 13;
        break;
      }
      switch (this->house) {

      case HouseholdClass::Small:
        multiplier = 0.5f;
        break;
      case HouseholdClass::Medium:
        multiplier = 1.f;
        break;
      case HouseholdClass::High:
        multiplier = 2.5f;
        break;
      }

      for (int i = 0; i < HOURS; i++) {
        const auto &[mu, s] = distParams[idx % HOURS];
        std::lognormal_distribution<long double> dist(mu, s);

        data[i] = (0.6f * data[i]) + (0.4f * (dist(gen) * multiplier));
        idx++;
      }

      constexpr float scalingFactor = -SOLARPRODUCES / 5.6f;
      if (solar) {
        __m256 vec256 = _mm256_set1_ps(scalingFactor);
        __m128 vec128 = _mm_set1_ps(scalingFactor);
        vec256 = _mm256_mul_ps(vec256, _mm256_load_ps(&solarCurve[0]));
        vec128 = _mm_mul_ps(vec128, _mm_load_ps(&solarCurve[8]));
        const __m256 data256 = _mm256_add_ps(vec256, _mm256_load_ps(&data[6]));
        const __m128 data128 = _mm_add_ps(vec128, _mm_load_ps(&data[14]));
        _mm256_storeu_ps(&data[6], data256);
        _mm_storeu_ps(&data[14], data128);
      }

      sum = calculate_sum();
    }
    float calculate_sum() const {
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

      return _mm_cvtss_f32(sums);
    }
  };

public:
  EnergyReadings() = delete;
  EnergyReadings(const unsigned long long size);
  std::vector<Consumer> consumers;
  std::array<float, HOURS> calculate_aggregate();
  bool simulate();
  std::array<unsigned, 2> clusters{0};
  long double mean;
  long double variance;
  float battery;
  alignas(32) std::array<float, HOURS> gridCapacity;
  double calculate_sum();

private:
  double calculate_mean();
  double calculate_variance();
  static constexpr std::array<float, HOURS> consumption = {
      0.25, 0.20, 0.20, 0.20, 0.25, 0.35, 0.60, 1.10, 0.90, 0.70, 0.60, 0.60,
      0.60, 0.65, 0.70, 0.80, 1.20, 1.80, 2.50, 2.20, 1.60, 1.10, 0.80, 0.40};

  static constexpr std::array<float, HOURS> varianceFactor = {
      0.05, 0.05, 0.05, 0.05, 0.08, 0.10, 0.20, 0.25, 0.20, 0.15, 0.15, 0.15,
      0.15, 0.15, 0.15, 0.20, 0.35, 0.40, 0.45, 0.40, 0.30, 0.25, 0.15,
  };
  alignas(32) static constexpr std::array<float, SOLARHOURS> solarCurve = {
      0.05f, 0.15f, 0.40f, 0.70f, 0.90f, 1.00f,
      0.95f, 0.75f, 0.45f, 0.20f, 0.05f, 0.00f};
  struct LogNormalParams {
    long double mu;
    long double s;
  };
  static const inline std::array<LogNormalParams, HOURS> distParams = []() {
    std::array<LogNormalParams, HOURS> params{};
    for (int i = 0; i < HOURS; ++i) {
      const long double mean = consumption[i];
      const long double sigma = mean * varianceFactor[i];

      params[i].mu =
          std::logl((mean * mean) / std::sqrtl(sigma * sigma + mean * mean));
      params[i].s =
          std::sqrtl(std::logl(1.l + (sigma * sigma) / (mean * mean)));
    }
    return params;
  }();
};
