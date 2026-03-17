#pragma once
#include "../Consumer/Consumer.h"

class EnergyReadings {
public:
  EnergyReadings() = delete;
  EnergyReadings(const unsigned long long size);
  bool simulate();
  long double mean;
  long double variance;
  float battery;
  double calculate_sum();

  std::vector<Consumer> consumers;
  std::array<float, HOURS> calculate_aggregate();
  alignas(32) std::array<float, HOURS> gridCapacity;
  std::array<unsigned, 2> clusters{0};

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
