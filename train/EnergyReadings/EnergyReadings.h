#pragma once
#include "../Consumer/Consumer.h"
#include <cmath>

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
  alignas(32) static constexpr std::array<float, SOLARHOURS> solarCurve = {
      0.05f, 0.15f, 0.40f, 0.70f, 0.90f, 1.00f,
      0.95f, 0.75f, 0.45f, 0.20f, 0.05f, 0.00f};

  static constexpr inline std::array<const LogNormalParams, HOURS> distParams =
      []() {
        constexpr auto calculate = [](const size_t i) {
          constexpr static std::array<float, HOURS> consumption = {
              0.25, 0.20, 0.20, 0.20, 0.25, 0.35, 0.60, 1.10,
              0.90, 0.70, 0.60, 0.60, 0.60, 0.65, 0.70, 0.80,
              1.20, 1.80, 2.50, 2.20, 1.60, 1.10, 0.80, 0.40};

          constexpr static std::array<float, HOURS> varianceFactor = {
              0.05, 0.05, 0.05, 0.05, 0.08, 0.10, 0.20, 0.25,
              0.20, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.20,
              0.35, 0.40, 0.45, 0.40, 0.30, 0.25, 0.15, 0.13};

          return LogNormalParams(
              std::logl((consumption[i] * consumption[i]) /
                        std::sqrtl(consumption[i] * varianceFactor[i] *
                                       consumption[i] * varianceFactor[i] +
                                   consumption[i] * consumption[i])),
              std::sqrtl(
                  std::logl(1.l + (consumption[i] * varianceFactor[i] *
                                   consumption[i] * varianceFactor[i]) /
                                      (consumption[i] * consumption[i]))));
        };

        constexpr std::array<const LogNormalParams, HOURS> params = {
            calculate(0),  calculate(1),  calculate(2),  calculate(3),
            calculate(4),  calculate(5),  calculate(6),  calculate(7),
            calculate(8),  calculate(9),  calculate(10), calculate(11),
            calculate(12), calculate(13), calculate(14), calculate(15),
            calculate(16), calculate(17), calculate(18), calculate(19),
            calculate(20), calculate(21), calculate(22), calculate(23),
        };
        static_assert(params[0].mu != 0.f, "");
        return params;
      }();
};
