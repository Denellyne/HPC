#pragma once

#include <array>
#include <immintrin.h>
#include <random>
constexpr unsigned short HOURS = 24;

#define SOLARPRODUCES 10.f
#define SOLARHOURS 12
struct LogNormalParams {
  long double mu;
  long double s;
};
enum class ConsumerClass {
  Day = 0,
  Night = 1,
};
enum class HouseholdClass {
  Small,
  Medium,
  High,
};
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
               const std::array<LogNormalParams, HOURS> &distParams,
               const std::array<float, SOLARHOURS> &solarCurve);
  void updateData(std::mt19937 &gen,
                  const std::array<LogNormalParams, HOURS> &distParams,
                  const std::array<float, SOLARHOURS> &solarCurve);
  float calculate_sum() const;

private:
};
