#include "Consumer.h"

void Consumer::setData(std::mt19937 &gen,
                       const std::array<LogNormalParams, HOURS> &distParams,
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
void Consumer::updateData(std::mt19937 &gen,
                          const std::array<LogNormalParams, HOURS> &distParams,
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
float Consumer::calculate_sum() const {
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
