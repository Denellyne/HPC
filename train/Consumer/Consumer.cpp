#include "Consumer.h"
#include <immintrin.h>
#include <xmmintrin.h>

void Consumer::setData(
    std::mt19937_64 &gen,
    const std::array<const LogNormalParams, HOURS> &distParams,
    const std::array<float, SOLARHOURS> &solarCurve) {
  int idx = 0;
  float multiplier = 0.f;
  if (this->flags & DAY)
    idx = 0;
  else
    idx = 13;
  if (this->flags & SMALL)
    multiplier = 0.6f;
  else if (this->flags & MEDIUM)
    multiplier = 1.f;
  else
    multiplier = 2.5f;

  for (int i = 0; i < HOURS; i++) {
    const auto &[mu, s] = distParams[idx % HOURS];

    std::lognormal_distribution<long double> dist(mu, s);

    data[i] = dist(gen) * multiplier;
    idx++;
  }
  constexpr float scalingFactor = -SOLARPRODUCES / 5.6f;
  if (this->flags & SOLAR) {
    __m256 vec256 = _mm256_set1_ps(scalingFactor);
    __m128 vec128 = _mm_set1_ps(scalingFactor);
    vec256 = _mm256_mul_ps(vec256, _mm256_load_ps(&solarCurve[0]));
    vec128 = _mm_mul_ps(vec128, _mm_load_ps(&solarCurve[8]));
    const __m256 data256 = _mm256_max_ps(
        _mm256_setzero_ps(), _mm256_add_ps(vec256, _mm256_load_ps(&data[6])));
    const __m128 data128 = _mm_max_ps(
        _mm_setzero_ps(), _mm_add_ps(vec128, _mm_load_ps(&data[14])));
    _mm256_storeu_ps(&data[6], data256);
    _mm_storeu_ps(&data[14], data128);
  }

  sum = calculate_sum();
}
void Consumer::updateData(
    std::mt19937_64 &gen,
    const std::array<const LogNormalParams, HOURS> &distParams,
    const std::array<float, SOLARHOURS> &solarCurve,
    const std::array<float, HOURS> &price) {
  int idx = 0;
  float multiplier = 0.f;
  if (this->flags & DAY)
    idx = 0;
  else
    idx = 13;
  if (this->flags & SMALL)
    multiplier = 0.6f;
  else if (this->flags & MEDIUM)
    multiplier = 1.f;
  else
    multiplier = 2.5f;

  // for (int i = 0; i < HOURS; i++) {
  //   const auto &[mu, s] = distParams[idx % HOURS];
  //
  //   std::lognormal_distribution<long double> dist(mu, s);
  //
  //   data[i] = (dist(gen) * multiplier) * 0.4f + (data[i] * 0.6f);
  //   idx++;
  // }
  for (int i = 0; i < HOURS; i += 8) {
    generateRandomData(gen, distParams, idx, i, multiplier);
    idx += 8;
  }

  if (this->flags & SOLAR) {
    constexpr float scalingFactor = -SOLARPRODUCES / 5.6f;
    __m256 vec256 = _mm256_set1_ps(scalingFactor);
    __m128 vec128 = _mm_set1_ps(scalingFactor);
    vec256 = _mm256_mul_ps(vec256, _mm256_load_ps(&solarCurve[0]));
    vec128 = _mm_mul_ps(vec128, _mm_load_ps(&solarCurve[8]));
    const __m256 data256 = _mm256_add_ps(vec256, _mm256_load_ps(&data[6]));
    const __m128 data128 = _mm_add_ps(vec128, _mm_load_ps(&data[14]));
    _mm256_storeu_ps(&data[6], data256);
    _mm_storeu_ps(&data[14], data128);
  }
  adjustByPrice(price);

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

__m256 Consumer::generateRandomData(
    std::mt19937_64 &gen,
    const std::array<const LogNormalParams, HOURS> &distParams,
    const unsigned distIdx, const unsigned idx, const float multiplier) {
  const auto generate8Floats = [&gen, &distParams, &distIdx, &multiplier]() {
    alignas(32) static thread_local float result[8];
    static thread_local std::uniform_real_distribution<double> uni(
        std::nextafter(0.0, 1.0), 1.0);
    for (int i = 0; i < 8; i++) {
      const auto &[mu, s] = distParams[(i + distIdx) % HOURS];
      const double u1 = uni(gen);
      const double u2 = uni(gen);
      const double z0 =
          std::sqrtf(-2.0f * std::logf(u1)) * std::cosf(2.0f * M_PI * u2);
      // std::lognormal_distribution<long double> dist(mu, s);
      result[i] = (std::expf(mu + s * z0) * 0.4f) * multiplier;
    }
    return _mm256_load_ps(result);
  };
  const __m256 result = _mm256_add_ps(
      generate8Floats(),
      _mm256_mul_ps(_mm256_set1_ps(0.6f), _mm256_load_ps(&data[idx])));
  _mm256_store_ps(&data[idx], result);

  return result;
}

void Consumer::adjustByPrice(const std::array<float, HOURS> &price) {
  const auto totalPrice = [&price, this]() {
    float result = 0;
    for (size_t i = 0; i < HOURS; i++)
      result += (price[i] * this->data[i]);
    return result;
  };

  const auto findPair = [this, &price](const int i, const float min,
                                       const float max) {
    int minIdx = i, maxIdx = i + 16;
    for (int idx = i; idx - i < 16; idx++) {
      int j = idx % HOURS;
      if (this->data[j] >= min &&
          this->data[j] * price[j] < this->data[minIdx] * price[minIdx])
        minIdx = j;
      else if (this->data[j] <= max &&
               this->data[j] * price[j] > this->data[maxIdx] * price[maxIdx])
        maxIdx = j;
    }
    std::array<int, 2> result = {minIdx, maxIdx};
    return result;
  };

  float min, max;
  int idx;
  if (this->flags & SMALL) {
    min = SMALLMIN;
    max = SMALLMAX;
  } else if (this->flags & HIGH) {
    min = HIGHMIN;
    max = HIGHMAX;
  } else {
    min = MEDIUMMIN;
    max = MEDIUMMAX;
  }

  if (this->flags & DAY)
    idx = 7;
  else
    idx = 19;
  float currentPrice = totalPrice();
  float previousPrice = currentPrice + 5.f;
  std::array<float, HOURS> prev = this->data;
  while (currentPrice < previousPrice) {
    prev = this->data;
    previousPrice = currentPrice;

    auto [x, y] = findPair(idx, min, max);
    float diff = std::min(abs(this->data[x] - max), abs(this->data[y] - min));
    this->data[x] += diff;
    this->data[y] -= diff;

    currentPrice = totalPrice();
  }
  this->data = std::move(prev);
}
