#include "EnergyReadings.h"
#include <immintrin.h>
#include <omp.h>

EnergyReadings::EnergyReadings(const unsigned long long size) {
  this->data.resize(size);

#pragma omp parallel
  {
    std::random_device rd;
    std::mt19937 gen(rd() + omp_get_thread_num());
    std::uniform_real_distribution<float> day(4.0f, 25.0f);
    std::uniform_real_distribution<float> night(0.0f, 6.0f);

    std::array<float, ARRSIZE> data;
#pragma omp for
    for (int i = 0; i < size; i++) {

      for (int j = 0; j < 6; j++) {
        data[j] = night(gen);
        data[j + 18] = night(gen);
      }

      for (int j = 6; j < 19; j++)
        data[j] = day(gen);

      this->data[i].data = data;
    }
  }

  this->mean = this->calculate_mean();
  this->variance = this->calculate_variance();
}

double EnergyReadings::calculate_mean() {
  double result = 0;
#pragma omp parallel for reduction(+ : result)
  for (const auto &entry : this->data)
    result += entry.mean();

  return result / this->data.size();
}
double EnergyReadings::calculate_variance() {
  double sum = 0;
  const __m256 meanVec = _mm256_set1_ps(static_cast<float>(this->mean));
  __m256 resultVec = _mm256_setzero_ps();

#pragma omp parallel firstprivate(resultVec)
  {
#pragma omp for reduction(+ : sum)
    for (const auto &entry : this->data) {
      __m256 v1 = _mm256_sub_ps(_mm256_load_ps(&entry.data[0]), meanVec);
      __m256 v2 = _mm256_sub_ps(_mm256_load_ps(&entry.data[8]), meanVec);
      __m256 v3 = _mm256_sub_ps(_mm256_load_ps(&entry.data[16]), meanVec);

      v1 = _mm256_mul_ps(v1, v1);
      v2 = _mm256_mul_ps(v2, v2);
      v3 = _mm256_mul_ps(v3, v3);

      resultVec =
          _mm256_add_ps(resultVec, _mm256_add_ps(v1, _mm256_add_ps(v2, v3)));
    }

    __m128 low = _mm_add_ps(low, _mm256_castps256_ps128(resultVec));

    const __m128 high = _mm256_extractf128_ps(resultVec, 1);

    low = _mm_add_ps(low, high);
    __m128 odd = _mm_movehdup_ps(low);
    __m128 sums = _mm_add_ps(low, odd);
    odd = _mm_movehl_ps(odd, sums);
    sums = _mm_add_ss(sums, odd);
    sum += _mm_cvtss_f32(sums);
  }
  return sum / (this->data.size() * ARRSIZE);
}

std::array<float, ARRSIZE> EnergyReadings::calculate_aggregate() {
  alignas(32) std::array<float, ARRSIZE> sum{};
  __m256 v1 = _mm256_setzero_ps();
  __m256 v2 = _mm256_setzero_ps();
  __m256 v3 = _mm256_setzero_ps();
#pragma omp parallel firstprivate(v1, v2, v3)
  {

#pragma omp for
    for (const auto &entry : this->data) {
      v1 = _mm256_add_ps(v1, _mm256_load_ps(&entry.data[0]));
      v2 = _mm256_add_ps(v2, _mm256_load_ps(&entry.data[8]));
      v3 = _mm256_add_ps(v3, _mm256_load_ps(&entry.data[16]));
    }

#pragma omp critical
    {
      __m256 temp;
      temp = _mm256_load_ps(&sum[0]);
      temp = _mm256_add_ps(temp, v1);
      _mm256_store_ps(&sum[0], temp);

      temp = _mm256_load_ps(&sum[8]);
      temp = _mm256_add_ps(temp, v2);
      _mm256_store_ps(&sum[8], temp);

      temp = _mm256_load_ps(&sum[16]);
      temp = _mm256_add_ps(temp, v3);
      _mm256_store_ps(&sum[16], temp);
    }
  }
  return sum;
}
