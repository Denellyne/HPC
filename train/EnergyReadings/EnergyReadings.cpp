#include "EnergyReadings.h"
#include <immintrin.h>
#include <omp.h>

EnergyReadings::EnergyReadings(const unsigned long long size) {
  this->battery = size * 3.0;
  for (int i = 0; i < HOURS; i++) {
    if (i >= 17 && i <= 21)
      this->gridCapacity[i] = (24.8f * size) / 24.f;
    else
      this->gridCapacity[i] = (14.0f * size) / 24.f;
  }
  this->consumers.resize(size);

#pragma omp parallel
  {
    std::random_device rd{};
    std::mt19937 gen(rd() ^ omp_get_thread_num());
    std::uniform_int_distribution<int> dist(0, 4999);
    std::uniform_int_distribution<int> producer(0, 49);

#pragma omp for
    for (size_t i = 0; i < size; i++) {
      ConsumerClass type = ConsumerClass::Day;
      int r = dist(gen);
      if (r < 1200)
        type = ConsumerClass::Night;
      else
        type = ConsumerClass::Day;
      HouseholdClass house = HouseholdClass::Medium;
      r = dist(gen);
      if (r < 300)
        house = HouseholdClass::High;
      else if (r < 1200)
        house = HouseholdClass::Small;
      else
        house = HouseholdClass::Medium;

      int isProducer = producer(gen);
      this->consumers[i].type = type;
      this->consumers[i].house = house;
      this->consumers[i].solar = (isProducer < 15);
      this->consumers[i].setData(gen, solarCurve);

#pragma omp atomic
      this->clusters[(int)type]++;
    }
  }

  this->mean = this->calculate_mean();
  this->variance = this->calculate_variance();
}

double EnergyReadings::calculate_sum() {
  double result = 0;
#pragma omp parallel for reduction(+ : result)
  for (const auto &entry : this->consumers)
    result += entry.mean();
  return result;
}
double EnergyReadings::calculate_mean() {
  return this->calculate_sum() / this->consumers.size();
}
double EnergyReadings::calculate_variance() {
  double sum = 0;
  const __m256 meanVec = _mm256_set1_ps(static_cast<float>(this->mean));
  __m256 resultVec = _mm256_setzero_ps();

#pragma omp parallel firstprivate(resultVec) reduction(+ : sum)
  {
#pragma omp for
    for (const auto &entry : this->consumers) {
      __m256 v1 = _mm256_sub_ps(_mm256_load_ps(&entry.data[0]), meanVec);
      __m256 v2 = _mm256_sub_ps(_mm256_load_ps(&entry.data[8]), meanVec);
      __m256 v3 = _mm256_sub_ps(_mm256_load_ps(&entry.data[16]), meanVec);

      v1 = _mm256_mul_ps(v1, v1);
      v2 = _mm256_mul_ps(v2, v2);
      v3 = _mm256_mul_ps(v3, v3);

      resultVec =
          _mm256_add_ps(resultVec, _mm256_add_ps(v1, _mm256_add_ps(v2, v3)));
    }

    __m128 low = _mm256_castps256_ps128(resultVec);

    const __m128 high = _mm256_extractf128_ps(resultVec, 1);

    low = _mm_add_ps(low, high);
    __m128 odd = _mm_movehdup_ps(low);
    __m128 sums = _mm_add_ps(low, odd);
    odd = _mm_movehl_ps(odd, sums);
    sums = _mm_add_ss(sums, odd);
    sum += _mm_cvtss_f32(sums);
  }
  return sum / (this->consumers.size() * HOURS);
}

std::array<float, HOURS> EnergyReadings::calculate_aggregate() {
  alignas(32) std::array<float, HOURS> sum{};
  __m256 v1 = _mm256_setzero_ps();
  __m256 v2 = _mm256_setzero_ps();
  __m256 v3 = _mm256_setzero_ps();
#pragma omp parallel firstprivate(v1, v2, v3)
  {

#pragma omp for
    for (const auto &entry : this->consumers) {
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

bool EnergyReadings::simulate() {

#pragma omp parallel
  {
    std::random_device rd{};
    std::mt19937 gen(rd() ^ omp_get_thread_num());

#pragma omp for
    for (size_t i = 0; i < this->consumers.size(); i++)
      this->consumers[i].updateData(gen, solarCurve);
  }

  for (size_t i = 0; i < 24; i++) {
    float grid = this->gridCapacity[i];
    float battery = this->battery;
#pragma omp parallel for reduction(+ : grid, battery)
    for (size_t j = 0; j < this->consumers.size(); j++) {
      if (this->consumers[j].data[i] < 0.f) {
        battery -= this->consumers[j].data[i];
        this->consumers[j].data[i] = 0.f;
        continue;
      }
      grid -= consumers[j].data[i];
    }
    if (grid < 0.f) {
      if (battery >= -grid)
        battery += grid;
      else {
        std::cout << "Grid failure at hour " << i + 1 << '\n';
        std::cout << "Battery: " << battery << " Grid: " << grid << '\n';
        return false;
      }
    } else
      battery += grid;
    this->battery = std::fmin(battery, this->consumers.size() * 5.0f);
  }

  this->mean = this->calculate_mean();
  this->variance = this->calculate_variance();

  return true;
}
