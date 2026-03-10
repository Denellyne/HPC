#include "EnergyReadings/EnergyReadings.h"

constexpr unsigned long long SIZE = 1e8;

int main(int argc, char *argv[]) {
  EnergyReadings readings = EnergyReadings(SIZE);
  printf("Mean: %LfkWh\n", readings.mean);
  printf("Variance: %Lf\n", std::sqrt(readings.variance));
  for (const auto &entry : readings.calculate_aggregate())
    std::cout << entry << ' ';

  std::cout << '\n';

  return 0;
}
