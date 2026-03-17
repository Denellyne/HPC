#include "EnergyReadings/EnergyReadings.h"

constexpr unsigned long long SIZE = 1e6;

int main(void) {
  EnergyReadings readings = EnergyReadings(SIZE);
  do {
    printf("\nBattery: %fMWh\n", readings.battery / 1000);
    printf("Sum: %fMWh\n", readings.calculate_sum() / 1000.f);
    printf("Mean: %LfKWh\n", readings.mean);
    printf("Variance: %Lf\n", std::sqrt(readings.variance));
    printf("Hour\tConsumption\n");
    int i = 1;

    for (const auto &entry : readings.calculate_aggregate())
      std::cout << i++ << '\t' << entry << '\n';

    printf("Day:%d Night:%d\n", readings.clusters[0], readings.clusters[1]);
  } while (readings.simulate());

  return 0;
}
