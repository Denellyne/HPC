#include "EnergyReadings/EnergyReadings.h"
#include <iostream>

// #define BENCH
int main(void) {
  EnergyReadings readings = EnergyReadings(SIZE);
#ifdef BENCH

  int x = 5;
  do {
    printf("\nBattery: %fMWh\n", readings.battery / 1000.f);
    printf("Sum: %fMWh\n", readings.calculate_sum() / 1000.f);
    printf("Mean: %fKWh\n", readings.mean);
    printf("Variance: %f\n", std::sqrt(readings.variance));
    printf("Hour\tConsumption\n");
    unsigned short i = 1;

    for (const auto &entry : readings.calculate_aggregate())
      std::cout << i++ << '\t' << entry << '\n';

    printf("Day:%d Night:%d\n", readings.clusters[0], readings.clusters[1]);
    readings.simulate();
  } while (x--);
#else

  do {
    printf("\nBattery: %fMWh\n", readings.battery / 1000.f);
    printf("Sum: %fMWh\n", readings.calculate_sum() / 1000.f);
    printf("Mean: %fKWh\n", readings.mean);
    printf("Variance: %f\n", std::sqrt(readings.variance));
    printf("Hour\tConsumption\n");
    unsigned short i = 1;

    for (const auto &entry : readings.calculate_aggregate())
      std::cout << i++ << '\t' << entry << '\n';
    for (const auto &entry : readings.consumers)
      if (entry.flags & SMALL) {
        for (const auto &x : entry.data) {
          std::cout << x << ' ';
        }
        std::cout << '\n';
        break;
      }

    printf("Day:%d Night:%d\n", readings.clusters[0], readings.clusters[1]);
  } while (readings.simulate());
#endif

  return 0;
}
