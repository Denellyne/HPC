#pragma once
#include "../EnergyReadings/EnergyReadings.h"

class Aggregator {
public:
private:
  const EnergyReadings readings;
  std::array<float, ARRSIZE> priceHour{};
};
