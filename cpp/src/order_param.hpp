#pragma once

#include <complex>
#include <vector>

struct SimulationResult {
  std::vector<double> times;
  std::vector<double> order;
};

double order_parameter(const std::vector<double>& phases);
double order_parameter_complex(const std::vector<std::complex<double>>& state);
