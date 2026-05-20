#include "order_param.hpp"

#include <cmath>

double order_parameter(const std::vector<double>& phases) {
  if (phases.empty()) {
    return 0.0;
  }

  double csum = 0.0;
  double ssum = 0.0;
  for (double theta : phases) {
    csum += std::cos(theta);
    ssum += std::sin(theta);
  }

  const double n = static_cast<double>(phases.size());
  return std::sqrt(csum * csum + ssum * ssum) / n;
}

double order_parameter_complex(const std::vector<std::complex<double>>& state) {
  if (state.empty()) {
    return 0.0;
  }

  std::complex<double> sum(0.0, 0.0);
  int count = 0;
  for (const auto& z : state) {
    double mag = std::abs(z);
    if (mag > 0.0) {
      sum += z / mag;
      ++count;
    }
  }

  if (count == 0) {
    return 0.0;
  }

  return std::abs(sum) / static_cast<double>(count);
}
