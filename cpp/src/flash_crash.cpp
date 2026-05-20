#include "flash_crash.hpp"

#include <cmath>
#include <stdexcept>

SimulationResult simulate_flash_crash(
    const Network& net,
    const std::vector<double>& omega,
    const std::vector<double>& theta0,
    double k_high,
    double k_low,
    double t_drop,
    double tmax,
    double dt) {
  if (omega.size() != theta0.size()) {
    throw std::runtime_error("omega and theta0 must have the same size");
  }
  if (net.size() != static_cast<int>(omega.size())) {
    throw std::runtime_error("network size mismatch");
  }
  if (dt <= 0.0 || tmax <= 0.0) {
    throw std::runtime_error("tmax and dt must be positive");
  }

  const int steps = static_cast<int>(tmax / dt);
  SimulationResult result;
  result.times.reserve(steps + 1);
  result.order.reserve(steps + 1);

  std::vector<double> theta = theta0;
  double t = 0.0;

  for (int step = 0; step <= steps; ++step) {
    result.times.push_back(t);
    result.order.push_back(order_parameter(theta));

    if (step == steps) {
      break;
    }

    const double k = (t < t_drop) ? k_high : k_low;
    std::vector<double> dtheta(theta.size(), 0.0);

    for (size_t i = 0; i < theta.size(); ++i) {
      double sum = 0.0;
      for (int j : net.adj[i]) {
        sum += std::sin(theta[j] - theta[i]);
      }
      double deg = static_cast<double>(net.adj[i].size());
      double coupling = (deg > 0.0) ? (k / deg) * sum : 0.0;
      dtheta[i] = omega[i] + coupling;
    }

    for (size_t i = 0; i < theta.size(); ++i) {
      theta[i] += dt * dtheta[i];
    }

    t += dt;
  }

  return result;
}
