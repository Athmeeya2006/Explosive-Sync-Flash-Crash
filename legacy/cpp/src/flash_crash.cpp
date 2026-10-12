#include "flash_crash.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "kuramoto_integrator.hpp"

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

  SimulationResult result;
  const int reserve_steps = static_cast<int>(tmax / dt) + 2;
  result.times.reserve(reserve_steps);
  result.order.reserve(reserve_steps);

  std::vector<double> theta = theta0;
  double t = 0.0;

  while (true) {
    result.times.push_back(t);
    result.order.push_back(order_parameter(theta));

    const double remaining = tmax - t;
    if (remaining <= 0.0) {
      break;
    }

    double dt_step = std::min(dt, remaining);
    if (t < t_drop && t + dt_step > t_drop) {
      dt_step = t_drop - t;
    }

    const double k = (t < t_drop) ? k_high : k_low;
    kuramoto_integrator::rk4_step(net, omega, k, dt_step, theta);
    t += dt_step;
  }

  return result;
}
