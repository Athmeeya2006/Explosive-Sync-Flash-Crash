#include "kuramoto.hpp"

#include <stdexcept>

#include "kuramoto_integrator.hpp"

SimulationResult simulate_kuramoto(
    const Network& net,
    const std::vector<double>& omega,
    const std::vector<double>& theta0,
    double k,
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

    kuramoto_integrator::rk4_step(net, omega, k, dt, theta);
    t += dt;
  }

  return result;
}
