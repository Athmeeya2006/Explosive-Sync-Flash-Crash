#include "kuramoto.hpp"

#include <cmath>
#include <stdexcept>

namespace {
void kuramoto_rhs(
    const Network& net,
    const std::vector<double>& omega,
    const std::vector<double>& theta,
    double k,
    std::vector<double>& dtheta) {
  const int n = static_cast<int>(theta.size());
  dtheta.assign(n, 0.0);

  for (int i = 0; i < n; ++i) {
    double sum = 0.0;
    for (int j : net.adj[i]) {
      sum += std::sin(theta[j] - theta[i]);
    }
    double deg = static_cast<double>(net.adj[i].size());
    double coupling = (deg > 0.0) ? (k / deg) * sum : 0.0;
    dtheta[i] = omega[i] + coupling;
  }
}

void rk4_step(
    const Network& net,
    const std::vector<double>& omega,
    double k,
    double dt,
    std::vector<double>& theta) {
  const int n = static_cast<int>(theta.size());
  std::vector<double> k1(n), k2(n), k3(n), k4(n), tmp(n);

  kuramoto_rhs(net, omega, theta, k, k1);
  for (int i = 0; i < n; ++i) {
    tmp[i] = theta[i] + 0.5 * dt * k1[i];
  }

  kuramoto_rhs(net, omega, tmp, k, k2);
  for (int i = 0; i < n; ++i) {
    tmp[i] = theta[i] + 0.5 * dt * k2[i];
  }

  kuramoto_rhs(net, omega, tmp, k, k3);
  for (int i = 0; i < n; ++i) {
    tmp[i] = theta[i] + dt * k3[i];
  }

  kuramoto_rhs(net, omega, tmp, k, k4);
  for (int i = 0; i < n; ++i) {
    theta[i] += (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
  }
}
}  // namespace

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

    rk4_step(net, omega, k, dt, theta);
    t += dt;
  }

  return result;
}
