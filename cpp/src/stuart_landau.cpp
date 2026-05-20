#include "stuart_landau.hpp"

#include <cmath>
#include <stdexcept>

namespace {
void sl_rhs(
    const Network& net,
    const std::vector<double>& omega,
    const std::vector<std::complex<double>>& z,
    const SLParams& params,
    std::vector<std::complex<double>>& dz) {
  const int n = static_cast<int>(z.size());
  dz.assign(n, std::complex<double>(0.0, 0.0));

  for (int i = 0; i < n; ++i) {
    std::complex<double> coupling(0.0, 0.0);
    for (int j : net.adj[i]) {
      coupling += z[j] - z[i];
    }

    double deg = static_cast<double>(net.adj[i].size());
    if (deg > 0.0) {
      coupling *= (params.k / deg);
    } else {
      coupling = std::complex<double>(0.0, 0.0);
    }

    double mag2 = std::norm(z[i]);
    std::complex<double> growth(params.alpha, omega[i]);
    dz[i] = (growth - mag2) * z[i] + coupling;
  }
}

void rk4_step(
    const Network& net,
    const std::vector<double>& omega,
    const SLParams& params,
    double dt,
    std::vector<std::complex<double>>& z) {
  const int n = static_cast<int>(z.size());
  std::vector<std::complex<double>> k1(n), k2(n), k3(n), k4(n), tmp(n);

  sl_rhs(net, omega, z, params, k1);
  for (int i = 0; i < n; ++i) {
    tmp[i] = z[i] + 0.5 * dt * k1[i];
  }

  sl_rhs(net, omega, tmp, params, k2);
  for (int i = 0; i < n; ++i) {
    tmp[i] = z[i] + 0.5 * dt * k2[i];
  }

  sl_rhs(net, omega, tmp, params, k3);
  for (int i = 0; i < n; ++i) {
    tmp[i] = z[i] + dt * k3[i];
  }

  sl_rhs(net, omega, tmp, params, k4);
  for (int i = 0; i < n; ++i) {
    z[i] += (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
  }
}
}  // namespace

SimulationResult simulate_stuart_landau(
    const Network& net,
    const std::vector<double>& omega,
    const std::vector<std::complex<double>>& z0,
    const SLParams& params,
    double tmax,
    double dt) {
  if (omega.size() != z0.size()) {
    throw std::runtime_error("omega and z0 must have the same size");
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

  std::vector<std::complex<double>> z = z0;
  double t = 0.0;

  for (int step = 0; step <= steps; ++step) {
    result.times.push_back(t);
    result.order.push_back(order_parameter_complex(z));

    if (step == steps) {
      break;
    }

    rk4_step(net, omega, params, dt, z);
    t += dt;
  }

  return result;
}
