#pragma once

#include <cmath>
#include <vector>

#include "network_gen.hpp"

namespace kuramoto_integrator {
inline void kuramoto_rhs(
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

inline void rk4_step(
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
}  // namespace kuramoto_integrator
