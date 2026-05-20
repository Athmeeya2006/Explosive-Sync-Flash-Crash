#pragma once

#include <complex>
#include <vector>

#include "network_gen.hpp"
#include "order_param.hpp"

struct SLParams {
  double alpha = 1.0;
  double k = 1.0;
};

SimulationResult simulate_stuart_landau(
    const Network& net,
    const std::vector<double>& omega,
    const std::vector<std::complex<double>>& z0,
    const SLParams& params,
    double tmax,
    double dt);
