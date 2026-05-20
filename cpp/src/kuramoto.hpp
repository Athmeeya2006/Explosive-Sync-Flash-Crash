#pragma once

#include <vector>

#include "network_gen.hpp"
#include "order_param.hpp"

SimulationResult simulate_kuramoto(
    const Network& net,
    const std::vector<double>& omega,
    const std::vector<double>& theta0,
    double k,
    double tmax,
    double dt);
