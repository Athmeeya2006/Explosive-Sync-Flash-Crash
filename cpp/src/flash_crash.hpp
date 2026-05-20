#pragma once

#include <vector>

#include "network_gen.hpp"
#include "order_param.hpp"

SimulationResult simulate_flash_crash(
    const Network& net,
    const std::vector<double>& omega,
    const std::vector<double>& theta0,
    double k_high,
    double k_low,
    double t_drop,
    double tmax,
    double dt);
