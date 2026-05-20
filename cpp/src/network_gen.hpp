#pragma once

#include <random>
#include <vector>

struct Network {
  std::vector<std::vector<int>> adj;

  int size() const {
    return static_cast<int>(adj.size());
  }
};

Network er_network(int n, double p, std::mt19937& rng);
Network ba_network(int n, int m, std::mt19937& rng);
