#include "network_gen.hpp"

#include <stdexcept>
#include <unordered_set>

namespace {
void add_edge(Network& net, int a, int b) {
  net.adj[a].push_back(b);
  net.adj[b].push_back(a);
}
}  // namespace

Network er_network(int n, double p, std::mt19937& rng) {
  if (n <= 0) {
    throw std::runtime_error("n must be positive");
  }
  if (p < 0.0 || p > 1.0) {
    throw std::runtime_error("p must be in [0, 1]");
  }

  Network net;
  net.adj.assign(n, {});

  std::uniform_real_distribution<double> dist(0.0, 1.0);
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      if (dist(rng) < p) {
        add_edge(net, i, j);
      }
    }
  }
  return net;
}

Network ba_network(int n, int m, std::mt19937& rng) {
  if (n <= 0) {
    throw std::runtime_error("n must be positive");
  }
  if (m < 1) {
    throw std::runtime_error("m must be at least 1");
  }
  if (n <= m + 1) {
    throw std::runtime_error("n must be greater than m + 1");
  }

  Network net;
  net.adj.assign(n, {});

  const int m0 = m + 1;
  for (int i = 0; i < m0; ++i) {
    for (int j = i + 1; j < m0; ++j) {
      add_edge(net, i, j);
    }
  }

  std::vector<int> repeated;
  repeated.reserve(m0 * m0);
  for (int i = 0; i < m0; ++i) {
    for (int k = 0; k < m0 - 1; ++k) {
      repeated.push_back(i);
    }
  }

  for (int node = m0; node < n; ++node) {
    std::unordered_set<int> targets;
    std::uniform_int_distribution<int> dist(0, static_cast<int>(repeated.size() - 1));

    while (static_cast<int>(targets.size()) < m) {
      int candidate = repeated[dist(rng)];
      targets.insert(candidate);
    }

    for (int target : targets) {
      add_edge(net, node, target);
      repeated.push_back(target);
    }

    for (int k = 0; k < static_cast<int>(targets.size()); ++k) {
      repeated.push_back(node);
    }
  }

  return net;
}
