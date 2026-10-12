#include "network_gen.hpp"

#include <stdexcept>

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
  std::vector<int> degree(n, 0);

  const int m0 = m + 1;
  for (int i = 0; i < m0; ++i) {
    for (int j = i + 1; j < m0; ++j) {
      add_edge(net, i, j);
      degree[i] += 1;
      degree[j] += 1;
    }
  }

  for (int node = m0; node < n; ++node) {
    std::vector<double> weights(node);
    for (int i = 0; i < node; ++i) {
      weights[i] = static_cast<double>(degree[i]);
    }

    std::vector<int> targets;
    targets.reserve(m);
    for (int edge = 0; edge < m; ++edge) {
      std::discrete_distribution<int> dist(weights.begin(), weights.end());
      int target = dist(rng);
      targets.push_back(target);
      weights[target] = 0.0;
    }

    for (int target : targets) {
      add_edge(net, node, target);
      degree[node] += 1;
      degree[target] += 1;
    }
  }

  return net;
}
