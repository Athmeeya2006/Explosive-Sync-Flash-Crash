#include <cmath>
#include <complex>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "flash_crash.hpp"
#include "kuramoto.hpp"
#include "network_gen.hpp"
#include "stuart_landau.hpp"

namespace {
constexpr const char* kVersion = "0.1.0";

struct Options {
  std::string system = "kuramoto";
  std::string network = "er";
  int n = 200;
  double p = 0.05;
  int m = 3;
  double k = 1.0;
  double tmax = 50.0;
  double dt = 0.05;
  unsigned int seed = 42;
  std::string freq_mode = "gaussian";
  double omega_mean = 0.0;
  double omega_std = 1.0;
  double alpha = 1.0;
  double k_high = 1.5;
  double k_low = 0.3;
  double t_drop = 20.0;
  std::string out = "data/order_param.csv";
};

void print_help() {
  std::cout
      << "Explosive sync / flash crash simulator\n"
      << "Usage: esfc [options]\n\n"
      << "Options:\n"
      << "  --system {kuramoto|stuart-landau|flash-crash}\n"
      << "  --network {er|ba}\n"
      << "  --n <int>             Number of nodes\n"
      << "  --p <float>           ER edge probability\n"
      << "  --m <int>             BA attachment parameter\n"
      << "  --k <float>           Coupling strength\n"
      << "  --k-high <float>      Flash-crash pre-drop coupling\n"
      << "  --k-low <float>       Flash-crash post-drop coupling\n"
      << "  --t-drop <float>      Time of coupling drop\n"
      << "  --tmax <float>        Simulation horizon\n"
      << "  --dt <float>          Time step\n"
      << "  --seed <int>          RNG seed\n"
      << "  --freq-mode {gaussian|degree-weighted}\n"
      << "  --omega-mean <float>  Mean of natural frequencies\n"
      << "  --omega-std <float>   Std dev of natural frequencies\n"
      << "  --alpha <float>       Stuart-Landau alpha parameter\n"
      << "  --out <path>          Output CSV path\n"
      << "  --version             Show version\n"
      << "  --help                Show this message\n";
}

std::string require_value(int& i, int argc, char** argv) {
  if (i + 1 >= argc) {
    throw std::runtime_error(std::string("Missing value for ") + argv[i]);
  }
  return std::string(argv[++i]);
}

Options parse_args(int argc, char** argv) {
  Options opts;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "--help") {
      print_help();
      std::exit(0);
    } else if (arg == "--version") {
      std::cout << "esfc " << kVersion << "\n";
      std::exit(0);
    } else if (arg == "--system") {
      opts.system = require_value(i, argc, argv);
    } else if (arg == "--network") {
      opts.network = require_value(i, argc, argv);
    } else if (arg == "--n") {
      opts.n = std::stoi(require_value(i, argc, argv));
    } else if (arg == "--p") {
      opts.p = std::stod(require_value(i, argc, argv));
    } else if (arg == "--m") {
      opts.m = std::stoi(require_value(i, argc, argv));
    } else if (arg == "--k") {
      opts.k = std::stod(require_value(i, argc, argv));
    } else if (arg == "--k-high") {
      opts.k_high = std::stod(require_value(i, argc, argv));
    } else if (arg == "--k-low") {
      opts.k_low = std::stod(require_value(i, argc, argv));
    } else if (arg == "--t-drop") {
      opts.t_drop = std::stod(require_value(i, argc, argv));
    } else if (arg == "--tmax") {
      opts.tmax = std::stod(require_value(i, argc, argv));
    } else if (arg == "--dt") {
      opts.dt = std::stod(require_value(i, argc, argv));
    } else if (arg == "--seed") {
      opts.seed = static_cast<unsigned int>(std::stoul(require_value(i, argc, argv)));
    } else if (arg == "--freq-mode") {
      opts.freq_mode = require_value(i, argc, argv);
    } else if (arg == "--omega-mean") {
      opts.omega_mean = std::stod(require_value(i, argc, argv));
    } else if (arg == "--omega-std") {
      opts.omega_std = std::stod(require_value(i, argc, argv));
    } else if (arg == "--alpha") {
      opts.alpha = std::stod(require_value(i, argc, argv));
    } else if (arg == "--out") {
      opts.out = require_value(i, argc, argv);
    } else {
      throw std::runtime_error("Unknown option: " + arg);
    }
  }
  return opts;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    Options opts = parse_args(argc, argv);
    const double two_pi = 2.0 * std::acos(-1.0);

    std::mt19937 rng(opts.seed);
    Network net = (opts.network == "ba")
        ? ba_network(opts.n, opts.m, rng)
        : er_network(opts.n, opts.p, rng);

    std::vector<double> omega(opts.n);
    if (opts.freq_mode == "gaussian") {
      std::normal_distribution<double> omega_dist(opts.omega_mean, opts.omega_std);
      for (int i = 0; i < opts.n; ++i) {
        omega[i] = omega_dist(rng);
      }
    } else if (opts.freq_mode == "degree-weighted") {
      const double omega_bar = (opts.omega_mean == 0.0) ? 1.0 : opts.omega_mean;
      for (int i = 0; i < opts.n; ++i) {
        omega[i] = omega_bar * static_cast<double>(net.adj[i].size());
      }
    } else {
      throw std::runtime_error("Unknown freq mode: " + opts.freq_mode);
    }

    SimulationResult result;
    if (opts.system == "kuramoto") {
      std::uniform_real_distribution<double> phase_dist(0.0, two_pi);
      std::vector<double> theta0(opts.n);
      for (int i = 0; i < opts.n; ++i) {
        theta0[i] = phase_dist(rng);
      }
      result = simulate_kuramoto(net, omega, theta0, opts.k, opts.tmax, opts.dt);
    } else if (opts.system == "stuart-landau") {
      std::uniform_real_distribution<double> phase_dist(0.0, two_pi);
      std::vector<std::complex<double>> z0(opts.n);
      for (int i = 0; i < opts.n; ++i) {
        double phase = phase_dist(rng);
        z0[i] = std::polar(1.0, phase);
      }
      SLParams params;
      params.alpha = opts.alpha;
      params.k = opts.k;
      result = simulate_stuart_landau(net, omega, z0, params, opts.tmax, opts.dt);
    } else if (opts.system == "flash-crash") {
      std::uniform_real_distribution<double> phase_dist(0.0, two_pi);
      std::vector<double> theta0(opts.n);
      for (int i = 0; i < opts.n; ++i) {
        theta0[i] = phase_dist(rng);
      }
      result = simulate_flash_crash(
          net,
          omega,
          theta0,
          opts.k_high,
          opts.k_low,
          opts.t_drop,
          opts.tmax,
          opts.dt);
    } else {
      throw std::runtime_error("Unknown system: " + opts.system);
    }

    std::filesystem::path out_path(opts.out);
    if (out_path.has_parent_path()) {
      std::filesystem::create_directories(out_path.parent_path());
    }

    std::ofstream out(out_path);
    if (!out) {
      throw std::runtime_error("Failed to open output file: " + opts.out);
    }

    out << "t,order\n";
    for (size_t i = 0; i < result.times.size(); ++i) {
      out << result.times[i] << "," << result.order[i] << "\n";
    }

    std::cout << "Wrote " << result.times.size() << " samples to " << opts.out << "\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "Error: " << ex.what() << "\n";
    return 1;
  }
}
