# Flash crash propagation, interactive simulation

Open `simulation/index.html` in a browser. No server, no build step, no internet.

The page integrates the **same dynamics as `research/model.py`**, live, on a network of 500 trading
desks, and shows the crash spreading node by node and edge by edge.

```
dθ_i = [ ω_i + (λ/⟨k⟩)·r_i·Σ_j A_ij sin(θ_j − θ_i) + ε·s_i·sin(−π/2 − θ_i) ] dt + σ dW_i + σ_c dW_c
```

## What you are looking at

| element | encodes |
|---|---|
| node colour | order flow `q_i = sin θ_i`, cyan buying, red selling, dark when flat |
| node size | degree `k_i` (how many desks watch it) |
| node glow | local agreement `r_i`, bright when its neighbours are unanimous |
| edge brightness | phase alignment `cos(θ_j − θ_i)`, a link lights up when its two ends lock |
| expanding ring | a desk that has just locked (`r_i` crossing 0.72), so the cascade front is visible |
| violet outline | desk halted by a circuit breaker |

Panels: synchrony `R` and order flow `Q`; the index price; the phase circle (every desk plotted at its
angle, with the order-parameter vector `R e^{iΨ}`); `R` vs `λ` tracing the hysteresis loop; and a phase
timeline colour-coded by market regime.

## The eight market phases

| phase | condition | meaning |
|---|---|---|
| INCOHERENT | `R < 0.25` | phases spread, orders cancel, a working market |
| CLUSTERING | `R ≥ 0.25` | sub-groups lock, `R` flickers upward |
| SHOCK APPLIED | `ε > 0` | a common sell signal drives a subset toward `θ = −π/2` |
| CASCADE | `dR/dt > 0.05`, `R > 0.3` | locked desks recruit neighbours; the front spreads from hubs outward |
| LOCKED CRASH | `R ≥ 0.75`, `Q < −0.3` | one giant cluster, everyone selling, self-sustaining |
| TRADING HALT | halt active | coupling severed; desks free-run at their own `ω` |
| DESYNCHRONISING | `dR/dt < −0.04` after a crash | diversity of horizons pulls the herd apart |
| RECOVERY | `R < 0.3` after a crash | if `λ > λ_b` the market can still re-crash |

## Controls

`λ` herding coupling · `σ` idiosyncratic news · `σ_c` common market mode · `f` herding fraction ·
shock size `ε` and reach `φ` · halt length `τ` · speed.

Buttons: sell shock (random or hub-targeted), trading halt, **auto ramp λ** (sweeps up then down and
draws the hysteresis loop), colour mode (order flow vs cyclic phase), network switch, reset.
Keys: `space` pause, `s` shock, `h` halt, `r` reset.

## Deep links

Scenarios are URL-addressable, which is also how the headless tests drive the page:

```
index.html?lam=3.45&warmup=120&shock=random          a locked crash
index.html?lam=3.45&warmup=120&shock=random&halt=8   a crash, then a circuit breaker
index.html?net=ba6&lam=2.8&warmup=100                the scale-free network
```
Parameters: `lam`, `sigma`, `sigmac`, `net` (`er12`|`ba6`), `warmup` (time units), `shock`
(`random`|`hubs`|`periphery`), `halt` (duration).

## The two networks

| key | network | E | ⟨k⟩ | κ | use |
|---|---|---|---|---|---|
| `er12` *(default)* | Erdős-Rényi | 3076 | 12.3 | 1.07 | **hysteresis**, the exact configuration S2-S5 measure |
| `ba6` | Barabási-Albert | 1491 | 5.96 | 2.29 | **hub targeting**, has hubs, but almost no bistable window |

On `er12` an adiabatic sweep here gives **λ_b = 3.1, λ_f = 4.1, max hysteresis gap 0.80**, reproducing
the window [3.21, 3.96] measured in S2 for the same configuration. `ba6` has width ≈ 0, consistent with
S2's finding that heterogeneous networks *resist* herding-induced bistability, so it is the wrong
network for the hysteresis story and the right one for the hub story.

## Files

| file | purpose |
|---|---|
| `index.html` | UI, rendering, phase detection |
| `sim.js` | the integrator, runs in the browser *and* in node, so it can be unit-tested |
| `network.js` | generated: both networks with precomputed force-directed layouts |
| `build_network.py` | regenerates `network.js` |
| `validate_physics.py` | **benchmark 1**, sim.js vs `research/model.py` |
| `validate_scenarios.js` | **benchmarks 2-8**, phases reachable, hysteresis, throughput |

## Benchmarks

```
research/.venv/bin/python simulation/build_network.py     # regenerate networks
research/.venv/bin/python simulation/validate_physics.py  # benchmark 1
node simulation/validate_scenarios.js                     # benchmarks 2-8
```

Current status, **all passing**:

| # | benchmark | result |
|---|---|---|
| 1 | sim.js reproduces `model.py` (σ=0, 2000 Heun steps, both networks, plain + herding + halted) | max \|Δθ\| = **4.3×10⁻¹³** |
| 2 | low λ stays incoherent | R = 0.071 |
| 3 | high λ synchronises | R = 0.980 |
| 4 | shock below the window = flash crash then recovery | peak R 0.506 → final 0.048 |
| 5 | shock inside the window = persistent crash | final R = 0.850 |
| 6 | trading halt desynchronises | R 0.850 → 0.005 |
| 7 | adiabatic sweep shows hysteresis | λ_f = 4.1, λ_b = 3.1, gap 0.795 |
| 8 | physics throughput | **160 µs/step**, 20 substeps = 3.2 ms/frame (budget 10) |

Checks 5 and 7 are enforced on `er12` only and reported as notes on `ba6`, because a network with no
bistable window has no second stable branch for a shock to leave the market on, "persistent" is
undefined there rather than failed.

## Note on performance

The integrator started at 523 µs/step. Two changes took it to 160 µs (3.3×): typed arrays for the
adjacency (it arrives from JSON as plain `Array`s), and lifting the halt test out of the inner
neighbour loop so the common case is branch-free. The halt flag is derived from `active` at the top of
each step rather than trusted as a manually-set field, setting `active` directly had silently broken
the physics, which benchmark 1 caught.
