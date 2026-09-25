"""Download the open-access reference library for this project.

Each entry is resolved through the arXiv API (an ID hint first, then a title search)
and only downloaded when the returned title closely matches the requested one.
Re-running skips files that already exist. Results are logged to index.csv.

    python references/fetch_references.py
"""

from __future__ import annotations

import csv
import difflib
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
ATOM = "{http://www.w3.org/2005/Atom}"
UA = {"User-Agent": "explosive-sync-refs/1.0 (research use)"}

# (short key, topic folder, exact title)
PAPERS: list[tuple[str, str, str]] = [
    # --- foundations: Kuramoto model ---
    ("acebron2005", "01_kuramoto_foundations", "The Kuramoto model: A simple paradigm for synchronization phenomena"),
    ("rodrigues2016", "01_kuramoto_foundations", "The Kuramoto model in complex networks"),
    ("ott_antonsen2008", "01_kuramoto_foundations", "Low dimensional behavior of large systems of globally coupled oscillators"),
    ("dorfler_bullo2014", "01_kuramoto_foundations", "Synchronization in complex networks of phase oscillators: A survey"),
    ("pazo2005", "01_kuramoto_foundations", "Thermodynamic limit of the first-order phase transition in the Kuramoto model"),
    ("arenas2008", "01_kuramoto_foundations", "Synchronization in complex networks"),
    # annealed mean-field threshold lambda_c = 2<k>/(pi g(0) <k^2>), used by research/s8b_theory_check.py
    ("ichinomiya2004", "01_kuramoto_foundations", "Frequency synchronization in a random oscillator network"),
    ("restrepo2005_onset", "01_kuramoto_foundations", "Onset of synchronization in large networks of coupled oscillators"),
    # bimodal g(omega): the frequency law used in the S8 sweep that is first-order without any correlation
    ("martens2009_bimodal", "01_kuramoto_foundations", "Exact results for the Kuramoto model with a bimodal frequency distribution"),
    # noisy Kuramoto: the anchor the sigma sweep in S8 currently lacks
    ("sakaguchi1988_noisy", "01_kuramoto_foundations", "Cooperative Phenomena in Coupled Oscillator Systems under External Fields"),
    ("bertini2010_noisy_kuramoto", "01_kuramoto_foundations", "Dynamical aspects of mean field plane rotators and the Kuramoto model"),
    # --- common noise and the synchrony -> observable mapping (prior art for S9's sigma_c) ---
    ("pikovsky2016_common_noise", "09_common_noise", "Interplay of coupling and common noise at the transition to synchrony in oscillator populations"),
    ("peter2019_microscopic_crosscorr", "09_common_noise", "Microscopic Cross-Correlations in the Finite-Size Kuramoto Model of Coupled Oscillators"),
    ("nagai2010_common_noise_sync", "09_common_noise", "Noise-Induced Synchronization of a Large Population of Globally Coupled Nonidentical Oscillators"),
    # --- explosive synchronization ---
    ("gomezgardenes2011", "02_explosive_sync", "Explosive synchronization transitions in scale-free networks"),
    ("leyva2012", "02_explosive_sync", "Explosive first-order transition to synchrony in networked chaotic oscillators"),
    ("peron2012", "02_explosive_sync", "Determination of the critical coupling of explosive synchronization transitions in scale-free networks by mean-field approximations"),
    ("coutinho2013", "02_explosive_sync", "Kuramoto model with frequency-degree correlations on complex networks"),
    ("ji2013", "02_explosive_sync", "Cluster explosive synchronization in complex networks"),
    ("leyva2013_weighted", "02_explosive_sync", "Explosive synchronization in weighted complex networks"),
    ("zou2014_basin", "02_explosive_sync", "Basin of attraction determines hysteresis in explosive synchronization"),
    ("zhang2015_adaptive", "02_explosive_sync", "Explosive synchronization in adaptive and multilayer networks"),
    ("boccaletti2016_review", "02_explosive_sync", "Explosive transitions in complex networks' structure and dynamics: percolation and synchronization"),
    ("danziger2019", "02_explosive_sync", "Dynamic interdependence and competition in multilayer networks"),
    ("dsouza2019_review", "02_explosive_sync", "Explosive phenomena in complex networks"),
    ("synchronization_bombs2022", "02_explosive_sync", "Self-organized explosive synchronization in complex networks: Emergence of synchronization bombs"),
    ("adaptive_competing2026", "02_explosive_sync", "Explosive Transitions in Complex Networks with Adaptive Competing Interactions"),
    ("typeI_neurons2025", "02_explosive_sync", "Explosive synchronization in networks of Type-I neurons with electrical synapses"),
    # --- higher-order interactions ---
    ("skardal_arenas2019", "03_higher_order", "Abrupt desynchronization and extensive multistability in globally coupled oscillator simplexes"),
    ("skardal_arenas2020", "03_higher_order", "Higher order interactions in complex networks of phase oscillators promote abrupt synchronization switching"),
    ("millan2020", "03_higher_order", "Explosive higher-order Kuramoto dynamics on simplicial complexes"),
    ("battiston2020", "03_higher_order", "Networks beyond pairwise interactions: structure and dynamics"),
    ("zhang2023_hyper_vs_simplicial", "03_higher_order", "Higher-order interactions shape collective dynamics differently in hypergraphs and simplicial complexes"),
    ("hoi_random_hypergraphs2025", "03_higher_order", "When higher-order interactions enhance synchronization: the case of the Kuramoto model on random hypergraphs"),
    # --- inertia / power grids (cascading desync analogues) ---
    ("rohden2012", "04_inertia_cascades", "Self-organized synchronization in decentralized power grids"),
    ("olmi2014", "04_inertia_cascades", "Hysteretic transitions in the Kuramoto model with inertia"),
    ("multiplex_inertia2017", "04_inertia_cascades", "Multiplexing induced explosive synchronization in Kuramoto oscillators with inertia"),
    # --- early warning signals & tipping ---
    ("scheffer2009", "05_early_warning", "Early-warning signals for critical transitions"),
    ("dakos2012_methods", "05_early_warning", "Methods for detecting early warnings of critical transitions in time series illustrated using simulated ecological data"),
    ("ashwin2012_tipping", "05_early_warning", "Tipping points in open systems: bifurcation, noise-induced and rate-dependent examples in the climate system"),
    ("kuehn2011", "05_early_warning", "A mathematical framework for critical transitions: bifurcations, fast-slow systems and stochastic dynamics"),
    ("hagstrom2021_phase_transitions_ews", "05_early_warning", "Phase Transitions and the Theory of Early Warning Indicators for Critical Transitions"),
    ("kato_masuda2026", "05_early_warning", "Early warning signals for synchronization transitions from partial observations"),
    ("ordinal_predictors2025", "05_early_warning", "Local predictors of explosive synchronization with ordinal methods"),
    ("forced_desync_ews2020", "05_early_warning", "Early warning signals for desynchronization in periodically forced systems"),
    ("bury2021_deep_learning_ews", "05_early_warning", "Deep learning for early warning signals of tipping points"),
    ("persistent_entropy2026", "05_early_warning", "Persistent Entropy as a Detector of Phase Transitions"),
    ("pseudo_coherence2026", "05_early_warning", "Pseudo-Coherence and Stochastic Synchronization: A Non-Normal Route to Collective Dynamics without Oscillators"),
    # --- finance: crashes, flash crashes, collective panic ---
    ("sornette2003_crashes", "06_finance", "Critical market crashes"),
    ("harmon2011_panic", "06_finance", "Anticipating economic market crises using measures of collective panic"),
    ("filimonov_sornette2012", "06_finance", "Quantifying reflexivity in financial markets: towards a prediction of flash crashes"),
    ("flash_crash_contagion2018", "06_finance", "Understanding Flash Crash Contagion and Systemic Risk: A Micro-Macro Agent-Based Approach"),
    ("johnson2013_ultrafast", "06_finance", "Abrupt rise of new machine ecology beyond human response time"),
    ("haldane_may2011", "06_finance", "Systemic risk in banking ecosystems"),
    ("bouchaud2010_endogenous", "06_finance", "The endogenous dynamics of markets: price impact and feedback loops"),
    ("synchronization_market_asymmetry2006", "06_finance", "Synchronization Model for Stock Market Asymmetry"),
    # --- novelty checks: resilience to targeted shocks, resetting, R-tipping ---
    ("mitra2018_nodal_robustness", "07_novelty_checks", "Identifying nodal properties that are crucial for the dynamical robustness of multi-stable networks"),
    ("resilience_targeted_attacks2019", "07_novelty_checks", "Resilience of networks of multi-stable chaotic systems to targetted attacks"),
    ("subsystem_resetting_kuramoto2026", "07_novelty_checks", "Analytical approach to subsystem resetting in generalized Kuramoto models"),
]


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", text.lower())).strip()


STOPWORDS = {"the", "and", "for", "from", "with", "into", "its", "are", "via", "beyond", "towards", "when", "case"}

# Candidate IDs (verified against the requested title before download; a wrong hint is simply rejected).
ID_HINTS: dict[str, str] = {
    "rodrigues2016": "1511.07139",
    "gomezgardenes2011": "1010.0960",
    "leyva2012": "1111.1813",
    "peron2012": "1205.1297",
    "coutinho2013": "1303.3298",
    "zhang2015_adaptive": "1407.6453",
    "boccaletti2016_review": "1606.04779",
    "danziger2019": "1705.00241",
    "synchronization_bombs2022": "2203.03728",
    "adaptive_competing2026": "2606.17944",
    "typeI_neurons2025": "2512.03600",
    "skardal_arenas2019": "1903.09946",
    "millan2020": "1912.04405",
    "battiston2020": "2006.01764",
    "zhang2023_hyper_vs_simplicial": "2203.03060",
    "hoi_random_hypergraphs2025": "2508.10992",
    "rohden2012": "1206.5591",
    "olmi2014": "1403.5429",
    "multiplex_inertia2017": "1711.10679",
    "dakos2012_methods": "1204.3389",
    "ashwin2012_tipping": "1103.0169",
    "kuehn2011": "1101.2899",
    "hagstrom2021_phase_transitions_ews": "2110.12287",
    "kato_masuda2026": "2608.28320",
    "ordinal_predictors2025": "2501.05202",
    "forced_desync_ews2020": "2003.11595",
    "bury2021_deep_learning_ews": "2106.10269",
    "pseudo_coherence2026": "2603.07206",
    "harmon2011_panic": "1102.2620",
    "flash_crash_contagion2018": "1805.08454",
    "johnson2013_ultrafast": "1309.3869",
    "bouchaud2010_endogenous": "1009.2928",
    "pazo2005": "cond-mat/0412456",
    "dorfler_bullo2014": "1402.2593",
    "acebron2005": "cond-mat/0506113",
    "synchronization_market_asymmetry2006": "physics/0604137",
    "pikovsky2016_common_noise": "1607.06383",
    "peter2019_microscopic_crosscorr": "1901.02779",
    "nagai2010_common_noise_sync": "1005.2833",
    "martens2009_bimodal": "0809.2129",
    "ichinomiya2004": "cond-mat/0403628",
    "restrepo2005_onset": "cond-mat/0411202",
    "bertini2010_noisy_kuramoto": "1001.2314",
    "mitra2018_nodal_robustness": "1801.02409",
    "resilience_targeted_attacks2019": "1911.08465",
    "subsystem_resetting_kuramoto2026": "2604.04769",
}


def _query(params: dict[str, object]) -> list[tuple[str, str, str]]:
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as resp:
        root = ET.fromstring(resp.read())
    time.sleep(3.1)  # arXiv API etiquette
    out = []
    for entry in root.findall(f"{ATOM}entry"):
        found = " ".join((entry.findtext(f"{ATOM}title") or "").split())
        arxiv_id = (entry.findtext(f"{ATOM}id") or "").rsplit("/abs/", 1)[-1]
        author = entry.findtext(f"{ATOM}author/{ATOM}name") or ""
        if found and found != "Error":
            out.append((arxiv_id, found, author))
    return out


def _best(title: str, cands: list[tuple[str, str, str]]) -> tuple[str, str, str] | None:
    scored = [(difflib.SequenceMatcher(None, norm(title), norm(c[1])).ratio(), c) for c in cands]
    scored = [x for x in scored if x[0] >= 0.85]
    return max(scored, key=lambda x: x[0])[1] if scored else None


def search_arxiv(title: str, key: str = "") -> tuple[str, str, str] | None:
    """Return (arxiv_id, found_title, first_author) for a verified title match, or None."""
    if key in ID_HINTS:
        hit = _best(title, _query({"id_list": ID_HINTS[key]}))
        if hit:
            return hit
    words = [w for w in norm(title.replace("-", " ")).split() if len(w) > 2 and w not in STOPWORDS]
    for field, n_words in (("ti", 6), ("all", 8)):
        query = " AND ".join(f"{field}:{w}" for w in words[:n_words])
        hit = _best(title, _query({"search_query": query, "max_results": 25}))
        if hit:
            return hit
    return None


def main() -> None:
    rows = []
    for key, folder, title in PAPERS:
        out_dir = HERE / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        existing = list(out_dir.glob(f"{key}__*.pdf"))
        if existing:
            print(f"[skip] {key}")
            rows.append({"key": key, "folder": folder, "title": title, "status": "present",
                         "arxiv_id": existing[0].stem.split("__")[-1].replace("_", "/"), "first_author": ""})
            continue
        try:
            hit = search_arxiv(title, key)
        except Exception as exc:  # network / parse errors: record and move on
            print(f"[error] {key}: {exc}")
            rows.append({"key": key, "folder": folder, "title": title, "status": f"error: {exc}",
                         "arxiv_id": "", "first_author": ""})
            continue
        if hit is None:
            print(f"[not on arXiv] {key}: {title}")
            rows.append({"key": key, "folder": folder, "title": title, "status": "not found on arXiv",
                         "arxiv_id": "", "first_author": ""})
            continue
        arxiv_id, found_title, author = hit
        base_id = re.sub(r"v\d+$", "", arxiv_id)
        pdf_path = out_dir / f"{key}__{base_id.replace('/', '_')}.pdf"
        try:
            req = urllib.request.Request(f"https://arxiv.org/pdf/{base_id}", headers=UA)
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
            if not data.startswith(b"%PDF"):
                raise ValueError("response is not a PDF")
            pdf_path.write_bytes(data)
            time.sleep(3.1)
            print(f"[ok] {key} -> {pdf_path.name}  ({found_title})")
            rows.append({"key": key, "folder": folder, "title": found_title, "status": "downloaded",
                         "arxiv_id": base_id, "first_author": author})
        except Exception as exc:
            print(f"[error] {key}: {exc}")
            rows.append({"key": key, "folder": folder, "title": found_title, "status": f"error: {exc}",
                         "arxiv_id": base_id, "first_author": author})

    with open(HERE / "index.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["key", "folder", "arxiv_id", "first_author", "status", "title"])
        writer.writeheader()
        writer.writerows(rows)
    ok = sum(r["status"] in ("downloaded", "present") for r in rows)
    print(f"\n{ok}/{len(rows)} papers available locally; see references/index.csv")


if __name__ == "__main__":
    main()
