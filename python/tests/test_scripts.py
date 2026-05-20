import sys
from unittest.mock import patch


from analysis.early_warning import main as ew_main
from analysis.finite_size_scaling import main as fss_main
from analysis.flash_crash_sim import main as fcs_main
from analysis.generate_networks import main as gn_main
from analysis.hysteresis_sweep import main as hs_main
from analysis.lyapunov_analysis import main as la_main
from analysis.run_kuramoto import main as rk_main
from analysis.run_stuart_landau import main as rsl_main

from plots.plot_degree_distribution import main as pdd_main
from plots.plot_early_warning import main as pew_main
from plots.plot_finite_size import main as pfs_main
from plots.plot_flash_crash import main as pfc_main
from plots.plot_order_param import main as pop_main
from plots.plot_percolation import main as pperc_main
from plots.plot_phase_diagram import main as ppd_main


def test_run_kuramoto(tmp_path):
    out = tmp_path / "k.csv"
    test_args = ["prog", "--n", "10", "--k", "1.0", "--tmax", "0.5", "--dt", "0.1", "--out", str(out)]
    with patch.object(sys, "argv", test_args):
        rk_main()
    assert out.exists()

def test_run_stuart_landau(tmp_path):
    out = tmp_path / "sl.csv"
    test_args = ["prog", "--n", "10", "--k", "1.0", "--tmax", "0.5", "--dt", "0.1", "--out", str(out)]
    with patch.object(sys, "argv", test_args):
        rsl_main()
    assert out.exists()

def test_flash_crash_sim(tmp_path):
    out = tmp_path / "fc.csv"
    test_args = ["prog", "--n", "10", "--k-high", "1.0", "--k-low", "0.5", "--t-drop", "0.2", "--tmax", "0.5", "--dt", "0.1", "--out", str(out)]
    with patch.object(sys, "argv", test_args):
        fcs_main()
    assert out.exists()

def test_finite_size_scaling(tmp_path):
    out = tmp_path / "fss.csv"
    test_args = ["prog", "--n-list", "10", "--k-list", "0.5", "--tmax", "0.5", "--dt", "0.1", "--n-replicas", "1", "--out", str(out)]
    with patch.object(sys, "argv", test_args):
        fss_main()
    assert out.exists()

def test_hysteresis_sweep(tmp_path):
    out = tmp_path / "hs.csv"
    test_args = ["prog", "--n", "10", "--k-min", "0.0", "--k-max", "1.0", "--k-steps", "2", "--tmax", "0.5", "--dt", "0.1", "--out", str(out)]
    with patch.object(sys, "argv", test_args):
        hs_main()
    assert out.exists()

def test_lyapunov_analysis(tmp_path):
    out = tmp_path / "lyap.csv"
    test_args = ["prog", "--n", "10", "--k-min", "0.0", "--k-max", "1.0", "--k-steps", "2", "--tmax", "0.5", "--t-transient", "0.2", "--dt", "0.1", "--out", str(out)]
    with patch.object(sys, "argv", test_args):
        la_main()
    assert out.exists()

def test_early_warning(tmp_path):
    # Need to generate input data first
    in_file = tmp_path / "in.csv"
    out_file = tmp_path / "out.csv"
    test_args_gen = ["prog", "--n", "10", "--tmax", "1.0", "--dt", "0.1", "--out", str(in_file)]
    with patch.object(sys, "argv", test_args_gen):
        rk_main()

    test_args = ["prog", "--input", str(in_file), "--window", "2", "--out", str(out_file)]
    with patch.object(sys, "argv", test_args):
        ew_main()
    assert out_file.exists()

def test_generate_networks(tmp_path):
    test_args = ["prog", "--model", "er", "--n", "10", "--p", "0.5", "--out", str(tmp_path)]
    with patch.object(sys, "argv", test_args):
        gn_main()
    assert (tmp_path / "er_n10.npy").exists()

def test_plots(tmp_path):
    # Generate data
    k_out = tmp_path / "k.csv"
    with patch.object(sys, "argv", ["prog", "--n", "10", "--tmax", "0.5", "--out", str(k_out)]):
        rk_main()

    ew_out = tmp_path / "ew.csv"
    with patch.object(sys, "argv", ["prog", "--input", str(k_out), "--window", "2", "--out", str(ew_out)]):
        ew_main()

    fss_out = tmp_path / "fss.csv"
    with patch.object(sys, "argv", ["prog", "--n-list", "10", "--k-list", "0.5", "--n-replicas", "1", "--tmax", "0.5", "--out", str(fss_out)]):
        fss_main()

    fc_out = tmp_path / "fc.csv"
    with patch.object(sys, "argv", ["prog", "--n", "10", "--tmax", "0.5", "--out", str(fc_out)]):
        fcs_main()

    # Test plot scripts
    p_k = tmp_path / "pk.png"
    with patch.object(sys, "argv", ["prog", "--input", str(k_out), "--out", str(p_k)]):
        pop_main()
    assert p_k.exists()

    p_ew = tmp_path / "pew.png"
    with patch.object(sys, "argv", ["prog", "--input", str(ew_out), "--out", str(p_ew)]):
        pew_main()
    assert p_ew.exists()

    p_fss = tmp_path / "pfss.png"
    with patch.object(sys, "argv", ["prog", "--input", str(fss_out), "--out", str(p_fss)]):
        pfs_main()
    assert p_fss.exists()

    p_pd = tmp_path / "ppd.png"
    with patch.object(sys, "argv", ["prog", "--input", str(fss_out), "--out", str(p_pd)]):
        ppd_main()
    assert p_pd.exists()

    p_fc = tmp_path / "pfc.png"
    with patch.object(sys, "argv", ["prog", "--input", str(fc_out), "--out", str(p_fc)]):
        pfc_main()
    assert p_fc.exists()

    p_dd = tmp_path / "pdd.png"
    with patch.object(sys, "argv", ["prog", "--n", "100", "--out", str(p_dd)]):
        pdd_main()
    assert p_dd.exists()

    p_perc = tmp_path / "pperc.png"
    with patch.object(sys, "argv", ["prog", "--n", "50", "--p-steps", "2", "--n-replicas", "2", "--out", str(p_perc)]):
        pperc_main()
    assert p_perc.exists()
