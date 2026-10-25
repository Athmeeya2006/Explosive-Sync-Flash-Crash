"""Black-background versions of existing figures, for the talk.

The data and the plots are unchanged: the colours are inverted and the hue is rotated by 180 degrees,
so white becomes black, black text becomes white, and the blue and orange curves keep their colours.
"""
import pathlib
from PIL import Image, ImageOps

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT = REPO / "presentation" / "assets"
OUT.mkdir(parents=True, exist_ok=True)

FIGURES = {
    "fig_sync_dark.png": REPO / "legacy" / "figures" / "order_param.png",
    "fig_herding_dark.png": REPO / "research" / "figures" / "s2_herding_hysteresis.png",
    "fig_fold_dark.png": REPO / "research" / "figures" / "p1_fold_coefficient.png",
    "fig_crash_anatomy_dark.png": REPO / "research" / "figures" / "s3_crash_anatomy.png",
    "fig_crash_phase_dark.png": REPO / "research" / "figures" / "s4_crash_phase_diagram.png",
    "fig_breakers_dark.png": REPO / "research" / "figures" / "s5_circuit_breakers.png",
    "fig_comovement_dark.png": REPO / "research" / "figures" / "s9_observable_mapping.png",
    "fig_flashcrash_dark.png": REPO / "research" / "empirical" / "figures" / "e6_flash_crash.png",
    "fig_minute_dark.png": REPO / "research" / "empirical" / "figures" / "e7_minute_event_study.png",
    "fig_realnet_dark.png": REPO / "research" / "empirical" / "figures" / "e5_real_network.png",
}

for name, src in FIGURES.items():
    img = ImageOps.invert(Image.open(src).convert("RGB"))
    h, s, v = img.convert("HSV").split()
    h = h.point(lambda x: (x + 128) % 256)
    Image.merge("HSV", (h, s, v)).convert("RGB").save(OUT / name)
    print("wrote", OUT / name)
