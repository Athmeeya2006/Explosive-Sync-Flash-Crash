"""Cut the repository's multi-panel figures into single panels for the talk, then make them black.

Nothing is redrawn: every panel is a crop of the original PNG (pixel boxes found from the white gaps
between panels), and the dark version only inverts the colours and rotates the hue by 180 degrees.
"""
import pathlib
from PIL import Image, ImageOps

REPO = pathlib.Path(__file__).resolve().parents[1]
FIG = REPO / "research" / "figures"
OUT = REPO / "presentation" / "assets"
OUT.mkdir(parents=True, exist_ok=True)


def dark(img):
    img = ImageOps.invert(img.convert("RGB"))
    h, s, v = img.convert("HSV").split()
    return Image.merge("HSV", (h.point(lambda x: (x + 128) % 256), s, v)).convert("RGB")


def side_by_side(*imgs, gap=30):
    h = max(i.height for i in imgs)
    out = Image.new("RGB", (sum(i.width for i in imgs) + gap * (len(imgs) - 1), h), "white")
    x = 0
    for i in imgs:
        out.paste(i, (x, (h - i.height) // 2))
        x += i.width + gap
    return out


def save(img, name):
    dark(img).save(OUT / name)
    print("wrote", OUT / name, img.size)


# S2: herding sweeps, top row = random (ER) network. Columns f = 0 ... 1, last = thresholds.
s2 = Image.open(FIG / "s2_herding_hysteresis.png").convert("RGB")
save(s2.crop((0, 90, 495, 728)), "panel_herding_f0.png")
save(s2.crop((1870, 90, 2300, 728)), "panel_herding_f1.png")

# S3: crash anatomy. Columns A, B, C; rows: coupling, R and Q, price, order-flow heatmap.
s3 = Image.open(FIG / "s3_crash_anatomy.png").convert("RGB")
for tag, (x0, x1) in {"A": (0, 868), "B": (868, 1676), "C": (1676, 2490)}.items():
    rq = s3.crop((x0, 575, x1, 930))
    price = s3.crop((x0, 985, x1, 1338))
    save(side_by_side(rq, price), f"panel_crash_{tag}.png")

# S5: circuit breakers, three panels.
s5 = Image.open(FIG / "s5_circuit_breakers.png").convert("RGB")
save(s5.crop((0, 100, 1059, 947)), "panel_halt_a.png")
save(s5.crop((1059, 100, 2107, 947)), "panel_halt_b.png")
save(s5.crop((2107, 100, 3174, 947)), "panel_halt_c.png")
