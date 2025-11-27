import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess

# --------------------------------------------------------------
# Select case
# --------------------------------------------------------------
cases = [d for d in os.listdir(".") if os.path.isdir(d) and "case" in d]

print("Available cases:")
for i, c in enumerate(cases):
    print(i, c)

idx = int(input("Select case index: "))
case = cases[idx]

case_dir = case                      # data lives directly here
x_file = os.path.join(case_dir, "mesh.txt")

# --------------------------------------------------------------
# Output folder videos_n
# --------------------------------------------------------------
out_root = "videos"
os.makedirs(out_root, exist_ok=True)

n = 0
while True:
    out_dir = os.path.join(out_root, f"videos_{n}")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        break
    n += 1

# --------------------------------------------------------------
duration_s = 10
max_frames = 200
dpi = 100

x = np.loadtxt(x_file)

# All .txt files in the case directory except mesh
y_files = [
    f for f in os.listdir(case_dir)
    if f.endswith(".txt") and f.lower() != "mesh.txt"
]

for fname in y_files:
    path = os.path.join(case_dir, fname)
    print(f"Processing {fname}...")

    y = np.loadtxt(path)
    if y.ndim == 1:
        print("  Skipped (not matrix)")
        continue

    n_frames_total, n_points = y.shape
    if n_frames_total > max_frames:
        step = int(np.ceil(n_frames_total / max_frames))
        y = y[::step, :]
        n_frames = y.shape[0]
    else:
        n_frames = n_frames_total

    fps = n_frames / duration_s

    tmp_dir = os.path.join(out_dir, "_tmp_" + os.path.splitext(fname)[0])
    os.makedirs(tmp_dir, exist_ok=True)

    for i in range(n_frames):
        plt.figure(figsize=(7, 4))
        plt.plot(x, y[i, :], lw=2)
        plt.xlim(x.min(), x.max())
        plt.ylim(np.min(y), np.max(y))
        plt.title(f"{fname} (Frame {i+1}/{n_frames})")
        plt.xlabel("Axial length [m]")
        plt.ylabel("Value")
        plt.grid(True)
        out_png = os.path.join(tmp_dir, f"frame_{i:05d}.png")
        plt.savefig(out_png, dpi=dpi)
        plt.close()

    out_mp4 = os.path.join(out_dir, os.path.splitext(fname)[0] + ".mp4")
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-framerate", f"{fps:.2f}",
        "-i", os.path.join(tmp_dir, "frame_%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        out_mp4
    ]
    subprocess.run(ffmpeg_cmd, check=True)

    for f in os.listdir(tmp_dir):
        os.remove(os.path.join(tmp_dir, f))
    os.rmdir(tmp_dir)

    print("  Saved ->", out_mp4)

print("Done.")
