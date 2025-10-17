import argparse
import threading
import time
import sys
import os
from itertools import cycle

import sd_mecha
from orbit import orbit

def _supports_tty():
    try:
        return sys.stdout.isatty()
    except Exception:
        return False

def _spinner_frames():
    if os.name == "nt":
        return ["o", "0", ".", "\\"]
    return ["⠋","⠙","⠸","⠴","⠦","⠇"]

def _animate_orbit(stop_event, label="orbit", extra=""):
    if not _supports_tty():
        return
    frames = cycle(_spinner_frames())
    letters = list(label)
    pos = 0
    up = "\x1b[1A"
    down = "\x1b[1B"
    clear = "\x1b[2K\r"
    while not stop_event.is_set():
        ch = next(frames)
        s = letters[:]
        s[pos % len(s)] = ch
        sys.stdout.write(up + clear + f"[{''.join(s)}]{(' ' + extra) if extra else ''}" + down)
        sys.stdout.flush()
        time.sleep(0.08)
        if ch in ("\\", "⠇"):
            pos += 1
    sys.stdout.write(up + clear + f"[{label}] done" + down + "\n")
    sys.stdout.flush()

def parse_args():
    p = argparse.ArgumentParser(description="ORBIT merge: inject B's orthogonal novelty into A.")
    p.add_argument("--modela", help="Path to base model A")
    p.add_argument("--modelb", help="Path to donor model B")
    p.add_argument("--output", required=True, help="Output path for merged model")

    # parameters
    p.add_argument("--alpha-par", type=float, default=0.25, help="Weight for parallel adjustment toward B (default: 0.25)")
    p.add_argument("--alpha-orth", type=float, default=0.50, help="Weight for orthogonal infusion from B (default: 0.50)")
    p.add_argument("--trust-k", type=float, default=3.0, help="MAD trust radius multiplier (default: 3.0)")
    p.add_argument("--eps", type=float, default=1e-8, help="Numerical epsilon (default: 1e-8)")
    p.add_argument("--coef-clip", type=float, default=8.0, help="Clamp for projection coefficient (default: 8.0; 0 disables)")

    return p.parse_args()

def main():
    args = parse_args()

    A = sd_mecha.model(args.modela)
    B = sd_mecha.model(args.modelb)

    recipe = orbit(
        A, B,
            alpha_par=args.alpha_par,
            alpha_orth=args.alpha_orth,
            trust_k=args.trust_k,
            eps=args.eps,
            coef_clip=args.coef_clip,
    )

    stop_event = threading.Event()
    extra = f"α∥={args.alpha_par:.2f} α⟂={args.alpha_orth:.2f} trust={args.trust_k:.1f}"
    spinner_thread = threading.Thread(target=_animate_orbit, args=(stop_event, "orbit", extra), daemon=True)
    spinner_thread.start()

    try:
        sd_mecha.merge(recipe, output=args.output)
    finally:
        stop_event.set()
        spinner_thread.join()
        sys.stdout.write(f"Saved → {args.output}\n")
        sys.stdout.flush()

if __name__ == "__main__":
    main()
