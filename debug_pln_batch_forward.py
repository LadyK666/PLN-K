from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models.pln_model import PLNModel  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser("PLN batch forward debug")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--image_size", type=int, default=448)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--pretrained_backbone", action="store_true")
    args = p.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    model = PLNModel(
        backbone_pretrained=args.pretrained_backbone,
        backbone_trainable=False,
        freeze_bn=True,
    ).to(device)
    model.eval()

    for B in [args.batch_size, min(32, args.batch_size), min(16, args.batch_size), min(8, args.batch_size)]:
        if B <= 0:
            continue
        x = torch.randn(B, 3, args.image_size, args.image_size, device=device)
        try:
            with torch.no_grad():
                outs = model(x)
        except RuntimeError as e:
            # Likely OOM. Try a smaller batch.
            print(f"Batch {B} failed: {e}")
            continue

        if not isinstance(outs, dict):
            raise TypeError(f"Expected dict output, got {type(outs)}")

        print(f"Batch {B} forward OK. Input={tuple(x.shape)}")
        for k, v in outs.items():
            print(f"  {k}: {tuple(v.shape)}")
        break


if __name__ == "__main__":
    main()

