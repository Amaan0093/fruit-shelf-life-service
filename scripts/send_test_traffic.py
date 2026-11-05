"""
Send simulated traffic to the API, to test monitoring end to end.
--darken simulates a change such as poor lighting.

    python scripts/send_test_traffic.py https://YOUR_ADDRESS --limit 250
    python scripts/send_test_traffic.py https://YOUR_ADDRESS --limit 250 --darken 0.6
"""
import argparse
import sys
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
import pandas as pd
from PIL import Image, ImageEnhance
from tqdm import tqdm

from config import ARTIFACTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--limit", type=int, default=250)
    parser.add_argument("--darken", type=float, help="Brightness factor, such as 0.6")
    args = parser.parse_args()

    rows = pd.read_csv(ARTIFACTS / "val.csv").sample(args.limit, random_state=7)
    failures = 0
    with httpx.Client(base_url=args.url.rstrip("/"), timeout=300) as client:
        for row in tqdm(list(rows.itertuples()), desc="sending"):
            with Image.open(row.path) as img:
                img = img.convert("RGB")
                if args.darken:
                    img = ImageEnhance.Brightness(img).enhance(args.darken)
                buffer = BytesIO()
                img.save(buffer, format="PNG")
            response = client.post(
                "/predict",
                data={"fruit": row.fruit},
                files={"file": ("image.png", buffer.getvalue(), "image/png")},
            )
            failures += response.status_code != 200
    print(f"Sent {len(rows)} images, {failures} failed")


if __name__ == "__main__":
    main()