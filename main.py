"""
Tatar OCR - Main CLI Entrypoint
"""

import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Tatar OCR Dataset & Augmentation Engine")
    parser.add_argument("--generate", action="store_true", help="Generate full 15k-20k synthetic dataset to disk")
    parser.add_argument("--samples", type=int, default=180, help="Samples per class (default: 180, ~18.3k images)")
    parser.add_argument("--out-dir", type=str, default="dataset/tatar_hw_dataset", help="Dataset destination folder")
    parser.add_argument("--img-size", type=int, default=64, help="Canvas resolution (default: 64)")
    parser.add_argument("--test", action="store_true", help="Run verification tests")
    args = parser.parse_args()

    if args.generate:
        from tatar_ocr_dataset import generate_batch_dataset_to_disk
        generate_batch_dataset_to_disk(
            out_dir=args.out_dir,
            samples_per_class=args.samples,
            img_size=args.img_size,
        )
    elif args.test:
        from test_pipeline import run_tests
        run_tests()
    else:
        print("Tatar OCR («Кара куян») Pipeline")
        print("Usage:")
        print("  - To launch interactive Marimo UI: uv run marimo run mp.py (or marimo edit mp.py)")
        print("  - To generate 15k-20k dataset:      python main.py --generate --samples 180")
        print("  - To run tests:                    python main.py --test")


if __name__ == "__main__":
    main()
