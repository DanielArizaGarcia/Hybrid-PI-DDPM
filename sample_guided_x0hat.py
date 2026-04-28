import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from tfm_shells.sampling.guided_x0hat import run_guided_sampling_x0hat


def main() -> None:
    parser = argparse.ArgumentParser(description="Run clean(x0-hat) physics-guided sampling")
    parser.add_argument("--config", default="configs/sample_guided_x0hat.yaml")
    args = parser.parse_args()
    run_guided_sampling_x0hat(Path(args.config))


if __name__ == "__main__":
    main()
