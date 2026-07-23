from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def run_step(script_name: str) -> None:
    subprocess.run([sys.executable, str(BASE_DIR / script_name)], check=True, cwd=BASE_DIR)


def main() -> None:
    for script_name in ["main.py", "analysis.py", "strategy.py"]:
        run_step(script_name)
    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
