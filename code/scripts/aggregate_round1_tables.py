from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from battery_thesis.results import aggregate_round1_tables


def main() -> None:
    parser = argparse.ArgumentParser(description='Aggregate round-one experiment summaries into results/tables/*.csv.')
    parser.add_argument('--results-root', required=True)
    args = parser.parse_args()

    outputs = aggregate_round1_tables(Path(args.results_root))
    for name, path in outputs.items():
        print(f'{name}: {path}')


if __name__ == '__main__':
    main()
