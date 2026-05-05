from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from battery_thesis.config import DELIVERABLES_ROOT, SAMPLES_ROOT, ensure_project_directories
from battery_thesis.documents import build_checkpoint_doc, build_experiment_doc
from battery_thesis.feature_catalog import FEATURE_ROWS


def main() -> None:
    ensure_project_directories()
    feature_dictionary_path = SAMPLES_ROOT / 'feature_dictionary.csv'
    with feature_dictionary_path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.writer(handle)
        writer.writerow(['feature_name', 'feature_group', 'fault_type', 'description_zh', 'unit'])
        writer.writerows(FEATURE_ROWS)

    build_experiment_doc(
        output_path=DELIVERABLES_ROOT / 'experiment_summary.docx',
        title='\u5b9e\u9a8c\u6458\u8981',
        sections=['\u5b9e\u9a8c\u8bbe\u8ba1', '\u7ed3\u679c\u5206\u6790', '\u56fe\u8868\u6e05\u5355'],
    )
    build_experiment_doc(
        output_path=DELIVERABLES_ROOT / 'method_summary.docx',
        title='\u65b9\u6cd5\u6458\u8981',
        sections=['\u7edf\u4e00\u5b9e\u9a8c\u6570\u636e\u96c6', '\u7279\u5f81\u5de5\u7a0b', '\u5171\u4eab\u7f16\u7801\u5668\u4e0e\u4e13\u5bb6\u5934'],
    )
    build_experiment_doc(
        output_path=DELIVERABLES_ROOT / 'innovation_summary.docx',
        title='\u521b\u65b0\u70b9\u6458\u8981',
        sections=['\u6570\u636e\u6784\u5efa\u521b\u65b0', '\u7279\u5f81\u8bbe\u8ba1\u521b\u65b0', '\u53cc\u4efb\u52a1\u6846\u67b6\u521b\u65b0'],
    )
    build_checkpoint_doc(
        output_path=DELIVERABLES_ROOT / 'implementation_checkpoint.docx',
        title='\u5b9e\u65bd\u68c0\u67e5\u70b9',
        sections=[
            (
                '\u5f53\u524d\u5b8c\u6210\u60c5\u51b5',
                [
                    '\u9879\u76ee\u865a\u62df\u73af\u5883\u5df2\u5b8c\u6210\u6838\u5fc3\u4f9d\u8d56\u5b89\u88c5\uff0c\u5305\u542b numpy\u3001pandas\u3001matplotlib\u3001scikit-learn\u3001lightgbm \u4e0e torch\u3002',
                    '\u4e2d\u7b49\u89c4\u6a21\u6837\u672c\u6784\u5efa\u3001LightGBM\u3001LSTM\u3001Transformer\u3001\u4e3b\u6a21\u578b\u4e0e\u4e00\u7ec4\u6d88\u878d\u7684\u672c\u5730\u53ef\u8fd0\u884c\u9a8c\u8bc1\u5df2\u5b8c\u6210\u3002',
                    '\u8bc6\u522b\u3001\u9884\u8b66\u3001\u6d88\u878d\u4e0e\u8bad\u7ec3\u66f2\u7ebf\u56fe\u5df2\u5bfc\u51fa\u4e3a\u6570\u636e\u70b9\u3001\u811a\u672c\u3001PNG\u3001SVG \u56db\u4ef6\u5957\u3002',
                ],
            ),
            (
                '\u5f53\u524d\u9650\u5236',
                [
                    '\u672c\u5730\u9a8c\u8bc1\u4f7f\u7528\u7684\u662f\u53d7\u9650\u8f66\u8f86\u5b50\u96c6\uff0c\u4e3b\u8981\u7528\u4e8e\u9a8c\u8bc1\u94fe\u8def\uff0c\u4e0d\u4f5c\u4e3a\u8bba\u6587\u6700\u7ec8\u6307\u6807\u3002',
                    '\u7b2c 1 \u8f6e\u6b63\u5f0f\u5b9e\u9a8c\u4ecd\u5efa\u8bae\u5728 Linux \u5355\u5361\u670d\u52a1\u5668\u4e0a\u8fd0\u884c\u5b8c\u6574 Data_set \u4e2d\u7b49\u89c4\u6a21\u6837\u672c\u6784\u5efa\u4e0e\u8bad\u7ec3\u3002',
                    'RAW_DATA \u6682\u4e0d\u5e76\u5165\u7b2c 1 \u8f6e\u8bad\u7ec3\uff0c\u540e\u7eed\u4f5c\u4e3a\u5916\u90e8\u9a8c\u8bc1\u6216\u8865\u5f3a\u5b9e\u9a8c\u4f7f\u7528\u3002',
                ],
            ),
            (
                '\u4e0b\u4e00\u6b65\u5efa\u8bae',
                [
                    '\u6309 deliverables/server_runbook.md \u5728\u670d\u52a1\u5668\u4e0a\u6267\u884c\u7b2c 1 \u8f6e\u6b63\u5f0f\u5b9e\u9a8c\u3002',
                    '\u5b8c\u6210\u670d\u52a1\u5668\u7ed3\u679c\u540e\uff0c\u57fa\u4e8e\u7b2c 1 \u8f6e\u7ed3\u679c\u7b5b\u9009\u548c\u52a0\u5f3a\u7279\u5f81\uff0c\u8fdb\u5165\u7b2c 2 \u8f6e\u8bba\u6587\u4e3b\u7ed3\u679c\u5b9e\u9a8c\u3002',
                ],
            ),
        ],
    )


if __name__ == '__main__':
    main()
