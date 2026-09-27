"""Build the final submission zip in the required structure.

    python make_package.py --team "<team_name>"

<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       ├── requirements.txt
│       └── reference/token_maps.json   # exact vocabulary maps of the submitted results
└── Documentation_template.md
"""
import argparse
import zipfile
from pathlib import Path

from config import OUTPUT, ROOT

CODE = ROOT / "code" / "business_entity_resolution"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True)
    args = ap.parse_args()
    zpath = ROOT / f"{args.team}_submission.zip"

    files = [(OUTPUT / "matching_results.tsv", "output/matching_results.tsv"),
             (OUTPUT / "candidate_pairs.tsv", "output/candidate_pairs.tsv"),
             (CODE / "README.md", "code/business_entity_resolution/README.md"),
             (CODE / "requirements.txt", "code/business_entity_resolution/requirements.txt"),
             (CODE / "reference" / "token_maps.json", "code/business_entity_resolution/reference/token_maps.json"),
             (ROOT / "Documentation_template.md", "Documentation_template.md")]
    files += [(p, f"code/business_entity_resolution/src/{p.name}") for p in sorted((CODE / "src").glob("*.py"))]
    missing = [str(src) for src, _ in files if not Path(src).exists()]
    if missing:
        raise SystemExit(f"missing files: {missing}")

    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for src, arc in files:
            z.write(src, arc)
    with zipfile.ZipFile(zpath) as z:
        for info in z.infolist():
            print(f"  {info.filename:<60} {info.file_size / 1e6:9.1f} MB")
    print(f"wrote {zpath} ({zpath.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
