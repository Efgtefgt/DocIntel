"""(Re)generate the synthetic sample dataset in data/samples.

Usage:  python scripts/generate_samples.py [output_dir]
Creates PNG images with matching .gt.txt ground truth, plus sample PDFs.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docintel.evaluation import synthetic_data as synth  # noqa: E402


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "data" / "samples"
    images = synth.write_benchmark_dataset(out)
    hard = synth.write_benchmark_dataset(out / "hard", suite="hard")
    print(f"Wrote {len(hard)} hard benchmark images to {out / 'hard'}")
    holdout = synth.write_benchmark_dataset(out / "holdout", suite="holdout")
    print(f"Wrote {len(holdout)} held-out images (never used for tuning) to {out / 'holdout'}")
    fresh = synth.write_benchmark_dataset(out / "holdout2", suite="holdout2")
    print(f"Wrote {len(fresh)} fresh held-out images to {out / 'holdout2'}")
    final = synth.write_benchmark_dataset(out / "holdout3", suite="holdout3")
    print(f"Wrote {len(final)} final held-out images to {out / 'holdout3'}")
    documents = synth.write_benchmark_dataset(out / "documents", suite="documents")
    print(f"Wrote {len(documents)} typed documents (document intelligence) to {out / 'documents'}")
    report = synth.render_text_image(synth.BENCHMARK_SAMPLES["incident_report"]["lines"])
    (out / "pdf").mkdir(exist_ok=True)
    (out / "pdf" / "digital.pdf").write_bytes(synth.make_digital_pdf(["Page one of a digital PDF.\nIt has embedded text.", "Page two.\nTotal: 1,250.00"]))
    (out / "pdf" / "scanned.pdf").write_bytes(synth.make_scanned_pdf([report]))
    (out / "pdf" / "mixed.pdf").write_bytes(synth.make_mixed_pdf("This first page contains embedded digital text.", report))
    (out / "pdf" / "table.pdf").write_bytes(synth.make_table_pdf([["Item", "Qty", "Price"], ["Apple", "3", "1.20"], ["Banana", "", "0.50"]]))
    synth.render_table_image([["Item", "Qty", "Price"], ["Apple", "3", "1.20"], ["Banana", "", "0.50"]]).save(out / "pdf" / "table_scan.png")
    print(f"Wrote {len(images)} benchmark images and 5 extra samples to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
