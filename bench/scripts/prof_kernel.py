"""Details of one kernel in an exported nsys profile, for the last token.

  python prof_kernel.py <name> <kernel> [--db PATH.sqlite]

By default reads FNS_MODELS/prof/fns-<name>.sqlite (FNS_MODELS defaults to ROOT/models);
--db gives the exported nsys sqlite file explicitly.

Prints each launch: duration, grid, registers, and shared memory.
"""

import argparse
import os
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS = Path(os.environ["FNS_MODELS"]) if os.environ.get("FNS_MODELS") else ROOT / "models"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="profile name: reads fns-<name>.sqlite")
    ap.add_argument("kernel", help="kernel name, e.g. k_attention_part")
    ap.add_argument("--db", type=Path, default=None, help="explicit path to the exported nsys sqlite file")
    o = ap.parse_args()
    db = o.db if o.db is not None else MODELS / "prof" / f"fns-{o.name}.sqlite"
    cur = sqlite3.connect(str(db)).cursor()
    (embed_id,) = cur.execute("select id from StringIds where value like 'k_embed_q8_0%'").fetchone()
    (kid,) = cur.execute("select id from StringIds where value = ?", (o.kernel,)).fetchone()
    starts = [s for (s,) in cur.execute("select start from CUPTI_ACTIVITY_KIND_KERNEL where shortName = ? order by start",
                                        (embed_id,))]
    a, b = starts[-2], starts[-1]
    rows = cur.execute("select end - start, gridX, gridY, gridZ, registersPerThread, staticSharedMemory, dynamicSharedMemory "
                       "from CUPTI_ACTIVITY_KIND_KERNEL where shortName = ? and start >= ? and start < ? order by start",
                       (kid, a, b)).fetchall()
    for d, gx, gy, gz, reg, ss, ds in rows:
        print(f"{d / 1e3:8.1f} us  grid ({gx},{gy},{gz})  reg {reg}  smem {ss}+{ds}")
    print(f"total {sum(r[0] for r in rows) / 1e3:.1f} us over {len(rows)} launches")


if __name__ == "__main__":
    main()
