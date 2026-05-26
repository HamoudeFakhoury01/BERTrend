#!/usr/bin/env python3
"""
Wire an existing .xlsx into the prospective_demo so it can be analyzed there.

The prospective_demo has no upload button: it reads a configured "feed" whose
data files (JSONL) are normally produced by a backend scheduler. This script
fakes that setup from a one-shot xlsx:

  1. writes the feed config TOML       (<CONFIG>/users/<user>/<model>_feed.toml)
  2. writes the analysis config TOML   (<CONFIG>/users/<user>/<model>_analysis.toml)
  3. converts the xlsx -> JSONL in the feed data dir, named so the pipeline's
     glob `*<feed_id>*.jsonl*` finds it.

Then, on the pod:
  python -m bertrend.bertrend_apps.prospective_demo.process_new_data train-new-model <user> <model>
  bash scripts/pod.sh prospective        # login admin / bertrend

Usage:
  python scripts/setup_prospective.py /path/to/cpts_2026.xlsx [--user admin] [--model cpts]
"""
import argparse
from pathlib import Path

import pandas as pd

# Resolve the live paths from the installed bertrend (correct on any host).
from bertrend import CONFIG_PATH, FEED_BASE_PATH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx", help="Path to the source .xlsx (text + timestamp columns)")
    ap.add_argument("--user", default="admin")
    ap.add_argument("--model", default="cpts", help="model_id / feed name")
    ap.add_argument("--granularity", type=int, default=30)
    ap.add_argument("--window", type=int, default=30)
    args = ap.parse_args()

    feed_id = f"feed_{args.model}"
    feed_rel = f"users/{args.user}/{feed_id}"           # relative to FEED_BASE_PATH
    feed_dir = FEED_BASE_PATH / "users" / args.user / feed_id
    cfg_dir = CONFIG_PATH / "users" / args.user
    feed_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)

    # 1. feed config -------------------------------------------------------
    feed_toml = cfg_dir / f"{args.model}_feed.toml"
    feed_toml.write_text(
        "[data-feed]\n"
        f'id = "{feed_id}"\n'
        f'feed_dir_path = "{feed_rel}"\n'
        'query = "manual upload"\n'
        'provider = "rss"\n'
        "max_results = 1000\n"
        f"number_of_days = 3650\n"
        'language = "fr"\n'
        'update_frequency = "1 0 * * 1"\n'
        "evaluate_articles_quality = false\n",
        encoding="utf-8",
    )

    # 2. analysis config ---------------------------------------------------
    analysis_toml = cfg_dir / f"{args.model}_analysis.toml"
    analysis_toml.write_text(
        "[model_config]\n"
        f"granularity = {args.granularity}\n"
        f"window_size = {args.window}\n"
        'language = "fr"\n'
        "split_by_paragraph = false\n\n"
        "[analysis_config]\n"
        "topic_evolution = true\n"
        "evolution_scenarios = true\n"
        "multifactorial_analysis = true\n\n"
        "[report_config]\n"
        "auto_send = false\n"
        "email_recipients = []\n"
        'report_title = "CPTS Val de Suippe — analyse"\n'
        "max_emerging_topics = 5\n"
        "max_strong_topics = 8\n",
        encoding="utf-8",
    )

    # 3. xlsx -> JSONL in the feed dir ------------------------------------
    df = pd.read_excel(args.xlsx)
    if "text" not in df.columns or "timestamp" not in df.columns:
        raise SystemExit(
            f"xlsx must have 'text' and 'timestamp' columns; found {list(df.columns)}"
        )
    out_jsonl = feed_dir / f"{feed_id}_data.jsonl"
    df.to_json(out_jsonl, orient="records", lines=True, date_format="iso", force_ascii=False)

    print("=== prospective_demo setup done ===")
    print(f"  feed config     : {feed_toml}")
    print(f"  analysis config : {analysis_toml}")
    print(f"  data (jsonl)    : {out_jsonl}  ({len(df)} rows)")
    print()
    print("Next, on the pod:")
    print(f"  python -m bertrend.bertrend_apps.prospective_demo.process_new_data train-new-model {args.user} {args.model}")
    print("  bash scripts/pod.sh prospective     # login admin / bertrend")


if __name__ == "__main__":
    main()
