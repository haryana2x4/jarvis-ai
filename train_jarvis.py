#!/usr/bin/env python3
"""
JARVIS AI Training & Benchmark Script
=====================================
Trains JarvisNeuralModel on the core + extended multi-domain intent dataset,
evaluates held-out generalization accuracy, and optionally ingests local
folders, GitHub repositories, or web URLs into Jarvis's persistent brain.
"""

import argparse
import json
from typing import List, Tuple

from jarvis_ai import Jarvis, JarvisConfig, SEED_TRAINING_DATA, default_brain_dir


EXTENDED_SAMPLES: List[Tuple[str, str]] = [
    ("find python repositories with highest stars on github", "github_research"),
    ("inspect github repo huggingface/transformers", "github_research"),
    ("search git repo for whisper speech recognition", "github_research"),
    ("verify from github and web if fastapi supports async", "verify_fact"),
    ("cross check whether numpy vectorization beats python loops", "verify_fact"),
    ("check sources and verify if pytorch supports cuda 12", "verify_fact"),
    ("extract product prices and links from website", "twist_website"),
    ("twist webpage into downloadable csv spreadsheet", "twist_website"),
    ("scrape documentation headings and code blocks from url", "twist_website"),
    ("search the whole web for latest python ai frameworks", "web_search"),
    ("look up arxiv papers on multi agent reasoning", "web_search"),
    ("plan write and test a data parser script automatically", "autonomous_task"),
    ("research topic save report.md and verify facts", "autonomous_task"),
    ("execute python snippet to benchmark sorting", "code_exec"),
    ("run pytest across tests directory", "code_exec"),
    ("read config.json in current directory", "file_ops"),
    ("list files in project workspace", "file_ops"),
    ("check my ram usage and cpu cores", "system_control"),
    ("open vscode editor", "system_control"),
    ("compute (128 * 64) + 512", "math_calc"),
    ("train jarvis on my projects directory", "train_learn"),
    ("teach jarvis new voice phrase", "train_learn"),
    ("recall saved facts about field shift", "memory_recall"),
    ("listen for hey jarvis wake word", "voice_control"),
    ("hey jarvis how is everything running", "chat_general"),
]

HELD_OUT_EVAL: List[Tuple[str, str]] = [
    ("search github for autonomous agent repos", "github_research"),
    ("verify if duckdb supports parquet reading from multiple sources", "verify_fact"),
    ("twist https://example.com into csv table", "twist_website"),
    ("search internet for recent llama model benchmarks", "web_search"),
    ("build a script test it and save results.json on your own", "autonomous_task"),
    ("run shell command git status", "code_exec"),
    ("open package.json and read it", "file_ops"),
    ("show current cpu and memory telemetry", "system_control"),
    ("calculate 45 * 12 - 90", "math_calc"),
    ("remember this new fact and retrain", "train_learn"),
    ("search your memory for previous project notes", "memory_recall"),
    ("start listening to voice commands", "voice_control"),
    ("good evening jarvis", "chat_general"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Train & evaluate JARVIS AI")
    parser.add_argument("--brain-dir", default=default_brain_dir())
    parser.add_argument("--epochs", type=int, default=65)
    parser.add_argument("--folder", action="append", default=[], help="Local folder(s) to index")
    parser.add_argument("--github", action="append", default=[], help="GitHub repo(s) owner/name to ingest")
    parser.add_argument("--url", action="append", default=[], help="Web URL(s) to twist and ingest")
    args = parser.parse_args()

    jarvis = Jarvis(JarvisConfig(brain_dir=args.brain_dir))
    dataset = list(SEED_TRAINING_DATA) + EXTENDED_SAMPLES
    train_metrics = jarvis.neural.fit(dataset, epochs=args.epochs, lr=0.025, save=True)
    eval_metrics = jarvis.neural.evaluate(HELD_OUT_EVAL)

    extras = {}
    for folder in args.folder:
        extras.setdefault("folders", []).append(jarvis.memory.ingest_folder(folder))
    for repo in args.github:
        extras.setdefault("github", []).append(jarvis.learn_github_repo(repo))
    for u in args.url:
        res = jarvis.twist(u, output_format="markdown")
        extras.setdefault("urls", []).append({"url": u, "ok": res.get("ok"), "title": res.get("title")})

    summary = {
        "brain_dir": args.brain_dir,
        "train_metrics": train_metrics,
        "held_out_eval": eval_metrics,
        "memory_stats": jarvis.memory.stats(),
        "extras": extras,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
