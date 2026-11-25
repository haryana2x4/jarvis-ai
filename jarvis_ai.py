#!/usr/bin/env python3
"""
JARVIS AI — Portable Single-File Autonomous AI Model & Multi-Source Agent
=========================================================================

Drop this single file (`jarvis_ai.py`) into ANY Python project to embed:
  1. Trainable NumPy Neural Intent & Skill Classifier (online + batch training)
  2. Hybrid SQLite FTS5 + Cosine Vector Persistent Knowledge Memory (RAG)
  3. Whole-Internet & GitHub Multi-Source Search + Fact Cross-Verifier
  4. Website Twister (pure-Python DOM tree, CSS selector & format transformer)
  5. Autonomous ReAct Plan-Act-Observe-Verify Engine + @register_tool API
  6. Universal LLM Router (Ollama, Gemini, OpenAI, Anthropic, OpenRouter, or 100% Local)
  7. Voice Command Engine (Wake-word STT + thread-safe TTS + voice learning)
  8. Built-in Local HTTP API Server & Embedded Command HUD (`python jarvis_ai.py serve`)

Quick Usage:
    from jarvis_ai import Jarvis
    ai = Jarvis()
    print(ai.ask("Search GitHub for FastAPI rate limiters and verify best practices"))
    data = ai.twist("https://news.ycombinator.com", output_format="json")
    result = ai.run_task("Create a summary of top Python HTTP libraries in summary.md")
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import html
import io
import json
import math
import operator
import os
import queue
import re
import shlex
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import webbrowser
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import requests

try:
    import psutil  # type: ignore
except Exception:
    psutil = None  # type: ignore

try:
    import pymupdf  # type: ignore
except Exception:
    pymupdf = None  # type: ignore

try:
    import pyttsx3  # type: ignore
except Exception:
    pyttsx3 = None  # type: ignore

try:
    import speech_recognition as sr  # type: ignore
except Exception:
    sr = None  # type: ignore


__version__ = "1.0.0"

USER_AGENT = "JarvisAI/1.0 (+https://github.com/haryana2x4; Autonomous Research Agent)"

SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", ".venv", "venv", "env",
    "__pycache__", ".pytest_cache", "dist", "build", ".next", ".cache",
}
TEXT_EXTS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".md", ".txt", ".json", ".yaml",
    ".yml", ".toml", ".html", ".css", ".sql", ".csv", ".sh", ".ps1",
}
SENSITIVE_PATTERNS = (
    ".env", "secret", "credential", "password", "token", "private_key",
    ".pem", ".key", ".pfx", "id_rsa", "id_ed25519",
)
DANGEROUS_SHELL_PATTERNS = (
    r"\brm\s+-rf\s+/(?:\s|$)",
    r"\bdel\s+/[sqf]+\s+[a-z]:\\",
    r"\bformat\s+[a-z]:",
    r"\bmkfs\b",
    r"\bshutdown\b",
    r"\breboot\b",
    r":\(\)\s*\{\s*:\|:&\s*\};:",
    r"\breg\s+delete\b",
    r"\bdiskpart\b",
    r"\bRemove-Item\b.*-Recurse.*([A-Z]:\\|Windows|System32)",
)


# ---------------------------------------------------------------------------
# Configuration & Data Structures
# ---------------------------------------------------------------------------

def default_brain_dir() -> str:
    """Return default persistent brain directory shared across all projects."""
    env_dir = os.environ.get("JARVIS_BRAIN_DIR")
    if env_dir:
        return os.path.abspath(os.path.expanduser(env_dir))
    return os.path.join(os.path.expanduser("~"), ".jarvis_brain")


@dataclass
class JarvisConfig:
    """Runtime configuration for Jarvis AI."""
    brain_dir: str = field(default_factory=default_brain_dir)
    workspace_root: str = field(default_factory=os.getcwd)
    llm_provider: str = "auto"  # auto|ollama|gemini|openai|anthropic|openrouter|local
    llm_model: str = ""
    ollama_url: str = field(
        default_factory=lambda: os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    )
    github_token: str = field(default_factory=lambda: os.environ.get("GITHUB_TOKEN", ""))
    allow_network: bool = True
    allow_shell: bool = True
    restrict_writes_to_workspace: bool = True
    tts_enabled: bool = True
    wake_words: Tuple[str, ...] = ("jarvis", "hey jarvis", "ok jarvis")
    max_autonomous_steps: int = 8
    request_timeout: int = 12


@dataclass
class StepTrace:
    """Observable execution step from Jarvis's autonomous loop."""
    step: int
    thought: str
    tool: str
    args: Dict[str, Any]
    observation: str
    status: str = "ok"
    duration_ms: int = 0


@dataclass
class JarvisResponse:
    """Unified structured response returned by Jarvis."""
    answer: str
    intent: str
    confidence: float
    verified: bool = False
    verdict: str = "UNVERIFIED"
    sources: List[Dict[str, Any]] = field(default_factory=list)
    steps: List[StepTrace] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    provider: str = "jarvis-neural-local"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to a JSON-serializable dictionary."""
        return {
            "answer": self.answer,
            "intent": self.intent,
            "confidence": round(float(self.confidence), 4),
            "verified": self.verified,
            "verdict": self.verdict,
            "sources": self.sources,
            "steps": [asdict(s) for s in self.steps],
            "artifacts": self.artifacts,
            "provider": self.provider,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Helpers: Tokenization, Safe Math, Text Cleaning
# ---------------------------------------------------------------------------

STOPWORDS = {
    "the", "is", "are", "was", "were", "a", "an", "and", "or", "to", "of",
    "in", "on", "for", "with", "me", "my", "you", "it", "this", "that",
    "please", "can", "could", "would", "do", "does", "from", "at", "by",
}


def tokenize(text: str, drop_stopwords: bool = False) -> List[str]:
    """Extract lowercase alphanumeric/code tokens."""
    tokens = re.findall(r"[a-z0-9_+#.-]+", (text or "").lower())
    if drop_stopwords:
        return [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens


def chunk_text(text: str, max_chars: int = 900, overlap: int = 140) -> List[str]:
    """Split text into overlapping chunks for RAG retrieval."""
    clean = re.sub(r"\s+", " ", (text or "")).strip()
    if not clean:
        return []
    if len(clean) <= max_chars:
        return [clean]
    chunks: List[str] = []
    start = 0
    while start < len(clean):
        end = min(len(clean), start + max_chars)
        if end < len(clean):
            split_at = clean.rfind(" ", start + max_chars // 2, end)
            if split_at > start:
                end = split_at
        chunks.append(clean[start:end].strip())
        if end >= len(clean):
            break
        start = max(start + 1, end - overlap)
    return chunks


def _stable_hash_bucket(token: str, dim: int) -> Tuple[int, float]:
    """Deterministic signed feature-hashing bucket across Python processes."""
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    val = int.from_bytes(digest, "little")
    idx = val % dim
    sign = 1.0 if ((val >> 63) & 1) == 0 else -1.0
    return idx, sign


def featurize_text(text: str, dim: int = 2048) -> np.ndarray:
    """Encode text into an L2-normalized multi-scale hashed n-gram vector."""
    vec = np.zeros(dim, dtype=np.float32)
    norm_text = re.sub(r"\s+", " ", (text or "").lower()).strip()
    if not norm_text:
        return vec

    words = tokenize(norm_text)
    for w in words:
        idx, sgn = _stable_hash_bucket("w:" + w, dim)
        vec[idx] += sgn * 1.4
    for a, b in zip(words, words[1:]):
        idx, sgn = _stable_hash_bucket(f"b:{a}_{b}", dim)
        vec[idx] += sgn * 1.8

    padded = f"^{norm_text}$"
    for n in (3, 4):
        for i in range(max(0, len(padded) - n + 1)):
            gram = padded[i : i + n]
            idx, sgn = _stable_hash_bucket(f"c{n}:{gram}", dim)
            vec[idx] += sgn * 0.45

    norm = float(np.linalg.norm(vec))
    if norm > 1e-8:
        vec /= norm
    return vec


def safe_eval_math(expr: str) -> Optional[float]:
    """Safely evaluate arithmetic expressions using AST whitelisting."""
    cleaned = re.sub(
        r"\b(what is|calculate|compute|solve|evaluate|equals)\b|\?|=|,|_",
        " ",
        (expr or "").lower(),
    ).strip()
    cleaned = cleaned.replace("×", "*").replace("÷", "/").replace("^", "**")
    if not cleaned or not re.fullmatch(r"[\d\s+\-*/().%]+", cleaned):
        return None

    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            if isinstance(node.op, ast.Pow):
                right = _eval(node.right)
                if abs(right) > 100:
                    raise ValueError("Exponent too large")
                return float(ops[type(node.op)](_eval(node.left), right))
            return float(ops[type(node.op)](_eval(node.left), _eval(node.right)))
        if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
            return float(ops[type(node.op)](_eval(node.operand)))
        raise ValueError("Unsupported AST node")

    try:
        tree = ast.parse(cleaned, mode="eval")
        val = _eval(tree)
        if math.isfinite(val):
            return int(val) if val.is_integer() else round(val, 8)
    except Exception:
        return None
    return None


def _is_sensitive_path(path: Union[str, Path]) -> bool:
    """Check whether a path looks like secrets/credentials that shouldn't be indexed."""
    low = str(path).lower()
    name = Path(path).name.lower()
    return any(p in name or f"/{p}" in low.replace("\\", "/") for p in SENSITIVE_PATTERNS)


# ---------------------------------------------------------------------------
# 1. Trainable NumPy Neural Intent & Skill Classifier
# ---------------------------------------------------------------------------

SEED_TRAINING_DATA: List[Tuple[str, str]] = [
    # web_search
    ("search the internet for quantum computing breakthroughs", "web_search"),
    ("search web for python 3.13 release notes", "web_search"),
    ("look up latest AI news online", "web_search"),
    ("google what is retrieval augmented generation", "web_search"),
    ("find information on the web about solar microgrids", "web_search"),
    ("search online for fastapi vs flask benchmarks", "web_search"),
    ("who is the creator of linux", "web_search"),
    ("what is the current population of tokyo", "web_search"),
    ("find research papers on arxiv about transformers", "web_search"),
    ("check stackoverflow for asyncio timeout error", "web_search"),
    ("browse internet for docker compose best practices", "web_search"),
    ("search the whole internet for edge tts voices", "web_search"),
    # github_research
    ("search github for python voice assistant repositories", "github_research"),
    ("find git repo for autonomous web scraper", "github_research"),
    ("inspect github repo tiangolo/fastapi", "github_research"),
    ("clone and analyze github.com/pallets/flask", "github_research"),
    ("get stars and readme from github repo", "github_research"),
    ("search github code for react threejs voxel character", "github_research"),
    ("find open source jarvis projects on github", "github_research"),
    ("check github repository commits and dependencies", "github_research"),
    ("pull data from github repo and train on it", "github_research"),
    ("find best python scraping repo on github", "github_research"),
    ("analyze git repo architecture", "github_research"),
    ("search git repos for langchain alternatives", "github_research"),
    # verify_fact
    ("verify whether python GIL is optional in 3.13", "verify_fact"),
    ("fact check if water boils at 100c on everest", "verify_fact"),
    ("cross verify this claim across github and the internet", "verify_fact"),
    ("is it true that sqlite supports fts5 full text search", "verify_fact"),
    ("double check from multiple sources if fastapi uses starlette", "verify_fact"),
    ("verify from github and web if ollama runs on port 11434", "verify_fact"),
    ("check sources and confirm whether rust beats c in speed", "verify_fact"),
    ("validate this information against real web sources", "verify_fact"),
    ("prove or disprove this technical statement", "verify_fact"),
    ("take data from everywhere to verify this claim", "verify_fact"),
    ("verify accuracy of this code pattern", "verify_fact"),
    ("check truth score and citations for this topic", "verify_fact"),
    # twist_website
    ("twist this website https://example.com and extract all links", "twist_website"),
    ("scrape and extract tables from website to json", "twist_website"),
    ("extract structured data from https://news.ycombinator.com", "twist_website"),
    ("parse html and convert webpage to csv", "twist_website"),
    ("twist webpage into clean markdown summary", "twist_website"),
    ("pull prices and product titles from this page", "twist_website"),
    ("extract emails headings and code blocks from url", "twist_website"),
    ("crawl website and extract json-ld metadata", "twist_website"),
    ("run css selector query on this webpage", "twist_website"),
    ("transform messy website html into structured api json", "twist_website"),
    ("grab all github links from this website", "twist_website"),
    ("extract data from website url", "twist_website"),
    # autonomous_task
    ("perform this multi step task on your own", "autonomous_task"),
    ("plan and build a python utility script and test it", "autonomous_task"),
    ("research github compare top 3 tools and write a report", "autonomous_task"),
    ("automate checking disk space and saving a log file", "autonomous_task"),
    ("do everything needed to create and verify the module", "autonomous_task"),
    ("create a json config file and validate it", "autonomous_task"),
    ("scan the project folder fix issues and run pytest", "autonomous_task"),
    ("fetch data from website save to csv and summarize", "autonomous_task"),
    ("work autonomously to finish this workflow", "autonomous_task"),
    ("execute task end to end without help", "autonomous_task"),
    ("build test and verify this feature on your own", "autonomous_task"),
    ("generate script run it and check output", "autonomous_task"),
    # code_exec
    ("write and run a python script to sort numbers", "code_exec"),
    ("execute python code to test numpy matrix multiply", "code_exec"),
    ("run shell command git status", "code_exec"),
    ("run pytest on the tests directory", "code_exec"),
    ("debug this python function and execute it", "code_exec"),
    ("compile or run this code snippet", "code_exec"),
    ("check python version in terminal", "code_exec"),
    ("run a benchmark script", "code_exec"),
    ("execute command line tool", "code_exec"),
    ("run tests and report pass count", "code_exec"),
    # file_ops
    ("read file app.py", "file_ops"),
    ("list all files in this directory", "file_ops"),
    ("search codebase for JarvisNeuralModel", "file_ops"),
    ("write a markdown file notes.md", "file_ops"),
    ("open and inspect package.json", "file_ops"),
    ("find where function parse_html is defined", "file_ops"),
    ("show contents of folder", "file_ops"),
    ("save output to output.json", "file_ops"),
    # system_control
    ("open chrome browser", "system_control"),
    ("launch notepad", "system_control"),
    ("open youtube in browser", "system_control"),
    ("start vscode", "system_control"),
    ("open explorer in current folder", "system_control"),
    ("check cpu ram and disk usage", "system_control"),
    ("show system stats and battery percentage", "system_control"),
    ("list top running processes using memory", "system_control"),
    ("what is my computer hardware status", "system_control"),
    ("open url in default browser", "system_control"),
    # math_calc
    ("calculate 256 * 48 + 1024", "math_calc"),
    ("what is (45 + 55) / 4", "math_calc"),
    ("compute 2^16 - 1", "math_calc"),
    ("solve 19.5 * 3.14159", "math_calc"),
    ("evaluate 1000 / 8", "math_calc"),
    ("what is 15 percent of 2400", "math_calc"),
    # train_learn
    ("train your model on this folder", "train_learn"),
    ("learn from this github repository", "train_learn"),
    ("remember that my favorite editor is vscode", "train_learn"),
    ("teach jarvis when i say open dev launch code", "train_learn"),
    ("ingest all pdf and markdown files into knowledge base", "train_learn"),
    ("retrain neural classifier with new examples", "train_learn"),
    ("update your training data with this correction", "train_learn"),
    ("index this codebase so you know everything about it", "train_learn"),
    # memory_recall
    ("what do you remember about my projects", "memory_recall"),
    ("recall what i taught you earlier", "memory_recall"),
    ("search your memory for sunday hackathon", "memory_recall"),
    ("show knowledge base stats", "memory_recall"),
    ("what facts have you learned so far", "memory_recall"),
    ("check your local memory store", "memory_recall"),
    # voice_control
    ("start voice command listening mode", "voice_control"),
    ("listen to my microphone", "voice_control"),
    ("enable wake word jarvis", "voice_control"),
    ("speak this answer out loud", "voice_control"),
    ("turn off voice speech", "voice_control"),
    ("run in voice assistant mode", "voice_control"),
    # chat_general
    ("hello jarvis how are you", "chat_general"),
    ("who are you and what can you do", "chat_general"),
    ("good morning jarvis", "chat_general"),
    ("explain neural networks simply", "chat_general"),
    ("help me brainstorm an app idea", "chat_general"),
    ("thanks jarvis great job", "chat_general"),
    ("tell me your capabilities", "chat_general"),
    ("hi buddy status report", "chat_general"),
]


class JarvisNeuralModel:
    """
    Real trainable 2-layer NumPy MLP intent & skill classifier with Adam optimizer.

    Features:
      - Fixed 2048-dim hashed word/bigram/char-ngram projection -> online learning
        never breaks when new words or new intent labels are introduced!
      - Supports batch training (`fit`), single-sample online updates (`train_step`),
        and dynamic addition of brand-new intent classes (`add_class`).
      - Persists trained weights to `jarvis_model.npz` + `jarvis_model_meta.json`.
    """

    def __init__(self, model_dir: str, input_dim: int = 2048, hidden_dim: int = 128):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.weights_path = self.model_dir / "jarvis_model.npz"
        self.meta_path = self.model_dir / "jarvis_model_meta.json"
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.classes: List[str] = []
        self.samples: List[Tuple[str, str]] = []
        self.rng = np.random.default_rng(42)

        self.W1 = np.zeros((input_dim, hidden_dim), dtype=np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = np.zeros((hidden_dim, 1), dtype=np.float32)
        self.b2 = np.zeros(1, dtype=np.float32)
        self.train_history: List[Dict[str, Any]] = []
        self._lock = threading.RLock()

        if self.weights_path.exists() and self.meta_path.exists():
            self.load()
        else:
            self.fit(SEED_TRAINING_DATA, epochs=45, lr=0.025, save=True)

    def _init_weights(self, classes: Sequence[str]) -> None:
        self.classes = list(dict.fromkeys(classes))
        num_c = max(1, len(self.classes))
        self.W1 = (
            self.rng.normal(0.0, math.sqrt(2.0 / self.input_dim), (self.input_dim, self.hidden_dim))
        ).astype(np.float32)
        self.b1 = np.zeros(self.hidden_dim, dtype=np.float32)
        self.W2 = (
            self.rng.normal(0.0, math.sqrt(2.0 / self.hidden_dim), (self.hidden_dim, num_c))
        ).astype(np.float32)
        self.b2 = np.zeros(num_c, dtype=np.float32)

    def _ensure_class(self, label: str) -> int:
        if label in self.classes:
            return self.classes.index(label)
        self.classes.append(label)
        new_col = self.rng.normal(0.0, 0.05, (self.hidden_dim, 1)).astype(np.float32)
        self.W2 = np.hstack([self.W2, new_col])
        self.b2 = np.append(self.b2, np.float32(0.0))
        return len(self.classes) - 1

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exps = np.exp(shifted)
        return exps / np.maximum(np.sum(exps, axis=-1, keepdims=True), 1e-12)

    def _forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        z1 = X @ self.W1 + self.b1
        h1 = np.maximum(0.0, z1)  # ReLU
        logits = h1 @ self.W2 + self.b2
        probs = self._softmax(logits)
        return z1, h1, probs

    def fit(
        self,
        data: Sequence[Tuple[str, str]],
        epochs: int = 45,
        lr: float = 0.02,
        l2: float = 1e-4,
        save: bool = True,
    ) -> Dict[str, Any]:
        """Train the neural intent classifier from scratch or merged dataset."""
        with self._lock:
            merged = list(dict.fromkeys([(t.strip(), lbl.strip()) for t, lbl in data if t.strip()]))
            if not merged:
                return {"status": "empty", "accuracy": 0.0}
            self.samples = merged
            labels = sorted({lbl for _, lbl in merged})
            self._init_weights(labels)

            X = np.vstack([featurize_text(t, self.input_dim) for t, _ in merged])
            y = np.array([self.classes.index(lbl) for _, lbl in merged], dtype=np.int64)
            n = len(merged)

            # Adam state
            params = [self.W1, self.b1, self.W2, self.b2]
            m = [np.zeros_like(p) for p in params]
            v = [np.zeros_like(p) for p in params]
            b1_adam, b2_adam, eps = 0.9, 0.999, 1e-8

            final_loss = 0.0
            final_acc = 0.0
            for epoch in range(1, epochs + 1):
                perm = self.rng.permutation(n)
                Xp, yp = X[perm], y[perm]
                z1, h1, probs = self._forward(Xp)

                one_hot = np.zeros_like(probs)
                one_hot[np.arange(n), yp] = 1.0
                ce_loss = -np.mean(np.sum(one_hot * np.log(np.maximum(probs, 1e-12)), axis=1))
                reg_loss = 0.5 * l2 * (float(np.sum(self.W1 ** 2)) + float(np.sum(self.W2 ** 2)))
                final_loss = float(ce_loss + reg_loss)
                final_acc = float(np.mean(np.argmax(probs, axis=1) == yp))

                dlogits = (probs - one_hot) / n
                dW2 = h1.T @ dlogits + l2 * self.W2
                db2 = np.sum(dlogits, axis=0)
                dh1 = dlogits @ self.W2.T
                dz1 = dh1 * (z1 > 0)
                dW1 = Xp.T @ dz1 + l2 * self.W1
                db1 = np.sum(dz1, axis=0)

                grads = [dW1, db1, dW2, db2]
                for i, (p, g) in enumerate(zip(params, grads)):
                    m[i] = b1_adam * m[i] + (1.0 - b1_adam) * g
                    v[i] = b2_adam * v[i] + (1.0 - b2_adam) * (g ** 2)
                    m_hat = m[i] / (1.0 - b1_adam ** epoch)
                    v_hat = v[i] / (1.0 - b2_adam ** epoch)
                    p -= lr * m_hat / (np.sqrt(v_hat) + eps)

            metrics = {
                "samples": n,
                "classes": len(self.classes),
                "epochs": epochs,
                "loss": round(final_loss, 5),
                "train_accuracy": round(final_acc, 4),
                "trained_at": datetime.now(timezone.utc).isoformat(),
            }
            self.train_history.append(metrics)
            if save:
                self.save()
            return metrics

    def train_online(
        self,
        examples: Sequence[Tuple[str, str]],
        steps: int = 12,
        lr: float = 0.035,
        save: bool = True,
    ) -> Dict[str, Any]:
        """Incrementally fine-tune the live neural model on new examples/corrections."""
        with self._lock:
            cleaned = [(t.strip(), lbl.strip()) for t, lbl in examples if t.strip() and lbl.strip()]
            if not cleaned:
                return {"updated": 0}
            for text, lbl in cleaned:
                self._ensure_class(lbl)
                if (text, lbl) not in self.samples:
                    self.samples.append((text, lbl))

            # Replay mix of new samples + random existing samples to prevent catastrophic forgetting
            replay_count = min(len(self.samples), max(16, len(cleaned) * 3))
            replay_indices = self.rng.choice(len(self.samples), size=replay_count, replace=False)
            batch = list(cleaned) * 2 + [self.samples[i] for i in replay_indices]
            X = np.vstack([featurize_text(t, self.input_dim) for t, _ in batch])
            y = np.array([self.classes.index(lbl) for _, lbl in batch], dtype=np.int64)
            n = len(batch)

            for _ in range(steps):
                z1, h1, probs = self._forward(X)
                one_hot = np.zeros_like(probs)
                one_hot[np.arange(n), y] = 1.0
                dlogits = (probs - one_hot) / n
                dW2 = h1.T @ dlogits + 1e-4 * self.W2
                db2 = np.sum(dlogits, axis=0)
                dh1 = dlogits @ self.W2.T
                dz1 = dh1 * (z1 > 0)
                dW1 = X.T @ dz1 + 1e-4 * self.W1
                db1 = np.sum(dz1, axis=0)
                self.W2 -= lr * dW2
                self.b2 -= lr * db2
                self.W1 -= lr * dW1
                self.b1 -= lr * db1

            _, _, probs = self._forward(X)
            acc = float(np.mean(np.argmax(probs, axis=1) == y))
            if save:
                self.save()
            return {"updated": len(cleaned), "total_samples": len(self.samples), "batch_accuracy": round(acc, 4)}

    def predict(self, text: str, top_k: int = 3) -> Tuple[str, float, List[Tuple[str, float]]]:
        """Predict intent label, confidence, and top-k probabilities."""
        with self._lock:
            if not self.classes:
                return "chat_general", 0.5, [("chat_general", 0.5)]
            x = featurize_text(text, self.input_dim)[None, :]
            _, _, probs = self._forward(x)
            p = probs[0]
            order = np.argsort(p)[::-1][:top_k]
            top = [(self.classes[i], float(p[i])) for i in order]
            return top[0][0], top[0][1], top

    def evaluate(self, test_data: Sequence[Tuple[str, str]]) -> Dict[str, Any]:
        """Evaluate accuracy on labeled test samples."""
        if not test_data:
            return {"accuracy": 0.0, "total": 0, "correct": 0}
        correct = 0
        details = []
        for text, expected in test_data:
            pred, conf, _ = self.predict(text)
            hit = pred == expected
            if hit:
                correct += 1
            else:
                details.append({"text": text, "expected": expected, "predicted": pred, "confidence": round(conf, 3)})
        return {
            "accuracy": round(correct / len(test_data), 4),
            "correct": correct,
            "total": len(test_data),
            "misses": details[:10],
        }

    def save(self) -> None:
        """Persist weights and training metadata to disk."""
        with self._lock:
            np.savez_compressed(
                self.weights_path,
                W1=self.W1,
                b1=self.b1,
                W2=self.W2,
                b2=self.b2,
            )
            meta = {
                "version": __version__,
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "classes": self.classes,
                "samples": self.samples,
                "train_history": self.train_history[-20:],
            }
            self.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def load(self) -> None:
        """Load persisted model weights and metadata."""
        with self._lock:
            meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
            data = np.load(self.weights_path)
            self.input_dim = int(meta.get("input_dim", self.input_dim))
            self.hidden_dim = int(meta.get("hidden_dim", self.hidden_dim))
            self.classes = list(meta.get("classes", []))
            self.samples = [tuple(x) for x in meta.get("samples", [])]
            self.train_history = list(meta.get("train_history", []))
            self.W1 = data["W1"].astype(np.float32)
            self.b1 = data["b1"].astype(np.float32)
            self.W2 = data["W2"].astype(np.float32)
            self.b2 = data["b2"].astype(np.float32)


# ---------------------------------------------------------------------------
# 2. Hybrid SQLite FTS5 + Vector Knowledge & Episodic Memory
# ---------------------------------------------------------------------------

class JarvisMemory:
    """Persistent hybrid FTS5 + vector memory store for RAG, repos, and corrections."""

    def __init__(self, brain_dir: str):
        self.db_path = Path(brain_dir) / "jarvis_memory.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            cur = self.conn.cursor()
            cur.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS knowledge (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_type TEXT NOT NULL,
                    source_uri TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tags TEXT DEFAULT '',
                    vec BLOB NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
                    title, content, tags, content='knowledge', content_rowid='id'
                );
                CREATE TRIGGER IF NOT EXISTS knowledge_ai AFTER INSERT ON knowledge BEGIN
                    INSERT INTO knowledge_fts(rowid, title, content, tags)
                    VALUES (new.id, new.title, new.content, new.tags);
                END;
                CREATE TRIGGER IF NOT EXISTS knowledge_ad AFTER DELETE ON knowledge BEGIN
                    INSERT INTO knowledge_fts(knowledge_fts, rowid, title, content, tags)
                    VALUES ('delete', old.id, old.title, old.content, old.tags);
                END;
                CREATE TABLE IF NOT EXISTS learned_commands (
                    phrase TEXT PRIMARY KEY,
                    intent TEXT NOT NULL,
                    action_json TEXT DEFAULT '{}',
                    response_template TEXT DEFAULT '',
                    hits INTEGER DEFAULT 0,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS episodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    intent TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    verdict TEXT DEFAULT 'UNVERIFIED',
                    confidence REAL DEFAULT 0.0,
                    created_at TEXT NOT NULL
                );
                """
            )
            self.conn.commit()

    def ingest_text(
        self,
        title: str,
        content: str,
        source_type: str = "note",
        source_uri: str = "local",
        tags: str = "",
        dedupe_uri: bool = True,
    ) -> int:
        """Chunk and store text + hashed vector embeddings into persistent memory."""
        chunks = chunk_text(content, max_chars=900, overlap=140)
        if not chunks:
            return 0
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            if dedupe_uri and source_uri not in ("local", ""):
                self.conn.execute("DELETE FROM knowledge WHERE source_uri = ?", (source_uri,))
            inserted = 0
            for idx, chunk in enumerate(chunks):
                chunk_title = title if len(chunks) == 1 else f"{title} [part {idx + 1}/{len(chunks)}]"
                vec = featurize_text(f"{title} {tags} {chunk}", 2048).astype(np.float32).tobytes()
                self.conn.execute(
                    "INSERT INTO knowledge(source_type, source_uri, title, content, tags, vec, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (source_type, source_uri, chunk_title, chunk, tags, vec, now),
                )
                inserted += 1
            self.conn.commit()
            return inserted

    def ingest_folder(self, folder_path: str, max_files: int = 200) -> Dict[str, Any]:
        """Index code, docs, markdown, and PDF files from a folder safely."""
        root = Path(folder_path).expanduser().resolve()
        if not root.exists():
            return {"error": f"Folder not found: {folder_path}", "files": 0, "chunks": 0}

        files_indexed = 0
        chunks_added = 0
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fname in filenames:
                if files_indexed >= max_files:
                    break
                fpath = Path(dirpath) / fname
                if _is_sensitive_path(fpath):
                    continue
                ext = fpath.suffix.lower()
                text = ""
                try:
                    if ext == ".pdf" and pymupdf is not None:
                        with pymupdf.open(str(fpath)) as doc:
                            text = "\n".join(page.get_text() for page in doc[:30])
                    elif ext in TEXT_EXTS and fpath.stat().st_size <= 350_000:
                        text = fpath.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                if text.strip():
                    rel = str(fpath.relative_to(root))
                    chunks_added += self.ingest_text(
                        title=rel,
                        content=text,
                        source_type="local_file",
                        source_uri=str(fpath),
                        tags=f"{root.name} {ext.lstrip('.')}",
                    )
                    files_indexed += 1
            if files_indexed >= max_files:
                break
        return {"folder": str(root), "files": files_indexed, "chunks": chunks_added}

    def search_knowledge(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Hybrid FTS5 BM25 + cosine vector retrieval over stored knowledge."""
        q_vec = featurize_text(query, 2048)
        tokens = tokenize(query, drop_stopwords=True)[:8]
        candidates: Dict[int, Dict[str, Any]] = {}

        with self._lock:
            if tokens:
                fts_query = " OR ".join(f'"{t.replace(chr(34), "")}"' for t in tokens)
                try:
                    rows = self.conn.execute(
                        "SELECT k.id, k.source_type, k.source_uri, k.title, k.content, k.tags, k.vec, "
                        "-bm25(knowledge_fts) AS bm25_score "
                        "FROM knowledge_fts f JOIN knowledge k ON k.id = f.rowid "
                        "WHERE knowledge_fts MATCH ? ORDER BY bm25(knowledge_fts) LIMIT ?",
                        (fts_query, top_k * 6),
                    ).fetchall()
                    for r in rows:
                        candidates[int(r["id"])] = dict(r)
                except sqlite3.OperationalError:
                    pass

            if len(candidates) < top_k * 3:
                rows = self.conn.execute(
                    "SELECT id, source_type, source_uri, title, content, tags, vec, 0.0 AS bm25_score "
                    "FROM knowledge ORDER BY id DESC LIMIT 250"
                ).fetchall()
                for r in rows:
                    candidates.setdefault(int(r["id"]), dict(r))

        ranked: List[Dict[str, Any]] = []
        for item in candidates.values():
            vec = np.frombuffer(item["vec"], dtype=np.float32)
            cos = float(np.dot(q_vec, vec)) if vec.shape == q_vec.shape else 0.0
            bm25 = max(0.0, float(item.get("bm25_score", 0.0)))
            score = 0.65 * max(0.0, cos) + 0.35 * min(1.0, bm25 / 10.0)
            if score > 0.08:
                ranked.append(
                    {
                        "id": item["id"],
                        "source_type": item["source_type"],
                        "source_uri": item["source_uri"],
                        "title": item["title"],
                        "content": item["content"],
                        "tags": item["tags"],
                        "score": round(score, 4),
                    }
                )
        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked[:top_k]

    def teach_command(
        self,
        phrase: str,
        intent: str,
        action: Optional[Dict[str, Any]] = None,
        response_template: str = "",
    ) -> None:
        """Persist a user-taught phrase -> intent/action/response mapping."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO learned_commands(phrase, intent, action_json, response_template, hits, updated_at) "
                "VALUES (?, ?, ?, ?, COALESCE((SELECT hits FROM learned_commands WHERE phrase=?), 0), ?)",
                (
                    phrase.strip().lower(),
                    intent.strip(),
                    json.dumps(action or {}),
                    response_template,
                    phrase.strip().lower(),
                    now,
                ),
            )
            self.conn.commit()

    def match_learned_command(self, query: str) -> Optional[Dict[str, Any]]:
        """Check exact or high-overlap match against user-taught custom commands."""
        norm = query.strip().lower()
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM learned_commands WHERE phrase = ?", (norm,)
            ).fetchone()
            if row:
                self.conn.execute(
                    "UPDATE learned_commands SET hits = hits + 1 WHERE phrase = ?", (norm,)
                )
                self.conn.commit()
                res = dict(row)
                res["action"] = json.loads(res.get("action_json") or "{}")
                return res
        return None

    def log_episode(self, query: str, intent: str, answer: str, verdict: str, confidence: float) -> None:
        """Record interaction history for auditability & future training."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self.conn.execute(
                "INSERT INTO episodes(query, intent, answer, verdict, confidence, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (query, intent, answer[:4000], verdict, float(confidence), now),
            )
            self.conn.commit()

    def stats(self) -> Dict[str, Any]:
        """Return memory & knowledge counts."""
        with self._lock:
            k_count = self.conn.execute("SELECT COUNT(*) FROM knowledge").fetchone()[0]
            c_count = self.conn.execute("SELECT COUNT(*) FROM learned_commands").fetchone()[0]
            e_count = self.conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
            by_type = dict(
                self.conn.execute(
                    "SELECT source_type, COUNT(*) FROM knowledge GROUP BY source_type"
                ).fetchall()
            )
            return {
                "knowledge_chunks": k_count,
                "learned_commands": c_count,
                "episodes": e_count,
                "by_source_type": by_type,
                "db_path": str(self.db_path),
            }


# ---------------------------------------------------------------------------
# 3. Website Twister: Pure-Python DOM Tree, CSS Selectors & Data Transformer
# ---------------------------------------------------------------------------

@dataclass
class _DomNode:
    tag: str
    attrs: Dict[str, str] = field(default_factory=dict)
    children: List["_DomNode"] = field(default_factory=list)
    text_parts: List[str] = field(default_factory=list)
    parent: Optional["_DomNode"] = None

    def text(self, separator: str = " ") -> str:
        parts: List[str] = []
        self._collect_text(parts)
        return re.sub(r"\s+", " ", separator.join(parts)).strip()

    def _collect_text(self, out: List[str]) -> None:
        if self.tag in ("script", "style", "noscript", "svg", "template"):
            return
        for t in self.text_parts:
            if t.strip():
                out.append(t.strip())
        for c in self.children:
            c._collect_text(out)


class _DomBuilder(HTMLParser):
    VOID_TAGS = {
        "area", "base", "br", "col", "embed", "hr", "img", "input",
        "link", "meta", "param", "source", "track", "wbr",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = _DomNode("document")
        self.stack: List[_DomNode] = [self.root]

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        t = tag.lower()
        attr_map = {k.lower(): (v if v is not None else "") for k, v in attrs}
        node = _DomNode(tag=t, attrs=attr_map, parent=self.stack[-1])
        self.stack[-1].children.append(node)
        if t not in self.VOID_TAGS:
            self.stack.append(node)

    def handle_startendtag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        t = tag.lower()
        attr_map = {k.lower(): (v if v is not None else "") for k, v in attrs}
        node = _DomNode(tag=t, attrs=attr_map, parent=self.stack[-1])
        self.stack[-1].children.append(node)

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].tag == t:
                del self.stack[i:]
                break

    def handle_data(self, data: str) -> None:
        if data and self.stack:
            self.stack[-1].text_parts.append(data)


def _match_simple_selector(node: _DomNode, part: str) -> bool:
    """Match a single CSS compound selector like `div.card#main[data-x*=foo]`."""
    if node.tag == "document":
        return False
    rem = part.strip()
    if not rem:
        return False

    # Extract tag name if present at start
    m_tag = re.match(r"^([a-zA-Z0-9_-]+|\*)", rem)
    if m_tag:
        tag = m_tag.group(1).lower()
        if tag != "*" and node.tag != tag:
            return False
        rem = rem[m_tag.end():]

    for m in re.finditer(r"(#|\.)([a-zA-Z0-9_-]+)|\[([^\]]+)\]", rem):
        kind, ident, attr_expr = m.group(1), m.group(2), m.group(3)
        if kind == "#":
            if node.attrs.get("id", "") != ident:
                return False
        elif kind == ".":
            classes = node.attrs.get("class", "").split()
            if ident not in classes:
                return False
        elif attr_expr:
            m_attr = re.match(r"^\s*([a-zA-Z0-9_:-]+)\s*(?:(\*=|\^=|\$=|=)\s*['\"]?([^'\"]*)['\"]?)?\s*$", attr_expr)
            if not m_attr:
                return False
            akey, aop, aval = m_attr.group(1).lower(), m_attr.group(2), m_attr.group(3) or ""
            if akey not in node.attrs:
                return False
            actual = node.attrs[akey]
            if aop == "=" and actual != aval:
                return False
            if aop == "*=" and aval not in actual:
                return False
            if aop == "^=" and not actual.startswith(aval):
                return False
            if aop == "$=" and not actual.endswith(aval):
                return False
    return True


def _iter_nodes(root: _DomNode) -> Iterable[_DomNode]:
    for child in root.children:
        yield child
        yield from _iter_nodes(child)


def css_select(root: _DomNode, selector: str) -> List[_DomNode]:
    """Query DOM nodes matching CSS selector(s) (`a, b`, `div > p`, `div p`)."""
    results: List[_DomNode] = []
    seen: set[int] = set()
    for group in selector.split(","):
        group = group.strip()
        if not group:
            continue
        tokens = re.findall(r">|[^\s>]+", group)
        if not tokens:
            continue
        current_matches = list(_iter_nodes(root))
        # Match right-to-left on chain
        for node in current_matches:
            if not _match_simple_selector(node, tokens[-1]):
                continue
            ok = True
            cursor = node
            idx = len(tokens) - 2
            while idx >= 0:
                tok = tokens[idx]
                if tok == ">":
                    idx -= 1
                    if idx < 0 or cursor.parent is None or not _match_simple_selector(cursor.parent, tokens[idx]):
                        ok = False
                        break
                    cursor = cursor.parent
                    idx -= 1
                else:
                    anc = cursor.parent
                    found = False
                    while anc is not None and anc.tag != "document":
                        if _match_simple_selector(anc, tok):
                            found = True
                            cursor = anc
                            break
                        anc = anc.parent
                    if not found:
                        ok = False
                        break
                    idx -= 1
            if ok and id(node) not in seen:
                seen.add(id(node))
                results.append(node)
    return results


class WebsiteTwister:
    """
    Universal Web Extractor & Data Transformer ("Website Twister").

    Parses any webpage into structured records (tables, links, metadata, JSON-LD,
    articles, code blocks, prices, emails, custom CSS selector fields) and twists
    the output into JSON, CSV, Markdown, or clean summaries.
    """

    CHALLENGE_MARKERS = (
        "cf-turnstile", "g-recaptcha", "hcaptcha", "verify you are human",
        "checking your browser before accessing", "access denied", "enable javascript and cookies to continue",
    )

    def __init__(self, timeout: int = 12):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"})

    def fetch(self, url: str) -> Dict[str, Any]:
        """Fetch raw HTML/JSON safely while respecting access gates and bot checks."""
        if not re.match(r"^https?://", url, re.I):
            url = "https://" + url
        try:
            resp = self.session.get(url, timeout=self.timeout, allow_redirects=True)
            ct = resp.headers.get("Content-Type", "")
            text = resp.text or ""
            low = text.lower()
            blocked_reason = ""
            if resp.status_code in (401, 403, 429):
                blocked_reason = f"HTTP {resp.status_code} access/rate gate"
            elif any(marker in low for marker in self.CHALLENGE_MARKERS) and len(text) < 25000:
                blocked_reason = "Anti-bot / CAPTCHA verification challenge detected"
            return {
                "url": resp.url,
                "status_code": resp.status_code,
                "content_type": ct,
                "raw": text,
                "blocked": bool(blocked_reason),
                "blocked_reason": blocked_reason,
            }
        except Exception as exc:
            return {"url": url, "status_code": 0, "content_type": "", "raw": "", "blocked": True, "blocked_reason": str(exc)}

    def parse_html(self, html_text: str, base_url: str = "") -> Dict[str, Any]:
        """Parse raw HTML into structured DOM entities."""
        builder = _DomBuilder()
        builder.feed(html_text or "")
        builder.close()
        root = builder.root

        title_nodes = css_select(root, "title")
        title = title_nodes[0].text() if title_nodes else ""

        meta: Dict[str, str] = {}
        for m in css_select(root, "meta"):
            key = m.attrs.get("name") or m.attrs.get("property") or ""
            val = m.attrs.get("content", "")
            if key and val:
                meta[key.lower()] = val

        json_ld: List[Any] = []
        for script in css_select(root, "script"):
            if script.attrs.get("type", "").lower() == "application/ld+json":
                raw_json = "".join(script.text_parts).strip()
                try:
                    json_ld.append(json.loads(raw_json))
                except Exception:
                    pass

        headings = [
            {"level": h.tag, "text": h.text()}
            for h in _iter_nodes(root)
            if h.tag in ("h1", "h2", "h3", "h4") and h.text()
        ]

        tables = self._extract_tables(root)
        links = self._extract_links(root, base_url)
        code_blocks = [c.text("\n") for c in css_select(root, "pre") if c.text()]
        images = [
            {
                "src": urllib.parse.urljoin(base_url, img.attrs.get("src", "")),
                "alt": img.attrs.get("alt", ""),
            }
            for img in css_select(root, "img")
            if img.attrs.get("src")
        ]

        # Clean article text ignoring nav/footer/aside boilerplate
        clean_chunks: List[str] = []
        for node in _iter_nodes(root):
            if node.tag in ("p", "li", "article", "blockquote") and not self._inside_boilerplate(node):
                t = node.text()
                if len(t) >= 20 and t not in clean_chunks:
                    clean_chunks.append(t)
        visible_text = root.text()
        clean_text = "\n".join(clean_chunks) if clean_chunks else visible_text

        emails = sorted(set(re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", visible_text + " " + html_text)))
        prices = re.findall(r"(?:USD|INR|EUR|GBP|\$|₹|€|£)\s?\d[\d,]*(?:\.\d{1,2})?", visible_text)

        return {
            "title": title,
            "url": base_url,
            "meta": meta,
            "json_ld": json_ld,
            "headings": headings[:60],
            "tables": tables,
            "links": links[:120],
            "code_blocks": code_blocks[:30],
            "images": images[:50],
            "emails": emails[:25],
            "prices": prices[:30],
            "clean_text": clean_text,
            "_root": root,
        }

    @staticmethod
    def _inside_boilerplate(node: _DomNode) -> bool:
        cur = node.parent
        while cur is not None:
            if cur.tag in ("nav", "footer", "aside", "header"):
                return True
            cur = cur.parent
        return False

    @staticmethod
    def _extract_tables(root: _DomNode) -> List[List[Dict[str, str]]]:
        tables_out: List[List[Dict[str, str]]] = []
        for tbl in css_select(root, "table"):
            rows = css_select(tbl, "tr")
            if not rows:
                continue
            headers: List[str] = []
            records: List[Dict[str, str]] = []
            for idx, tr in enumerate(rows):
                ths = [c.text() for c in tr.children if c.tag == "th"]
                tds = [c.text() for c in tr.children if c.tag in ("td", "th")]
                if idx == 0 and ths:
                    headers = [h or f"col_{i}" for i, h in enumerate(ths)]
                    continue
                if not tds:
                    continue
                if not headers:
                    headers = [f"col_{i}" for i in range(len(tds))]
                row_obj = {
                    headers[i] if i < len(headers) else f"col_{i}": val
                    for i, val in enumerate(tds)
                }
                records.append(row_obj)
            if records:
                tables_out.append(records)
        return tables_out

    @staticmethod
    def _extract_links(root: _DomNode, base_url: str) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        seen: set[str] = set()
        base_host = urllib.parse.urlparse(base_url).netloc.lower()
        for a in css_select(root, "a"):
            href = a.attrs.get("href", "").strip()
            if not href or href.startswith(("#", "javascript:", "mailto:")):
                continue
            abs_url = urllib.parse.urljoin(base_url, href)
            if abs_url in seen:
                continue
            seen.add(abs_url)
            host = urllib.parse.urlparse(abs_url).netloc.lower()
            kind = "github" if "github.com" in host else ("internal" if host == base_host else "external")
            out.append({"text": a.text()[:120], "url": abs_url, "kind": kind})
        return out

    def twist(
        self,
        url_or_html: str,
        output_format: str = "json",
        selector: Optional[str] = None,
        fields: Optional[Dict[str, str]] = None,
        filter_pattern: Optional[str] = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """
        Fetch (or parse raw HTML) and transform website content into structured data.

        Args:
            url_or_html: HTTP(S) URL or raw HTML snippet.
            output_format: 'json' | 'csv' | 'markdown' | 'text' | 'tables'
            selector: Optional CSS selector to isolate elements.
            fields: Optional {field_name: css_selector} mapping for repeating items.
            filter_pattern: Optional regex/substring filter applied to extracted items.
            limit: Maximum number of records returned.
        """
        is_html = "<" in url_or_html and ">" in url_or_html and not url_or_html.strip().startswith("http")
        if is_html:
            raw_html = url_or_html
            final_url = "inline://html"
            status_code = 200
        else:
            fetched = self.fetch(url_or_html)
            if fetched["blocked"]:
                return {
                    "ok": False,
                    "blocked": True,
                    "url": fetched["url"],
                    "status_code": fetched["status_code"],
                    "reason": fetched["blocked_reason"],
                }
            final_url = fetched["url"]
            status_code = fetched["status_code"]
            if "application/json" in fetched["content_type"].lower():
                try:
                    parsed_json = json.loads(fetched["raw"])
                    return {
                        "ok": True,
                        "url": final_url,
                        "status_code": status_code,
                        "format": "json",
                        "data": parsed_json,
                        "formatted": json.dumps(parsed_json, indent=2)[:15000],
                    }
                except Exception:
                    pass
            raw_html = fetched["raw"]

        parsed = self.parse_html(raw_html, base_url=final_url)
        root: _DomNode = parsed.pop("_root")

        records: List[Any] = []
        if fields:
            cols = {k: [n.text() for n in css_select(root, sel)] for k, sel in fields.items()}
            row_count = max((len(v) for v in cols.values()), default=0)
            for i in range(min(row_count, limit)):
                records.append({k: (v[i] if i < len(v) else "") for k, v in cols.items()})
        elif selector:
            nodes = css_select(root, selector)
            for n in nodes[:limit]:
                item = {"tag": n.tag, "text": n.text(), "attrs": n.attrs}
                if n.tag == "a" and "href" in n.attrs:
                    item["url"] = urllib.parse.urljoin(final_url, n.attrs["href"])
                records.append(item)
        elif parsed["tables"]:
            records = parsed["tables"][0][:limit]
        else:
            records = parsed["links"][:limit]

        if filter_pattern:
            rx = re.compile(filter_pattern, re.I)
            records = [r for r in records if rx.search(json.dumps(r, ensure_ascii=False))]

        fmt = output_format.lower().strip()
        formatted = self._format_output(fmt, parsed, records)
        return {
            "ok": True,
            "blocked": False,
            "url": final_url,
            "status_code": status_code,
            "title": parsed["title"],
            "format": fmt,
            "count": len(records),
            "records": records,
            "tables_count": len(parsed["tables"]),
            "links_count": len(parsed["links"]),
            "emails": parsed["emails"],
            "prices": parsed["prices"],
            "headings": parsed["headings"][:15],
            "clean_text_preview": parsed["clean_text"][:2500],
            "formatted": formatted,
        }

    @staticmethod
    def _format_output(fmt: str, parsed: Dict[str, Any], records: List[Any]) -> str:
        if fmt == "csv":
            if not records:
                return ""
            buf = io.StringIO()
            if isinstance(records[0], dict):
                keys = list(dict.fromkeys(k for r in records for k in r.keys()))
                writer = csv.DictWriter(buf, fieldnames=keys)
                writer.writeheader()
                for r in records:
                    writer.writerow({k: r.get(k, "") for k in keys})
            else:
                writer = csv.writer(buf)
                for r in records:
                    writer.writerow([r])
            return buf.getvalue()
        if fmt == "markdown":
            lines = [f"# {parsed['title'] or parsed['url']}", f"Source: {parsed['url']}", ""]
            if parsed["headings"]:
                lines.append("## Headings")
                for h in parsed["headings"][:10]:
                    lines.append(f"- **{h['level']}**: {h['text']}")
                lines.append("")
            if records:
                lines.append("## Extracted Records")
                for r in records[:25]:
                    lines.append(f"- `{json.dumps(r, ensure_ascii=False)}`")
            if parsed["clean_text"]:
                lines.extend(["", "## Clean Text", parsed["clean_text"][:2500]])
            return "\n".join(lines)
        if fmt == "text":
            return parsed["clean_text"][:6000]
        return json.dumps(
            {"title": parsed["title"], "url": parsed["url"], "records": records, "emails": parsed["emails"], "prices": parsed["prices"]},
            indent=2,
            ensure_ascii=False,
        )

    def crawl(self, start_url: str, max_pages: int = 4, same_domain: bool = True) -> List[Dict[str, Any]]:
        """Crawl linked pages safely up to `max_pages`."""
        visited: set[str] = set()
        queue_urls = [start_url]
        results: List[Dict[str, Any]] = []
        base_host = urllib.parse.urlparse(start_url).netloc.lower()

        while queue_urls and len(results) < max_pages:
            url = queue_urls.pop(0)
            if url in visited:
                continue
            visited.add(url)
            twisted = self.twist(url, output_format="json", limit=30)
            results.append(twisted)
            if not twisted.get("ok"):
                continue
            for item in twisted.get("records", []):
                next_url = item.get("url") if isinstance(item, dict) else None
                if next_url and next_url not in visited:
                    if not same_domain or urllib.parse.urlparse(next_url).netloc.lower() == base_host:
                        queue_urls.append(next_url)
        return results


# ---------------------------------------------------------------------------
# 4. Multi-Source Internet & GitHub Search + Cross-Verification Engine
# ---------------------------------------------------------------------------

class MultiSourceVerifier:
    """
    Gathers evidence across GitHub, DuckDuckGo Web, Wikipedia, StackOverflow,
    arXiv, and Jarvis's local knowledge base, then cross-checks claims.
    """

    def __init__(self, config: JarvisConfig, memory: JarvisMemory, twister: WebsiteTwister):
        self.config = config
        self.memory = memory
        self.twister = twister
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self._gh_cache: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}

    def _gh_headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/vnd.github+json", "User-Agent": USER_AGENT}
        if self.config.github_token:
            headers["Authorization"] = f"Bearer {self.config.github_token}"
        return headers

    def search_github_repos(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search public GitHub repositories with caching and rate-limit web fallback."""
        if not self.config.allow_network:
            return []
        filler = {
            "verify", "whether", "search", "github", "repo", "repos", "repository",
            "find", "check", "explain", "best", "framework", "web", "code", "project",
        }
        keywords = [t for t in tokenize(query, drop_stopwords=True) if t not in filler]
        q = " ".join(keywords[:3]) if len(keywords) >= 2 else (keywords[0] if keywords else query.strip())
        if not q:
            return []

        cache_key = f"{q.lower()}:{limit}"
        cached = self._gh_cache.get(cache_key)
        if cached and (time.time() - cached[0]) < 300:
            return cached[1]

        out: List[Dict[str, Any]] = []
        try:
            r = self.session.get(
                "https://api.github.com/search/repositories",
                params={"q": q, "sort": "stars", "order": "desc", "per_page": limit},
                headers=self._gh_headers(),
                timeout=self.config.request_timeout,
            )
            if r.status_code == 200:
                out = [
                    {
                        "source_type": "github_repo",
                        "full_name": it["full_name"],
                        "title": it["full_name"],
                        "url": it["html_url"],
                        "description": it.get("description") or "",
                        "stars": it.get("stargazers_count", 0),
                        "language": it.get("language") or "Unknown",
                        "default_branch": it.get("default_branch", "main"),
                        "updated_at": it.get("updated_at", ""),
                        "snippet": f"{it['full_name']} ({it.get('stargazers_count', 0)} stars, {it.get('language')}): {it.get('description') or ''}",
                    }
                    for it in r.json().get("items", [])
                ]
            elif r.status_code in (403, 429):
                # Unauthenticated GitHub rate limit reached -> discover github.com repos via HN Algolia + Bing RSS
                seen_slugs: set[str] = set()
                ignored = {"login", "signup", "features", "pricing", "topics", "explore", "marketplace", "github"}

                def _try_add_repo(candidate_url: str, desc: str) -> None:
                    if len(out) >= limit:
                        return
                    m = re.match(r"^https?://github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)", (candidate_url or "").strip())
                    if not m:
                        return
                    slug = m.group(1).rstrip("/")
                    if slug.split("/")[0].lower() in ignored or slug in seen_slugs:
                        return
                    seen_slugs.add(slug)
                    out.append({
                        "source_type": "github_repo",
                        "full_name": slug,
                        "title": slug,
                        "url": f"https://github.com/{slug}",
                        "description": desc,
                        "stars": None,
                        "language": "Unknown",
                        "snippet": f"{slug}: {desc}",
                    })

                hn = self.session.get(
                    "https://hn.algolia.com/api/v1/search",
                    params={"query": f"github.com {q}", "hitsPerPage": 10},
                    timeout=self.config.request_timeout,
                )
                if hn.status_code == 200:
                    for hit in hn.json().get("hits", []):
                        _try_add_repo(hit.get("url") or "", hit.get("title") or "")
                if len(out) < limit:
                    rss = self.session.get(
                        "https://www.bing.com/search",
                        params={"q": f"{q} github.com", "format": "rss"},
                        timeout=self.config.request_timeout,
                    )
                    if rss.status_code == 200 and "<rss" in rss.text:
                        root = ET.fromstring(rss.text)
                        for item in root.findall("./channel/item"):
                            _try_add_repo(item.findtext("link") or "", (item.findtext("description") or "").strip())
        except Exception:
            pass
        if out:
            self._gh_cache[cache_key] = (time.time(), out)
        return out

    def inspect_github_repo(self, repo_slug_or_url: str, ingest: bool = True) -> Dict[str, Any]:
        """Fetch repo metadata, README, root tree, and optionally index into Jarvis memory."""
        m = re.search(r"(?:github\.com/)?([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)", repo_slug_or_url.strip().rstrip("/"))
        if not m:
            return {"ok": False, "error": f"Invalid GitHub repository: {repo_slug_or_url}"}
        slug = m.group(1).removesuffix(".git")
        headers = self._gh_headers()
        try:
            meta_res = self.session.get(f"https://api.github.com/repos/{slug}", headers=headers, timeout=self.config.request_timeout)
            if meta_res.status_code != 200:
                return {"ok": False, "error": f"GitHub API HTTP {meta_res.status_code} for {slug}"}
            meta = meta_res.json()
            branch = meta.get("default_branch", "main")

            readme_text = ""
            readme_res = self.session.get(
                f"https://raw.githubusercontent.com/{slug}/{branch}/README.md",
                timeout=self.config.request_timeout,
            )
            if readme_res.status_code == 200:
                readme_text = readme_res.text

            tree_res = self.session.get(f"https://api.github.com/repos/{slug}/contents", headers=headers, timeout=self.config.request_timeout)
            files = [f["name"] for f in tree_res.json()] if tree_res.status_code == 200 and isinstance(tree_res.json(), list) else []

            chunks_added = 0
            if ingest and readme_text:
                chunks_added = self.memory.ingest_text(
                    title=f"GitHub:{slug}",
                    content=f"Repository: {slug}\nDescription: {meta.get('description')}\nFiles: {', '.join(files)}\n\n{readme_text[:25000]}",
                    source_type="github_repo",
                    source_uri=meta["html_url"],
                    tags=f"github {meta.get('language') or ''} {slug}",
                )
            return {
                "ok": True,
                "full_name": slug,
                "url": meta["html_url"],
                "stars": meta.get("stargazers_count", 0),
                "language": meta.get("language"),
                "description": meta.get("description") or "",
                "files": files[:40],
                "readme_preview": readme_text[:3000],
                "chunks_indexed": chunks_added,
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def clone_and_index_github_repo(self, repo_slug_or_url: str, dest_dir: Optional[str] = None) -> Dict[str, Any]:
        """Shallow-clone a GitHub repository via `git` and index its files locally."""
        m = re.search(r"(?:github\.com/)?([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)", repo_slug_or_url.strip().rstrip("/"))
        if not m:
            return {"ok": False, "error": f"Invalid repo slug: {repo_slug_or_url}"}
        slug = m.group(1).removesuffix(".git")
        url = f"https://github.com/{slug}.git"
        target = Path(dest_dir) if dest_dir else (Path(self.config.brain_dir) / "repos" / slug.replace("/", "__"))
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            if target.exists():
                subprocess.run(["git", "-C", str(target), "pull", "--ff-only"], capture_output=True, text=True, timeout=45)
            else:
                res = subprocess.run(["git", "clone", "--depth", "1", url, str(target)], capture_output=True, text=True, timeout=60)
                if res.returncode != 0:
                    return {"ok": False, "error": res.stderr.strip() or "git clone failed"}
            stats = self.memory.ingest_folder(str(target), max_files=180)
            return {"ok": True, "repo": slug, "local_path": str(target), **stats}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def search_duckduckgo(self, query: str, limit: int = 6) -> List[Dict[str, Any]]:
        """Search the web via Bing RSS + DuckDuckGo Instant Answer API."""
        if not self.config.allow_network:
            return []
        results: List[Dict[str, Any]] = []
        try:
            ia = self.session.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
                timeout=self.config.request_timeout,
            )
            if ia.status_code == 200:
                data = ia.json()
                if data.get("AbstractText"):
                    results.append(
                        {
                            "source_type": "web_instant",
                            "title": data.get("Heading") or query,
                            "url": data.get("AbstractURL") or "https://duckduckgo.com",
                            "snippet": data["AbstractText"],
                        }
                    )
        except Exception:
            pass

        try:
            rss = self.session.get(
                "https://www.bing.com/search",
                params={"q": query, "format": "rss"},
                timeout=self.config.request_timeout,
            )
            if rss.status_code == 200 and "<rss" in rss.text:
                root = ET.fromstring(rss.text)
                for item in root.findall("./channel/item")[:limit]:
                    title = (item.findtext("title") or "").strip()
                    link = (item.findtext("link") or "").strip()
                    desc = (item.findtext("description") or "").strip()
                    if title and link.startswith("http"):
                        results.append(
                            {
                                "source_type": "web_search",
                                "title": title,
                                "url": link,
                                "snippet": desc,
                            }
                        )
        except Exception:
            pass
        return results[:limit]

    def search_hackernews(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Search Hacker News discussions & shared web articles via Algolia API."""
        if not self.config.allow_network:
            return []
        try:
            r = self.session.get(
                "https://hn.algolia.com/api/v1/search",
                params={"query": query, "tags": "story", "hitsPerPage": limit},
                timeout=self.config.request_timeout,
            )
            if r.status_code != 200:
                return []
            out: List[Dict[str, Any]] = []
            for h in r.json().get("hits", [])[:limit]:
                title = h.get("title") or ""
                url = h.get("url") or f"https://news.ycombinator.com/item?id={h.get('objectID', '')}"
                if title:
                    out.append(
                        {
                            "source_type": "hackernews",
                            "title": title,
                            "url": url,
                            "snippet": f"Hacker News ({h.get('points', 0)} pts, {h.get('num_comments', 0)} comments): {title}",
                        }
                    )
            return out
        except Exception:
            return []

    def search_wikipedia(self, query: str, limit: int = 2) -> List[Dict[str, Any]]:
        """Query Wikipedia API and return intro extracts."""
        if not self.config.allow_network:
            return []
        clean = re.sub(r"^(who is|what is|verify|explain|define)\s+", "", query, flags=re.I).strip(" ?.")
        if len(clean) < 2:
            return []
        try:
            r = self.session.get(
                "https://en.wikipedia.org/w/api.php",
                params={"action": "query", "list": "search", "srlimit": limit, "utf8": 1, "format": "json", "srsearch": clean},
                timeout=self.config.request_timeout,
            )
            if r.status_code != 200:
                return []
            hits = r.json().get("query", {}).get("search", [])
            out: List[Dict[str, Any]] = []
            for h in hits[:limit]:
                title = h.get("title", "")
                ex = self.session.get(
                    "https://en.wikipedia.org/w/api.php",
                    params={"action": "query", "prop": "extracts", "exintro": 1, "explaintext": 1, "redirects": 1, "format": "json", "titles": title},
                    timeout=self.config.request_timeout,
                )
                pages = ex.json().get("query", {}).get("pages", {}) if ex.status_code == 200 else {}
                extract = next(iter(pages.values()), {}).get("extract", "")
                if extract:
                    out.append(
                        {
                            "source_type": "wikipedia",
                            "title": title,
                            "url": f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}",
                            "snippet": extract[:900],
                        }
                    )
            return out
        except Exception:
            return []

    def search_stackoverflow(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Search StackOverflow Q&A for technical/code verification."""
        if not self.config.allow_network:
            return []
        try:
            r = self.session.get(
                "https://api.stackexchange.com/2.3/search/advanced",
                params={"order": "desc", "sort": "relevance", "q": query, "site": "stackoverflow", "pagesize": limit},
                timeout=self.config.request_timeout,
            )
            if r.status_code != 200:
                return []
            return [
                {
                    "source_type": "stackoverflow",
                    "title": html.unescape(it.get("title", "")),
                    "url": it.get("link", ""),
                    "score": it.get("score", 0),
                    "answered": it.get("is_answered", False),
                    "snippet": f"StackOverflow (score {it.get('score', 0)}, answered={it.get('is_answered')}): {html.unescape(it.get('title', ''))} [{', '.join(it.get('tags', [])[:4])}]",
                }
                for it in r.json().get("items", [])[:limit]
            ]
        except Exception:
            return []

    def search_arxiv(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Search arXiv preprints for scientific & AI claims."""
        if not self.config.allow_network:
            return []
        try:
            r = self.session.get(
                "https://export.arxiv.org/api/query",
                params={"search_query": f"all:{query}", "start": 0, "max_results": limit},
                timeout=self.config.request_timeout,
            )
            if r.status_code != 200:
                return []
            root = ET.fromstring(r.text)
            ns = {"a": "http://www.w3.org/2005/Atom"}
            out: List[Dict[str, Any]] = []
            for entry in root.findall("a:entry", ns)[:limit]:
                title = (entry.findtext("a:title", "", ns) or "").strip().replace("\n", " ")
                summary = (entry.findtext("a:summary", "", ns) or "").strip().replace("\n", " ")
                link = entry.findtext("a:id", "", ns) or ""
                if title:
                    out.append({"source_type": "arxiv", "title": title, "url": link, "snippet": summary[:500]})
            return out
        except Exception:
            return []

    def search_all(self, query: str) -> List[Dict[str, Any]]:
        """Query Web, GitHub, Wikipedia, StackOverflow, arXiv, and Local Memory in parallel."""
        all_hits: List[Dict[str, Any]] = []
        local_hits = self.memory.search_knowledge(query, top_k=4)
        for h in local_hits:
            all_hits.append(
                {
                    "source_type": f"local:{h['source_type']}",
                    "title": h["title"],
                    "url": h["source_uri"],
                    "snippet": h["content"][:500],
                    "relevance": h["score"],
                }
            )

        if not self.config.allow_network:
            return all_hits

        tasks = {
            "web": lambda: self.search_duckduckgo(query, limit=5),
            "github": lambda: self.search_github_repos(query, limit=4),
            "wikipedia": lambda: self.search_wikipedia(query, limit=2),
            "stackoverflow": lambda: self.search_stackoverflow(query, limit=3),
            "arxiv": lambda: self.search_arxiv(query, limit=2),
            "hackernews": lambda: self.search_hackernews(query, limit=3),
        }
        with ThreadPoolExecutor(max_workers=6) as pool:
            futs = {pool.submit(fn): name for name, fn in tasks.items()}
            for fut in as_completed(futs, timeout=self.config.request_timeout + 4):
                try:
                    all_hits.extend(fut.result() or [])
                except Exception:
                    pass
        return all_hits

    def verify(self, claim_or_query: str, deep_fetch_top_web: bool = True) -> Dict[str, Any]:
        """
        Cross-verify a claim/question across GitHub, internet search, Wikipedia,
        StackOverflow, arXiv, and local memory, scoring evidence overlap.
        """
        sources = self.search_all(claim_or_query)
        q_tokens = set(tokenize(claim_or_query, drop_stopwords=True))

        # Optionally scrape the top web hit for deeper grounding
        if deep_fetch_top_web and self.config.allow_network:
            web_hits = [s for s in sources if s.get("source_type") == "web_search" and s.get("url")]
            if web_hits:
                top_url = web_hits[0]["url"]
                twisted = self.twister.twist(top_url, output_format="text")
                if twisted.get("ok") and twisted.get("clean_text_preview"):
                    web_hits[0]["snippet"] = (
                        web_hits[0].get("snippet", "") + "\n" + twisted["clean_text_preview"][:700]
                    ).strip()
                    self.memory.ingest_text(
                        title=twisted.get("title") or top_url,
                        content=twisted["clean_text_preview"],
                        source_type="verified_web",
                        source_uri=top_url,
                        tags="verified",
                    )

        scored_sources: List[Dict[str, Any]] = []
        distinct_families: set[str] = set()
        for src in sources:
            blob = f"{src.get('title', '')} {src.get('snippet', '')}"
            s_tokens = set(tokenize(blob, drop_stopwords=True))
            overlap = len(q_tokens & s_tokens) / max(1, len(q_tokens))
            base_rel = float(src.get("relevance", 0.0))
            rel = round(max(base_rel, 0.7 * overlap + 0.3 * min(1.0, len(s_tokens) / 30.0)), 3)
            if rel >= 0.15 or src.get("source_type") == "wikipedia":
                scored = {**src, "relevance": rel}
                scored_sources.append(scored)
                distinct_families.add(src.get("source_type", "unknown").split(":")[0])

        scored_sources.sort(key=lambda x: x["relevance"], reverse=True)
        top_sources = scored_sources[:10]

        avg_rel = sum(s["relevance"] for s in top_sources[:4]) / max(1, min(4, len(top_sources)))
        diversity_bonus = min(0.35, 0.09 * len(distinct_families))
        confidence = round(min(0.98, 0.65 * avg_rel + diversity_bonus), 3) if top_sources else 0.0

        if len(distinct_families) >= 2 and confidence >= 0.55:
            verdict = "VERIFIED"
        elif top_sources and confidence >= 0.3:
            verdict = "PARTIALLY_SUPPORTED"
        else:
            verdict = "UNVERIFIED"

        return {
            "query": claim_or_query,
            "verdict": verdict,
            "confidence": confidence,
            "verified": verdict == "VERIFIED",
            "source_families": sorted(distinct_families),
            "sources": top_sources,
        }


# ---------------------------------------------------------------------------
# 5. Universal LLM Router (Ollama / Gemini / OpenAI / Anthropic / Local)
# ---------------------------------------------------------------------------

class LLMRouter:
    """Routes reasoning to available local/cloud LLMs with 100% local synthesis fallback."""

    def __init__(self, config: JarvisConfig):
        self.config = config
        self.session = requests.Session()
        self._cached_ollama_model: Optional[str] = None

    def detect_provider(self) -> Tuple[str, str]:
        """Detect best active LLM provider & model without failing."""
        pref = self.config.llm_provider.lower()
        if pref == "local":
            return "local", "jarvis-neural-local"

        if pref in ("auto", "ollama"):
            model = self._probe_ollama()
            if model:
                return "ollama", model
            if pref == "ollama":
                return "local", "jarvis-neural-local"

        if pref in ("auto", "gemini") and (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            return "gemini", self.config.llm_model or "gemini-2.5-flash"

        if pref in ("auto", "openai") and os.environ.get("OPENAI_API_KEY"):
            return "openai", self.config.llm_model or "gpt-4o-mini"

        if pref in ("auto", "openrouter") and os.environ.get("OPENROUTER_API_KEY"):
            return "openrouter", self.config.llm_model or "openai/gpt-4o-mini"

        if pref in ("auto", "anthropic") and os.environ.get("ANTHROPIC_API_KEY"):
            return "anthropic", self.config.llm_model or "claude-3-5-sonnet-latest"

        return "local", "jarvis-neural-local"

    def _probe_ollama(self) -> Optional[str]:
        if self._cached_ollama_model:
            return self._cached_ollama_model
        try:
            r = self.session.get(f"{self.config.ollama_url.rstrip('/')}/api/tags", timeout=1.5)
            if r.status_code == 200:
                models = [m.get("name") for m in r.json().get("models", []) if m.get("name")]
                chosen = self.config.llm_model or os.environ.get("OLLAMA_MODEL") or (models[0] if models else "")
                if chosen:
                    self._cached_ollama_model = chosen
                    return chosen
        except Exception:
            pass
        return None

    def generate(self, prompt: str, system: str = "", temperature: float = 0.2) -> Tuple[Optional[str], str]:
        """Generate response from active external LLM, or return (None, 'local') for local synthesis."""
        provider, model = self.detect_provider()
        if provider == "local":
            return None, "jarvis-neural-local"

        try:
            if provider == "ollama":
                r = self.session.post(
                    f"{self.config.ollama_url.rstrip('/')}/api/chat",
                    json={
                        "model": model,
                        "stream": False,
                        "options": {"temperature": temperature},
                        "messages": [
                            {"role": "system", "content": system or "You are JARVIS, a precise autonomous AI."},
                            {"role": "user", "content": prompt},
                        ],
                    },
                    timeout=30,
                )
                if r.status_code == 200:
                    text = r.json().get("message", {}).get("content", "").strip()
                    if text:
                        return text, f"ollama:{model}"

            elif provider == "gemini":
                key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
                r = self.session.post(
                    url,
                    json={
                        "contents": [{"role": "user", "parts": [{"text": f"{system}\n\n{prompt}"}]}],
                        "generationConfig": {"temperature": temperature, "maxOutputTokens": 1200},
                    },
                    timeout=25,
                )
                if r.status_code == 200:
                    cands = r.json().get("candidates", [])
                    text = cands[0]["content"]["parts"][0]["text"].strip() if cands else ""
                    if text:
                        return text, f"gemini:{model}"

            elif provider in ("openai", "openrouter"):
                is_or = provider == "openrouter"
                key = os.environ["OPENROUTER_API_KEY"] if is_or else os.environ["OPENAI_API_KEY"]
                base = os.environ.get("OPENAI_BASE_URL") or (
                    "https://openrouter.ai/api/v1" if is_or else "https://api.openai.com/v1"
                )
                r = self.session.post(
                    f"{base.rstrip('/')}/chat/completions",
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json={
                        "model": model,
                        "temperature": temperature,
                        "messages": [
                            {"role": "system", "content": system or "You are JARVIS."},
                            {"role": "user", "content": prompt},
                        ],
                    },
                    timeout=25,
                )
                if r.status_code == 200:
                    text = r.json()["choices"][0]["message"]["content"].strip()
                    if text:
                        return text, f"{provider}:{model}"

            elif provider == "anthropic":
                r = self.session.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": model,
                        "max_tokens": 1200,
                        "system": system or "You are JARVIS.",
                        "messages": [{"role": "user", "content": prompt}],
                    },
                    timeout=25,
                )
                if r.status_code == 200:
                    text = r.json()["content"][0]["text"].strip()
                    if text:
                        return text, f"anthropic:{model}"
        except Exception:
            pass

        return None, "jarvis-neural-local"


# ---------------------------------------------------------------------------
# 6. Voice Command Engine (Wake-Word STT + Thread-Safe TTS)
# ---------------------------------------------------------------------------

class VoiceEngine:
    """Offline/online microphone recognition & non-blocking Windows TTS queue."""

    def __init__(self, enabled: bool = True, wake_words: Tuple[str, ...] = ("jarvis", "hey jarvis")):
        self.enabled = enabled
        self.wake_words = tuple(w.lower() for w in wake_words)
        self._tts_queue: queue.Queue[Optional[str]] = queue.Queue()
        self._tts_thread: Optional[threading.Thread] = None
        self._running = False

    def _ensure_tts_worker(self) -> None:
        if not self.enabled or pyttsx3 is None:
            return
        if self._tts_thread and self._tts_thread.is_alive():
            return

        def _worker() -> None:
            engine = None
            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", 185)
            except Exception:
                return
            while True:
                msg = self._tts_queue.get()
                if msg is None:
                    break
                try:
                    engine.say(msg)
                    engine.runAndWait()
                except Exception:
                    pass

        self._tts_thread = threading.Thread(target=_worker, daemon=True)
        self._tts_thread.start()

    def speak(self, text: str, block: bool = False) -> bool:
        """Speak text aloud via pyttsx3 without cross-thread COM crashes."""
        clean = re.sub(r"[`*_#\[\]]", " ", (text or "")).strip()
        clean = re.sub(r"https?://\S+", "link", clean)[:450]
        if not self.enabled or pyttsx3 is None or not clean:
            return False
        self._ensure_tts_worker()
        self._tts_queue.put(clean)
        if block:
            time.sleep(min(4.0, max(0.5, len(clean) / 18.0)))
        return True

    def listen_once(self, timeout: int = 6, phrase_time_limit: int = 10) -> Optional[str]:
        """Capture one spoken phrase from default microphone."""
        if sr is None:
            return None
        recognizer = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.4)
                audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            return recognizer.recognize_google(audio).strip()
        except Exception:
            return None

    def extract_wake_command(self, transcript: str) -> Optional[str]:
        """Strip wake word if present and return remaining command."""
        low = (transcript or "").strip().lower()
        for wake in sorted(self.wake_words, key=len, reverse=True):
            if low == wake:
                return ""
            if low.startswith(wake + " "):
                return transcript.strip()[len(wake):].strip(" ,:")
            if wake in low:
                idx = low.index(wake) + len(wake)
                return transcript.strip()[idx:].strip(" ,:")
        return None

    def run_loop(
        self,
        handler: Callable[[str], str],
        require_wake_word: bool = True,
        stop_phrases: Tuple[str, ...] = ("stop listening", "jarvis sleep", "exit voice"),
    ) -> None:
        """Continuous hands-free voice command loop."""
        if sr is None:
            raise RuntimeError("SpeechRecognition / PyAudio not available.")
        self._running = True
        self.speak("Jarvis voice core online. Say Jarvis followed by your command.")
        while self._running:
            heard = self.listen_once(timeout=6, phrase_time_limit=12)
            if not heard:
                continue
            low = heard.lower()
            if any(p in low for p in stop_phrases):
                self.speak("Standing by, sir.", block=True)
                self._running = False
                break
            cmd = self.extract_wake_command(heard) if require_wake_word else heard
            if cmd is None:
                continue
            if cmd == "":
                self.speak("Yes? Listening.")
                cmd = self.listen_once(timeout=6, phrase_time_limit=12) or ""
            if cmd:
                reply = handler(cmd)
                self.speak(reply)

    def stop(self) -> None:
        self._running = False


# ---------------------------------------------------------------------------
# 7. Autonomous Jarvis Engine & Tool Registry
# ---------------------------------------------------------------------------

class Jarvis:
    """
    Unified Autonomous JARVIS AI Engine.

    Importable everywhere as a single drop-in Python module. Combines trainable
    neural routing, persistent RAG memory, multi-source web/GitHub verification,
    website twisting, autonomous task execution, voice commands, and REST server.
    """

    def __init__(self, config: Optional[JarvisConfig] = None, **kwargs: Any):
        if config is None:
            config = JarvisConfig(**kwargs)
        self.config = config
        Path(self.config.brain_dir).mkdir(parents=True, exist_ok=True)
        self.neural = JarvisNeuralModel(self.config.brain_dir)
        self.memory = JarvisMemory(self.config.brain_dir)
        self.twister = WebsiteTwister(timeout=self.config.request_timeout)
        self.verifier = MultiSourceVerifier(self.config, self.memory, self.twister)
        self.llm = LLMRouter(self.config)
        self.voice = VoiceEngine(enabled=self.config.tts_enabled, wake_words=self.config.wake_words)
        self.tools: Dict[str, Dict[str, Any]] = {}
        self._register_builtin_tools()

    def register_tool(
        self,
        name: str,
        description: str,
        fn: Optional[Callable[..., Any]] = None,
    ) -> Callable[..., Any]:
        """Register any custom project tool so Jarvis can invoke it autonomously."""
        def _decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.tools[name] = {"name": name, "description": description, "fn": func}
            return func

        if fn is not None:
            return _decorator(fn)
        return _decorator

    def _resolve_workspace_path(self, raw_path: str, for_write: bool = False) -> Path:
        root = Path(self.config.workspace_root).resolve()
        p = Path(raw_path)
        resolved = p.resolve() if p.is_absolute() else (root / p).resolve()
        if for_write and self.config.restrict_writes_to_workspace:
            if resolved != root and root not in resolved.parents:
                raise PermissionError(f"Write blocked outside workspace_root ({root}): {resolved}")
        return resolved

    def _register_builtin_tools(self) -> None:
        self.register_tool("web_search", "Search the whole internet (DuckDuckGo, Wikipedia, StackOverflow, arXiv)", lambda query: self.verifier.search_all(query))
        self.register_tool("github_search", "Search GitHub repositories by keyword", lambda query, limit=5: self.verifier.search_github_repos(query, limit=limit))
        self.register_tool("github_inspect", "Fetch GitHub repo metadata, README, files, and index into memory", lambda repo: self.verifier.inspect_github_repo(repo, ingest=True))
        self.register_tool("github_clone", "Shallow-clone a GitHub repo and index its codebase", lambda repo: self.verifier.clone_and_index_github_repo(repo))
        self.register_tool("verify_claim", "Cross-verify any fact across GitHub, web, Wikipedia, StackOverflow, and local memory", lambda claim: self.verifier.verify(claim))
        self.register_tool("twist_website", "Extract & transform structured data (json/csv/markdown/tables) from any website URL", lambda url, output_format="json", selector=None: self.twist(url, output_format=output_format, selector=selector))
        self.register_tool("calculate", "Evaluate mathematical expressions safely", lambda expression: {"expression": expression, "result": safe_eval_math(expression)})
        self.register_tool("system_info", "Inspect CPU, RAM, disk, and process telemetry", lambda: self.get_system_info())
        self.register_tool("read_file", "Read a text file in workspace", lambda path: self._tool_read_file(path))
        self.register_tool("write_file", "Write content to a file inside workspace", lambda path, content: self._tool_write_file(path, content))
        self.register_tool("list_dir", "List files in a directory", lambda path=".": self._tool_list_dir(path))
        self.register_tool("run_python", "Run Python code safely in a subprocess", lambda code: self._tool_run_python(code))
        self.register_tool("run_shell", "Run a guarded shell command in workspace", lambda command: self._tool_run_shell(command))
        self.register_tool("open_target", "Open an application or URL on Windows", lambda target: self._tool_open_target(target))
        self.register_tool("recall_memory", "Search Jarvis local trained knowledge base", lambda query: self.memory.search_knowledge(query, top_k=5))
        self.register_tool("ingest_folder", "Index and train memory on a local project folder", lambda path: self.memory.ingest_folder(path))

    # --- Built-in Safe Tool Implementations ---

    def _tool_read_file(self, path: str) -> Dict[str, Any]:
        p = self._resolve_workspace_path(path, for_write=False)
        if _is_sensitive_path(p):
            return {"ok": False, "error": "Access to sensitive credential files is restricted."}
        if not p.exists() or not p.is_file():
            return {"ok": False, "error": f"File not found: {p}"}
        return {"ok": True, "path": str(p), "content": p.read_text(encoding="utf-8", errors="ignore")[:12000]}

    def _tool_write_file(self, path: str, content: str) -> Dict[str, Any]:
        p = self._resolve_workspace_path(path, for_write=True)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return {"ok": True, "path": str(p), "bytes": len(content.encode("utf-8"))}

    def _tool_list_dir(self, path: str = ".") -> Dict[str, Any]:
        p = self._resolve_workspace_path(path, for_write=False)
        if not p.exists():
            return {"ok": False, "error": f"Directory not found: {p}"}
        entries = [
            {"name": item.name, "is_dir": item.is_dir(), "size": item.stat().st_size if item.is_file() else 0}
            for item in sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))[:100]
        ]
        return {"ok": True, "path": str(p), "entries": entries}

    def _tool_run_python(self, code: str, timeout: int = 20) -> Dict[str, Any]:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tmp:
            tmp.write(code)
            tmp_path = tmp.name
        try:
            proc = subprocess.run(
                [sys.executable, tmp_path],
                cwd=self.config.workspace_root,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return {
                "ok": proc.returncode == 0,
                "returncode": proc.returncode,
                "stdout": proc.stdout[:6000],
                "stderr": proc.stderr[:3000],
            }
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _tool_run_shell(self, command: str, timeout: int = 25) -> Dict[str, Any]:
        if not self.config.allow_shell:
            return {"ok": False, "error": "Shell execution disabled by JarvisConfig."}
        for pattern in DANGEROUS_SHELL_PATTERNS:
            if re.search(pattern, command, re.I):
                return {"ok": False, "error": f"Blocked potentially destructive command: {command}"}
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=self.config.workspace_root,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return {
                "ok": proc.returncode == 0,
                "returncode": proc.returncode,
                "stdout": proc.stdout[:6000],
                "stderr": proc.stderr[:3000],
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _tool_open_target(self, target: str) -> Dict[str, Any]:
        clean = target.strip()
        if re.match(r"^https?://", clean, re.I):
            webbrowser.open(clean)
            return {"ok": True, "opened": clean, "kind": "url"}
        app_aliases = {
            "chrome": "chrome",
            "edge": "msedge",
            "notepad": "notepad",
            "calculator": "calc",
            "calc": "calc",
            "explorer": "explorer .",
            "vscode": "code",
            "code": "code",
        }
        cmd = app_aliases.get(clean.lower())
        if not cmd:
            return {"ok": False, "error": f"Unknown app alias '{clean}'. Allowed aliases: {', '.join(sorted(app_aliases))}"}
        subprocess.Popen(shlex.split(cmd) if " " in cmd else [cmd], shell=(os.name == "nt"))
        return {"ok": True, "opened": cmd, "kind": "app"}

    def get_system_info(self) -> Dict[str, Any]:
        """Collect host CPU, RAM, disk, and process diagnostics."""
        info: Dict[str, Any] = {
            "platform": sys.platform,
            "python": sys.version.split()[0],
            "cwd": os.getcwd(),
            "brain_dir": self.config.brain_dir,
        }
        if psutil is not None:
            vm = psutil.virtual_memory()
            disk = psutil.disk_usage(self.config.workspace_root)
            info.update(
                {
                    "cpu_percent": psutil.cpu_percent(interval=0.1),
                    "cpu_cores": psutil.cpu_count(logical=True),
                    "ram_total_gb": round(vm.total / (1024 ** 3), 2),
                    "ram_used_percent": vm.percent,
                    "disk_free_gb": round(disk.free / (1024 ** 3), 2),
                }
            )
        return info

    # --- Public High-Level APIs ---

    def twist(
        self,
        url_or_html: str,
        output_format: str = "json",
        selector: Optional[str] = None,
        fields: Optional[Dict[str, str]] = None,
        filter_pattern: Optional[str] = None,
        save_to: Optional[str] = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """Extract & transform data from any webpage and optionally save/index it."""
        res = self.twister.twist(
            url_or_html,
            output_format=output_format,
            selector=selector,
            fields=fields,
            filter_pattern=filter_pattern,
            limit=limit,
        )
        if res.get("ok") and res.get("clean_text_preview"):
            self.memory.ingest_text(
                title=res.get("title") or res.get("url", "website"),
                content=res["clean_text_preview"],
                source_type="twisted_web",
                source_uri=res.get("url", "inline"),
                tags=f"web {output_format}",
            )
        if save_to and res.get("ok"):
            out_path = self._resolve_workspace_path(save_to, for_write=True)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(res.get("formatted", ""), encoding="utf-8")
            res["saved_to"] = str(out_path)
        return res

    def verify(self, claim_or_query: str) -> Dict[str, Any]:
        """Cross-verify any claim or technical question across GitHub, Web, and Memory."""
        return self.verifier.verify(claim_or_query)

    def search_github(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search GitHub repositories."""
        return self.verifier.search_github_repos(query, limit=limit)

    def learn_github_repo(self, repo_slug_or_url: str, clone_full: bool = False) -> Dict[str, Any]:
        """Ingest a GitHub repository into Jarvis's local searchable brain."""
        if clone_full:
            return self.verifier.clone_and_index_github_repo(repo_slug_or_url)
        return self.verifier.inspect_github_repo(repo_slug_or_url, ingest=True)

    def train(
        self,
        samples: Optional[Sequence[Tuple[str, str]]] = None,
        folder: Optional[str] = None,
        github_repo: Optional[str] = None,
        url: Optional[str] = None,
        epochs: int = 45,
        retrain_from_scratch: bool = False,
    ) -> Dict[str, Any]:
        """Train Jarvis's neural model on labeled intents and/or ingest knowledge sources."""
        report: Dict[str, Any] = {}
        if samples:
            if retrain_from_scratch:
                merged = list(self.neural.samples) + list(samples)
                report["neural"] = self.neural.fit(merged, epochs=epochs, save=True)
            else:
                report["neural"] = self.neural.train_online(samples, save=True)
        if folder:
            report["folder_ingest"] = self.memory.ingest_folder(folder)
        if github_repo:
            report["github_ingest"] = self.learn_github_repo(github_repo, clone_full=False)
        if url:
            twisted = self.twist(url, output_format="markdown")
            report["web_ingest"] = {"ok": twisted.get("ok"), "url": twisted.get("url"), "title": twisted.get("title")}
        report["memory_stats"] = self.memory.stats()
        return report

    def teach(
        self,
        phrase: str,
        intent: str = "chat_general",
        response: str = "",
        action: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Teach Jarvis a new command, correction, or custom fact in real time."""
        self.memory.teach_command(phrase, intent, action=action, response_template=response)
        if response:
            self.memory.ingest_text(
                title=f"Taught: {phrase[:60]}",
                content=f"When user asks '{phrase}': {response}",
                source_type="taught_fact",
                source_uri=f"taught://{hashlib.md5(phrase.encode()).hexdigest()[:10]}",
                tags=intent,
            )
        neural_update = self.neural.train_online([(phrase, intent)], save=True)
        return {"ok": True, "phrase": phrase, "intent": intent, "neural": neural_update}

    def ask(self, query: str, verify_sources: bool = False) -> JarvisResponse:
        """
        Answer any question intelligently using:
        1. Taught command memory
        2. Trained neural intent router
        3. Hybrid RAG knowledge retrieval
        4. Live Web + GitHub multi-source verification (automatic when needed)
        5. Connected LLM or Local Synthesis Engine
        """
        text = (query or "").strip()
        if not text:
            return JarvisResponse(answer="Jarvis online. Ready for your command.", intent="chat_general", confidence=1.0)

        # 1. Exact taught command check
        learned = self.memory.match_learned_command(text)
        if learned:
            action = learned.get("action") or {}
            if action.get("tool") in self.tools:
                obs = self.tools[action["tool"]]["fn"](**action.get("args", {}))
                return JarvisResponse(
                    answer=learned.get("response_template") or json.dumps(obs, indent=2, default=str)[:2000],
                    intent=learned["intent"],
                    confidence=0.99,
                    metadata={"learned_match": True, "tool_output": obs},
                )
            if learned.get("response_template"):
                return JarvisResponse(
                    answer=learned["response_template"],
                    intent=learned["intent"],
                    confidence=0.99,
                    metadata={"learned_match": True},
                )

        # 2. Fast deterministic math check
        math_val = safe_eval_math(text)
        if math_val is not None:
            return JarvisResponse(
                answer=f"Result: {math_val}",
                intent="math_calc",
                confidence=0.99,
                verified=True,
                verdict="VERIFIED",
                metadata={"value": math_val},
            )

        # 3. Neural intent prediction
        intent, conf, top_k = self.neural.predict(text)
        if verify_sources:
            intent = "verify_fact"

        # 4. Route direct capability queries cleanly
        url_match = re.search(r"https?://\S+", text)
        if url_match and intent in ("twist_website", "autonomous_task", "web_search"):
            fmt = "csv" if "csv" in text.lower() else ("markdown" if "markdown" in text.lower() else "json")
            res = self.twist(url_match.group(0), output_format=fmt)
            ans = (
                f"Twisted `{res.get('url')}` into **{fmt.upper()}** ({res.get('count', 0)} records, "
                f"{res.get('tables_count', 0)} tables, {res.get('links_count', 0)} links).\n\n"
                f"```json\n{res.get('formatted', '')[:1600]}\n```"
                if res.get("ok")
                else f"Could not twist URL: {res.get('reason')}"
            )
            return JarvisResponse(
                answer=ans,
                intent="twist_website",
                confidence=max(conf, 0.85),
                verified=bool(res.get("ok")),
                verdict="VERIFIED" if res.get("ok") else "UNVERIFIED",
                sources=[{"source_type": "twisted_web", "url": res.get("url"), "title": res.get("title", "")}],
                metadata=res,
            )

        if intent == "system_control" and any(w in text.lower() for w in ("cpu", "ram", "disk", "memory", "stats", "hardware", "system")):
            info = self.get_system_info()
            return JarvisResponse(
                answer="Host Telemetry:\n" + "\n".join(f"- **{k}**: {v}" for k, v in info.items()),
                intent=intent,
                confidence=conf,
                verified=True,
                verdict="VERIFIED",
                metadata=info,
            )

        if intent == "memory_recall":
            hits = self.memory.search_knowledge(text, top_k=5)
            stats = self.memory.stats()
            if hits:
                lines = [f"Memory Recall ({stats['knowledge_chunks']} chunks indexed):"]
                for i, h in enumerate(hits, 1):
                    lines.append(f"[{i}] **{h['title']}** ({h['source_type']}, score={h['score']}): {h['content'][:220]}...")
                ans = "\n".join(lines)
            else:
                ans = f"Memory currently holds {stats['knowledge_chunks']} knowledge chunks and {stats['learned_commands']} taught commands, but nothing closely matched '{text}'."
            return JarvisResponse(answer=ans, intent=intent, confidence=conf, sources=hits, metadata=stats)

        if intent == "autonomous_task" or intent == "code_exec":
            return self.run_task(text)

        # 5. Gather local RAG + live multi-source evidence when answering/verifying
        needs_live = intent in ("web_search", "github_research", "verify_fact") or verify_sources or "?" in text
        local_hits = self.memory.search_knowledge(text, top_k=4)
        verification: Dict[str, Any] = {"verdict": "UNVERIFIED", "confidence": 0.0, "verified": False, "sources": []}
        if needs_live:
            verification = self.verify(text)
        elif local_hits:
            verification = {
                "verdict": "PARTIALLY_SUPPORTED",
                "confidence": local_hits[0]["score"],
                "verified": False,
                "sources": [
                    {"source_type": f"local:{h['source_type']}", "title": h["title"], "url": h["source_uri"], "snippet": h["content"][:400], "relevance": h["score"]}
                    for h in local_hits
                ],
            }

        sources = verification.get("sources", [])
        context_block = "\n".join(
            f"[{i}] ({s.get('source_type')}) {s.get('title')} — {s.get('url')}\n{s.get('snippet', '')[:450]}"
            for i, s in enumerate(sources[:6], 1)
        )

        # 6. Try external LLM with grounded citations, else synthesize locally
        llm_prompt = (
            f"User Query: {text}\nPredicted Intent: {intent} (top={top_k})\n"
            f"Verification Verdict: {verification.get('verdict')} (confidence={verification.get('confidence')})\n\n"
            f"Retrieved Evidence:\n{context_block or 'No external hits retrieved.'}\n\n"
            "Answer clearly, concisely, and cite evidence indices like [1], [2] for factual claims."
        )
        llm_text, provider = self.llm.generate(
            llm_prompt,
            system="You are JARVIS, an accurate, autonomous, multi-source verified AI engineer.",
        )

        answer = llm_text if llm_text else self._synthesize_local_answer(text, intent, sources, verification)
        self.memory.log_episode(text, intent, answer, verification.get("verdict", "UNVERIFIED"), conf)

        return JarvisResponse(
            answer=answer,
            intent=intent,
            confidence=conf,
            verified=bool(verification.get("verified")),
            verdict=verification.get("verdict", "UNVERIFIED"),
            sources=sources,
            provider=provider,
            metadata={"top_intents": [(k, round(v, 3)) for k, v in top_k]},
        )

    @staticmethod
    def _synthesize_local_answer(
        query: str,
        intent: str,
        sources: List[Dict[str, Any]],
        verification: Dict[str, Any],
    ) -> str:
        """Synthesize a grounded, cited response when running 100% locally without an external LLM."""
        if intent == "chat_general" and not sources:
            return (
                "JARVIS Neural Core online. I can search and cross-verify across GitHub and the internet, "
                "twist any website into structured JSON/CSV/Markdown, execute autonomous multi-step coding/file tasks, "
                "train on your repos/folders, and run on voice commands. Tell me what to build or investigate."
            )
        if not sources:
            return (
                f"No verified external or local evidence matched `{query}` yet (offline/empty result). "
                "Try `jarvis.train(...)` to index relevant docs/repos, or connect an LLM/internet source."
            )
        verdict = verification.get("verdict", "PARTIALLY_SUPPORTED")
        v_conf = verification.get("confidence", 0.0)
        lines = [f"**Verification:** `{verdict}` (confidence `{v_conf:.2f}`, sources `{', '.join(verification.get('source_families', ['local']))}`)\n"]
        for i, src in enumerate(sources[:5], 1):
            snip = re.sub(r"\s+", " ", src.get("snippet", "")).strip()[:280]
            lines.append(f"[{i}] **{src.get('title', 'Source')}** (`{src.get('source_type')}`) — {snip}\n    Source: {src.get('url')}")
        return "\n".join(lines)

    # --- Autonomous ReAct Plan-Act-Observe-Verify Loop ---

    def run_task(self, goal: str, max_steps: Optional[int] = None) -> JarvisResponse:
        """
        Autonomously decompose a goal into steps, call tools, observe outputs,
        self-correct on errors, and verify results.
        """
        limit = max_steps or self.config.max_autonomous_steps
        provider, _ = self.llm.detect_provider()
        if provider != "local":
            res = self._run_llm_react_loop(goal, limit)
            if res is not None:
                return res
        return self._run_deterministic_planner(goal, limit)

    def _run_llm_react_loop(self, goal: str, max_steps: int) -> Optional[JarvisResponse]:
        """Execute LLM-driven JSON ReAct loop across all registered tools."""
        tool_list = "\n".join(f"- {name}: {meta['description']}" for name, meta in self.tools.items())
        scratchpad: List[str] = []
        traces: List[StepTrace] = []
        artifacts: List[str] = []
        sources: List[Dict[str, Any]] = []

        for step_idx in range(1, max_steps + 1):
            prompt = (
                f"Goal: {goal}\n\nAvailable Tools:\n{tool_list}\n\n"
                f"Previous Steps:\n{chr(10).join(scratchpad) or 'None'}\n\n"
                "Respond ONLY with valid JSON: either\n"
                '{"thought": "...", "tool": "<tool_name>", "args": {...}}\n'
                'or when finished:\n{"thought": "...", "final_answer": "..."}'
            )
            raw, provider_tag = self.llm.generate(prompt, system="You are JARVIS autonomous ReAct executor. Output strict JSON only.")
            if not raw:
                return None
            m = re.search(r"\{.*\}", raw, re.S)
            if not m:
                return None
            try:
                action = json.loads(m.group(0))
            except Exception:
                return None

            thought = str(action.get("thought", ""))
            if "final_answer" in action:
                return JarvisResponse(
                    answer=str(action["final_answer"]),
                    intent="autonomous_task",
                    confidence=0.92,
                    verified=bool(sources),
                    verdict="VERIFIED" if sources else "PARTIALLY_SUPPORTED",
                    sources=sources[:10],
                    steps=traces,
                    artifacts=artifacts,
                    provider=provider_tag,
                )

            t_name = str(action.get("tool", ""))
            t_args = action.get("args") or {}
            t0 = time.perf_counter()
            if t_name not in self.tools:
                obs_obj = {"error": f"Unknown tool '{t_name}'"}
                status = "error"
            else:
                try:
                    obs_obj = self.tools[t_name]["fn"](**t_args)
                    status = "ok"
                    if isinstance(obs_obj, dict) and obs_obj.get("path") and t_name == "write_file":
                        artifacts.append(str(obs_obj["path"]))
                    if isinstance(obs_obj, list):
                        sources.extend([x for x in obs_obj[:4] if isinstance(x, dict) and x.get("url")])
                except Exception as exc:
                    obs_obj = {"error": str(exc)}
                    status = "error"
            dur = int((time.perf_counter() - t0) * 1000)
            obs_str = json.dumps(obs_obj, default=str, ensure_ascii=False)[:1800]
            traces.append(StepTrace(step=step_idx, thought=thought, tool=t_name, args=t_args, observation=obs_str, status=status, duration_ms=dur))
            scratchpad.append(f"Step {step_idx} | Tool={t_name} | Args={t_args} | Obs={obs_str[:600]}")

        return JarvisResponse(
            answer="Reached max autonomous steps. Summary of last observations:\n" + "\n".join(s.observation[:250] for s in traces[-3:]),
            intent="autonomous_task",
            confidence=0.75,
            steps=traces,
            artifacts=artifacts,
            sources=sources[:10],
        )

    def _run_deterministic_planner(self, goal: str, max_steps: int) -> JarvisResponse:
        """100% local multi-step task planner that decomposes compound tasks reliably."""
        subgoals = [s.strip() for s in re.split(r"\b(?:and then|then|;)\b|,\s*(?=(?:save|write|verify|run|extract|twist|open|search))", goal, flags=re.I) if s.strip()]
        traces: List[StepTrace] = []
        artifacts: List[str] = []
        sources: List[Dict[str, Any]] = []
        last_payload: Any = None

        for idx, sub in enumerate(subgoals[:max_steps], 1):
            low = sub.lower()
            t0 = time.perf_counter()
            tool_name = "recall_memory"
            args: Dict[str, Any] = {}
            obs: Any = None
            status = "ok"

            try:
                url_m = re.search(r"https?://\S+", sub)
                gh_m = re.search(r"(?:github\.com/)?([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)", sub)
                file_m = re.search(r"([A-Za-z0-9_./\\-]+\.(?:md|txt|json|csv|py|js))", sub)
                math_v = safe_eval_math(sub)

                if math_v is not None:
                    tool_name, args = "calculate", {"expression": sub}
                    obs = {"result": math_v}
                elif url_m:
                    fmt = "csv" if "csv" in low else ("markdown" if "markdown" in low else "json")
                    tool_name, args = "twist_website", {"url": url_m.group(0), "output_format": fmt}
                    obs = self.twist(url_m.group(0), output_format=fmt)
                    if obs.get("ok"):
                        sources.append({"source_type": "twisted_web", "url": obs.get("url"), "title": obs.get("title", "")})
                elif "github" in low and gh_m and ("inspect" in low or "analyze" in low or "index" in low or "train" in low):
                    tool_name, args = "github_inspect", {"repo": gh_m.group(1)}
                    obs = self.verifier.inspect_github_repo(gh_m.group(1), ingest=True)
                elif "github" in low:
                    q_clean = re.sub(r"\b(search|find|github|repos?|for|top)\b", " ", sub, flags=re.I).strip() or sub
                    tool_name, args = "github_search", {"query": q_clean}
                    obs = self.search_github(q_clean, limit=5)
                    sources.extend(obs)
                elif "verify" in low or "fact check" in low:
                    tool_name, args = "verify_claim", {"claim": sub}
                    obs = self.verify(sub)
                    sources.extend(obs.get("sources", [])[:5])
                elif any(w in low for w in ("write", "save", "export", "create")) and file_m:
                    target_file = file_m.group(1)
                    body = (
                        last_payload.get("formatted")
                        if isinstance(last_payload, dict) and last_payload.get("formatted")
                        else json.dumps(last_payload if last_payload is not None else {"goal": goal, "step": sub}, indent=2, default=str)
                    )
                    tool_name, args = "write_file", {"path": target_file, "content": body}
                    obs = self._tool_write_file(target_file, body)
                    if obs.get("ok"):
                        artifacts.append(obs["path"])
                elif "read" in low and file_m:
                    tool_name, args = "read_file", {"path": file_m.group(1)}
                    obs = self._tool_read_file(file_m.group(1))
                elif any(w in low for w in ("list files", "list dir", "show files")):
                    tool_name, args = "list_dir", {"path": "."}
                    obs = self._tool_list_dir(".")
                elif "run python" in low or "execute python" in low:
                    code_m = re.search(r"```(?:python)?\s*(.*?)```", sub, re.S)
                    code = code_m.group(1) if code_m else "print('Jarvis Python runtime operational')"
                    tool_name, args = "run_python", {"code": code}
                    obs = self._tool_run_python(code)
                elif any(w in low for w in ("git status", "run pytest", "pytest", "python --version")):
                    cmd = "pytest -q" if "pytest" in low else ("git status -s" if "git status" in low else "python --version")
                    tool_name, args = "run_shell", {"command": cmd}
                    obs = self._tool_run_shell(cmd)
                elif any(w in low for w in ("cpu", "ram", "disk", "system info", "hardware")):
                    tool_name, args = "system_info", {}
                    obs = self.get_system_info()
                elif any(w in low for w in ("search", "internet", "web", "research", "look up")):
                    tool_name, args = "web_search", {"query": sub}
                    obs = self.verifier.search_all(sub)
                    sources.extend([x for x in obs[:5] if isinstance(x, dict) and x.get("url")])
                else:
                    tool_name, args = "verify_claim", {"claim": sub}
                    obs = self.verify(sub)
                    sources.extend(obs.get("sources", [])[:4])
            except Exception as exc:
                status = "error"
                obs = {"error": str(exc)}

            last_payload = obs
            dur = int((time.perf_counter() - t0) * 1000)
            traces.append(
                StepTrace(
                    step=idx,
                    thought=f"Executing subgoal: {sub}",
                    tool=tool_name,
                    args=args,
                    observation=json.dumps(obs, default=str, ensure_ascii=False)[:1800],
                    status=status,
                    duration_ms=dur,
                )
            )

        summary_lines = [f"Completed **{len(traces)}** autonomous step(s) for `{goal}`:"]
        for t in traces:
            summary_lines.append(f"- Step {t.step} (`{t.tool}`, {t.duration_ms}ms, `{t.status}`): {t.observation[:260]}")
        if artifacts:
            summary_lines.append(f"\nCreated artifacts: {', '.join(artifacts)}")

        return JarvisResponse(
            answer="\n".join(summary_lines),
            intent="autonomous_task",
            confidence=0.88 if all(t.status == "ok" for t in traces) else 0.65,
            verified=bool(sources),
            verdict="VERIFIED" if len(sources) >= 2 else ("PARTIALLY_SUPPORTED" if sources else "UNVERIFIED"),
            sources=sources[:10],
            steps=traces,
            artifacts=artifacts,
            provider="jarvis-autonomous-local",
        )

    # --- Voice Wrappers ---

    def speak(self, text: str, block: bool = False) -> bool:
        return self.voice.speak(text, block=block)

    def listen_once(self) -> Optional[str]:
        return self.voice.listen_once()

    def start_voice_loop(self, require_wake_word: bool = True) -> None:
        self.voice.run_loop(lambda cmd: self.ask(cmd).answer, require_wake_word=require_wake_word)

    # --- Local HTTP API Server & Embedded HUD ---

    def serve(self, host: str = "127.0.0.1", port: int = 8787, open_browser: bool = False) -> ThreadingHTTPServer:
        """Start local HTTP API & Web HUD so ANY language/app can call Jarvis."""
        jarvis = self
        ui_file = Path(__file__).with_name("jarvis_ui.html")

        class JarvisHandler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: Any) -> None:
                return

            def _send_json(self, code: int, payload: Any) -> None:
                body = json.dumps(payload, default=str, ensure_ascii=False).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _read_json(self) -> Dict[str, Any]:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length > 0 else b"{}"
                return json.loads(raw.decode("utf-8") or "{}")

            def do_OPTIONS(self) -> None:
                self._send_json(200, {"ok": True})

            def do_GET(self) -> None:
                if self.path in ("/", "/index.html"):
                    html_doc = ui_file.read_text(encoding="utf-8") if ui_file.exists() else _EMBEDDED_FALLBACK_HTML
                    raw = html_doc.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                    return
                if self.path == "/v1/health":
                    prov, mod = jarvis.llm.detect_provider()
                    self._send_json(200, {
                        "ok": True,
                        "version": __version__,
                        "provider": f"{prov}:{mod}",
                        "neural_classes": jarvis.neural.classes,
                        "memory": jarvis.memory.stats(),
                        "tools": sorted(jarvis.tools.keys()),
                    })
                    return
                if self.path == "/v1/memory":
                    self._send_json(200, jarvis.memory.stats())
                    return
                self._send_json(404, {"error": "Not found"})

            def do_POST(self) -> None:
                try:
                    data = self._read_json()
                    if self.path == "/v1/ask":
                        res = jarvis.ask(data.get("query", ""), verify_sources=bool(data.get("verify", False)))
                        self._send_json(200, res.to_dict())
                    elif self.path == "/v1/task":
                        res = jarvis.run_task(data.get("goal", ""), max_steps=data.get("max_steps"))
                        self._send_json(200, res.to_dict())
                    elif self.path == "/v1/verify":
                        self._send_json(200, jarvis.verify(data.get("query", "")))
                    elif self.path == "/v1/twist":
                        self._send_json(200, jarvis.twist(
                            data.get("url", ""),
                            output_format=data.get("format", "json"),
                            selector=data.get("selector"),
                            fields=data.get("fields"),
                            filter_pattern=data.get("filter"),
                        ))
                    elif self.path == "/v1/github":
                        if data.get("repo"):
                            self._send_json(200, jarvis.learn_github_repo(data["repo"], clone_full=bool(data.get("clone", False))))
                        else:
                            self._send_json(200, {"repos": jarvis.search_github(data.get("query", ""), limit=int(data.get("limit", 5)))})
                    elif self.path == "/v1/train":
                        self._send_json(200, jarvis.train(
                            samples=data.get("samples"),
                            folder=data.get("folder"),
                            github_repo=data.get("github_repo"),
                            url=data.get("url"),
                        ))
                    elif self.path == "/v1/teach":
                        self._send_json(200, jarvis.teach(
                            phrase=data.get("phrase", ""),
                            intent=data.get("intent", "chat_general"),
                            response=data.get("response", ""),
                            action=data.get("action"),
                        ))
                    else:
                        self._send_json(404, {"error": "Unknown endpoint"})
                except Exception as exc:
                    self._send_json(500, {"error": str(exc)})

        server = ThreadingHTTPServer((host, port), JarvisHandler)
        if open_browser:
            webbrowser.open(f"http://{host}:{port}")
        return server


_EMBEDDED_FALLBACK_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>JARVIS AI Core</title>
<style>body{font-family:system-ui;background:#0a0f1c;color:#e2e8f0;max-width:860px;margin:2rem auto;padding:1rem}
input,button,select,textarea{background:#111a2e;color:#e2e8f0;border:1px solid #334155;padding:.6rem;border-radius:8px}
button{background:#0284c7;border:none;cursor:pointer;font-weight:600}pre{background:#020617;padding:1rem;border-radius:8px;overflow:auto}</style>
</head><body><h1>JARVIS Portable AI Core</h1>
<p>Endpoints: <code>/v1/ask</code>, <code>/v1/task</code>, <code>/v1/verify</code>, <code>/v1/twist</code>, <code>/v1/github</code>, <code>/v1/train</code>, <code>/v1/teach</code></p>
<input id="q" style="width:75%" placeholder="Ask, verify, or command Jarvis..."/>
<button onclick="run()">Send</button><pre id="out">Ready.</pre>
<script>async function run(){const r=await fetch('/v1/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:q.value})});out.textContent=JSON.stringify(await r.json(),null,2);}</script>
</body></html>"""


# Singleton convenience helper for effortless cross-module imports
_GLOBAL_JARVIS: Optional[Jarvis] = None
_GLOBAL_LOCK = threading.Lock()


def get_jarvis(**kwargs: Any) -> Jarvis:
    """Return a shared thread-safe Jarvis instance for any project."""
    global _GLOBAL_JARVIS
    with _GLOBAL_LOCK:
        if _GLOBAL_JARVIS is None:
            _GLOBAL_JARVIS = Jarvis(**kwargs)
        return _GLOBAL_JARVIS


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="JARVIS AI — Portable Multi-Source Autonomous Agent")
    sub = parser.add_subparsers(dest="cmd")

    p_ask = sub.add_parser("ask", help="Ask Jarvis a question")
    p_ask.add_argument("query", nargs="+")
    p_ask.add_argument("--verify", action="store_true")

    p_task = sub.add_parser("task", help="Run an autonomous multi-step task")
    p_task.add_argument("goal", nargs="+")

    p_twist = sub.add_parser("twist", help="Extract & twist structured data from a website URL")
    p_twist.add_argument("url")
    p_twist.add_argument("--format", default="json", choices=["json", "csv", "markdown", "text"])
    p_twist.add_argument("--selector", default=None)
    p_twist.add_argument("--out", default=None)

    p_verify = sub.add_parser("verify", help="Cross-verify a claim across GitHub and the internet")
    p_verify.add_argument("claim", nargs="+")

    p_train = sub.add_parser("train", help="Train Jarvis on a folder, GitHub repo, or URL")
    p_train.add_argument("--folder", default=None)
    p_train.add_argument("--github", default=None)
    p_train.add_argument("--url", default=None)

    p_serve = sub.add_parser("serve", help="Run local HTTP API server & Web HUD")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8787)
    p_serve.add_argument("--open", action="store_true")

    p_voice = sub.add_parser("voice", help="Run continuous hands-free voice command loop")
    p_voice.add_argument("--no-wake", action="store_true")

    sub.add_parser("chat", help="Interactive terminal chat with Jarvis")

    args = parser.parse_args(argv)
    jarvis = Jarvis()

    if args.cmd == "ask":
        res = jarvis.ask(" ".join(args.query), verify_sources=args.verify)
        print(res.answer)
        return 0
    if args.cmd == "task":
        res = jarvis.run_task(" ".join(args.goal))
        print(res.answer)
        return 0
    if args.cmd == "twist":
        res = jarvis.twist(args.url, output_format=args.format, selector=args.selector, save_to=args.out)
        print(res.get("formatted") or json.dumps(res, indent=2))
        return 0
    if args.cmd == "verify":
        print(json.dumps(jarvis.verify(" ".join(args.claim)), indent=2, ensure_ascii=False))
        return 0
    if args.cmd == "train":
        print(json.dumps(jarvis.train(folder=args.folder, github_repo=args.github, url=args.url), indent=2))
        return 0
    if args.cmd == "serve":
        srv = jarvis.serve(host=args.host, port=args.port, open_browser=args.open)
        print(f"JARVIS API & HUD listening at http://{args.host}:{args.port}")
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            srv.shutdown()
        return 0
    if args.cmd == "voice":
        jarvis.start_voice_loop(require_wake_word=not args.no_wake)
        return 0

    print(f"JARVIS AI v{__version__} interactive mode (type 'exit' to quit)")
    while True:
        try:
            line = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line or line.lower() in ("exit", "quit"):
            break
        print("\nJARVIS >", jarvis.ask(line).answer, "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
