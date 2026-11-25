"""Comprehensive unit, integration, and single-file portability tests for JARVIS AI."""

import json
import shutil
import subprocess
import sys
import threading
import urllib.request
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from jarvis_ai import (  # noqa: E402
    Jarvis,
    JarvisConfig,
    JarvisNeuralModel,
    SEED_TRAINING_DATA,
    safe_eval_math,
)


@pytest.fixture
def isolated_jarvis(tmp_path: Path) -> Jarvis:
    brain = tmp_path / "brain"
    ws = tmp_path / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    cfg = JarvisConfig(
        brain_dir=str(brain),
        workspace_root=str(ws),
        llm_provider="local",
        allow_network=False,
        tts_enabled=False,
    )
    return Jarvis(cfg)


def test_neural_training_persistence_and_dynamic_class(tmp_path: Path) -> None:
    brain = tmp_path / "neural_brain"
    model1 = JarvisNeuralModel(str(brain))
    metrics = model1.fit(SEED_TRAINING_DATA, epochs=45, lr=0.025, save=True)
    assert metrics["train_accuracy"] >= 0.90
    assert (brain / "jarvis_model.npz").exists()

    # Reload into brand new instance and check identical prediction
    model2 = JarvisNeuralModel(str(brain))
    pred, conf, _ = model2.predict("search github for python scraper repos")
    assert pred == "github_research"
    assert conf > 0.45

    # Teach a brand-new unseen intent class dynamically
    upd = model2.train_online(
        [
            ("launch orbital rocket telemetry", "deploy_rocket"),
            ("ignite falcon booster stage one", "deploy_rocket"),
        ],
        steps=20,
    )
    assert upd["updated"] == 2
    pred_new, _, _ = model2.predict("launch orbital rocket telemetry")
    assert pred_new == "deploy_rocket"


def test_memory_hybrid_search_and_sensitive_skip(isolated_jarvis: Jarvis, tmp_path: Path) -> None:
    proj = tmp_path / "sample_repo"
    proj.mkdir()
    (proj / "README.md").write_text(
        "# Solar Microgrid Controller\nUses MPPT inverter telemetry and battery SOC prediction.",
        encoding="utf-8",
    )
    (proj / ".env").write_text("SECRET_API_KEY=do_not_index_this", encoding="utf-8")
    (proj / "private.key").write_text("PRIVATE KEY DATA", encoding="utf-8")

    stats = isolated_jarvis.memory.ingest_folder(str(proj))
    assert stats["files"] == 1  # .env and private.key skipped

    hits = isolated_jarvis.memory.search_knowledge("MPPT inverter battery telemetry")
    assert hits
    assert "Solar Microgrid" in hits[0]["content"]
    assert not isolated_jarvis.memory.search_knowledge("do_not_index_this")


def test_website_twister_dom_selectors_tables_and_formats(isolated_jarvis: Jarvis) -> None:
    sample_html = """
    <!doctype html>
    <html>
      <head>
        <title>AI Tools Index</title>
        <meta name="description" content="Top developer tools" />
        <script type="application/ld+json">{"@type":"Dataset","name":"Tools"}</script>
      </head>
      <body>
        <nav><a href="/ignore">Nav Link</a></nav>
        <main id="content">
          <h1>Open Source AI Frameworks</h1>
          <div class="card"><h2>Jarvis</h2><span class="price">$0</span><a href="https://github.com/example/jarvis">Repo</a></div>
          <div class="card"><h2>LangCore</h2><span class="price">$49</span><a href="https://example.com/lang">Site</a></div>
          <table>
            <tr><th>Tool</th><th>Language</th><th>Stars</th></tr>
            <tr><td>FastAPI</td><td>Python</td><td>80000</td></tr>
            <tr><td>ThreeJS</td><td>JavaScript</td><td>105000</td></tr>
          </table>
          <p>Contact engineering at team@example.org for benchmark details.</p>
        </main>
      </body>
    </html>
    """
    # Default table extraction into CSV
    csv_res = isolated_jarvis.twist(sample_html, output_format="csv", save_to="tools.csv")
    assert csv_res["ok"] is True
    assert csv_res["tables_count"] == 1
    assert "FastAPI,Python,80000" in csv_res["formatted"]
    assert "team@example.org" in csv_res["emails"]
    assert "$49" in csv_res["prices"]
    assert Path(csv_res["saved_to"]).exists()

    # Field mapping with CSS selectors
    mapped = isolated_jarvis.twist(
        sample_html,
        output_format="json",
        fields={"name": "div.card h2", "price": "div.card > span.price"},
    )
    assert mapped["records"] == [
        {"name": "Jarvis", "price": "$0"},
        {"name": "LangCore", "price": "$49"},
    ]

    # Direct selector filter
    gh = isolated_jarvis.twist(sample_html, selector="div.card a[href*=github]")
    assert gh["count"] == 1
    assert gh["records"][0]["url"] == "https://github.com/example/jarvis"


def test_website_twister_detects_antibot_without_bypass(isolated_jarvis: Jarvis) -> None:
    class FakeResp:
        url = "https://protected.example.com"
        status_code = 403
        headers = {"Content-Type": "text/html"}
        text = "<html><body><div class='cf-turnstile'>Verify you are human</div></body></html>"

    with patch.object(isolated_jarvis.twister.session, "get", return_value=FakeResp()):
        res = isolated_jarvis.twist("https://protected.example.com")
    assert res["ok"] is False
    assert res["blocked"] is True
    assert "403" in res["reason"]


def test_autonomous_task_custom_tool_and_safety(isolated_jarvis: Jarvis) -> None:
    @isolated_jarvis.register_tool("reverse_words", "Reverse words in a string")
    def reverse_words(text: str) -> str:
        return " ".join(reversed(text.split()))

    assert "reverse_words" in isolated_jarvis.tools
    assert isolated_jarvis.tools["reverse_words"]["fn"]("hello jarvis world") == "world jarvis hello"

    # Math + file output autonomous workflow
    res = isolated_jarvis.run_task("calculate 25 * 4 + 17 then write output to calc.txt")
    assert res.intent == "autonomous_task"
    assert len(res.steps) >= 2
    assert res.artifacts
    saved_text = Path(res.artifacts[0]).read_text(encoding="utf-8")
    assert "117" in saved_text

    # Safety guards: destructive shell & out-of-workspace writes blocked
    blocked_shell = isolated_jarvis._tool_run_shell("rm -rf /")
    assert blocked_shell["ok"] is False
    with pytest.raises(PermissionError):
        isolated_jarvis._tool_write_file("../../outside_escape.txt", "nope")


def test_safe_eval_math_and_teach_loop(isolated_jarvis: Jarvis) -> None:
    assert safe_eval_math("what is (15 + 5) * 3?") == 60
    assert safe_eval_math("import os") is None

    isolated_jarvis.teach(
        "run diagnostics",
        intent="system_control",
        action={"tool": "system_info", "args": {}},
    )
    ans = isolated_jarvis.ask("run diagnostics")
    assert ans.metadata.get("learned_match") is True
    assert "python" in ans.metadata["tool_output"]


def test_multisource_verifier_cross_checks_sources(tmp_path: Path) -> None:
    cfg = JarvisConfig(brain_dir=str(tmp_path / "vbrain"), workspace_root=str(tmp_path), llm_provider="local", allow_network=True, tts_enabled=False)
    ai = Jarvis(cfg)
    ai.memory.ingest_text("Local SQLite Notes", "SQLite supports FTS5 full-text search index natively.", source_type="note")

    with (
        patch.object(ai.verifier, "search_duckduckgo", return_value=[{"source_type": "web_search", "title": "SQLite FTS5 Extension", "url": "https://sqlite.org/fts5.html", "snippet": "SQLite FTS5 provides full-text search."}]),
        patch.object(ai.verifier, "search_github_repos", return_value=[{"source_type": "github_repo", "title": "sqlite/sqlite", "url": "https://github.com/sqlite/sqlite", "snippet": "Official SQLite mirror supporting FTS5 full-text search"}]),
        patch.object(ai.verifier, "search_wikipedia", return_value=[]),
        patch.object(ai.verifier, "search_stackoverflow", return_value=[]),
        patch.object(ai.verifier, "search_arxiv", return_value=[]),
        patch.object(ai.twister, "twist", return_value={"ok": False}),
    ):
        report = ai.verify("Does SQLite support FTS5 full-text search?")
    assert report["verdict"] == "VERIFIED"
    assert report["confidence"] >= 0.55
    assert len(report["source_families"]) >= 2


def test_voice_wake_word_parser(isolated_jarvis: Jarvis) -> None:
    assert isolated_jarvis.voice.extract_wake_command("Hey Jarvis, twist example.com") == "twist example.com"
    assert isolated_jarvis.voice.extract_wake_command("jarvis") == ""
    assert isolated_jarvis.voice.extract_wake_command("random background conversation") is None


def test_http_api_server_endpoints(isolated_jarvis: Jarvis) -> None:
    srv = isolated_jarvis.serve(host="127.0.0.1", port=0)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/health", timeout=5) as r:
            health = json.loads(r.read().decode("utf-8"))
        assert health["ok"] is True
        assert "twist_website" in health["tools"]

        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/ask",
            data=json.dumps({"query": "what is 7 * 8"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            ask_res = json.loads(r.read().decode("utf-8"))
        assert "56" in ask_res["answer"]
    finally:
        srv.shutdown()
        srv.server_close()


def test_single_file_portability_in_isolated_project(tmp_path: Path) -> None:
    """Copy ONLY jarvis_ai.py into an empty folder and confirm standalone execution."""
    proj_dir = tmp_path / "brand_new_project"
    proj_dir.mkdir()
    shutil.copy2(ROOT_DIR / "jarvis_ai.py", proj_dir / "jarvis_ai.py")

    isolated_brain = tmp_path / "portable_brain"
    script = (
        "from jarvis_ai import Jarvis, JarvisConfig; "
        f"ai = Jarvis(JarvisConfig(brain_dir={str(isolated_brain)!r}, workspace_root={str(proj_dir)!r}, allow_network=False, tts_enabled=False)); "
        "r1 = ai.ask('calculate 99 + 1'); "
        "r2 = ai.twist('<html><body><h1>Hello</h1><p>World wide web test content here.</p></body></html>', output_format='markdown'); "
        "assert '100' in r1.answer; "
        "assert r2['ok'] and 'Hello' in r2['formatted']; "
        "print('PORTABLE_OK')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(proj_dir),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert "PORTABLE_OK" in proc.stdout
