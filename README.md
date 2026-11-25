# JARVIS AI — Portable Single-File Autonomous AI Model

A drop-in, self-contained AI engine (`jarvis_ai.py`) you can copy into **any** Python project or expose to **any** JavaScript / Node.js / React / Web app via `jarvis_bridge.js` or its built-in HTTP API server.

## Files

- [jarvis_ai.py](jarvis_ai.py) — **Single portable file** containing everything:
  - `JarvisNeuralModel`: Real trainable 2-layer NumPy MLP intent & skill classifier with Adam optimizer + online incremental fine-tuning (`jarvis_model.npz`)
  - `JarvisMemory`: Hybrid SQLite FTS5 BM25 + vector RAG knowledge store + taught command persistence (`jarvis_memory.db`)
  - `WebsiteTwister`: Pure-Python HTML DOM tree, CSS selector engine, table/JSON-LD/price/email/link extractor, and JSON/CSV/Markdown transformer
  - `MultiSourceVerifier`: Parallel search & cross-verification across GitHub repositories, Bing Web RSS, DuckDuckGo Instant Answer, Wikipedia, StackOverflow, Hacker News, arXiv, and local RAG memory
  - `LLMRouter`: Auto-connects to local Ollama, Gemini, OpenAI, OpenRouter, or Anthropic when configured, with 100% local synthesis fallback
  - `VoiceEngine`: Wake-word detection (`"jarvis"`, `"hey jarvis"`), SpeechRecognition mic capture, and thread-safe `pyttsx3` TTS
  - `Jarvis` & `ThreadingHTTPServer`: Autonomous multi-step ReAct task runner, `@register_tool` extensibility, CLI, and Web HUD
- [jarvis_bridge.js](jarvis_bridge.js) — Zero-dependency JS/TS/Node/Browser client
- [jarvis_ui.html](jarvis_ui.html) — Real-time Command HUD (served automatically at `http://127.0.0.1:8787`)
- [train_jarvis.py](train_jarvis.py) — Training, evaluation, and repository/folder indexing script
- [tests/test_jarvis.py](tests/test_jarvis.py) — Automated test suite (including single-file portability proof)

## Use Anywhere in Any Python Project

Copy **only** `jarvis_ai.py` into your project folder (by default, all projects share your trained brain at `~/.jarvis_brain`, or pass `brain_dir="./my_brain"`):

```python
from jarvis_ai import Jarvis

ai = Jarvis()

# 1. Ask + cross-verify across GitHub, web, Wikipedia, StackOverflow & arXiv
res = ai.ask("Does SQLite support FTS5 full-text search?", verify_sources=True)
print(res.verdict, res.confidence, res.answer)

# 2. Twist any website into JSON, CSV, or Markdown
data = ai.twist("https://news.ycombinator.com", selector="span.titleline > a", output_format="json")

# 3. Autonomous multi-step tasks + custom project tools
@ai.register_tool("deploy_preview", "Deploy current app preview")
def deploy_preview(env: str = "staging"):
    return {"deployed": env, "status": "ok"}

task_res = ai.run_task("calculate 256 * 12 then write result to report.txt")

# 4. Train / teach in real time
ai.teach("open workspace", intent="system_control", action={"tool": "open_target", "args": {"target": "vscode"}})
ai.train(github_repo="tiangolo/fastapi", folder="./my_project_docs")
```

## Use in JavaScript / Node.js / React (`SUNDAY`, `terrawise-ai`, Web Apps)

Start the local server once:

```bash
python jarvis_ai.py serve --port 8787
```

Then in any Node or browser file using [jarvis_bridge.js](jarvis_bridge.js):

```js
const { JarvisClient } = require('./jarvis_bridge.js');
const jarvis = new JarvisClient('http://127.0.0.1:8787');

const reply = await jarvis.ask('Search GitHub for FastAPI rate limiters', true);
const tableCsv = await jarvis.twist('https://example.com', { format: 'csv' });
```

## CLI Modes

```bash
python jarvis_ai.py serve --open              # Launch Web HUD + HTTP API on :8787
python jarvis_ai.py voice                     # Continuous hands-free "Jarvis ..." voice loop
python jarvis_ai.py ask "verify python GIL" --verify
python jarvis_ai.py twist https://news.ycombinator.com --selector "span.titleline > a" --format json
python jarvis_ai.py task "calculate 42 * 99 then write answer.txt"
python jarvis_ai.py train --folder ./my_repo --github tiangolo/fastapi
python train_jarvis.py                        # Re-train & print held-out accuracy metrics
```

## Optional Environment Variables

- `JARVIS_BRAIN_DIR`: Override shared brain path (default `~/.jarvis_brain`)
- `GITHUB_TOKEN`: Raise GitHub search limits from 10/min to 30/min and higher REST quotas
- `OLLAMA_HOST` / `OLLAMA_MODEL`: Connect local Ollama (default `http://127.0.0.1:11434`)
- `GEMINI_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, or `ANTHROPIC_API_KEY`: Optional LLM cortex upgrade
