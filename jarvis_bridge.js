/**
 * JARVIS AI Universal JavaScript / Node.js / Browser Bridge
 * Drop this file into ANY JS/TS/React/Node project to connect to JARVIS.
 *
 * Works in:
 *   - Browser / React / Vite (`<script src="jarvis_bridge.js">` or ES import)
 *   - Node.js (`const { JarvisClient } = require('./jarvis_bridge.js')`)
 */

(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    const exported = factory();
    root.JarvisClient = exported.JarvisClient;
  }
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  class JarvisClient {
    /**
     * @param {string} baseUrl - Running Jarvis HTTP server URL (default http://127.0.0.1:8787)
     */
    constructor(baseUrl = 'http://127.0.0.1:8787') {
      this.baseUrl = baseUrl.replace(/\/+$/, '');
    }

    async _request(path, method = 'GET', body = null) {
      const opts = {
        method,
        headers: { 'Content-Type': 'application/json' },
      };
      if (body !== null) {
        opts.body = JSON.stringify(body);
      }
      const res = await fetch(`${this.baseUrl}${path}`, opts);
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || `Jarvis HTTP ${res.status}`);
      }
      return data;
    }

    health() {
      return this._request('/v1/health', 'GET');
    }

    memory() {
      return this._request('/v1/memory', 'GET');
    }

    ask(query, verify = false) {
      return this._request('/v1/ask', 'POST', { query, verify });
    }

    runTask(goal, maxSteps = 8) {
      return this._request('/v1/task', 'POST', { goal, max_steps: maxSteps });
    }

    verify(query) {
      return this._request('/v1/verify', 'POST', { query });
    }

    twist(url, { format = 'json', selector = null, fields = null, filter = null } = {}) {
      return this._request('/v1/twist', 'POST', { url, format, selector, fields, filter });
    }

    searchGithub(query, limit = 5) {
      return this._request('/v1/github', 'POST', { query, limit });
    }

    learnGithubRepo(repo, clone = false) {
      return this._request('/v1/github', 'POST', { repo, clone });
    }

    train({ samples = null, folder = null, githubRepo = null, url = null } = {}) {
      return this._request('/v1/train', 'POST', {
        samples,
        folder,
        github_repo: githubRepo,
        url,
      });
    }

    teach(phrase, { intent = 'chat_general', response = '', action = null } = {}) {
      return this._request('/v1/teach', 'POST', { phrase, intent, response, action });
    }
  }

  return { JarvisClient };
});
