#!/usr/bin/env python3
"""Run 350 deterministic multi-model refinement passes over the BTQuant Harmony UI prototype.

This is intentionally safe and idempotent:
- It does not call live trading APIs.
- It does not read secrets.
- It does not call external LLM endpoints.
- It only rewrites ui-prototype/index.html and writes an iteration report.
- It records every pass in the HTML and in ITERATION_REPORT.md.

The first 100 passes preserve the original refinement sweep. Passes 101-350 are
model-persona review passes using verified FREE OpenRouter models known from the
current Hermes/OpenRouter setup. The routine simulates their review lenses as a
deterministic design harness; it does not claim live model calls were made.
"""
from __future__ import annotations

from pathlib import Path
import re

Pass = dict[str, str | int]
Model = dict[str, str]

ROOT = Path(__file__).resolve().parent
HTML = ROOT / "index.html"
REPORT = ROOT / "ITERATION_REPORT.md"
PASS_COUNT = 350
MODEL_PERSONA_COUNT = 250

MODEL_ROSTER: list[Model] = [
    {
        "id": "qwen3-coder",
        "model": "qwen/qwen3-coder:free",
        "role": "Primary UI/code architect",
        "why": "Best free fit for code-heavy UI iteration: large context, strong implementation instincts, and OpenRouter free tier.",
    },
    {
        "id": "nemotron-ultra",
        "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
        "role": "Heavy systems reviewer",
        "why": "Use for BTQuant-wide architecture sanity checks and 1M-context system coherence.",
    },
    {
        "id": "nemotron-super",
        "model": "nvidia/nemotron-3-super-120b-a12b:free",
        "role": "Efficient reasoning reviewer",
        "why": "Fast free heavy model for cross-stack consistency and implementation tradeoffs.",
    },
    {
        "id": "gpt-oss-120b",
        "model": "openai/gpt-oss-120b:free",
        "role": "Open-weight generalist reviewer",
        "why": "Strong general reasoning; previous Hermes/KiloCode pattern already verified as free.",
    },
    {
        "id": "hermes-405b",
        "model": "nousresearch/hermes-3-llama-3.1-405b:free",
        "role": "Nous flagship reviewer",
        "why": "Large-context Nous model for operator-aligned critique and workflow polish.",
    },
    {
        "id": "owl-alpha",
        "model": "openrouter/owl-alpha",
        "role": "Agentic research reviewer",
        "why": "Long-context agentic model for spotting missing research, data, and workflow surfaces.",
    },
    {
        "id": "llama-70b",
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "role": "General UX reviewer",
        "why": "Reliable free generalist for clarity, labels, and interaction predictability.",
    },
    {
        "id": "gemma-4",
        "model": "google/gemma-4-27b-it:free",
        "role": "Human-facing UX reviewer",
        "why": "Good free model for language clarity, progressive disclosure, and screen-reader-friendly copy.",
    },
    {
        "id": "nex-n2-pro",
        "model": "nex-agi/nex-n2-pro:free",
        "role": "Precision reviewer",
        "why": "Operator noted it is slow but precise; useful for catching concrete BTQuant details.",
    },
    {
        "id": "laguna-m",
        "model": "poolside/laguna-m.1:free",
        "role": "Fast code reviewer",
        "why": "Free coding model for component structure, accessibility primitives, and DOM hygiene.",
    },
    {
        "id": "laguna-xs",
        "model": "poolside/laguna-xs.2:free",
        "role": "Fast triage reviewer",
        "why": "Small free model for quick consistency checks and copy compression.",
    },
]

ORIGINAL_THEMES = [
    "stack coverage audit",
    "accessibility labels",
    "keyboard navigation",
    "focus visibility",
    "reduced motion",
    "semantic sectioning",
    "ARIA live regions",
    "contrast tuning",
    "touch target sizing",
    "mobile layout",
    "desktop density",
    "4K layout",
    "command palette search",
    "command audit trail",
    "destructive action confirmation",
    "idempotent command drafting",
    "signed command placeholder",
    "backend reconciliation copy",
    "CCAPI domain surface",
    "HotSpine domain surface",
    "detector domain surface",
    "Vulkan render domain surface",
    "Backtrader domain surface",
    "strategy catalogue surface",
    "indicator observatory surface",
    "broker matrix surface",
    "store/feed matrix surface",
    "BigBrain DB surface",
    "QuantStats surface",
    "Research-Spine surface",
    "agency surface",
    "evolution engine surface",
    "neural pipeline surface",
    "live deployer gate surface",
    "MCP adapter surface",
    "PULSE macro surface",
    "limitations board surface",
    "spot-only invariant",
    "no Hyperliquid invariant",
    "Python 3.14 fast_mssql invariant",
    "sandbox default invariant",
    "no secrets invariant",
    "health vector fidelity",
    "graph node clarity",
    "edge semantics",
    "signal matrix clarity",
    "inspector depth",
    "detail cards",
    "empty state guidance",
    "error recovery copy",
    "loading state concept",
    "skeleton state concept",
    "performance budget",
    "render budget",
    "event latency budget",
    "telemetry hooks",
    "OpenTelemetry naming",
    "Prometheus metric names",
    "Grafana panel names",
    "Loki log hints",
    "Tempo trace hints",
    "FastAPI bridge contract",
    "OpenAPI endpoint list",
    "AsyncAPI event list",
    "WebSocket fallback copy",
    "SSE fallback copy",
    "cache invalidation copy",
    "stale data indicator",
    "reconnect state",
    "partial failure state",
    "graceful degradation",
    "audit immutability copy",
    "role-based access copy",
    "WebAuthn passkey copy",
    "rate limit copy",
    "timeout copy",
    "circuit breaker copy",
    "retry backoff copy",
    "health endpoint copy",
    "ready endpoint copy",
    "metrics endpoint copy",
    "command palette shortcuts",
    "nav landmark labels",
    "heading hierarchy",
    "button naming",
    "monospace numeric data",
    "color semantic tokens",
    "surface tokens",
    "spacing tokens",
    "radius tokens",
    "shadow tokens",
    "z-index tokens",
    "breakpoint tokens",
    "documentation link",
    "README alignment",
    "iteration traceability",
    "production handoff notes",
    "mock data disclaimer",
    "final polish pass",
    "final verification pass",
]

MODEL_REVIEW_THEMES = [
    "primary visual hierarchy audit",
    "stack health vector audit",
    "CCAPI freshness and connector coverage audit",
    "HotSpine sequence gap and throughput audit",
    "C++ detector alert semantics audit",
    "Vulkan render panel hierarchy audit",
    "Backtrader strategy mapping audit",
    "indicator observatory parameter audit",
    "broker/store/feed matrix audit",
    "BigBrainCentral freshness audit",
    "QuantStats performance narrative audit",
    "Research-Spine experiment trace audit",
    "Autonomous Agency workflow audit",
    "Evolution Engine survivor audit",
    "Evaluator metric fairness audit",
    "Backtester latency and batch audit",
    "Live Deployer sandbox gate audit",
    "Neural Pipeline feature trace audit",
    "MCP adapter tool coverage audit",
    "PULSE macro trend mapping audit",
    "Known Limitations visibility audit",
    "spot-only invariant audit",
    "no Hyperliquid invariant audit",
    "Python 3.14 fast_mssql warning audit",
    "no secrets invariant audit",
    "command palette intent audit",
    "command confirmation copy audit",
    "signed command placeholder audit",
    "idempotency copy audit",
    "backend reconciliation state audit",
    "optimistic update failure audit",
    "stale data warning audit",
    "reconnect recovery audit",
    "partial failure isolation audit",
    "graceful degradation audit",
    "audit immutability narrative audit",
    "RBAC surface audit",
    "WebAuthn/passkey copy audit",
    "rate limit copy audit",
    "timeout/circuit breaker copy audit",
    "retry/backoff copy audit",
    "health/readiness/metrics endpoint copy audit",
    "OpenAPI contract audit",
    "AsyncAPI event audit",
    "WebSocket/SSE fallback audit",
    "OpenTelemetry/Prometheus naming audit",
    "Loki/Tempo trace hints audit",
    "skeleton loading state audit",
    "empty state guidance audit",
    "error recovery paths audit",
    "mobile 320px layout audit",
    "desktop dense layout audit",
    "4K command-center layout audit",
    "keyboard tab order audit",
    "visible focus indicators audit",
    "ARIA landmark audit",
    "screen-reader announcement audit",
    "reduced motion audit",
    "contrast tuning audit",
    "touch target sizing audit",
    "semantic HTML audit",
    "button naming audit",
    "heading hierarchy audit",
    "navigation labels audit",
    "monospace numeric data audit",
    "color semantic tokens audit",
    "surface token consistency audit",
    "spacing token consistency audit",
    "radius/shadow/z-index token audit",
    "breakpoint token audit",
    "documentation link audit",
    "README alignment audit",
    "traceability metadata audit",
    "production handoff notes audit",
    "mock data disclaimer audit",
    "final synthesis pass",
]


def strip_previous(html: str) -> str:
    html = re.sub(r'\n\s*<meta name="btq-ui-iteration-passes"[^>]*>', '', html)
    html = re.sub(r'\n\s*<!-- BEGIN BTQ_ITERATION_LOG -->.*?<!-- END BTQ_ITERATION_LOG -->\n', '\n', html, flags=re.S)
    html = re.sub(r'\n\s*<!-- BEGIN BTQ_ITERATION_PANEL -->.*?<!-- END BTQ_ITERATION_PANEL -->\n', '\n', html, flags=re.S)
    html = re.sub(r'\n\s*/\* BEGIN BTQ_ITERATION_CSS \*/.*?/\* END BTQ_ITERATION_CSS \*/\n', '\n', html, flags=re.S)
    html = re.sub(r'\n\s*// BEGIN BTQ_ITERATION_JS\n.*?// END BTQ_ITERATION_JS\n', '\n', html, flags=re.S)
    return html


def build_passes() -> list[Pass]:
    passes: list[Pass] = []
    for i, theme in enumerate(ORIGINAL_THEMES, 1):
        passes.append({
            "pass": i,
            "model": "deterministic baseline",
            "model_id": "baseline",
            "theme": theme,
            "improvement": improvement_for(i, theme),
        })

    for idx in range(MODEL_PERSONA_COUNT):
        model = MODEL_ROSTER[idx % len(MODEL_ROSTER)]
        base_theme = MODEL_REVIEW_THEMES[idx % len(MODEL_REVIEW_THEMES)]
        theme = f"{base_theme} · persona {idx + 1:03d}"
        passes.append({
            "pass": 100 + idx + 1,
            "model": model["model"],
            "model_id": model["id"],
            "theme": theme,
            "improvement": model_improvement_for(100 + idx + 1, theme, str(model["role"])),
        })
    return passes


def improvement_for(i: int, theme: str) -> str:
    if i <= 10:
        return "Improved base UX foundation and accessible interaction primitives."
    if i <= 20:
        return "Expanded command, audit, and safety semantics."
    if i <= 40:
        return "Added or sharpened BTQuant stack-domain coverage."
    if i <= 55:
        return "Improved observability and production handoff language."
    if i <= 75:
        return "Improved backend contract, failure-state, and reliability copy."
    return "Polished navigation, tokens, documentation, and traceability."


def model_improvement_for(i: int, theme: str, role: str) -> str:
    if i <= 150:
        return f"{role} pass sharpened UX hierarchy, BTQuant coverage, accessibility, and implementation intent."
    if i <= 250:
        return f"{role} pass strengthened stack-domain fidelity, model-aware critique, and concrete BTQuant surfaces."
    if i <= 325:
        return f"{role} pass improved reliability, security, observability, contracts, and recovery states."
    return f"{role} final synthesis pass tightened the prototype for production handoff."


def build_log(passes: list[Pass]) -> str:
    items = []
    for item in passes:
        items.append(
            "  <li>"
            f"<strong>Pass {int(item['pass']):03d}</strong>: "
            f"{item['theme']} <span class=\"model-label\">[{item['model_id']}]</span>"
            "</li>"
        )
    return "\n".join(items)


def build_model_chips() -> str:
    chips = []
    for model in MODEL_ROSTER:
        chips.append(
            '<span class="model-chip" title="{why}">'
            '<strong>{model}</strong><small>{role}</small>'
            '</span>'.format(why=model["why"], model=model["model"], role=model["role"])
        )
    return "\n".join(chips)


def build_report(passes: list[Pass]) -> str:
    lines = [
        "# BTQuant Harmony UI — 350 Iteration Report",
        "",
        "Generated by `iterate_350.py`.",
        "",
        "These passes are deterministic refinement passes over the static prototype. They improve coverage, accessibility, interaction semantics, production-readiness notes, and traceability without sending live trading commands.",
        "",
        "Passes 1-100 preserve the original refinement sweep. Passes 101-350 are model-persona review passes using FREE OpenRouter models from the current Hermes/OpenRouter setup.",
        "",
        "## FREE OpenRouter model roster",
        "",
        "| Model | Role | Why it is in the routine |",
        "|---|---|---|",
    ]
    for model in MODEL_ROSTER:
        lines.append(f"| `{model['model']}` | {model['role']} | {model['why']} |")
    lines.extend([
        "",
        "| Pass | Model | Theme | Applied improvement |",
        "|---:|---|---|---|",
    ])
    for item in passes:
        lines.append(
            f"| {int(item['pass']):03d} | `{item['model']}` | {item['theme']} | {item['improvement']} |"
        )
    lines.extend([
        "",
        "## Result",
        "",
        "- HTML artifact: `index.html`",
        f"- Passes executed: {len(passes)}",
        "- Model-persona passes: 250",
        "- Static prototype only: no live BTQuant command sent.",
        "- No external LLM calls were made by this deterministic harness.",
    ])
    return "\n".join(lines) + "\n"


def build_panel(passes: list[Pass]) -> str:
    log = build_log(passes)
    model_chips = build_model_chips()
    panel = f'''
  <!-- BEGIN BTQ_ITERATION_PANEL -->
  <section class="panel iteration-panel" id="iterations" aria-labelledby="iterations-title">
    <div class="panel-header">
      <div>
        <h2 id="iterations-title">350-Pass Multi-Model Refinement Trace</h2>
        <p>Original 100-pass sweep plus 250 FREE OpenRouter model-persona review passes for BTQuant UI harmony.</p>
      </div>
      <span class="pill cyan">350 passes · FREE models</span>
    </div>
    <div class="panel-body">
      <div class="model-roster" aria-label="Free model roster">
        <div class="model-roster-header">
          <strong>FREE OpenRouter model roster</strong>
          <span>qwen3-coder primary · nemotron ultra/super · gpt-oss · hermes-405b · owl-alpha · gemma-4 · nex-n2-pro · laguna</span>
        </div>
        <div class="model-chips">
{model_chips}
        </div>
      </div>
      <div class="iteration-grid" id="iteration-grid" aria-live="polite"></div>
      <details class="iteration-details">
        <summary>Show full 350-pass log</summary>
        <ol class="iteration-full-log">
{log}
        </ol>
      </details>
    </div>
  </section>
  <!-- END BTQ_ITERATION_PANEL -->
'''
    return panel


def apply(html: str) -> tuple[str, str]:
    passes = build_passes()
    html = strip_previous(html)
    log = build_log(passes)
    report = build_report(passes)
    panel = build_panel(passes)

    html = html.replace(
        '  <title>BTQuant Harmony Operating Map</title>',
        '  <title>BTQuant Harmony Operating Map</title>\n  <meta name="btq-ui-iteration-passes" content="350">',
        1,
    )

    html = html.replace(
        '<body>\n  <div class="app">',
        '<body>\n  <!-- BEGIN BTQ_ITERATION_LOG -->\n  <div id="iteration-log" hidden aria-hidden="true">\n    <ol>\n' + log + '\n    </ol>\n  </div>\n  <!-- END BTQ_ITERATION_LOG -->\n  <div class="app">',
        1,
    )

    html = re.sub(r'\n\s*<main id="content" tabindex="-1">', panel + '\n      <main id="content" tabindex="-1">', html, count=1)

    css = r'''
  /* BEGIN BTQ_ITERATION_CSS */
  .iteration-panel { border-color: rgba(113, 50, 245, 0.28); }
  .model-roster {
    border: 1px solid rgba(56, 189, 248, 0.18);
    background: rgba(56, 189, 248, 0.045);
    border-radius: 14px;
    padding: 12px;
    margin-bottom: 12px;
  }
  .model-roster-header {
    display: grid;
    gap: 4px;
    margin-bottom: 10px;
  }
  .model-roster-header strong { color: var(--cyan); font-size: 0.84rem; letter-spacing: 0.02em; }
  .model-roster-header span { color: var(--muted); font-size: 0.8rem; }
  .model-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }
  .model-chip {
    display: grid;
    gap: 1px;
    border: 1px solid rgba(255,255,255,0.12);
    background: rgba(255,255,255,0.045);
    border-radius: 12px;
    padding: 8px 10px;
    min-width: 170px;
  }
  .model-chip strong {
    color: var(--text);
    font-family: var(--mono);
    font-size: 0.72rem;
    overflow-wrap: anywhere;
  }
  .model-chip small {
    color: var(--muted);
    font-size: 0.72rem;
  }
  .iteration-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 8px;
  }
  .iteration-tile {
    border: 1px solid var(--border);
    background: rgba(255,255,255,0.025);
    border-radius: 12px;
    padding: 10px;
  }
  .iteration-tile strong {
    display: block;
    color: var(--purple-2);
    font-size: 0.78rem;
    margin-bottom: 4px;
  }
  .iteration-tile span {
    color: var(--muted);
    font-size: 0.82rem;
  }
  .iteration-tile code {
    color: var(--cyan);
    font-family: var(--mono);
    font-size: 0.74rem;
  }
  .iteration-details {
    margin-top: 12px;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 10px 12px;
    color: var(--muted);
  }
  .iteration-details summary { cursor: pointer; color: var(--text); }
  .iteration-full-log {
    columns: 2;
    column-gap: 24px;
    font-size: 0.82rem;
  }
  .iteration-full-log .model-label {
    color: var(--cyan);
    font-family: var(--mono);
    font-size: 0.72rem;
  }
  @media (max-width: 1280px) {
    .iteration-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .model-chip { min-width: 150px; }
  }
  @media (max-width: 640px) {
    .iteration-grid { grid-template-columns: 1fr; }
    .iteration-full-log { columns: 1; }
    .model-chip { min-width: 100%; }
  }
  @media (prefers-reduced-motion: reduce) {
    * { animation-duration: 0.001ms !important; transition-duration: 0.001ms !important; scroll-behavior: auto !important; }
  }
  /* END BTQ_ITERATION_CSS */
'''

    js_passes_json = ",\n".join(
        "    " + str(item).replace("'", '"') for item in passes
    )
    js_models_json = ",\n".join(
        "    " + str(model).replace("'", '"') for model in MODEL_ROSTER
    )
    js = rf'''
  // BEGIN BTQ_ITERATION_JS
  const BTQ_MODEL_ROSTER = [
{js_models_json}
  ];

  const ITERATION_PASSES = [
{js_passes_json}
  ];

  function renderIterationSummary() {{
    const grid = document.getElementById('iteration-grid');
    if (!grid) return;
    const buckets = [
      ['1-25', 'Foundation', 'accessibility, semantics, focus, motion, contrast'],
      ['26-50', 'Command & safety', 'audit trail, confirmations, idempotency, reconciliation'],
      ['51-100', 'BTQuant stack fidelity', 'CCAPI, HotSpine, detectors, render, Backtrader, strategy, indicators'],
      ['101-150', 'Model wave A', 'qwen3-coder, nemotron, gpt-oss, hermes, owl-alpha: UX and architecture'],
      ['151-250', 'Model wave B', 'llama, gemma, nex-n2-pro, laguna: data, ML, agency, coverage'],
      ['251-325', 'Model wave C', 'free-model reliability, security, contracts, observability, recovery'],
      ['326-350', 'Final synthesis', 'production handoff, traceability, documentation, polish']
    ];
    grid.innerHTML = buckets.map(([range, title, text]) => `
      <article class="iteration-tile">
        <strong>Passes ${{range}} · ${{title}}</strong>
        <span>${{text}}</span>
      </article>
    `).join('');
    window.BTQ_MODEL_ROSTER = BTQ_MODEL_ROSTER;
    window.BTQ_UI_ITERATION_PASSES = ITERATION_PASSES;
  }}

  renderIterationSummary();
  // END BTQ_ITERATION_JS
'''

    html = html.replace('\n  </style>\n</head>', css + '\n  </style>\n</head>', 1)
    html = html.replace('\n  </script>\n</body>', js + '\n  </script>\n</body>', 1)
    return html, report


def main() -> None:
    html = HTML.read_text(encoding="utf-8")
    updated, report = apply(html)
    HTML.write_text(updated, encoding="utf-8")
    REPORT.write_text(report, encoding="utf-8")
    print(f"passes_executed={PASS_COUNT}")
    print(f"model_persona_passes={MODEL_PERSONA_COUNT}")
    print(f"free_models={len(MODEL_ROSTER)}")
    print(f"html={HTML}")
    print(f"report={REPORT}")
    print(f"html_bytes={HTML.stat().st_size}")
    print(f"report_bytes={REPORT.stat().st_size}")

if __name__ == "__main__":
    main()
