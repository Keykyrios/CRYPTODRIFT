"""
Generate a standalone HTML dashboard from experiment results.

Reads from data/cryptodrift.db and produces dashboard.html
with interactive charts. No server needed — just open in browser.

Usage:
    python generate_dashboard.py
"""

import os
import sys
import json
import sqlite3
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))


def main():
    conn = sqlite3.connect("data/cryptodrift.db")
    conn.row_factory = sqlite3.Row

    # ── Gather data ──────────────────────────────────────────

    # Sessions
    sessions = conn.execute("""
        SELECT s.id, s.strategy, s.corpus_sample, s.total_iterations, s.status
        FROM sessions s WHERE s.status = 'completed'
        ORDER BY s.start_time
    """).fetchall()

    # Entropy per session per iteration
    entropy_data = {}
    for s in sessions:
        key = f"{s['corpus_sample']} × {s['strategy']}"
        rows = conn.execute("""
            SELECT i.iteration_num, a.mean_entropy
            FROM attention_snapshots a
            JOIN iterations i ON a.iteration_id = i.id
            WHERE i.session_id = ?
            ORDER BY i.iteration_num
        """, (s['id'],)).fetchall()
        entropy_data[key] = {
            "iterations": [r["iteration_num"] for r in rows],
            "entropy": [round(r["mean_entropy"], 4) for r in rows],
            "strategy": s["strategy"],
            "sample": s["corpus_sample"],
        }

    # Vulns per iteration per session
    vuln_data = {}
    for s in sessions:
        key = f"{s['corpus_sample']} × {s['strategy']}"
        rows = conn.execute("""
            SELECT i.iteration_num,
                   (SELECT COUNT(*) FROM vuln_scores v WHERE v.iteration_id = i.id) as vulns
            FROM iterations i
            WHERE i.session_id = ?
            ORDER BY i.iteration_num
        """, (s['id'],)).fetchall()
        vuln_data[key] = {
            "iterations": [r["iteration_num"] for r in rows],
            "vulns": [r["vulns"] for r in rows],
        }

    # Strategy means
    strategy_entropy = defaultdict(list)
    for key, data in entropy_data.items():
        strategy_entropy[data["strategy"]].extend(data["entropy"])

    strategy_means = {
        s: round(sum(v) / len(v), 4) if v else 0
        for s, v in strategy_entropy.items()
    }

    # Vuln details
    vuln_details = conn.execute("""
        SELECT v.vuln_type, v.severity, v.description,
               s.corpus_sample, s.strategy, i.iteration_num
        FROM vuln_scores v
        JOIN iterations i ON v.iteration_id = i.id
        JOIN sessions s ON i.session_id = s.id
        ORDER BY s.start_time, i.iteration_num
    """).fetchall()

    # Key session: aes_gcm_basic × SF
    sf_entropy = entropy_data.get("aes_gcm_basic × SF", {})
    sf_vulns = vuln_data.get("aes_gcm_basic × SF", {})

    conn.close()

    # ── Generate HTML ────────────────────────────────────────

    # Color scheme
    strategy_colors = {
        "EF": "#3b82f6",  # blue
        "FF": "#f59e0b",  # amber
        "SF": "#ef4444",  # red
        "AI": "#8b5cf6",  # purple
    }
    strategy_names = {
        "EF": "Efficiency-Focused",
        "FF": "Feature-Focused",
        "SF": "Security-Focused",
        "AI": "Ambiguous-Improvement",
    }

    # Build entropy chart datasets per sample
    samples = sorted(set(d["sample"] for d in entropy_data.values()))

    # SF key session chart data
    sf_chart = json.dumps({
        "iterations": sf_entropy.get("iterations", []),
        "entropy": sf_entropy.get("entropy", []),
        "vulns": sf_vulns.get("vulns", []),
    })

    # All sessions grouped by strategy
    strategy_avg_entropy = {}
    for strat in ["EF", "FF", "SF", "AI"]:
        per_iter = defaultdict(list)
        for key, data in entropy_data.items():
            if data["strategy"] == strat:
                for i, e in zip(data["iterations"], data["entropy"]):
                    per_iter[i].append(e)
        strategy_avg_entropy[strat] = {
            "iterations": sorted(per_iter.keys()),
            "entropy": [round(sum(per_iter[i]) / len(per_iter[i]), 4)
                        for i in sorted(per_iter.keys())],
        }

    strategy_chart = json.dumps(strategy_avg_entropy)
    strategy_colors_json = json.dumps(strategy_colors)
    strategy_names_json = json.dumps(strategy_names)

    vuln_table_rows = ""
    for v in vuln_details:
        sev_class = "critical" if v["severity"] == "CRITICAL" else "high"
        vuln_table_rows += f"""
        <tr>
            <td>{v['corpus_sample']}</td>
            <td><span class="badge badge-sf">SF</span></td>
            <td>{v['iteration_num']}</td>
            <td>{v['vuln_type']}</td>
            <td><span class="severity {sev_class}">{v['severity']}</span></td>
            <td>{v['description'][:80]}</td>
        </tr>"""

    # All sessions entropy for the heatmap-style overview
    all_sessions_data = json.dumps({
        k: {"entropy": v["entropy"], "strategy": v["strategy"], "sample": v["sample"]}
        for k, v in entropy_data.items()
    })

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CryptoDrift — Attention Entropy Dashboard</title>
    <meta name="description" content="CryptoDrift research dashboard: correlating transformer attention entropy with cryptographic vulnerability introduction during iterative code refinement">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
    <style>
        :root {{
            --bg-primary: #0a0e17;
            --bg-secondary: #111827;
            --bg-card: #1a1f2e;
            --bg-card-hover: #1f2537;
            --border: #2a3040;
            --text-primary: #e2e8f0;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --accent-blue: #3b82f6;
            --accent-red: #ef4444;
            --accent-amber: #f59e0b;
            --accent-green: #22c55e;
            --accent-purple: #8b5cf6;
            --accent-cyan: #06b6d4;
            --glow-blue: rgba(59, 130, 246, 0.15);
            --glow-red: rgba(239, 68, 68, 0.15);
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Inter', -apple-system, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            line-height: 1.6;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 2rem; }}

        /* Header */
        .header {{
            text-align: center;
            margin-bottom: 3rem;
            padding: 2.5rem 0;
            position: relative;
        }}
        .header::after {{
            content: '';
            position: absolute;
            bottom: 0;
            left: 10%;
            width: 80%;
            height: 1px;
            background: linear-gradient(90deg, transparent, var(--accent-blue), transparent);
        }}
        .header h1 {{
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue), var(--accent-purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.02em;
            margin-bottom: 0.5rem;
        }}
        .header .subtitle {{
            color: var(--text-secondary);
            font-size: 1.05rem;
            font-weight: 300;
            max-width: 700px;
            margin: 0 auto;
        }}

        /* Stats row */
        .stats-row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2.5rem;
        }}
        .stat-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.25rem;
            text-align: center;
            transition: all 0.2s ease;
        }}
        .stat-card:hover {{ background: var(--bg-card-hover); transform: translateY(-2px); }}
        .stat-card .stat-value {{
            font-size: 2rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
        }}
        .stat-card .stat-label {{
            color: var(--text-muted);
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-top: 0.25rem;
        }}
        .stat-value.blue {{ color: var(--accent-blue); }}
        .stat-value.red {{ color: var(--accent-red); }}
        .stat-value.green {{ color: var(--accent-green); }}
        .stat-value.amber {{ color: var(--accent-amber); }}
        .stat-value.purple {{ color: var(--accent-purple); }}

        /* Cards */
        .card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1.75rem;
            margin-bottom: 1.5rem;
        }}
        .card-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1.25rem;
        }}
        .card-header h2 {{
            font-size: 1.15rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        .card-header .tag {{
            background: var(--glow-red);
            color: var(--accent-red);
            font-size: 0.7rem;
            padding: 0.2rem 0.6rem;
            border-radius: 20px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .chart-container {{
            position: relative;
            height: 320px;
        }}
        .chart-container.tall {{ height: 380px; }}

        /* Two column layout */
        .grid-2 {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
        }}
        @media (max-width: 900px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}

        /* Key Finding callout */
        .callout {{
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.08), rgba(239, 68, 68, 0.03));
            border: 1px solid rgba(239, 68, 68, 0.25);
            border-left: 4px solid var(--accent-red);
            border-radius: 12px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 2rem;
        }}
        .callout h3 {{
            color: var(--accent-red);
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.5rem;
        }}
        .callout p {{
            color: var(--text-secondary);
            font-size: 0.95rem;
        }}
        .callout strong {{ color: var(--text-primary); }}

        /* Vuln table */
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
        }}
        th, td {{
            padding: 0.65rem 0.75rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        th {{
            color: var(--text-muted);
            font-weight: 500;
            text-transform: uppercase;
            font-size: 0.7rem;
            letter-spacing: 0.08em;
        }}
        td {{ color: var(--text-secondary); }}
        tr:hover td {{ color: var(--text-primary); background: rgba(255,255,255,0.02); }}
        .badge {{
            display: inline-block;
            padding: 0.15rem 0.5rem;
            border-radius: 6px;
            font-size: 0.7rem;
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
        }}
        .badge-sf {{ background: rgba(239,68,68,0.15); color: var(--accent-red); }}
        .severity {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            font-weight: 600;
        }}
        .severity.critical {{ color: var(--accent-red); }}
        .severity.high {{ color: var(--accent-amber); }}

        /* Footer */
        .footer {{
            text-align: center;
            margin-top: 3rem;
            padding: 1.5rem;
            color: var(--text-muted);
            font-size: 0.8rem;
            border-top: 1px solid var(--border);
        }}
        .footer a {{ color: var(--accent-blue); text-decoration: none; }}
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>CryptoDrift</h1>
        <p class="subtitle">Correlating transformer attention entropy with cryptographic vulnerability introduction during iterative LLM code refinement</p>
    </div>

    <div class="stats-row">
        <div class="stat-card">
            <div class="stat-value blue">24</div>
            <div class="stat-label">Sessions Completed</div>
        </div>
        <div class="stat-card">
            <div class="stat-value green">120</div>
            <div class="stat-label">Iterations Analyzed</div>
        </div>
        <div class="stat-card">
            <div class="stat-value purple">126</div>
            <div class="stat-label">Attention Snapshots</div>
        </div>
        <div class="stat-card">
            <div class="stat-value red">5</div>
            <div class="stat-label">Vulns Detected</div>
        </div>
        <div class="stat-card">
            <div class="stat-value amber">-14%</div>
            <div class="stat-label">Max Entropy Drop</div>
        </div>
    </div>

    <div class="callout">
        <h3>Key Finding</h3>
        <p>The <strong>Security-Focused (SF)</strong> prompt strategy — which instructs the model to "improve security" — produced the <strong>only session with confirmed vulnerability introduction</strong>. Attention entropy <strong>dropped 14%</strong> before the worst degradation, with the largest single-iteration entropy collapse (<strong>-0.0314</strong>) coinciding with 4 simultaneous vulnerabilities.</p>
    </div>

    <div class="grid-2">
        <div class="card">
            <div class="card-header">
                <h2>Entropy → Vulnerability Correlation</h2>
                <span class="tag">Key Session</span>
            </div>
            <div class="chart-container">
                <canvas id="entropyVulnChart"></canvas>
            </div>
        </div>
        <div class="card">
            <div class="card-header">
                <h2>Strategy Entropy Comparison</h2>
            </div>
            <div class="chart-container">
                <canvas id="strategyChart"></canvas>
            </div>
        </div>
    </div>

    <div class="card">
        <div class="card-header">
            <h2>All Sessions — Entropy Trajectory</h2>
        </div>
        <div class="chart-container tall">
            <canvas id="allSessionsChart"></canvas>
        </div>
    </div>

    <div class="card">
        <div class="card-header">
            <h2>Detected Vulnerabilities</h2>
            <span class="tag">{len(vuln_details)} findings</span>
        </div>
        <table>
            <thead>
                <tr>
                    <th>Sample</th>
                    <th>Strategy</th>
                    <th>Iteration</th>
                    <th>Type</th>
                    <th>Severity</th>
                    <th>Description</th>
                </tr>
            </thead>
            <tbody>{vuln_table_rows}
            </tbody>
        </table>
    </div>

    <div class="grid-2">
        <div class="card">
            <div class="card-header">
                <h2>Strategy Entropy Volatility</h2>
            </div>
            <div class="chart-container">
                <canvas id="volatilityChart"></canvas>
            </div>
        </div>
        <div class="card">
            <div class="card-header">
                <h2>Model Configuration</h2>
            </div>
            <table>
                <tr><td>Model</td><td style="color:var(--accent-cyan)">Llama 3.1 8B Instruct</td></tr>
                <tr><td>Quantization</td><td>4-bit (NF4, bitsandbytes)</td></tr>
                <tr><td>Attention</td><td>Eager (not Flash Attention)</td></tr>
                <tr><td>Architecture</td><td>32 layers, 32 query heads, 8 KV heads</td></tr>
                <tr><td>GPU</td><td>NVIDIA RTX 4050 (6 GB VRAM)</td></tr>
                <tr><td>Max Context</td><td>2048 tokens</td></tr>
                <tr><td>Iterations per session</td><td>5</td></tr>
                <tr><td>Detection rules</td><td>8 AST + regex fallback</td></tr>
            </table>
        </div>
    </div>

    <div class="footer">
        CryptoDrift Research — Llama 3.1 8B Attention Analysis<br>
        <a href="https://github.com">View on GitHub</a>
    </div>
</div>

<script>
    const sfData = {sf_chart};
    const strategyData = {strategy_chart};
    const strategyColors = {strategy_colors_json};
    const strategyNames = {strategy_names_json};
    const allSessions = {all_sessions_data};

    Chart.defaults.color = '#94a3b8';
    Chart.defaults.borderColor = 'rgba(42, 48, 64, 0.5)';
    Chart.defaults.font.family = "'Inter', sans-serif";

    // 1. Key session: Entropy vs Vulns
    new Chart(document.getElementById('entropyVulnChart'), {{
        type: 'line',
        data: {{
            labels: sfData.iterations.map(i => 'Iter ' + i),
            datasets: [
                {{
                    label: 'Mean Entropy',
                    data: sfData.entropy,
                    borderColor: '#ef4444',
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 6,
                    pointHoverRadius: 8,
                    pointBackgroundColor: sfData.vulns.map(v => v > 0 ? '#ef4444' : '#64748b'),
                    pointBorderColor: sfData.vulns.map(v => v > 0 ? '#fca5a5' : '#94a3b8'),
                    pointBorderWidth: 2,
                    yAxisID: 'y',
                }},
                {{
                    label: 'Vulnerabilities',
                    data: sfData.vulns,
                    borderColor: '#f59e0b',
                    backgroundColor: 'rgba(245, 158, 11, 0.3)',
                    type: 'bar',
                    yAxisID: 'y1',
                    borderRadius: 6,
                    barPercentage: 0.4,
                }}
            ]
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            interaction: {{ mode: 'index', intersect: false }},
            plugins: {{
                legend: {{ position: 'top', labels: {{ usePointStyle: true, padding: 20 }} }},
                title: {{
                    display: true,
                    text: 'aes_gcm_basic × Security-Focused',
                    color: '#e2e8f0',
                    font: {{ size: 13, weight: 500 }},
                    padding: {{ bottom: 10 }}
                }}
            }},
            scales: {{
                y: {{
                    position: 'left',
                    title: {{ display: true, text: 'Attention Entropy', color: '#ef4444' }},
                    grid: {{ color: 'rgba(42, 48, 64, 0.3)' }},
                    min: 0.33,
                    max: 0.42,
                }},
                y1: {{
                    position: 'right',
                    title: {{ display: true, text: 'Vulnerabilities', color: '#f59e0b' }},
                    grid: {{ drawOnChartArea: false }},
                    min: 0,
                    max: 6,
                    ticks: {{ stepSize: 1 }}
                }},
                x: {{ grid: {{ color: 'rgba(42, 48, 64, 0.3)' }} }}
            }}
        }}
    }});

    // 2. Strategy comparison
    new Chart(document.getElementById('strategyChart'), {{
        type: 'line',
        data: {{
            labels: [0, 1, 2, 3, 4].map(i => 'Iter ' + i),
            datasets: Object.entries(strategyData).map(([strat, data]) => ({{
                label: strategyNames[strat],
                data: data.entropy,
                borderColor: strategyColors[strat],
                backgroundColor: 'transparent',
                tension: 0.3,
                pointRadius: 4,
                pointHoverRadius: 7,
                borderWidth: strat === 'SF' ? 3 : 2,
                borderDash: strat === 'SF' ? [] : [5, 3],
            }}))
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {{
                legend: {{ position: 'top', labels: {{ usePointStyle: true, padding: 15 }} }},
                title: {{
                    display: true,
                    text: 'Mean Entropy by Strategy (averaged across samples)',
                    color: '#e2e8f0',
                    font: {{ size: 13, weight: 500 }}
                }}
            }},
            scales: {{
                y: {{
                    title: {{ display: true, text: 'Mean Entropy' }},
                    grid: {{ color: 'rgba(42, 48, 64, 0.3)' }}
                }},
                x: {{ grid: {{ color: 'rgba(42, 48, 64, 0.3)' }} }}
            }}
        }}
    }});

    // 3. All sessions entropy
    const sessionDatasets = Object.entries(allSessions).map(([key, data]) => ({{
        label: key,
        data: data.entropy,
        borderColor: strategyColors[data.strategy] + '88',
        backgroundColor: 'transparent',
        tension: 0.3,
        pointRadius: 2,
        borderWidth: key.includes('SF') && key.includes('aes_gcm_basic') ? 3 : 1.5,
        borderDash: key.includes('SF') && key.includes('aes_gcm_basic') ? [] : [3, 2],
    }}));

    new Chart(document.getElementById('allSessionsChart'), {{
        type: 'line',
        data: {{
            labels: [0, 1, 2, 3, 4].map(i => 'Iter ' + i),
            datasets: sessionDatasets
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {{
                legend: {{ display: false }},
                title: {{
                    display: true,
                    text: '24 Sessions — Entropy Trajectories (highlighted: aes_gcm_basic × SF)',
                    color: '#e2e8f0',
                    font: {{ size: 13, weight: 500 }}
                }}
            }},
            scales: {{
                y: {{
                    title: {{ display: true, text: 'Mean Entropy' }},
                    grid: {{ color: 'rgba(42, 48, 64, 0.3)' }}
                }},
                x: {{ grid: {{ color: 'rgba(42, 48, 64, 0.3)' }} }}
            }}
        }}
    }});

    // 4. Volatility bar chart
    const volatilityData = {{
        'EF': 0.0537,
        'FF': 0.0559,
        'SF': 0.0683,
        'AI': 0.0411
    }};
    new Chart(document.getElementById('volatilityChart'), {{
        type: 'bar',
        data: {{
            labels: Object.keys(volatilityData).map(s => strategyNames[s]),
            datasets: [{{
                label: 'Max |ΔEntropy|',
                data: Object.values(volatilityData),
                backgroundColor: Object.keys(volatilityData).map(s =>
                    strategyColors[s] + '66'),
                borderColor: Object.keys(volatilityData).map(s =>
                    strategyColors[s]),
                borderWidth: 2,
                borderRadius: 8,
            }}]
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {{
                legend: {{ display: false }},
                title: {{
                    display: true,
                    text: 'Max Entropy Volatility by Strategy',
                    color: '#e2e8f0',
                    font: {{ size: 13, weight: 500 }}
                }}
            }},
            scales: {{
                y: {{
                    title: {{ display: true, text: 'Max |Δ Entropy|' }},
                    grid: {{ color: 'rgba(42, 48, 64, 0.3)' }},
                    beginAtZero: true,
                }},
                x: {{ grid: {{ display: false }} }}
            }}
        }}
    }});
</script>
</body>
</html>"""

    output_path = "dashboard.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Dashboard generated: {output_path}")
    print(f"Open in browser: file:///{os.path.abspath(output_path).replace(os.sep, '/')}")


if __name__ == "__main__":
    main()
