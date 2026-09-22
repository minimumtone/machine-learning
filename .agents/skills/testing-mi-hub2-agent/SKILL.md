---
name: testing-mi-hub2-agent
description: How to run and test the mi_hub2 state-driven research agent (Streamlit UI, FastAPI, acceptance tests) in minimumtone/machine-learning
---

# Testing mi_hub2 research agent

## Run the apps (from repo root)
- Streamlit UI: `cd mi_hub2 && PYTHONPATH=src streamlit run src/mi_hub/agent/ui_streamlit.py --server.port 8501 --server.headless true`
- FastAPI: `cd mi_hub2 && PYTHONPATH=src uvicorn mi_hub.agent.api:app --port 8800` (all routes are prefixed with `/api/agent`, check `/openapi.json`)
- Acceptance tests: `cd mi_hub2 && python -m pytest tests/ -q`
- Dependencies: `pip install streamlit fastapi uvicorn pytest httpx pandas pydantic`
- Sessions persist to `~/mi_hub_data/agent_sessions/*.json` — delete this dir for a clean slate. Check persisted `agent_state` with `grep -o '"agent_state": "[^"]*"' ~/mi_hub_data/agent_sessions/*.json`.

## API payload gotchas
- `POST /api/agent/goals` expects `{"statement": "..."}` (not `goal_text`).
- `POST /api/agent/approvals/{approval_id}` requires `{"session_id": "...", "approve": true}` in the body.

## Golden path via UI
1. Sidebar: default goal → 「セッション開始」 → plan v1 with 6 tasks.
2. Chat 「実行を続けて」 → 5 tasks run, stops at run_models_bulk with awaiting_approval; approve in 承認 tab.
3. Chat 「実行を続けて」 again → 「正常終了: 承認済み検証計画が完了」, progress 100%, state completed. (On older builds with `max_iterations=5` this blocked with 「資源制約: 反復回数上限に到達」; if that happens, workaround is 「再開」 then 「実行を続けて」. Fixed as of PR #463 branch.)
4. Approval also works via chat: type 「承認」 / 「却下」 when a pending approval exists.
5. Chat commands: 一時停止 / 再開 / 終了.

## PR #463+ features (GraphRAG / loop pills / timeline / ML scripts)
- Start Streamlit with `OPENAI_API_KEY` in env to enable LLM intent classification and script generation (bind the secret via exec env; do not write it to files). Without it only deterministic fallback commands work — chat script requests will NOT generate scripts.
- Loop pills (Goal→Observe→Plan→Human Check→Act→Evaluate→Replan) at top of Agent pane; current stage is red. `completed`/`paused` show 「現在の状態: …（ループ外）」.
- Chat a request like 「…をscikit-learnで予測するデモをやろう。図もPNGで保存して。」 → script proposed (未実行) → approve in 承認 tab → runs with exit code, generated files go to `~/mi_hub_data/agent_sessions/workspaces/SESSION-*/`, PNG shown inline in 証拠 tab.
- 計画 tab: 「研究プロセス改善提案を生成」 (GraphRAG); 仮説 tab: 「研究目標から仮説候補を生成」 and 「利用ログから辞書を更新」; タイムライン tab merges audit log + approvals + errors newest-first.
- Sidebar 「事例レポートを書き出す」 writes `case_report.md` into the session workspace and shows a download button.
- Streamlit tab devinids shift between rerenders — re-read the DOM before clicking tabs; to clear the sidebar goal textarea use click → Ctrl+A → Delete → type (plain type appends into existing text).

## PR #493+ features (Falsifier / 3-value judgement)
- New sessions now generate a 7-task plan: `FalsifierAgent / search_counter_evidence` sits right after `HypothesisAgent / generate_hypotheses`.
- 仮説 tab expander shows structured fields: 想定機構 / 適用範囲 / 予測 / 反証条件 / 別機構の候補（Falsifier） / 「反証検討で収集した証拠: N 件」.
- After evaluation completes, chat shows 【判定案】…「この判定を確定しますか？」. In the 仮説 tab, below 検証承認/却下, there is 判定案（ルール評価）+ 4-criteria table (効果の有意性/効果方向の一致/独立系列での再現/データ点数, ○/×) with 「判定を確定」「保留のまま続行」 buttons — long expander, scroll the Agent pane to reach them.
- 「保留のまま続行」 keeps status unchanged and logs `judgement_deferred` (actor=human); 「判定を確定」 shows （研究者確定済み）, removes buttons, sets hypothesis status to the verdict (e.g. inconclusive), and persists `judgement.confirmed_by_human: true` in the session JSON. Confirmation survives session reload.
- The deterministic demo data yields verdict 保留（Inconclusive）(fails 独立系列での再現).
- Expected test count as of PR #493: 95 passed. `ruff check src tests` with a recent local ruff (0.16) flags a few pre-existing BLE001/I001/UP024 findings — likely a version/config mismatch with maintainer's setup, don't treat as a PR regression without confirming the CI ruff config.

## Known bugs to watch
- (Fixed as of PR #463 branch) `observe()` used to overwrite persisted `agent_state` to `observing` on every rerender. Now read-only: the 状態 metric and persisted JSON correctly show awaiting_approval/completed. Still worth re-checking after refactors.
- OPENAI_API_KEY is optional for the core loop (deterministic fallback), but required for LLM intent classification, chat script generation, and free-form chat replies.
- Expected test count: 95 passed with `PYTHONPATH=src python -m pytest tests -q` (was 49 at #463, 32 pre-#463).
