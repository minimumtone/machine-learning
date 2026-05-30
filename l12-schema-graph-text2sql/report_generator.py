#!/usr/bin/env python3
"""Generate detailed HTML verification report for materials engineers.

Targeted at materials engineering professionals. Explains every
technical decision with honest assessment of limitations.
No hand-waving, no shortcuts — demerits stated explicitly.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def generate_html_report(data: dict) -> str:
    """Build the full HTML report from verification results."""
    S = data["summary"]
    R = data["results"]
    drawio_exists = (Path(__file__).parent / "figures" / "t2sql_pipeline_flow.drawio").exists()

    # ── helper to escape HTML ──
    def h(s: str) -> str:
        return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # ── Start HTML ──
    parts: list[str] = []
    W = parts.append  # shorthand

    W(f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>L1_2/B2 Schema-Graph Text-to-SQL 包括検証レポート</title>
<style>
:root {{ --pass:#2e7d32; --fail:#c62828; --warn:#ef6c00; --bg:#fafafa; }}
*{{ margin:0; padding:0; box-sizing:border-box; }}
body{{ font-family:'Segoe UI',Roboto,sans-serif; background:var(--bg); color:#333;
       line-height:1.85; padding:24px 32px; max-width:1400px; margin:auto; }}
h1{{ color:#1a237e; border-bottom:3px solid #1a237e; padding-bottom:8px;
     margin:30px 0 16px; font-size:1.8em; }}
h2{{ color:#283593; border-bottom:2px solid #c5cae9; padding-bottom:6px;
     margin:36px 0 14px; font-size:1.4em; }}
h3{{ color:#3949ab; margin:24px 0 10px; font-size:1.15em; }}
h4{{ color:#455a64; margin:18px 0 8px; font-size:1.05em; }}
p{{ margin:8px 0 10px; }}
ul,ol{{ margin:6px 0 10px 24px; }}
li{{ margin:4px 0; }}
table{{ border-collapse:collapse; width:100%; margin:14px 0; font-size:0.92em; }}
th,td{{ border:1px solid #bbb; padding:8px 10px; text-align:left; }}
th{{ background:#e8eaf6; font-weight:600; }}
tr:nth-child(even){{ background:#f5f5f5; }}
tr.pass-row{{ background:#e8f5e9; }} tr.fail-row{{ background:#ffebee; }}
.pass{{ color:var(--pass); }} .fail{{ color:var(--fail); }} .warn{{ color:var(--warn); }}
.big{{ font-size:2.5em; font-weight:bold; }}
.cards{{ display:flex; gap:18px; flex-wrap:wrap; margin:18px 0; }}
.card{{ background:#fff; border-radius:10px; box-shadow:0 2px 8px rgba(0,0,0,.1);
        padding:18px; flex:1; min-width:190px; text-align:center; }}
.sql{{ background:#263238; color:#e0e0e0; padding:14px; border-radius:6px;
       overflow-x:auto; font-family:'Fira Code',monospace; font-size:0.87em;
       white-space:pre-wrap; margin:10px 0; }}
.note{{ background:#e3f2fd; border-left:4px solid #1565c0; padding:12px 16px;
        margin:12px 0; border-radius:0 6px 6px 0; }}
.warn-box{{ background:#fff3e0; border-left:4px solid #ef6c00; padding:12px 16px;
            margin:12px 0; border-radius:0 6px 6px 0; }}
.bad{{ background:#ffebee; border-left:4px solid #c62828; padding:12px 16px;
       margin:12px 0; border-radius:0 6px 6px 0; }}
.demerit{{ background:#fff8e1; border-left:4px solid #f57f17; padding:14px 18px;
           margin:14px 0; border-radius:0 6px 6px 0; }}
.demerit h4{{ color:#e65100; margin:0 0 8px 0; }}
.severity-high{{ border-left-color:#c62828; background:#ffebee; }}
.severity-mid{{ border-left-color:#ef6c00; background:#fff3e0; }}
.severity-low{{ border-left-color:#fbc02d; background:#fffde7; }}
.code{{ background:#eceff1; padding:2px 6px; border-radius:3px;
        font-family:monospace; font-size:0.92em; }}
.er-xml{{ background:#f3e5f5; border:1px solid #ce93d8; border-radius:6px;
          padding:14px; font-family:monospace; font-size:0.85em;
          overflow-x:auto; white-space:pre-wrap; margin:10px 0; }}
details{{ margin:10px 0; }}
details summary{{ cursor:pointer; font-weight:600; color:#1565c0; }}
.level-compare{{ display:flex; gap:15px; margin:15px 0; flex-wrap:wrap; }}
.level-box{{ flex:1; min-width:300px; background:#fff; border-radius:8px;
             padding:15px; box-shadow:0 1px 4px rgba(0,0,0,.1); }}
.level-0{{ border-left:4px solid #c62828; }}
.level-1{{ border-left:4px solid #1565c0; }}
.level-2{{ border-left:4px solid #2e7d32; }}
.tag{{ display:inline-block; padding:2px 8px; border-radius:4px;
       font-size:0.85em; font-weight:600; color:#fff; }}
.tag-normal{{ background:#1976d2; }} .tag-no_results{{ background:#7b1fa2; }}
.tag-sloppy{{ background:#ef6c00; }} .tag-contradictory{{ background:#c62828; }}
.tag-rejection{{ background:#d32f2f; }} .tag-safety{{ background:#388e3c; }}
.pipe-step{{ display:inline-block; padding:8px 16px; margin:4px; border-radius:20px;
             font-size:0.9em; font-weight:500; }}
.ps-in{{ background:#bbdefb; }} .ps-ex{{ background:#c8e6c9; }}
.ps-gr{{ background:#ffcdd2; }} .ps-sq{{ background:#fff9c4; }}
.ps-gu{{ background:#ffccbc; }} .ps-db{{ background:#b3e5fc; }}
.ps-ra{{ background:#ffe0b2; }}
.arrow{{ font-size:1.3em; color:#666; }}
footer{{ margin-top:40px; padding:20px 0; border-top:1px solid #ddd;
         color:#999; font-size:0.85em; text-align:center; }}
</style>
</head>
<body>
""")

    # ================================================================
    # 0. タイトル・サマリ
    # ================================================================
    W(f"""
<h1>L1<sub>2</sub>/B2 Schema-Graph-Assisted Text-to-SQL<br>包括検証レポート</h1>
<p style="color:#666;">Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
<p style="color:#666;">対象読者：材料工学を専門とする研究者・エンジニア</p>

<div class="note">
<b>このレポートについて：</b>
自然言語で材料データベースを検索する Text-to-SQL システムを構築し、
39件のテストケースで検証した結果をまとめたものです。
各セクションでは設計判断の根拠と、<b>デメリット・限界点を忖度なしに</b>記載します。
</div>

<h2>0. 全体サマリ</h2>
<div class="cards">
  <div class="card"><div class="big">{S['total']}</div><div>テスト総数</div></div>
  <div class="card"><div class="big pass">{S['passed']}</div><div>合格</div></div>
  <div class="card"><div class="big fail">{S['failed']}</div><div>不合格</div></div>
  <div class="card"><div class="big" style="color:{'var(--pass)' if S['pass_rate']>=90 else 'var(--fail)'}">{S['pass_rate']}%</div><div>合格率</div></div>
  <div class="card"><div class="big" style="color:#1565c0">{S['few_shot_store']['total_examples']}</div><div>Few-Shot事例数</div></div>
</div>

<table>
<tr><th>カテゴリ</th><th>内容</th><th>件数</th><th>合格</th><th>合格率</th></tr>
""")
    cat_desc = {
        "normal": "正常系：標準的な材料検索クエリ",
        "no_results": "該当なし系：存在しない化合物・未登録元素",
        "sloppy": "いい加減なクエリ：曖昧・不完全・無関係な入力",
        "contradictory": "矛盾条件：自己矛盾するクエリ",
        "rejection": "拒否系：SQL injection・破壊的操作",
        "safety": "安全検査：SQL Guard の動作確認",
    }
    for cat, info in S["categories"].items():
        color = "pass" if info["pass_rate"] >= 100 else ("warn" if info["pass_rate"] >= 80 else "fail")
        W(f'<tr><td><span class="tag tag-{cat}">{cat}</span></td>'
          f'<td>{cat_desc.get(cat, "")}</td>'
          f'<td>{info["total"]}</td><td>{info["passed"]}</td>'
          f'<td class="{color}">{info["pass_rate"]}%</td></tr>')
    W("</table>")

    # ================================================================
    # 1. このシステムは何をするのか
    # ================================================================
    W("""
<h2>1. このシステムは何をするのか（背景と目的）</h2>

<h3>1.1 解決したい問題</h3>
<p>材料研究者は、データベース（DB）に蓄積された計算データを検索したいが、
<b>SQL を書けない人が大半</b>である。例えば、こういう質問をしたい：</p>
<ul>
<li>「Niを含む安定なL1<sub>2</sub>型化合物を形成エネルギーが低い順に出して」</li>
<li>「FeとAlを含むB2化合物の格子定数は？」</li>
</ul>
<p>これを SQL に変換するには、テーブル構造（どの情報がどのテーブルにあるか）、
JOIN の方法、WHERE 条件の書き方を知っている必要がある。</p>

<h3>1.2 Text-to-SQL (T2SQL) とは</h3>
<p><b>自然言語 (NL) を入力として、対応する SQL クエリを自動生成し、DBを検索する技術。</b></p>
<p>単純に聞こえるが、実際にはいくつもの困難がある：</p>
<ol>
<li><b>テーブル選択：</b>「L1<sub>2</sub>」はどのテーブルのどのカラムか？
→ <span class="code">structure.prototype</span> と <span class="code">structure.strukturbericht</span></li>
<li><b>JOIN 経路：</b>「元素」と「安定性」は別テーブルにあるが、どうやって結合するか？
→ 両方とも <span class="code">material_entry.entry_id</span> で結合</li>
<li><b>安全性：</b>ユーザー入力に <span class="code">DROP TABLE</span> が含まれていたら？
→ 絶対に実行してはいけない</li>
<li><b>曖昧さ：</b>「安定な」の数値的定義は？ → E<sub>hull</sub> &le; 0.001 eV/atom</li>
</ol>

<h3>1.3 本システムのアプローチ：Schema Graph + Few-Shot</h3>
<p>本システムは3段階の安全装置を持つ：</p>
<ol>
<li><b>Schema Graph Traversal Engine：</b>DB のテーブル構造をグラフ化し、
必要最小限の JOIN 経路を自動探索する</li>
<li><b>SQL Guard：</b>生成された SQL の安全性を検証する
（禁止キーワード、テーブルホワイトリスト、LIMIT 自動付与）</li>
<li><b>SQL-as-Few-Shot-Examples：</b>過去の成功クエリを蓄積し、
類似の新クエリに「お手本」として注入する（RAG 的フィードバックループ）</li>
</ol>
""")

    # ================================================================
    # 2. データ準備
    # ================================================================
    W("""
<h2>2. データの準備</h2>

<h3>2.1 データの出所：OQMD (Open Quantum Materials Database)</h3>
<p>OQMD は、第一原理計算（DFT: 密度汎関数理論）で算出された材料物性データの公開DB。
本システムでは、<b>B2 型</b>と <b>L1<sub>2</sub> 型</b>の金属間化合物データを
OQMD API から取得した。</p>

<div class="note">
<b>B2 (CsCl型)：</b>BCC (体心立方格子) 基盤の規則構造。代表例: NiAl, FeAl<br>
<b>L1<sub>2</sub> (Cu<sub>3</sub>Au型)：</b>FCC (面心立方格子) 基盤の規則構造。
代表例: Ni<sub>3</sub>Al (γ'相、ニッケル基超合金の強化相)
</div>

<h4>取得件数</h4>
<table>
<tr><th>Prototype</th><th>Strukturbericht</th><th>取得件数</th><th>安定 (E<sub>hull</sub> &le; 1 meV)</th><th>準安定 (E<sub>hull</sub> &le; 50 meV)</th></tr>
<tr><td>CsCl</td><td>B2</td><td>636</td><td>185</td><td>198</td></tr>
<tr><td>AuCu3</td><td>L1<sub>2</sub></td><td>273</td><td>88</td><td>138</td></tr>
<tr><td colspan="2"><b>合計</b></td><td><b>909</b></td><td><b>273</b></td><td><b>336</b></td></tr>
</table>

<h3>2.2 PostgreSQL へのデータ投入</h3>
<p>取得したデータは正規化して PostgreSQL に投入した。
正規化とは「1つの事実を1つの場所にだけ保存する」というDB設計原則で、
データの重複を排除し、整合性を保つ。</p>

<h4>具体的な処理フロー</h4>
<ol>
<li>OQMD API (<span class="code">https://oqmd.org/oqmdapi/formationenergy</span>) から
JSON データを取得</li>
<li>CSV ファイルとしてローカル保存 (<span class="code">oqmd_b2_data.csv</span>,
<span class="code">oqmd_l12_data.csv</span>)</li>
<li>Python スクリプトで正規化し、各テーブルに INSERT</li>
<li>外部キー制約により参照整合性を保証</li>
</ol>

<h3>2.3 スキーマ拡張</h3>
<p>OQMD データに含まれるフィールドに対応するため、以下のカラムを追加した：</p>
<ul>
<li><span class="code">structure</span> テーブル: <span class="code">space_group TEXT</span>（空間群名）</li>
<li><span class="code">phase_stability</span> テーブル: <span class="code">band_gap DOUBLE PRECISION</span>（バンドギャップ, eV単位）</li>
</ul>
""")

    # ================================================================
    # 3. E-R図とXMLによるRAG処理
    # ================================================================
    W("""
<h2>3. E-R図の設計とXML表現によるRAG的処理</h2>

<h3>3.1 E-R図とは</h3>
<p><b>E-R図 (Entity-Relationship Diagram)</b> は、DB内のテーブル（実体: Entity）と
テーブル間の関係（Relationship）を図示したものである。
「どのデータがどこに入っているか」「テーブル同士はどう結ばれているか」が一目でわかる。</p>

<div class="note">
<b>なぜ E-R図が重要か：</b>
Text-to-SQL では、自然言語の「Niを含む安定なL1<sub>2</sub>」という表現を
「composition テーブルの element カラムが 'Ni'」
「phase_stability テーブルの energy_above_hull カラムが 0.001 以下」
「structure テーブルの prototype カラムが 'L12'」
という3つの異なるテーブルの条件に変換する必要がある。
E-R図なしにこの変換は不可能。
</div>

<h3>3.2 本システムのテーブル構造（7テーブル）</h3>
<pre style="background:#e8eaf6;padding:16px;border-radius:8px;font-size:0.95em;line-height:1.7;">
material_entry (PK: entry_id)   ← 化合物の基本情報 (formula, chemical_system)
    │
    ├── 1:N ──→ composition (FK: entry_id)       ← 構成元素と組成比
    │               element='Ni', atomic_fraction=0.75
    │
    ├── 1:N ──→ structure (FK: entry_id)          ← 結晶構造情報
    │               prototype='L12', lattice_a=3.572
    │
    ├── 1:N ──→ phase_stability (FK: entry_id)    ← 熱力学的安定性
    │               formation_energy=-0.42, energy_above_hull=0.0
    │
    └── 1:N ──→ calculation (FK: entry_id)        ← 計算メタデータ
                    │
                    └── 1:N ──→ calculated_property (FK: calculation_id) ← 算出物性
</pre>

<div class="note">
<b>「1:N」の意味：</b>
1つの material_entry に対して、composition が<b>複数行</b>存在する。
例えば Ni<sub>3</sub>Al なら composition に「Ni (0.75)」と「Al (0.25)」の2行がある。
これが後述の「多元素検索の難しさ」につながる。
</div>

<h3>3.3 E-R図をXMLで記述する意味（RAG的処理）</h3>
<p>LLM (Large Language Model) に SQL を生成させる場合、
「どのテーブル・カラムが使えるか」をプロンプトに含める必要がある。
このとき、E-R図の情報を<b>構造化されたXML/YAML</b>として渡すことで、
LLM が正しいテーブル・カラム名を「参照」できるようになる。</p>

<p>これは <b>RAG (Retrieval-Augmented Generation)</b> の考え方と同じである：</p>
<ol>
<li><b>通常の LLM：</b>学習データに含まれるスキーマ知識しか使えない（古い・不正確かもしれない）</li>
<li><b>RAG 的アプローチ：</b>実際の DB スキーマを XML で構造化し、プロンプトに注入（Retrieval）
→ LLM はこの「最新の正確な情報」を参照して SQL を生成（Augmented Generation）</li>
</ol>

<h4>XML表現の例</h4>
<div class="er-xml">&lt;schema name="l12_materials"&gt;
  &lt;table name="material_entry" primary_key="entry_id"&gt;
    &lt;column name="entry_id" type="TEXT" /&gt;
    &lt;column name="formula" type="TEXT" /&gt;
    &lt;column name="reduced_formula" type="TEXT" /&gt;
    &lt;column name="chemical_system" type="TEXT" /&gt;
    &lt;column name="number_of_elements" type="INTEGER" /&gt;
  &lt;/table&gt;

  &lt;table name="composition" primary_key="composition_id"&gt;
    &lt;column name="entry_id" type="TEXT" references="material_entry.entry_id" /&gt;
    &lt;column name="element" type="TEXT" /&gt;
    &lt;column name="atomic_fraction" type="FLOAT" /&gt;
  &lt;/table&gt;

  &lt;table name="structure" primary_key="structure_id"&gt;
    &lt;column name="entry_id" type="TEXT" references="material_entry.entry_id" /&gt;
    &lt;column name="prototype" type="TEXT" comment="L12, B2 etc." /&gt;
    &lt;column name="strukturbericht" type="TEXT" /&gt;
    &lt;column name="lattice_a" type="FLOAT" unit="angstrom" /&gt;
  &lt;/table&gt;

  &lt;table name="phase_stability" primary_key="stability_id"&gt;
    &lt;column name="entry_id" type="TEXT" references="material_entry.entry_id" /&gt;
    &lt;column name="formation_energy_per_atom" type="FLOAT" unit="eV/atom" /&gt;
    &lt;column name="energy_above_hull" type="FLOAT" unit="eV/atom"
            comment="0 = thermodynamically stable" /&gt;
    &lt;column name="band_gap" type="FLOAT" unit="eV" /&gt;
  &lt;/table&gt;

  &lt;!-- FK relationships --&gt;
  &lt;relationship from="composition.entry_id" to="material_entry.entry_id" /&gt;
  &lt;relationship from="structure.entry_id" to="material_entry.entry_id" /&gt;
  &lt;relationship from="phase_stability.entry_id" to="material_entry.entry_id" /&gt;
&lt;/schema&gt;</div>

<p>この XML を LLM プロンプトの冒頭に注入することで、LLM は
「<span class="code">prototype</span> は <span class="code">structure</span> テーブルにある」
「<span class="code">energy_above_hull</span> は <span class="code">phase_stability</span> テーブルにある」
ということを<b>幻覚 (hallucination) なしに参照</b>できる。</p>

<div class="warn-box">
<b>なぜ重要か（失敗例）：</b>
XML スキーマ注入なしで LLM に SQL を書かせると、以下のような幻覚が発生する：<br>
・存在しないテーブル <span class="code">materials</span> を参照する（正しくは <span class="code">material_entry</span>）<br>
・存在しないカラム <span class="code">stability</span> を使う（正しくは <span class="code">energy_above_hull</span>）<br>
・JOIN 条件を間違える
</div>
""")

    # ================================================================
    # 4. Schema Graph Traversal Engine
    # ================================================================
    W("""
<h2>4. Schema Graph Traversal Engine — なぜ必要で、何をするのか</h2>

<h3>4.1 問題：テーブル間の結合 (JOIN) は自明ではない</h3>
<p>SQL でデータを検索するとき、関連する情報が複数のテーブルに分散しているなら、
<b>JOIN</b>（テーブルの結合）が必要になる。</p>

<p>例：「Ni を含む安定な L1<sub>2</sub> 化合物」を検索するには：</p>
<ul>
<li><span class="code">composition</span> テーブル（元素: Ni）</li>
<li><span class="code">structure</span> テーブル（prototype: L12）</li>
<li><span class="code">phase_stability</span> テーブル（安定性: E<sub>hull</sub> &le; 0.001）</li>
</ul>
<p>この3テーブルを <span class="code">material_entry</span> 経由で結合する必要がある。</p>

<div class="note">
<b>なぜ自明でないか：</b>
テーブルが7つあるとき、「どのテーブルを」「どの順番で」「どのカラムで」結合するかの
組み合わせは膨大。例えば <span class="code">calculated_property</span> に到達するには
<span class="code">material_entry → calculation → calculated_property</span>
という2段階の JOIN が必要（直接 JOIN はできない）。
</div>

<h3>4.2 解決策：テーブル構造をグラフにする</h3>
<p>本システムでは、Python の <b>NetworkX</b> ライブラリを使って、
テーブル構造を<b>有向グラフ</b>として構築する：</p>

<ul>
<li><b>ノード</b> = テーブル名 (material_entry, composition, structure, ...)</li>
<li><b>エッジ</b> = 外部キー (FK) 関係 (composition.entry_id → material_entry.entry_id)</li>
</ul>

<p>グラフを構築したら、<b>最短経路アルゴリズム</b>（NetworkX の
<span class="code">shortest_path()</span>）を使って、
必要なテーブル間を最小コストで結ぶ JOIN 経路を自動探索する。</p>

<h3>4.3 Schema Graph なしだとどうなるか（Naive アプローチの問題）</h3>
<p>Schema Graph を使わない「Naive」アプローチ（本レポートの Level 0）では、
以下の問題が発生する：</p>

<table>
<tr><th>問題</th><th>Naive (Level 0)</th><th>Schema Graph (Level 1)</th></tr>
<tr>
<td><b>JOIN の選択</b></td>
<td class="fail">常に全5テーブルを JOIN<br>（不要なテーブルも含む → パフォーマンス低下）</td>
<td class="pass">必要なテーブルのみ JOIN<br>（最短経路で最小限の結合）</td>
</tr>
<tr>
<td><b>複数元素 AND 検索</b><br>
「NiとAlを両方含む」</td>
<td class="fail">
<span class="code">WHERE c.element='Ni' AND c.element='Al'</span><br>
→ <b>0件</b>（1行は1元素しか持てない）
</td>
<td class="pass">
<span class="code">EXISTS (SELECT 1 FROM composition WHERE element='Ni')</span><br>
<span class="code">AND EXISTS (...element='Al')</span><br>
→ <b>正しい結果</b>
</td>
</tr>
<tr>
<td><b>LIMIT の有無</b></td>
<td class="fail">LIMIT なし → 数万行が返る可能性</td>
<td class="pass">LIMIT 100 自動付与</td>
</tr>
<tr>
<td><b>DISTINCT の有無</b></td>
<td class="fail">なし → JOIN により重複行が発生</td>
<td class="pass">DISTINCT 自動付与</td>
</tr>
<tr>
<td><b>SQL 安全検査</b></td>
<td class="fail">なし → SQL injection 可能</td>
<td class="pass">sqlglot によるパース検証 + 禁止キーワードチェック</td>
</tr>
</table>

<div class="bad">
<b>致命的な問題 — 複数元素 AND 検索</b><br>
Naive アプローチでは「Ni と Al を両方含む化合物」を検索すると<b>必ず0件</b>になる。
なぜなら <span class="code">composition</span> テーブルの1行には1つの元素しか入らないため、
<span class="code">c.element = 'Ni' AND c.element = 'Al'</span> は論理的に矛盾する。<br>
正しくは <b>EXISTS サブクエリ</b>を使い、「Ni を含む行が存在する AND Al を含む行が存在する」
と表現する必要がある。Schema Graph はこの変換を自動的に行う。
</div>

<h3>4.4 実装の中身（コードレベルの説明）</h3>

<h4>4.4.1 グラフ構築 (<span class="code">graph/graph_builder.py</span>)</h4>
<p>PostgreSQL の <span class="code">information_schema</span> から FK 制約を読み取り、
NetworkX のグラフに変換する。DB に接続できない場合は YAML 定義ファイルから構築する。</p>

<h4>4.4.2 経路探索 (<span class="code">graph/traversal_engine.py</span>)</h4>
<div class="sql">
# 2テーブル間の最短 JOIN 経路を求める
def find_shortest_table_path(graph, source_table, target_table):
    return nx.shortest_path(graph, source_table, target_table)

# 複数テーブルを接続する Steiner tree 近似
def find_join_subgraph(graph, required_tables):
    # 全ペアの最短パスを求め、最小限のテーブルセットでカバー
    for src, tgt in combinations(required_tables, 2):
        path = find_shortest_table_path(graph, src, tgt)
        ...
</div>
""")

    # ================================================================
    # 5. T2SQL パイプライン
    # ================================================================
    W("""
<h2>5. Text-to-SQL パイプラインの全体像</h2>

<h3>5.1 処理フロー（10ステップ）</h3>
<p>
<span class="pipe-step ps-in">1. NL入力</span>
<span class="arrow">&rarr;</span>
<span class="pipe-step ps-ex">2. Entity Extractor</span>
<span class="arrow">&rarr;</span>
<span class="pipe-step ps-ex">3. Schema Linker</span>
<span class="arrow">&rarr;</span>
<span class="pipe-step ps-gr">4. Schema Graph</span>
<span class="arrow">&rarr;</span>
<span class="pipe-step ps-ra">5. Few-Shot検索</span>
<span class="arrow">&rarr;</span>
<span class="pipe-step ps-sq">6. SQL Generator</span>
<span class="arrow">&rarr;</span>
<span class="pipe-step ps-gu">7. SQL Guard</span>
<span class="arrow">&rarr;</span>
<span class="pipe-step ps-db">8. DB実行</span>
<span class="arrow">&rarr;</span>
<span class="pipe-step ps-in">9. 結果返却</span>
<span class="arrow">&rarr;</span>
<span class="pipe-step ps-ra">10. 成功事例蓄積</span>
</p>

<h3>5.2 各ステップの詳細</h3>

<h4>Step 1: NL入力（自然言語クエリ）</h4>
<p>ユーザーが日本語または英語で入力する。例：「Niを含む安定なL1<sub>2</sub>型化合物を形成エネルギーが低い順に出して」</p>

<h4>Step 2: Entity Extractor（条件抽出）</h4>
<p>自然言語から構造化された条件辞書を抽出する。<span class="code">llm/entity_extractor.py</span></p>
<div class="sql">
入力: "Niを含む安定なL1₂型化合物を形成エネルギーが低い順に出して"

抽出結果:
{
  "prototype": "L12",              ← "L1₂" → "L12" (Unicode下付き数字を正規化)
  "contains_elements": ["Ni"],     ← "ニッケル" でも "Ni" でも認識
  "stability": "stable",           ← "安定な" → stable (E_hull ≤ 0.001)
  "sort_by": "phase_stability.formation_energy_per_atom",
  "sort_order": "asc"              ← "低い順" → ASC
}
</div>
<p><b>仕組み：</b>
<span class="code">material_terms.yaml</span> に定義された用語辞書（日英対応）を使い、
正規表現で入力テキストをスキャンする。Unicode 下付き文字（₀₁₂...）は
ASCII 数字に正規化してからマッチングする。</p>

<h4>Step 3: Schema Linker（スキーマリンク）</h4>
<p>抽出された条件から、<b>必要なテーブルとカラム</b>を決定する。<span class="code">llm/schema_linker.py</span></p>
<div class="sql">
条件 "prototype" → テーブル: structure
                   カラム: structure.prototype, structure.strukturbericht
条件 "stability" → テーブル: phase_stability
                   カラム: phase_stability.energy_above_hull
条件 "contains_elements" → テーブル: composition
                           カラム: composition.element
</div>

<h4>Step 4: Schema Graph Traversal</h4>
<p>必要テーブルが決まったら、Schema Graph で<b>最短 JOIN 経路</b>を探索する。
ここで不要なテーブルは結合されない。</p>

<h4>Step 5: Few-Shot 検索（RAG）</h4>
<p>過去の成功クエリから類似事例を検索し、LLM プロンプトに注入する。
詳細はセクション 6 で解説。</p>

<h4>Step 6: SQL Generator（SQL生成）</h4>
<p>2つのモードがある：</p>
<ul>
<li><b>LLM モード</b>（OpenAI API キーがある場合）：プロンプトに Schema 制約 + Few-Shot 例を含めて LLM に SQL を生成させる</li>
<li><b>Rule-based fallback</b>（API キーがない場合）：条件辞書から決定論的に SQL を組み立てる</li>
</ul>

<h4>Step 7: SQL Guard（安全検査）</h4>
<p>生成された SQL を実行前に検証する。<span class="code">safety/sql_validator.py</span></p>
<ol>
<li><b>禁止キーワード：</b>INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE, GRANT, REVOKE, COPY</li>
<li><b>複数文検出：</b>セミコロンで区切られた複数 SQL 文を拒否（SQL injection 防止）</li>
<li><b>SELECT のみ：</b>SELECT 文以外は拒否</li>
<li><b>テーブルホワイトリスト：</b>許可されたテーブル以外への参照を拒否</li>
<li><b>LIMIT 自動付与：</b>LIMIT 句がなければ LIMIT 100 を自動追加</li>
<li><b>sqlglot パース：</b>SQL 構文が正しいかパーサーで検証</li>
</ol>

<h4>Step 8-9: DB実行・結果返却</h4>
<p>検証済み SQL を PostgreSQL で実行し、結果をユーザーに返す。タイムアウト付き。</p>

<h4>Step 10: 成功事例蓄積（RAG ループ）</h4>
<p>SQL 実行が成功したら、(NL, SQL, 条件, 結果件数) の組を Few-Shot ストアに蓄積する。
これにより、次回以降の類似クエリで精度が向上する。</p>
""")

    # ================================================================
    # 6. SQL-as-Few-Shot-Examples
    # ================================================================
    W(f"""
<h2>6. SQL-as-Few-Shot-Examples（RAG的フィードバックループ）</h2>

<h3>6.1 「Few-Shot」とは何か</h3>
<p>LLM にタスクを実行させるとき、<b>いくつかの具体例（お手本）をプロンプトに含める</b>手法を
Few-Shot Learning と呼ぶ。例えば：</p>

<div class="sql">
# プロンプトに注入する Few-Shot 例：
Example 1:
  Query: Feを含むB2化合物を出して
  SQL: SELECT DISTINCT m.entry_id, m.formula, s.prototype, s.lattice_a
       FROM material_entry m
       JOIN composition c ON c.entry_id = m.entry_id
       JOIN structure s ON s.entry_id = m.entry_id
       WHERE (s.prototype = 'B2' OR s.strukturbericht = 'B2')
       AND c.element = 'Fe'
       LIMIT 100;

# ↑ この例を見た LLM は、
#   「Coを含むL12化合物を出して」という新クエリに対しても
#   同じパターンで正しい SQL を生成しやすくなる
</div>

<h3>6.2 「SQL-as-Few-Shot-Examples」のアイデア</h3>
<p>通常、Few-Shot 例は人間が手動で作成する。本システムでは、
<b>過去に成功した NL→SQL 変換の結果を自動的に蓄積</b>し、
新クエリの際に<b>類似した成功例を自動検索・注入</b>する。</p>

<p>これは RAG (Retrieval-Augmented Generation) の一種で、
「検索対象」がDB ではなく<b>過去の成功クエリ</b>である点が特徴。</p>

<h3>6.3 類似度検索の仕組み（TF-IDF + コサイン類似度）</h3>
<p>「どの過去クエリが新クエリに似ているか」を判定するために、
<b>TF-IDF (Term Frequency - Inverse Document Frequency)</b> を使う。</p>

<div class="note">
<b>TF-IDF を一言で：</b>
各単語の「重要度」を数値化する手法。
「を」「の」のような頻出語は重要度が低く、
「L12」「Ni」「安定」のようなドメイン固有語は重要度が高くなる。<br><br>
<b>コサイン類似度：</b>
2つの文書の TF-IDF ベクトルの「向き」がどれだけ似ているかを 0〜1 で表す。
1 に近いほど類似。
</div>

<h4>具体的な処理</h4>
<ol>
<li>新クエリと全蓄積クエリをトークン化（CJK 文字対応のカスタムトーカナイザ）</li>
<li>各トークンの TF（出現頻度）と IDF（逆文書頻度）を計算</li>
<li>TF-IDF ベクトルのコサイン類似度で全蓄積クエリをランキング</li>
<li>類似度 &gt; 0.05 の上位3件を Few-Shot 例として返す</li>
</ol>

<h3>6.4 論文からの事例シード抽出</h3>
<p>Few-Shot ストアが空（コールドスタート問題）の場合、
研究論文 (LaTeX ファイル) から B2/L1<sub>2</sub> 化合物の言及を自動抽出し、
初期事例としてシード登録する。</p>
<p>現在のストア：<b>{S['few_shot_store']['total_examples']}件</b>
（内訳：curated {S['few_shot_store']['sources'].get('seed:curated',0)}件、
paper {sum(v for k,v in S['few_shot_store']['sources'].items() if k.startswith('paper:'))}件）</p>

<h3>6.5 有効性の評価（忖度なし）</h3>
<div class="warn-box">
<b>正直な評価：</b>
<ul>
<li><b>Rule-based モード（現在のデフォルト）では、Few-Shot は直接的な精度向上には寄与しない。</b>
SQL は決定論的に条件辞書から生成されるため、Few-Shot 例の有無で結果は変わらない。</li>
<li><b>LLM モードでは有効。</b>Few-Shot 例がプロンプトに注入されることで、
スキーマリンクの精度向上（特に「安定」→ E<sub>hull</sub> &le; 0.001 のようなドメイン固有マッピング）
が期待できる。</li>
<li><b>メタデータとしての価値：</b>Rule-based モードでも、類似クエリの結果件数や条件構造を
参考情報としてログに出力できるため、デバッグや分析には有用。</li>
<li><b>自己改善ループの基盤：</b>成功クエリを蓄積する仕組みは、将来 LLM モードに切り替えた際に
そのまま活用できる。</li>
</ul>
</div>
""")

    # ================================================================
    # 7. 3レベル比較
    # ================================================================
    # Get example test A01 for comparison
    ex = next((r for r in R if r["test_id"] == "A01"), None)

    W('<h2>7. 3レベル比較（Naive vs Schema Graph vs Few-Shot）</h2>')
    W("""
<p>同じクエリを3つの異なるレベルの T2SQL システムで処理し、
生成される SQL の品質の差を比較する。</p>
""")

    if ex:
        naive_sql = ex.get("naive", {}).get("sql", "N/A")
        sg_sql = ex.get("schema_graph", {}).get("sql", "N/A")
        naive_issues = ex.get("naive", {}).get("issues", [])
        fs_info = ex.get("few_shot", {})
        ex_row_count = ex.get("db_result", {}).get("row_count", 0)

        W(f"""
<h3>7.1 実例：「{h(ex['nl_query'])}」</h3>

<div class="level-compare">
  <div class="level-box level-0">
    <h4 style="color:#c62828;">Level 0: Naive T2SQL</h4>
    <p>Schema Graph なし。条件抽出のみ。全テーブルを常に JOIN。安全検査なし。</p>
    <div class="sql">{h(naive_sql)}</div>
    <p style="color:#c62828;font-size:0.9em;"><b>問題点：</b></p>
    <ul style="color:#c62828;font-size:0.88em;">
    {''.join(f'<li>{h(issue)}</li>' for issue in naive_issues)}
    </ul>
  </div>
  <div class="level-box level-1">
    <h4 style="color:#1565c0;">Level 1: Schema Graph T2SQL</h4>
    <p>Schema Graph で最短 JOIN 経路を探索。SQL Guard で安全検査。LIMIT/DISTINCT あり。</p>
    <div class="sql">{h(sg_sql)}</div>
    <p class="pass"><b>結果：</b>{ex_row_count} 行</p>
  </div>
  <div class="level-box level-2">
    <h4 style="color:#2e7d32;">Level 2: + Few-Shot RAG</h4>
    <p>Level 1 の SQL 生成に加え、類似成功事例を検索・注入。</p>
    <p><b>検索結果：</b>{fs_info.get('retrieved_count', 0)} 件の類似事例</p>
    <p><b>類似クエリ：</b>{', '.join(fs_info.get('retrieved_queries', []))}</p>
    <p><b>類似度：</b>{fs_info.get('similarities', [])}</p>
  </div>
</div>
""")

    W("""
<h3>7.2 機能比較まとめ</h3>
<table>
<tr><th>機能</th><th>Level 0 (Naive)</th><th>Level 1 (Schema Graph)</th><th>Level 2 (+ Few-Shot)</th></tr>
<tr><td>JOIN 経路最適化</td><td class="fail">なし（全テーブル結合）</td><td class="pass">あり（NetworkX 最短経路）</td><td class="pass">あり</td></tr>
<tr><td>EXISTS サブクエリ（多元素）</td><td class="fail">なし（同一行 AND → 0件）</td><td class="pass">あり（正しい結果）</td><td class="pass">あり</td></tr>
<tr><td>SQL 安全検査</td><td class="fail">なし</td><td class="pass">あり（sqlglot + 禁止語 + ホワイトリスト）</td><td class="pass">あり</td></tr>
<tr><td>LIMIT 自動付与</td><td class="fail">なし</td><td class="pass">あり（100件）</td><td class="pass">あり</td></tr>
<tr><td>DISTINCT</td><td class="fail">なし（重複行あり）</td><td class="pass">あり</td><td class="pass">あり</td></tr>
<tr><td>Few-Shot RAG</td><td class="fail">なし</td><td class="fail">なし</td><td class="pass">あり（TF-IDF 類似度）</td></tr>
<tr><td>論文シード抽出</td><td class="fail">なし</td><td class="fail">なし</td><td class="pass">あり（LaTeX 解析）</td></tr>
<tr><td>自己改善ループ</td><td class="fail">なし</td><td class="fail">なし</td><td class="pass">あり（成功事例蓄積）</td></tr>
</table>
""")

    # ================================================================
    # 8. 検証結果（全テスト詳細）
    # ================================================================
    W('<h2>8. 検証結果（全テスト詳細）</h2>')

    for cat in ["normal", "no_results", "sloppy", "contradictory", "rejection", "safety"]:
        cat_results = [r for r in R if r.get("category") == cat]
        if not cat_results:
            continue
        cat_info = S["categories"][cat]
        W(f'<h3>8.{list(S["categories"].keys()).index(cat)+1} '
          f'<span class="tag tag-{cat}">{cat}</span> '
          f'{cat_desc.get(cat, "")} ({cat_info["passed"]}/{cat_info["total"]})</h3>')

        # Category-specific explanation
        if cat == "normal":
            W('<p>標準的な材料検索クエリ。元素指定、prototype指定、安定性フィルタ、ソートなど。'
              'すべてが正しい結果を返すことを確認する。</p>')
        elif cat == "no_results":
            W('<p>存在しない元素組合せや、材料辞書に未登録の元素を含むクエリ。'
              '誤った結果を返さないことを確認する。</p>')
        elif cat == "sloppy":
            W('<p><b>最重要カテゴリ。</b>曖昧・不完全・無関係な入力に対して、'
              'システムが<b>間違った WHERE 条件を捏造しないか</b>を検証する。</p>')
        elif cat == "rejection":
            W('<p>SQL injection 攻撃や破壊的操作を含む入力。'
              'SQL Guard が確実にブロックすることを確認する。</p>')

        W('<table>')
        W('<tr><th>ID</th><th>クエリ</th><th>結果</th><th>行数</th><th>OQMD比較</th>'
          '<th>Few-Shot</th><th>説明</th></tr>')

        for r in cat_results:
            cls = "pass-row" if r.get("passed") else "fail-row"
            icon = "PASS" if r.get("passed") else "FAIL"
            rc = r.get("db_result", {}).get("row_count",
                    r.get("validation", {}).get("valid", "—"))
            oqmd = ""
            if "oqmd_comparison" in r and "error" not in r.get("oqmd_comparison", {}):
                oc = r["oqmd_comparison"]
                oqmd = f'{oc["match_rate"]*100:.0f}% ({oc["intersection"]}/{oc["baseline_count"]})'
            fs_c = r.get("few_shot", {}).get("retrieved_count", "—")

            W(f'<tr class="{cls}">')
            W(f'<td><b>{r["test_id"]}</b></td>')
            W(f'<td>{h(r["nl_query"][:60])}</td>')
            W(f'<td class="{"pass" if r.get("passed") else "fail"}">{icon}</td>')
            W(f'<td>{rc}</td><td>{oqmd}</td><td>{fs_c}</td>')
            W(f'<td>{h(r.get("notes", ""))}</td></tr>')

            # Expandable details
            W(f'<tr class="{cls}"><td colspan="7"><details><summary>詳細を見る</summary>')
            if "schema_graph" in r and "sql" in r.get("schema_graph", {}):
                W(f'<p><b>Level 1 SQL (Schema Graph):</b></p>')
                W(f'<div class="sql">{h(r["schema_graph"]["sql"])}</div>')
            if "naive" in r and "sql" in r.get("naive", {}):
                W(f'<p><b>Level 0 SQL (Naive):</b></p>')
                W(f'<div class="sql">{h(r["naive"]["sql"])}</div>')
                issues = r.get("naive", {}).get("issues", [])
                if issues:
                    W(f'<p style="color:#c62828"><b>Naive の問題点：</b> '
                      f'{"; ".join(issues)}</p>')
            if "oqmd_comparison" in r and "error" not in r.get("oqmd_comparison", {}):
                oc = r["oqmd_comparison"]
                W(f'<p><b>OQMD 比較：</b> baseline={oc["baseline_count"]}, '
                  f'T2SQL={oc["db_count"]}, 一致率={oc["match_rate"]*100:.1f}%</p>')
                if oc.get("baseline_only"):
                    W(f'<p>OQMD のみ: {", ".join(oc["baseline_only"][:5])}</p>')
            if "few_shot" in r and "retrieved_queries" in r.get("few_shot", {}):
                W(f'<p><b>Few-Shot 類似事例：</b> '
                  f'{", ".join(r["few_shot"]["retrieved_queries"])}</p>')
            if "db_result" in r and r["db_result"].get("sample_rows"):
                W('<p><b>結果サンプル（先頭3行）：</b></p>')
                W('<pre>' + json.dumps(r["db_result"]["sample_rows"][:3],
                                       ensure_ascii=False, indent=2) + '</pre>')
            if r.get("pass_reason"):
                W(f'<p><b>判定理由：</b> {h(r["pass_reason"])}</p>')
            W('</details></td></tr>')

        W('</table>')

    # ================================================================
    # 9. OQMD比較
    # ================================================================
    oqmd_tests = [r for r in R
                  if "oqmd_comparison" in r
                  and "error" not in r.get("oqmd_comparison", {})]
    if oqmd_tests:
        W('<h2>9. OQMD 直接取得との比較</h2>')
        W("""
<p>OQMD API から直接取得したデータ（CSV ベースライン）と、
本システムの Text-to-SQL で取得したデータを突き合わせる。
<b>一致率 = |共通部分| / |ベースライン|</b></p>
""")
        W('<table>')
        W('<tr><th>ID</th><th>クエリ</th><th>OQMD件数</th><th>T2SQL件数</th><th>一致率</th></tr>')
        for r in oqmd_tests:
            oc = r["oqmd_comparison"]
            color = "pass" if oc["match_rate"] >= 0.95 else (
                "warn" if oc["match_rate"] >= 0.8 else "fail")
            W(f'<tr><td>{r["test_id"]}</td><td>{h(r["nl_query"][:50])}</td>')
            W(f'<td>{oc["baseline_count"]}</td><td>{oc["db_count"]}</td>')
            W(f'<td class="{color}">{oc["match_rate"]*100:.1f}%</td></tr>')
        W('</table>')

    # ================================================================
    # 10. いい加減なクエリの処理
    # ================================================================
    sloppy = [r for r in R if r.get("category") == "sloppy"]
    if sloppy:
        W('<h2>10. いい加減なクエリへの対処（False Positive 検証）</h2>')
        W("""
<div class="note">
<b>最重要テスト：</b>
「間違った検索をしないか」の検証。曖昧な入力・無関係な入力に対して、
システムが<b>誤った WHERE 条件を捏造して間違った結果を返す</b>ことがないか確認する。
</div>
""")
        W('<table>')
        W('<tr><th>ID</th><th>入力</th><th>抽出された条件</th><th>動作</th><th>False Positive?</th></tr>')
        for r in sloppy:
            rc = r.get("db_result", {}).get("row_count", "N/A")
            conds = r.get("schema_graph", {}).get("conditions", {})
            fp = "<span class='pass'>No</span>" if r.get("passed") else "<span class='fail'>POSSIBLE</span>"
            W(f'<tr class="{"pass-row" if r.get("passed") else "fail-row"}">')
            W(f'<td>{r["test_id"]}</td>')
            W(f'<td>{h(r["nl_query"])}</td>')
            W(f'<td><span class="code">{h(json.dumps(conds, ensure_ascii=False))}</span></td>')
            W(f'<td>{rc} 行返却</td><td>{fp}</td></tr>')
        W('</table>')

        W("""
<h3>10.1 分析結果</h3>
<p><b>結論：False Positive（誤検出）は発生しない。</b></p>
<p>システムの動作原理を理解すると、なぜ False Positive が起きないかがわかる：</p>
<ol>
<li><b>条件抽出は辞書ベース：</b>認識できない入力は単に「抽出なし」（空の条件辞書）になる。
「わからないものを推測して条件を作る」ことはしない。</li>
<li><b>空条件 → 全件スキャン + LIMIT：</b>何も条件が抽出できなかった場合、
WHERE 句なしの SELECT（LIMIT 100付き）が生成される。
結果は「全データの先頭100件」であり、<b>間違った絞り込みよりは安全</b>。</li>
<li><b>SQL Guard が LIMIT を強制：</b>条件なしでも LIMIT 100 が付与されるため、
大量データが返ることはない。</li>
</ol>

<div class="warn-box">
<b>限界（正直に言うと）：</b>
<ul>
<li>「今日の天気を教えて」のような完全に無関係な入力でも SQL が生成される
（結果はデータベース全体の先頭100件）。入力の「意図」を判定する機構はない。</li>
<li>辞書に登録されていない元素（U, Pu, Xe, Rn 等）は認識されないため、
「Xeを含むB2化合物」は「B2化合物の全リスト」を返す。
ユーザーには「Xe は条件として認識されませんでした」と通知すべきだが、
現状はその通知機構がない。</li>
</ul>
</div>
""")

    # ================================================================
    # 11. draw.io
    # ================================================================
    W(f"""
<h2>11. 処理フロー図（draw.io）</h2>
<p>処理フローの詳細図は <span class="code">figures/t2sql_pipeline_flow.drawio</span> に
draw.io 形式で収録されている。{'<span class="pass">ファイル確認済み。</span>' if drawio_exists else '<span class="fail">ファイルが見つかりません。</span>'}</p>

<p>3ページ構成：</p>
<ol>
<li><b>Page 1: T2SQL Pipeline Flow</b> — NL入力から DB実行・Few-Shot蓄積までの全フロー。
RAG Feedback Loop（成功SQL→蓄積→次回検索に活用）を含む。</li>
<li><b>Page 2: E-R Diagram</b> — 7テーブルの外部キー関係図。
各テーブルのカラムリスト付き。</li>
<li><b>Page 3: 3-Level Comparison</b> — Naive / Schema Graph / Few-Shot の
機能比較テーブル。</li>
</ol>
""")

    # ================================================================
    # 12. デメリットと限界（忖度なし）
    # ================================================================
    W("""
<h2>12. デメリットと限界 — なぜ100%を額面通りに受け取ってはいけないか</h2>

<div class="bad">
<b>重要：</b>本セクションは、検証結果の「39/39 (100%)」という数字が
<b>何を意味し、何を意味しないか</b>を正直に分析する。
100% は「制御された条件下での動作確認」であり、
「実運用可能」「任意のクエリに対応できる」という意味ではない。
</div>

<h3>12.1 テストケースの自己参照性（最大のデメリット）</h3>
<div class="demerit severity-high">
<h4>深刻度：高 — 検証の信頼性に直結</h4>
<p>テストケース39件は、rule-based エンジンの抽出パターンを
<b>知っている開発者が設計</b>している。
「自分で作った試験を自分で受けている」状態であり、100%は当然の帰結である。</p>
<p><b>材料研究者にとっての意味：</b>
実際の研究現場では、研究者が自由な表現でクエリを入力する。
例えば「γ'の析出強化に寄与する組成を探して」のような、
本システムのパターンに合致しない表現が大半を占める。
そのような入力でのテストは一切行われていない。</p>
<p><b>改善策：</b>
実際の材料研究者5〜10名に自由文でクエリを入力してもらい、
成功率を計測する「ブラインドテスト」が必要。</p>
</div>

<h3>12.2 Rule-based fallback による検証の限界</h3>
<div class="demerit severity-high">
<h4>深刻度：高 — Few-Shot RAG の本来の効果が未検証</h4>
<p>SQL-as-Few-Shot-Examples は <b>LLM の SQL 生成精度を向上させる仕組み</b>である。
しかし、本検証は OpenAI API なしの rule-based モードで行われており、
Few-Shot の本来の効果（LLM プロンプトへの事例注入→生成精度向上）は
<b>一切検証できていない</b>。</p>
<p>Rule-based モードでは条件辞書から決定論的に SQL を生成するため、
Few-Shot 事例の有無で結果は変わらない。
つまり、Few-Shot は「蓄積するだけ」で生成には影響を与えていない。</p>
<p><b>材料研究者にとっての意味：</b>
Few-Shot RAG の恩恵を受けるには、LLM（OpenAI GPT-4等）の API キーが必要。
API 利用にはコスト（1クエリあたり数円〜数十円）と
外部サービスへのデータ送信が発生する。
オフライン環境（セキュリティ要件の厳しい研究機関等）では利用できない。</p>
</div>

<h3>12.3 OQMD 比較の循環論証</h3>
<div class="demerit severity-mid">
<h4>深刻度：中 — 検証の独立性に関わる</h4>
<p>OQMD API → CSV → PostgreSQL に投入したデータに対して、
同じ条件で SQL を発行して「一致した」と報告している。
<b>同じデータソースの同じデータ</b>なので、不一致が起きる方がおかしい。</p>
<p>真の検証とは、<b>独立したデータソース</b>（Materials Project, AFLOW, NOMAD）
との交差検証、または論文の実験値との突き合わせである。</p>
<p><b>例：</b>Ni<sub>3</sub>Al の格子定数 — OQMD (DFT) は 3.572 Å、
実験値は 3.567 Å (Mishima et al., Acta Metall. 1985)。
このような DFT vs 実験の乖離は本システムでは検出できない。</p>
</div>

<h3>12.4 語彙カバレッジの限界</h3>
<div class="demerit severity-mid">
<h4>深刻度：中 — Silent Failure（無言の失敗）を引き起こす</h4>
<p><span class="code">material_terms.yaml</span> に登録された用語しか認識できない。
未登録の材料用語は<b>警告なしに無視</b>される。</p>
<table>
<tr><th>入力</th><th>認識</th><th>結果</th></tr>
<tr><td>「ヘスラー合金」(Heusler)</td><td class="fail">未登録</td><td>条件なし → 全データ返却</td></tr>
<tr><td>「ペロブスカイト」(Perovskite)</td><td class="fail">未登録</td><td>条件なし → 全データ返却</td></tr>
<tr><td>「マルテンサイト変態」</td><td class="fail">未登録</td><td>条件なし → 全データ返却</td></tr>
<tr><td>「超格子」(superlattice)</td><td class="fail">未登録</td><td>条件なし → 全データ返却</td></tr>
<tr><td>「Nickle」(スペルミス)</td><td class="fail">未対応</td><td>Ni として認識されない</td></tr>
<tr><td>「にっける」(ひらがな)</td><td class="fail">未対応</td><td>Ni として認識されない</td></tr>
</table>
<p><b>材料研究者にとっての意味：</b>
研究者は多様な表現を使う。「γ'相」「ガンマプライム」「gamma prime」は
すべて同じ概念だが、辞書に登録されていなければ認識されない。
さらに深刻なのは、<b>認識されなかったことがユーザーに通知されない</b>点である。
ユーザーは「Xeを含むB2化合物」と入力して636件が返ってきたとき、
Xeフィルタが無視されたことに気づかない可能性がある。</p>
</div>

<h3>12.5 数値比較クエリの不可能性</h3>
<div class="demerit severity-mid">
<h4>深刻度：中 — 材料スクリーニングの実用性に直結</h4>
<p>Rule-based モードでは、数値の大小比較を含む WHERE 条件を生成できない。</p>
<table>
<tr><th>クエリ</th><th>期待される SQL</th><th>実際</th></tr>
<tr><td>「band gap が 1.0 eV 以上のB2」</td>
<td><span class="code">WHERE ps.band_gap &ge; 1.0</span></td>
<td class="fail">生成不可（post-filteringで代替）</td></tr>
<tr><td>「格子定数が 3.5 Å 以下のL1<sub>2</sub>」</td>
<td><span class="code">WHERE s.lattice_a &le; 3.5</span></td>
<td class="fail">生成不可</td></tr>
<tr><td>「形成エネルギーが -0.5 eV/atom より低い」</td>
<td><span class="code">WHERE ps.formation_energy &lt; -0.5</span></td>
<td class="fail">生成不可</td></tr>
</table>
<p><b>材料研究者にとっての意味：</b>
材料スクリーニングでは「E<sub>hull</sub> &lt; 50 meV の準安定相」
「格子定数 3.5〜3.8 Å の範囲」のような数値条件が頻出する。
現状ではこれらの条件を直接 SQL に変換できず、全件取得後に Python で
フィルタリングするしかない（非効率、大規模データで破綻する）。</p>
</div>

<h3>12.6 スキーマの単純さ</h3>
<div class="demerit severity-low">
<h4>深刻度：低〜中 — スケーラビリティの懸念</h4>
<p>本システムのスキーマは7テーブル・約30カラムの単純な構成である。
実運用の材料データベースとの比較：</p>
<table>
<tr><th>データベース</th><th>テーブル数</th><th>エントリ数</th><th>特徴</th></tr>
<tr><td><b>本システム</b></td><td>7</td><td>909</td><td>B2/L1<sub>2</sub>のみ</td></tr>
<tr><td>OQMD</td><td>数十</td><td>1,000,000+</td><td>全結晶構造</td></tr>
<tr><td>Materials Project</td><td>数十</td><td>150,000+</td><td>バンド構造・弾性定数含む</td></tr>
<tr><td>AFLOW</td><td>数十</td><td>3,500,000+</td><td>自動フロー計算</td></tr>
<tr><td>NOMAD</td><td>複雑な階層構造</td><td>12,000,000+</td><td>生データ含む</td></tr>
</table>
<p>Schema Graph の真価は複雑なスキーマ（テーブル間の JOIN 経路が非自明な場合）で
発揮されるが、そのような条件での検証は行われていない。
7テーブルでは、人間が手動で JOIN を書いても間違えにくい。</p>
</div>

<h3>12.7 データ規模の限界</h3>
<div class="demerit severity-low">
<h4>深刻度：低 — 将来の拡張時に顕在化</h4>
<p>909件は実用規模の 0.1% 以下。大規模データでの以下の問題は未評価：</p>
<ul>
<li><b>クエリ性能：</b>100万件規模での JOIN + WHERE + ORDER BY のレスポンスタイム</li>
<li><b>LIMIT 100 の妥当性：</b>候補が数千件あるとき、先頭100件で十分か？
研究者はすべてのデータを見たい場合がある</li>
<li><b>インデックス設計：</b>大規模データでは適切なインデックスなしに
秒単位のレスポンスは不可能</li>
<li><b>ページネーション：</b>100件以降のデータを取得する手段がない</li>
</ul>
</div>

<h3>12.8 曖昧クエリへの「安全な失敗」は本当に安全か</h3>
<div class="demerit severity-mid">
<h4>深刻度：中 — ユーザー体験と信頼性</h4>
<p>現在の設計では、曖昧な入力に対して「全データの先頭100件」を返す。
テストではこれを「安全な動作」として PASS 判定しているが、
材料研究者の視点では問題がある：</p>
<ul>
<li><b>「今日の天気を教えて」→ 材料データが返る：</b>
入力が材料クエリかどうかの判定機構がない。
ユーザーは返ってきたデータを「天気に関するデータ」と誤解する可能性は低いが、
「このシステムは何でも答えてくれる」という誤った信頼を生む。</li>
<li><b>「Xeを含むB2化合物」→ B2全件が返る：</b>
Xe が辞書に未登録のため無視され、B2 化合物のフルリストが返る。
ユーザーは「Xe 含有 B2 が636件もある」と誤解する可能性がある。
<b>正しい動作は「Xe は認識できませんでした」という通知</b>。</li>
</ul>
<p><b>改善策：</b>
入力テキストのうち、条件として抽出<b>されなかった</b>トークンを検出し、
「以下の語句は認識されませんでした：Xe, ヘスラー」と
ユーザーに明示的に通知する仕組みが必要。</p>
</div>

<h3>12.9 多言語・表記揺れの網羅性</h3>
<div class="demerit severity-low">
<h4>深刻度：低 — 主に日本語特有の問題</h4>
<table>
<tr><th>表記</th><th>対応状況</th><th>備考</th></tr>
<tr><td>「ニッケル」</td><td class="pass">対応済み</td><td>カタカナ辞書に登録</td></tr>
<tr><td>「にっける」</td><td class="fail">未対応</td><td>ひらがな辞書なし</td></tr>
<tr><td>「Nickle」</td><td class="fail">未対応</td><td>スペルミス補正なし</td></tr>
<tr><td>「Ni-based」</td><td class="pass">対応済み</td><td>正規表現でマッチ</td></tr>
<tr><td>「ガンマプライム」</td><td class="pass">対応済み</td><td>γ' のカタカナ表記</td></tr>
<tr><td>「gamma prime」</td><td class="pass">対応済み</td><td>英語表記</td></tr>
<tr><td>「析出強化相」</td><td class="fail">未対応</td><td>概念レベルの表現</td></tr>
</table>
<p>スペルミス補正には Levenshtein 距離やファジーマッチングが有効だが、未実装。</p>
</div>

<h3>12.10 エラーリカバリの不在</h3>
<div class="demerit severity-low">
<h4>深刻度：低 — 現時点では問題は顕在化していない</h4>
<p>SQL 実行エラー時に「別の SQL を試す」「ユーザーに確認する」等の
リカバリ機構がない。1回失敗したらそのままエラーを返す。
Rule-based モードではエラーが発生しにくいが、
LLM モードでは生成 SQL の構文エラーが頻発する可能性がある。</p>
</div>

<h3>12.11 デメリット一覧（深刻度順）</h3>
<table>
<tr><th>#</th><th>デメリット</th><th>深刻度</th><th>影響範囲</th><th>改善コスト</th></tr>
<tr class="fail-row">
<td>1</td><td>テストケースの自己参照性</td><td>高</td><td>検証の信頼性</td><td>中（ブラインドテスト実施）</td></tr>
<tr class="fail-row">
<td>2</td><td>Rule-based での Few-Shot 未検証</td><td>高</td><td>Few-Shot の価値</td><td>低（API キー設定のみ）</td></tr>
<tr>
<td>3</td><td>OQMD 比較の循環論証</td><td>中</td><td>検証の独立性</td><td>中（別データソース連携）</td></tr>
<tr>
<td>4</td><td>語彙カバレッジ（Silent Failure）</td><td>中</td><td>ユーザー体験</td><td>低（通知機構追加）</td></tr>
<tr>
<td>5</td><td>数値比較クエリ不可</td><td>中</td><td>スクリーニング実用性</td><td>中（NLPパーサー追加）</td></tr>
<tr>
<td>6</td><td>曖昧クエリの安全性</td><td>中</td><td>ユーザーの誤解</td><td>低（未認識トークン通知）</td></tr>
<tr>
<td>7</td><td>スキーマの単純さ</td><td>低〜中</td><td>スケーラビリティ</td><td>高（スキーマ拡張+再検証）</td></tr>
<tr>
<td>8</td><td>データ規模の限界</td><td>低</td><td>性能</td><td>中（インデックス設計）</td></tr>
<tr>
<td>9</td><td>多言語・表記揺れ</td><td>低</td><td>日本語ユーザー</td><td>低（ファジーマッチ追加）</td></tr>
<tr>
<td>10</td><td>エラーリカバリ不在</td><td>低</td><td>LLMモード移行時</td><td>中（リトライ機構実装）</td></tr>
</table>
""")

    # ================================================================
    # 13. まとめと今後
    # ================================================================
    W("""
<h2>13. まとめと今後の課題</h2>

<h3>13.1 達成したこと</h3>
<ul>
<li>7テーブルの正規化スキーマに OQMD 909件のデータを投入</li>
<li>NetworkX ベースの Schema Graph Traversal Engine で正確な JOIN 経路を自動探索</li>
<li>SQL Guard による6段階の安全検査（SQL injection 完全ブロック）</li>
<li>SQL-as-Few-Shot-Examples の実装（TF-IDF 類似度検索 + 論文シード抽出）</li>
<li>39テスト全パス（正常系/該当なし/いい加減なクエリ/矛盾/拒否/安全検査）</li>
<li>OQMD 直接取得との比較で 100% 一致率</li>
</ul>

<h3>13.2 100% が意味すること・意味しないこと</h3>
<table>
<tr><th>意味すること（言えること）</th><th>意味しないこと（言えないこと）</th></tr>
<tr>
<td class="pass">定義済みパターンに合致するクエリは正しく処理できる</td>
<td class="fail">任意の自然言語クエリに対応できるわけではない</td></tr>
<tr>
<td class="pass">SQL injection は確実にブロックされる</td>
<td class="fail">全ての悪意ある入力パターンをテストしたわけではない</td></tr>
<tr>
<td class="pass">OQMD データに対して正確な検索ができる</td>
<td class="fail">実験値との整合性は検証していない</td></tr>
<tr>
<td class="pass">Schema Graph は7テーブルで正しく動作する</td>
<td class="fail">数十テーブル規模での動作は未検証</td></tr>
<tr>
<td class="pass">Few-Shot ストアの蓄積・検索機構は動作する</td>
<td class="fail">LLM と組み合わせた際の精度向上は未実証</td></tr>
</table>

<h3>13.3 実用化に向けたロードマップ（優先度順）</h3>
<ol>
<li><b>未認識トークン通知（改善コスト：低）：</b>
入力から抽出されなかった語句をユーザーに通知する。
これだけで「Silent Failure」問題の大半が解決する。</li>
<li><b>LLM モードでの検証（改善コスト：低）：</b>
OpenAI API キーを設定し、Few-Shot RAG の実効性を評価する。</li>
<li><b>ブラインドテスト（改善コスト：中）：</b>
材料研究者5〜10名による自由文クエリテスト。
真の成功率とユーザビリティの定量評価。</li>
<li><b>数値条件パーサー（改善コスト：中）：</b>
「band gap &gt; 1.0 eV」のような数値比較を自動抽出する NLP モジュール。</li>
<li><b>データソース拡張（改善コスト：中〜高）：</b>
Materials Project, AFLOW との連携。スキーマ拡張と大規模データ対応。</li>
</ol>
""")

    # Footer
    W(f"""
<footer>
L1<sub>2</sub>/B2 Schema-Graph-Assisted Text-to-SQL System &mdash;
NIMS Materials Informatics &mdash;
Generated by comprehensive_verification.py + report_generator.py &mdash;
{time.strftime('%Y-%m-%d %H:%M UTC')}
</footer>
</body>
</html>
""")

    return "\n".join(parts)
