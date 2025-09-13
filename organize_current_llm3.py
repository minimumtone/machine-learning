#!/usr/bin/env python3
"""
プログラム・データファイル自動カタログ化ツール (改良版)

カレントディレクトリ内の Python/R/Notebook/CSV/テキストファイルを OpenAI API で解析し、
適切なプロジェクト構造に整理してドキュメント化します。

改良点:
- セキュリティ: 環境変数によるAPI キー管理
- ファイル対応: CSV データファイルとテキストファイルのサポート
- 専用分析: ファイル種別に応じた適切な LLM プロンプト
"""

import argparse
import datetime as dt
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple

import chardet
from openai import OpenAI

# ============================
# ============================

README_TPL = """# {title}

**作成日**: {date_ymd}  
**言語**: {summary}  
**LLM解析**: {llm_model}

## 概要

{summary}


{env_setup}


{run_howto}


{tree}


{deps}
"""

USAGE_TPL = """# 使用方法

{usage_body}
"""

HOW_IT_WORKS_TPL = """# 実装解説

{how_body}
"""


# ============================
# ============================

def read_text_auto(path: Path) -> str:
    try:
        with open(path, "rb") as f:
            raw = f.read()
        detected = chardet.detect(raw)
        encoding = detected.get("encoding", "utf-8")
        if encoding is None:
            encoding = "utf-8"
        return raw.decode(encoding, errors="replace")
    except Exception:
        return path.read_text(encoding="utf-8", errors="replace")


def ensure_unique_dir(base_path: Path) -> Path:
    if not base_path.exists():
        return base_path
    counter = 1
    while True:
        new_path = Path(f"{base_path}_{counter:02d}")
        if not new_path.exists():
            return new_path
        counter += 1


def normalize_slug(text: str) -> str:
    import re
    text = re.sub(r"[^\w\s\-_]", "", text)
    text = re.sub(r"[\s\-_]+", "_", text)
    return text.strip("_")[:32]


def created_datetime(path: Path) -> dt.datetime:
    try:
        ts = path.stat().st_ctime
    except Exception:
        ts = time.time()
    return dt.datetime.fromtimestamp(ts)


def detect_language(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".py":
        return "python"
    if ext in (".r", ".R", ".rscript"):
        return "r"
    if ext == ".ipynb":
        return "notebook"
    if ext in (".rmd", ".Rmd"):
        return "rmd"
    if ext == ".csv":
        return "csv"
    if ext == ".txt":
        return "text"
    return "unknown"


def scan_sources(root: Path, include_ext: Tuple[str, ...], ignore_dirs: Tuple[str, ...]) -> List[Path]:
    files: List[Path] = []
    for p in root.rglob("*"):
        if any(ignored in p.parts for ignored in ignore_dirs):
            continue
        if p.is_file() and p.suffix.lower() in include_ext:
            files.append(p)
    return files


# ============================
# ============================

class LLMClient:
    def __init__(self, model: str, timeout: int = 120):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is required")
        self.client = OpenAI(api_key=self.api_key, timeout=timeout)
        self.model = model

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.0, max_tokens: int = 4096) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {e}")


def extract_json(text: str) -> Dict:
    import re
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise ValueError(f"Failed to extract JSON from LLM response: {text[:200]}...")


def smart_sample_code(code: str, max_chars: int = 80000) -> str:
    if len(code) <= max_chars:
        return code
    return code[:max_chars] + "\n\n# ... (truncated)"


# ============================
# ============================

def analyze_csv_file(path: Path) -> tuple[str, dict]:
    try:
        import pandas as pd
        file_size = path.stat().st_size
        size_mb = file_size / (1024 * 1024)
        
        if size_mb > 10:
            df = pd.read_csv(path, nrows=100)
        else:
            df = pd.read_csv(path)
        
        info = {
            'rows': len(df) if size_mb <= 10 else f"約{len(df)}行（サンプル）",
            'columns': len(df.columns),
            'size': f"{size_mb:.1f}MB"
        }
        
        sample = df.head(10).to_csv(index=False)
        
        return sample, info
    except Exception as e:
        return f"# CSV読み込みエラー: {e}", {'rows': 'N/A', 'columns': 'N/A', 'size': 'N/A'}


def analyze_text_file(path: Path, max_chars: int = 5000) -> tuple[str, dict]:
    try:
        content = read_text_auto(path)
        lines = content.split('\n')
        
        info = {
            'chars': len(content),
            'lines': len(lines)
        }
        
        if len(content) > max_chars:
            sample = content[:max_chars] + "\n\n[... 以下省略 ...]"
        else:
            sample = content
            
        return sample, info
    except Exception as e:
        return f"# テキスト読み込みエラー: {e}", {'chars': 'N/A', 'lines': 'N/A'}


# ============================
# ============================

def make_tree_text(project_root: Path) -> str:
    lines = []
    for p in sorted(project_root.rglob("*")):
        if p.is_file():
            rel = p.relative_to(project_root)
            lines.append(f"- {rel}")
    return "\n".join(lines) if lines else "（ファイル構成を取得できませんでした）"


def deps_text_block(py_deps: Set[str], r_deps: Set[str]) -> str:
    lines = []
    if py_deps:
        lines.append("### Python")
        lines.extend(f"- {dep}" for dep in sorted(py_deps))
    if r_deps:
        lines.append("### R")
        lines.extend(f"- {dep}" for dep in sorted(r_deps))
    return "\n".join(lines) if lines else "依存関係なし"


def env_setup_block(has_python: bool, has_r: bool) -> str:
    lines = []
    if has_python:
        lines.extend([
            "### Python環境",
            "```bash",
            "pip install -r env/requirements.txt",
            "```"
        ])
    if has_r:
        lines.extend([
            "### R環境",
            "```r",
            "source('env/install_packages.R')",
            "```"
        ])
    return "\n".join(lines) if lines else "特別な環境構築は不要です。"


def usage_text_block(language: str, filename: str, usage_ja: str) -> str:
    if language == "python":
        base = f"```bash\npython {filename}\n```"
    elif language in ("r", "rmd"):
        base = f"```r\nsource('{filename}')\n```"
    elif language == "notebook":
        base = f"Jupyter Notebook で `{filename}` を開いて実行してください。"
    elif language == "csv":
        base = f"```python\nimport pandas as pd\ndf = pd.read_csv('{filename}')\n```"
    elif language == "text":
        base = f"テキストエディタで `{filename}` を参照してください。"
    else:
        base = f"ファイル `{filename}` を適切なツールで開いてください。"
    
    if usage_ja:
        return f"{base}\n\n{usage_ja}"
    return base


# ============================
# LLM 解析プロンプト
# ============================

LLM_SYSTEM = (
    "あなたは熟練のソフトウェアアーキテクト兼テクニカルライターです。"
    "ユーザーの試作コードを素早く把握し、目的に沿った簡潔で正確な日本語ドキュメントを作成してください。"
    "出力は必ずスキーマに厳密に従ったJSONオブジェクト形式のみで返してください。"
)

def llm_prompt_for_code(path: Path, language: str, code_excerpt: str) -> List[Dict[str, str]]:
    user_content = f"""
# 指示
以下のスクリプト（{language}）を読み、**JSONオブジェクトのみ**で返答してください。次のスキーマに厳密に従ってください。

- title_ja: 60文字以内のタイトル
- short_slug: 32文字以内。ディレクトリ名に使える短いスラッグ（日本語可、スペースは_）
- summary_ja: 2〜6文で、何をするコードか、前提、入出力、想定ユースケースを要約
- usage_ja: 使い方（コマンド例や引数説明）。必要ならコードブロックを使用
- deps_python: Pythonの外部依存パッケージ名配列（なければ[]）
- deps_r: Rの外部依存パッケージ名配列（なければ[]）
- cli_options: コマンドライン引数の推定一覧（例: [{{"flags": "-i, --input", "help": "入力CSV", "type":"str", "default": ""}}]）
- how_it_works_ja: 実装の読み方（主要関数/クラス/処理フローを箇条書きで）
- risk_notes_ja: 注意点（データ前提、精度、副作用、セキュリティ等）。なければ「特になし」
- tags_ja: 3〜8個のタグ（"FFT","可視化","機械学習" 等）

**JSON以外のテキストは一切出力しないでください。**

# ファイル
- path: {path.name}

# コード（抜粋）
```{language}
{code_excerpt}
```
"""
    return [
        {"role": "system", "content": LLM_SYSTEM},
        {"role": "user", "content": user_content.strip()},
    ]


def llm_prompt_for_csv(path: Path, csv_sample: str, csv_info: dict) -> List[Dict[str, str]]:
    user_content = f"""
# 指示
以下のCSVデータファイルを分析し、**JSONオブジェクトのみ**で返答してください。次のスキーマに厳密に従ってください。

- title_ja: 60文字以内のタイトル
- short_slug: 32文字以内。ディレクトリ名に使える短いスラッグ（日本語可、スペースは_）
- summary_ja: データの内容、形式、用途を2〜6文で要約
- usage_ja: データの使い方（読み込み方法、主要な列の説明）
- data_schema: 列名と型の説明（例: [{{"column": "time", "type": "float", "description": "時間[秒]"}}]）
- data_stats: データの統計情報（行数、列数、データ範囲など）
- how_it_works_ja: データの構造と内容の説明
- risk_notes_ja: データ使用時の注意点（精度、前提条件等）。なければ「特になし」
- tags_ja: 3〜8個のタグ（"時系列データ","シミュレーション","科学計算" 等）

**JSON以外のテキストは一切出力しないでください。**

- path: {path.name}
- 行数: {csv_info.get('rows', 'N/A')}
- 列数: {csv_info.get('columns', 'N/A')}
- ファイルサイズ: {csv_info.get('size', 'N/A')}

```csv
{csv_sample}
```
"""
    return [
        {"role": "system", "content": "あなたは熟練のデータサイエンティスト兼テクニカルライターです。CSVデータファイルを分析し、その内容と用途を正確に把握して日本語ドキュメントを作成してください。出力は必ずスキーマに厳密に従ったJSONオブジェクト形式のみで返してください。"},
        {"role": "user", "content": user_content.strip()},
    ]


def llm_prompt_for_text(path: Path, text_sample: str, text_info: dict) -> List[Dict[str, str]]:
    user_content = f"""
# 指示
以下のテキストファイルを分析し、**JSONオブジェクトのみ**で返答してください。次のスキーマに厳密に従ってください。

- title_ja: 60文字以内のタイトル
- short_slug: 32文字以内。ディレクトリ名に使える短いスラッグ（日本語可、スペースは_）
- summary_ja: 文書の内容、目的、対象読者を2〜6文で要約
- usage_ja: 文書の使い方（参照方法、関連する作業など）
- content_type: 文書の種類（"技術文書","説明書","データソース","学術資料" 等）
- key_topics: 主要なトピックや概念のリスト
- how_it_works_ja: 文書の構成と主要な内容の説明
- risk_notes_ja: 文書使用時の注意点（情報の正確性、更新日等）。なければ「特になし」
- tags_ja: 3〜8個のタグ（"材料科学","合金","技術解説" 等）

**JSON以外のテキストは一切出力しないでください。**

- path: {path.name}
- 文字数: {text_info.get('chars', 'N/A')}
- 行数: {text_info.get('lines', 'N/A')}

```
{text_sample}
```
"""
    return [
        {"role": "system", "content": "あなたは熟練のテクニカルライター兼情報アーキテクトです。テキスト文書を分析し、その内容と用途を正確に把握して日本語ドキュメントを作成してください。出力は必ずスキーマに厳密に従ったJSONオブジェクト形式のみで返してください。"},
        {"role": "user", "content": user_content.strip()},
    ]


# ============================
# 結果コンテナ
# ============================

@dataclass
class LLMResult:
    title_ja: str = ""
    short_slug: str = ""
    summary_ja: str = ""
    usage_ja: str = ""
    deps_python: List[str] = field(default_factory=list)
    deps_r: List[str] = field(default_factory=list)
    cli_options: List[Dict[str, str]] = field(default_factory=list)
    how_it_works_ja: str = ""
    risk_notes_ja: str = ""
    tags_ja: List[str] = field(default_factory=list)
    data_schema: List[Dict[str, str]] = field(default_factory=list)
    data_stats: str = ""
    content_type: str = ""
    key_topics: List[str] = field(default_factory=list)


@dataclass
class ProjectMeta:
    title: str
    language: str
    created: str
    source_file: str
    slug: str
    summary: str
    python_deps: List[str]
    r_deps: List[str]
    argparse: List[Dict[str, str]]
    llm_model: str


# ============================
# LLM 呼び出し
# ============================

def call_llm_for_file(llm: LLMClient, path: Path, language: str, content: str, max_chars: int) -> LLMResult:
    if language == "csv":
        sample, info = analyze_csv_file(path)
        msgs = llm_prompt_for_csv(path, sample, info)
    elif language == "text":
        sample, info = analyze_text_file(path, max_chars)
        msgs = llm_prompt_for_text(path, sample, info)
    else:
        excerpt = smart_sample_code(content, max_chars=max_chars)
        msgs = llm_prompt_for_code(path, language, excerpt)
    
    content = llm.chat(msgs, temperature=0.15, max_tokens=4096)
    data = extract_json(content)

    def get(d: Dict, key: str, default):
        v = d.get(key, default)
        return v if v is not None else default

    def get_str(d: Dict, key: str, default: str = "") -> str:
        v = d.get(key, default)
        if v is None:
            return default
        if isinstance(v, list):
            return "\n".join(str(item) for item in v)
        return str(v)

    def get_list(d: Dict, key: str, default: list = None) -> list:
        if default is None:
            default = []
        v = d.get(key, default)
        if v is None:
            return default
        if isinstance(v, str):
            return [v] if v else []
        if isinstance(v, list):
            return v
        return [str(v)]

    return LLMResult(
        title_ja=get_str(data, "title_ja", ""),
        short_slug=normalize_slug(get_str(data, "short_slug", "") or get_str(data, "title_ja", "")),
        summary_ja=get_str(data, "summary_ja", ""),
        usage_ja=get_str(data, "usage_ja", ""),
        deps_python=list(dict.fromkeys(get_list(data, "deps_python", []))),
        deps_r=list(dict.fromkeys(get_list(data, "deps_r", []))),
        cli_options=get(data, "cli_options", []) or [],
        how_it_works_ja=get_str(data, "how_it_works_ja", ""),
        risk_notes_ja=get_str(data, "risk_notes_ja", ""),
        tags_ja=list(dict.fromkeys(get_list(data, "tags_ja", []))),
        data_schema=get(data, "data_schema", []) or [],
        data_stats=get_str(data, "data_stats", ""),
        content_type=get_str(data, "content_type", ""),
        key_topics=list(dict.fromkeys(get_list(data, "key_topics", []))),
    )


# ============================
# 出力（env / docs など）
# ============================

def write_requirements(env_dir: Path, deps: Set[str]) -> None:
    env_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# Auto-generated by LLM. Please review and edit as needed."]
    lines.extend(sorted(d for d in deps if d))
    (env_dir / "requirements.txt").write_text("\n".join(lines), encoding="utf-8")


def write_install_r(env_dir: Path, pkgs: Set[str]) -> None:
    env_dir.mkdir(parents=True, exist_ok=True)
    vec = ", ".join(f'"{p}"' for p in sorted(p for p in pkgs if p))
    lines = [
        "# Auto-generated by LLM. Please review and edit as needed.",
        'options(repos = c(CRAN = "https://cloud.r-project.org"))',
        f"pkgs_to_install <- c({vec})",
        "if (length(pkgs_to_install) > 0) {",
        "  install.packages(pkgs_to_install)",
        "} else {",
        '  message("No R packages detected by LLM.")',
        "}",
    ]
    (env_dir / "install_packages.R").write_text("\n".join(lines), encoding="utf-8")


def make_docs(project_root: Path, meta: ProjectMeta, env_text: str, run_text: str, deps_text: str, how_body: str) -> None:
    docs_dir = project_root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    readme_content = README_TPL.format(
        title=meta.title,
        date_ymd=meta.created.split("T")[0],
        summary=meta.summary or "（LLMによる要約）",
        env_setup=env_text,
        run_howto=run_text,
        tree=make_tree_text(project_root),
        deps=deps_text,
        llm_model=meta.llm_model,
    )
    (docs_dir / "README.md").write_text(readme_content, encoding="utf-8")
    (docs_dir / "USAGE.md").write_text(USAGE_TPL.format(usage_body=run_text), encoding="utf-8")
    (docs_dir / "HOW_IT_WORKS.md").write_text(HOW_IT_WORKS_TPL.format(how_body=how_body), encoding="utf-8")


def organize_one_file(file_path: Path, out_root: Path, move: bool, llm: LLMClient, max_chars: int) -> ProjectMeta:
    lang = detect_language(file_path)
    if lang == "unknown":
        raise ValueError(f"{file_path.name} は対応していないファイル形式です。")

    created_dt = created_datetime(file_path)
    date_prefix = created_dt.strftime("%Y%m%d")
    
    if lang in ("csv", "text"):
        code_raw = ""
    else:
        code_raw = read_text_auto(file_path)

    start = time.time()
    llm_res = call_llm_for_file(llm, file_path, lang, code_raw, max_chars=max_chars)
    elapsed = time.time() - start
    print(f"[info] LLM 解析 OK: {file_path.name} ({elapsed:.1f}s)")

    title = llm_res.title_ja or file_path.stem
    slug = llm_res.short_slug or normalize_slug(title)
    summary = llm_res.summary_ja
    py_deps = set(llm_res.deps_python or [])
    r_deps = set(llm_res.deps_r or [])
    usage_ja = llm_res.usage_ja
    how_ja = llm_res.how_it_works_ja or ""
    cli_opts = llm_res.cli_options or []
    risk_notes = llm_res.risk_notes_ja or ""

    lang_key = {"python": "python", "r": "r", "rmd": "r", "notebook": "nb", "csv": "data", "text": "docs"}.get(lang, "misc")
    proj_dir = ensure_unique_dir(out_root / f"{date_prefix}-{lang_key}-{slug}")
    proj_dir.mkdir(parents=True, exist_ok=True)

    if lang == "notebook":
        subdir = "notebooks"
    elif lang in ("csv", "text"):
        subdir = "data" if lang == "csv" else "docs"
    else:
        subdir = "src"
    
    out_src_dir = proj_dir / subdir
    out_src_dir.mkdir(exist_ok=True)
    destination_path = out_src_dir / file_path.name
    if move:
        shutil.move(str(file_path), str(destination_path))
    else:
        shutil.copy2(file_path, destination_path)

    env_dir = proj_dir / "env"
    if py_deps:
        write_requirements(env_dir, py_deps)
    if r_deps:
        write_install_r(env_dir, r_deps)

    env_text = env_setup_block(bool(py_deps), bool(r_deps))
    run_text = usage_text_block(lang, file_path.name, usage_ja)
    deps_text = deps_text_block(py_deps, r_deps)
    how_text = (how_ja + (f"\n\n## 注意点\n{risk_notes}" if risk_notes and risk_notes != "特になし" else "")).strip()

    meta = ProjectMeta(
        title=title,
        language=lang,
        created=created_dt.isoformat(),
        source_file=file_path.name,
        slug=slug,
        summary=summary,
        python_deps=sorted(py_deps),
        r_deps=sorted(r_deps),
        argparse=cli_opts,
        llm_model=llm.model,
    )

    make_docs(proj_dir, meta, env_text, run_text, deps_text, how_text)

    (proj_dir / "project.json").write_text(json.dumps(asdict(meta), ensure_ascii=False, indent=2), encoding="utf-8")

    return meta


# ============================
# CLI エントリポイント
# ============================

def main():
    parser = argparse.ArgumentParser(description="カレントディレクトリの試作スクリプト・データファイルを OpenAI API で解析・整理してドキュメント化します。")
    parser.add_argument("--out", default="organized", help="出力先ルートディレクトリ（既定: ./organized）")
    parser.add_argument("--move", action="store_true", help="元ファイルを移動します（既定はコピー）")
    parser.add_argument("--ext", default=".py,.r,.R,.ipynb,.rmd,.Rmd,.csv,.txt", help="対象の拡張子（カンマ区切り）")
    parser.add_argument("--ignore-dirs", default=".git,.venv,venv,env,__pycache__,organized,node_modules,build,dist,.ipynb_checkpoints",
                        help="無視するディレクトリ名（カンマ区切り）")
    parser.add_argument("--max-chars", type=int, default=80000, help="LLMへ送信するコードの最大文字数（既定: 80000）")
    parser.add_argument("--model", default="gpt-4o-mini", help="使用するOpenAIモデル名（既定: gpt-4o-mini）")
    parser.add_argument("--timeout", type=int, default=120, help="OpenAI APIのタイムアウト秒数（既定: 120）")
    parser.add_argument("--dry-run", action="store_true", help="実際のファイル操作は行わず、対象ファイルのみ表示します")

    args = parser.parse_args()

    cwd = Path(".").resolve()
    out_root = cwd / args.out

    include_ext = tuple(e.strip().lower() for e in args.ext.split(","))
    ignore_dirs = tuple(d.strip() for d in args.ignore_dirs.split(","))
    ignore_dirs_with_out = ignore_dirs + (args.out,)

    try:
        self_path = Path(__file__).resolve()
    except NameError:
        self_path = None

    all_files = scan_sources(cwd, include_ext, ignore_dirs_with_out)
    candidates = [f for f in all_files if not self_path or f.resolve() != self_path]

    if not candidates:
        print("対象ファイルが見つかりませんでした。")
        return

    print(f"検出された対象ファイル数: {len(candidates)}")
    if args.dry_run:
        print("--- 対象ファイル一覧 (dry-run) ---")
        for f in candidates:
            print(f"- {f.relative_to(cwd)} [{detect_language(f)}]")
        print("------------------------------------")
        return

    out_root.mkdir(parents=True, exist_ok=True)

    try:
        llm = LLMClient(model=args.model, timeout=args.timeout)
    except RuntimeError as e:
        print(f"エラー: {e}")
        return

    metas: List[ProjectMeta] = []
    print(candidates)
    for f in candidates:
        try:
            meta = organize_one_file(f, out_root, move=args.move, llm=llm, max_chars=args.max_chars)
            metas.append(meta)
            print(f"[ok] {f.name} -> '{meta.slug}' プロジェクトとして整理しました。")
        except Exception as e:
            print(f"[error] {f.name} の処理中にエラー: {e}")

    if metas:
        summary_md_path = out_root / "SUMMARY.md"
        summary_lines = ["# 整理サマリ", f"生成日時: {dt.datetime.now().isoformat()}", ""]
        for m in sorted(metas, key=lambda x: x.created):
            created_date = dt.datetime.fromisoformat(m.created).strftime("%Y-%m-%d")
            summary_lines.append(f"- **[{m.slug}](./{m.slug})** ({created_date}, {m.language}) - {m.title}")
        summary_md_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"\n完了: {out_root.relative_to(cwd)} に {len(metas)} 個のプロジェクトを生成しました。")


if __name__ == "__main__":
    main()
