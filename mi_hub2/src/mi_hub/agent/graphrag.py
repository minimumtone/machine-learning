"""GraphRAG ナレッジプロバイダ — 論文から知識グラフを構築し新規仮説候補を生成する。

- 材料工学向けトークナイザ: MeCab（fugashi）＋材料用語ユーザ辞書（最長一致で連結）。
  fugashi が無い環境では正規表現フォールバック。
- 知識グラフ: 文単位の共起からエンティティ間エッジを張り、出典（doc_id）を保持。
- 新規仮説候補: 直接エッジは無いが共通の隣接概念を持つエンティティ対を「橋渡し」
  として提示する（採否の科学的判断は研究者が行う）。
- 利用ログ: 検索・仮説生成を JSONL に記録し、ログ中の未知語をユーザ辞書へ
  取り込む更新（update_from_logs）を可能にする。
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from .tools import KnowledgeProvider

try:
    import fugashi

    _TAGGER: Any = fugashi.Tagger()
except ImportError:  # 実行環境に MeCab が無い場合は正規表現で代替
    _TAGGER = None

# 材料工学の初期ユーザ辞書（利用ログから追記される）
DEFAULT_MATERIALS_TERMS = [
    "高エントロピー合金", "相安定性", "中距離秩序", "短距離秩序", "格子定数",
    "混合エンタルピー", "混合エントロピー", "原子半径差", "生成エンタルピー",
    "欠陥形成エネルギー", "アンチサイト", "クリープ寿命", "相境界", "状態図",
    "第一原理計算", "擬ポテンシャル", "交換相関汎関数", "スーパーセル",
    "HEA", "VEC", "SQS", "DFT", "CALPHAD", "MLIP", "B2", "L12", "FCC", "BCC",
    "Al-Mn-Al", "Ni-Al", "Hume-Rothery",
]

_TERM_RE = re.compile(r"[A-Za-z][A-Za-z0-9$_\-]+|[\u4e00-\u9fff\u30a0-\u30ff]{2,}")


class MaterialsTokenizer:
    """MeCab＋材料用語辞書のトークナイザ。辞書語は最長一致で1トークンに連結する。"""

    def __init__(self, user_terms: list[str] | None = None):
        self.user_terms = sorted(
            set(user_terms or DEFAULT_MATERIALS_TERMS), key=len, reverse=True
        )

    def add_terms(self, terms: list[str]) -> None:
        self.user_terms = sorted(
            set(self.user_terms) | set(terms), key=len, reverse=True
        )

    def tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        rest = text
        while rest:
            matched = None
            for term in self.user_terms:
                if rest.startswith(term):
                    matched = term
                    break
            if matched:
                tokens.append(matched)
                rest = rest[len(matched):]
            else:
                tokens.append(rest[0])
                rest = rest[1:]
        # 辞書語以外の残り文字列を形態素解析（または正規表現）で分割し直す
        out: list[str] = []
        buf = ""
        for t in tokens:
            if t in self.user_terms:
                if buf:
                    out.extend(self._segment(buf))
                    buf = ""
                out.append(t)
            else:
                buf += t
        if buf:
            out.extend(self._segment(buf))
        return [t for t in out if t.strip()]

    @staticmethod
    def _segment(text: str) -> list[str]:
        if _TAGGER is not None:
            return [w.surface for w in _TAGGER(text)]
        return _TERM_RE.findall(text)

    def extract_entities(self, text: str) -> list[str]:
        """辞書語＋名詞連続をエンティティとして抽出する。"""
        entities: list[str] = []
        for term in self.user_terms:
            if term in text:
                entities.append(term)
        if _TAGGER is not None:
            noun_run: list[str] = []
            for w in _TAGGER(text):
                pos = w.feature.pos1 if hasattr(w.feature, "pos1") else ""
                if pos == "名詞":
                    noun_run.append(w.surface)
                else:
                    if len(noun_run) >= 2:
                        entities.append("".join(noun_run))
                    noun_run = []
            if len(noun_run) >= 2:
                entities.append("".join(noun_run))
        else:
            entities.extend(_TERM_RE.findall(text))
        seen: set[str] = set()
        uniq = []
        for e in entities:
            if e not in seen and len(e) >= 2:
                seen.add(e)
                uniq.append(e)
        return uniq


class GraphRAGProvider(KnowledgeProvider):
    """論文群から構築した知識グラフを検索し、新規仮説候補も返すプロバイダ。"""

    def __init__(self, data_dir: str, name: str = "graphrag"):
        super().__init__(name)
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.graph_path = os.path.join(data_dir, "graph.json")
        self.log_path = os.path.join(data_dir, "usage_log.jsonl")
        self.dict_path = os.path.join(data_dir, "user_terms.json")
        terms = None
        if os.path.exists(self.dict_path):
            with open(self.dict_path, encoding="utf-8") as f:
                terms = json.load(f)
        self.tokenizer = MaterialsTokenizer(terms)
        self.docs: dict[str, dict[str, Any]] = {}
        self.edges: dict[str, dict[str, list[str]]] = {}  # ent -> ent -> [doc_id]
        if os.path.exists(self.graph_path):
            with open(self.graph_path, encoding="utf-8") as f:
                data = json.load(f)
            self.docs = data.get("docs", {})
            self.edges = data.get("edges", {})

    # ---------- 構築 ----------
    def add_document(self, doc_id: str, title: str, text: str,
                     source: str = "") -> list[str]:
        """論文（要旨等）を取り込み、文単位の共起でグラフへ追加する。"""
        self.docs[doc_id] = {"title": title, "text": text, "source": source}
        all_entities: set[str] = set()
        for sentence in re.split(r"[。.\n]", text):
            ents = self.tokenizer.extract_entities(sentence)
            all_entities.update(ents)
            for i, a in enumerate(ents):
                for b in ents[i + 1:]:
                    self._add_edge(a, b, doc_id)
        self._save_graph()
        return sorted(all_entities)

    def _add_edge(self, a: str, b: str, doc_id: str) -> None:
        for x, y in ((a, b), (b, a)):
            docs = self.edges.setdefault(x, {}).setdefault(y, [])
            if doc_id not in docs:
                docs.append(doc_id)

    def _save_graph(self) -> None:
        with open(self.graph_path, "w", encoding="utf-8") as f:
            json.dump({"docs": self.docs, "edges": self.edges}, f,
                      ensure_ascii=False, indent=1)

    # ---------- 検索（KnowledgeProvider インターフェイス） ----------
    def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        q_ents = [e for e in self.tokenizer.extract_entities(query)
                  if e in self.edges]
        hits: dict[str, list[str]] = {}
        for e in q_ents:
            for doc_ids in self.edges.get(e, {}).values():
                for d in doc_ids:
                    hits.setdefault(d, []).append(e)
        ranked = sorted(hits.items(), key=lambda kv: -len(set(kv[1])))
        results = []
        for doc_id, matched in ranked[:limit]:
            doc = self.docs.get(doc_id, {})
            results.append({
                "title": doc.get("title", doc_id),
                "claim": doc.get("text", "")[:300],
                "evidence_type": "literature",
                "keywords": sorted(set(matched)),
                "limitations": ["GraphRAG検索結果。原典の確認が必要"],
                "source": doc.get("source", ""),
                "doc_id": doc_id,
            })
        self._log("search", query=query, matched_entities=q_ents,
                  n_results=len(results))
        return results

    # ---------- 新規仮説候補 ----------
    def propose_hypotheses(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """直接エッジが無いが共通概念を介して繋がるエンティティ対を仮説候補にする。"""
        q_ents = [e for e in self.tokenizer.extract_entities(query)
                  if e in self.edges]
        proposals: list[dict[str, Any]] = []
        seen_pairs: set[tuple[str, str]] = set()
        for a in q_ents:
            for bridge, docs_ab in self.edges.get(a, {}).items():
                for c, docs_bc in self.edges.get(bridge, {}).items():
                    if c == a or c in self.edges.get(a, {}):
                        continue  # 既知の直接関係は新規性なし
                    pair = tuple(sorted((a, c)))
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    docs = sorted(set(docs_ab) | set(docs_bc))
                    proposals.append({
                        "statement": (
                            f"「{a}」と「{c}」は「{bridge}」を介して関連する可能性がある"
                            "（両者を直接結ぶ報告は取込済み文献に無い）"
                        ),
                        "entities": [a, bridge, c],
                        "supporting_docs": [
                            {"doc_id": d,
                             "title": self.docs.get(d, {}).get("title", d)}
                            for d in docs
                        ],
                        "note": "新規仮説の候補。採否・検証設計は研究者が判断する",
                    })
        proposals = proposals[:limit]
        self._log("propose_hypotheses", query=query, n_proposals=len(proposals))
        return proposals

    # ---------- 利用ログとログ駆動更新 ----------
    def _log(self, action: str, **payload: Any) -> None:
        entry = {"ts": time.time(), "action": action, **payload}
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def update_from_logs(self, min_count: int = 2) -> list[str]:
        """利用ログ中の頻出未知語をユーザ辞書へ追加し、永続化する。"""
        if not os.path.exists(self.log_path):
            return []
        counts: dict[str, int] = {}
        with open(self.log_path, encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("action") != "search":
                    continue
                for tok in MaterialsTokenizer._segment(entry.get("query", "")):
                    if len(tok) >= 3 and tok not in self.tokenizer.user_terms:
                        counts[tok] = counts.get(tok, 0) + 1
        new_terms = sorted(t for t, c in counts.items() if c >= min_count)
        if new_terms:
            self.tokenizer.add_terms(new_terms)
            with open(self.dict_path, "w", encoding="utf-8") as f:
                json.dump(self.tokenizer.user_terms, f, ensure_ascii=False, indent=1)
            self._log("dictionary_updated", added_terms=new_terms)
        return new_terms


# 初期投入用の文献抜粋（デモ・受入試験用。実文献の取込は add_document で行う）
SEED_PAPERS: list[dict[str, str]] = [
    {
        "doc_id": "PAPER-mro-almn",
        "title": "Al-Mn系合金における中距離秩序の安定性（抜粋）",
        "text": "Al-Mn-Al配置では中距離秩序が安定であると報告されている。"
                "DFTによる距離依存のエネルギー計算は、特定のAl-Mn間距離で"
                "エネルギー極小を示す。中距離秩序は電子構造の変化と関連する。",
        "source": "seed",
    },
    {
        "doc_id": "PAPER-hea-features",
        "title": "高エントロピー合金の相安定性予測特徴量（抜粋）",
        "text": "高エントロピー合金の相安定性はVEC、原子半径差、混合エンタルピー、"
                "混合エントロピーで整理できる。Hume-Rothery則の拡張として、"
                "VECはFCCとBCCの相選択と相関する。",
        "source": "seed",
    },
    {
        "doc_id": "PAPER-sqs-mlip",
        "title": "SQSとMLIPによる不規則固溶体の計算（抜粋）",
        "text": "SQSはDFTで不規則固溶体を扱う標準手法だが計算コストが高い。"
                "MLIPは第一原理計算の精度を保ちながらSQSの探索を高速化する。"
                "高エントロピー合金への適用が進んでいる。",
        "source": "seed",
    },
    {
        "doc_id": "PAPER-calphad-dft",
        "title": "CALPHADにおけるDFT生成エンタルピーの利用と誤差（抜粋）",
        "text": "CALPHAD評価ではDFTの生成エンタルピーが入力になる。"
                "DFTの計算誤差は状態図の相境界の位置に影響を与えるため、"
                "感度解析による不確かさ評価が必要である。",
        "source": "seed",
    },
    {
        "doc_id": "PAPER-creep",
        "title": "Ni基超合金のクリープ寿命と組織因子（抜粋）",
        "text": "クリープ寿命はL12析出相の体積分率、格子定数ミスフィット、"
                "拡散係数に依存する。CALPHAD計算による相分率予測は"
                "クリープ寿命モデルの特徴量として有効である。",
        "source": "seed",
    },
]


def build_default_provider(data_dir: str) -> GraphRAGProvider:
    """初期文献入りの GraphRAG プロバイダを構築する（既存グラフがあれば再利用）。"""
    provider = GraphRAGProvider(data_dir)
    if not provider.docs:
        for p in SEED_PAPERS:
            provider.add_document(p["doc_id"], p["title"], p["text"], p["source"])
    return provider
