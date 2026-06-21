#!/usr/bin/env python3
"""Generate a MeCab user dictionary for materials science terms.

Sources:
  1. material_terms.yaml — property terms, prototype aliases, element names
  2. Hardcoded DB/OQMD/pipeline terms
  3. materials_engineering_vocab.csv — 525-word domain vocabulary
     (metals/alloys, crystallography, thermodynamics, diffusion, mechanical,
      fracture, heat treatment, DFT, MD, materials informatics)

Output:
  - llm/mecab_materials.csv (source CSV for mecab-dict-index)
  - llm/mecab_materials.dic (compiled binary dictionary)

Usage:
  python scripts/build_mecab_materials_dict.py
  # Then verify:
  python -c "import MeCab; t=MeCab.Tagger('-u llm/mecab_materials.dic'); print(t.parse('バルクモジュラスが高いNi₃Al'))"
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import yaml  # noqa: E402 — must follow sys.path manipulation


def load_material_terms():
    """Extract Japanese material terms from material_terms.yaml."""
    terms_path = PROJECT / "llm" / "material_terms.yaml"
    with open(terms_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    entries = []

    # Property terms (Japanese aliases)
    for prop_key, prop_data in data.get("property_terms", {}).items():
        for alias in prop_data.get("aliases", []):
            if any(ord(c) > 0x2FFF for c in alias):  # CJK chars
                entries.append((alias, "材料特性", prop_key))

    # Prototype aliases (Japanese)
    for proto, aliases in data.get("prototype_aliases", {}).items():
        for alias in aliases:
            if any(ord(c) > 0x2FFF for c in alias):
                entries.append((alias, "結晶構造", proto))

    # Element names (Japanese)
    for elem, elem_data in data.get("elements", {}).items():
        for alias in elem_data.get("aliases", []):
            if any(ord(c) > 0x2FFF for c in alias):
                entries.append((alias, "元素", elem))

    # Stability terms
    for term_key, term_data in data.get("stability_terms", {}).items():
        desc = term_data.get("description", "")
        entries.append((term_key, "安定性", desc))

    return entries


def get_additional_materials_terms():
    """Additional materials science terms not in material_terms.yaml."""
    terms = [
        # Composite terms (multi-word Japanese)
        ("バルクモジュラス", "材料特性", "bulk_modulus"),
        ("せん断弾性率", "材料特性", "shear_modulus"),
        ("ヤング率", "材料特性", "youngs_modulus"),
        ("ポアソン比", "材料特性", "poisson_ratio"),
        ("格子定数", "材料特性", "lattice_constant"),
        ("生成エンタルピー", "材料特性", "formation_enthalpy"),
        ("形成エネルギー", "材料特性", "formation_energy"),
        ("生成エネルギー", "材料特性", "formation_energy"),
        ("キュリー温度", "材料特性", "curie_temperature"),
        ("デバイ温度", "材料特性", "debye_temperature"),
        ("熱伝導率", "材料特性", "thermal_conductivity"),
        ("磁気モーメント", "材料特性", "total_magnetization"),
        ("磁気異方性エネルギー", "材料特性", "magnetic_anisotropy"),
        ("磁気秩序", "材料特性", "magnetic_ordering"),
        ("表面エネルギー", "材料特性", "surface_energy"),
        ("粒界エネルギー", "材料特性", "grain_boundary_energy"),
        ("空孔形成エネルギー", "材料特性", "vacancy_formation"),
        ("バンドギャップ", "材料特性", "band_gap"),
        ("直接バンドギャップ", "材料特性", "direct_band_gap"),
        ("ハル上エネルギー", "材料特性", "energy_above_hull"),
        ("フェルミ面でのDOS", "材料特性", "dos_at_fermi"),
        ("体積弾性率", "材料特性", "bulk_modulus"),
        ("原子あたり体積", "材料特性", "volume_per_atom"),
        ("空間群", "材料特性", "space_group"),
        ("仕事関数", "材料特性", "work_function"),
        ("グリュナイゼン定数", "材料特性", "gruneisen_parameter"),
        ("スピン偏極", "材料特性", "spin_polarized"),
        ("強磁性", "材料特性", "ferromagnetic"),

        # Crystal structure terms
        ("ガンマプライム", "結晶構造", "L12_gamma_prime"),
        ("規則化FCC", "結晶構造", "L12"),
        ("規則化BCC", "結晶構造", "B2"),
        ("岩塩型", "結晶構造", "NaCl"),
        ("金属間化合物", "材料分類", "intermetallic"),
        ("固溶体", "材料分類", "solid_solution"),
        ("超合金", "材料分類", "superalloy"),
        ("ニッケル基超合金", "材料分類", "ni_based_superalloy"),

        # Structural concepts
        ("結晶構造", "構造概念", "crystal_structure"),
        ("空間群番号", "構造概念", "space_group_number"),
        ("原子分率", "構造概念", "atomic_fraction"),
        ("化学量論比", "構造概念", "stoichiometry"),
        ("組成", "構造概念", "composition"),
        ("格子体積", "構造概念", "lattice_volume"),

        # Computational methods
        ("第一原理計算", "計算手法", "dft"),
        ("密度汎関数理論", "計算手法", "dft"),
        ("擬ポテンシャル", "計算手法", "pseudopotential"),
        ("平面波基底", "計算手法", "plane_wave_basis"),

        # DB/Query terms
        ("相安定性", "DB概念", "phase_stability"),
        ("弾性テンソル", "DB概念", "elastic_tensor"),
        ("状態密度", "DB概念", "density_of_states"),
        ("バンド構造", "DB概念", "band_structure"),
        ("欠陥構造", "DB概念", "material_defect"),
        ("計算条件", "DB概念", "calculation"),

        # OQMD-specific terms (Phase 2)
        ("安定構造", "OQMD", "stable_structure"),
        ("準安定構造", "OQMD", "metastable_structure"),
        ("凸包", "OQMD", "convex_hull"),
        ("エネルギー凸包", "OQMD", "energy_convex_hull"),
        ("生成エンタルピー", "OQMD", "formation_enthalpy"),
        ("全エネルギー", "OQMD", "total_energy"),
        ("原子あたりエネルギー", "OQMD", "energy_per_atom"),
        ("磁性状態", "OQMD", "magnetic_state"),
        ("非磁性", "OQMD", "non_magnetic"),
        ("反強磁性", "OQMD", "antiferromagnetic"),
        ("フェリ磁性", "OQMD", "ferrimagnetic"),
        ("単元素", "OQMD", "pure_element"),
        ("単体", "OQMD", "elemental_substance"),
        ("同素体", "OQMD", "allotrope"),
    ]
    return terms


def load_engineering_vocab():
    """Load the 525-word materials engineering vocabulary CSV."""
    vocab_path = PROJECT / "llm" / "materials_engineering_vocab.csv"
    if not vocab_path.exists():
        print(f"  WARNING: {vocab_path} not found, skipping")
        return []
    entries = []
    with open(vocab_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            word_ja, word_en, category = row[0].strip(), row[1].strip(), row[2].strip()
            # Only include terms with Japanese characters and length >= 2
            if len(word_ja) >= 2 and any(
                ord(c) > 0x2FFF or 0x3040 <= ord(c) <= 0x30FF for c in word_ja
            ):
                entries.append((word_ja, category, word_en))
    return entries


def _get_dicdir():
    """Find the MeCab system dictionary *source* directory (needs pos-id.def etc)."""
    # System ipadic has the full source files needed for compilation
    for d in ["/usr/share/mecab/dic/ipadic",
              "/usr/lib/x86_64-linux-gnu/mecab/dic/ipadic"]:
        if os.path.exists(d) and os.path.exists(os.path.join(d, "pos-id.def")):
            return d
    raise RuntimeError("No MeCab ipadic source dictionary found. Install: apt-get install mecab-ipadic-utf8")


def generate_mecab_csv(entries, output_path):
    """Generate MeCab user dictionary CSV.
    
    Format: 表層形,左文脈ID,右文脈ID,コスト,品詞,品詞細分類1,品詞細分類2,品詞細分類3,活用型,活用形,原形,読み,発音
    For user dictionaries, left/right context IDs and cost are set to let MeCab auto-assign.
    """
    # Deduplicate by surface form
    seen = set()
    unique_entries = []
    for surface, category, key in entries:
        if surface not in seen and len(surface) >= 2:
            seen.add(surface)
            unique_entries.append((surface, category, key))

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for surface, category, key in sorted(unique_entries, key=lambda x: x[0]):
            # Lower cost = higher priority (prefer our terms over default splitting)
            cost = max(-10000, -len(surface) * 500)
            # IPAdic format: 表層形,左ID,右ID,コスト,品詞,細分類1,細分類2,細分類3,活用型,活用形,原形,読み,発音
            writer.writerow([
                surface, "", "", str(cost),
                "名詞", "一般", "*", "*", "*", "*",
                surface, "*", "*",
            ])

    return len(unique_entries)


def compile_mecab_dict(csv_path, dic_path):
    """Compile CSV to binary .dic using mecab-dict-index."""
    dicdir = _get_dicdir()

    mecab_dict_index = "/usr/lib/mecab/mecab-dict-index"
    if not os.path.exists(mecab_dict_index):
        for alt in ["/usr/bin/mecab-dict-index",
                     "/usr/local/libexec/mecab/mecab-dict-index"]:
            if os.path.exists(alt):
                mecab_dict_index = alt
                break

    cmd = [
        mecab_dict_index,
        "-d", dicdir,
        "-u", str(dic_path),
        "-f", "utf-8",
        "-t", "utf-8",
        str(csv_path),
    ]

    print(f"Compiling: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"STDERR: {result.stderr}")
        raise RuntimeError(f"mecab-dict-index failed: {result.returncode}")
    print(f"Compiled: {dic_path}")


def main():
    print("=== MeCab Materials Dictionary Builder ===")
    
    # Phase 1: Extract from existing code
    print("\n[Phase 1] Extracting from material_terms.yaml...")
    yaml_terms = load_material_terms()
    print(f"  Found {len(yaml_terms)} terms from YAML")

    # Phase 2: Additional materials terms (including OQMD)
    print("\n[Phase 2] Adding additional materials terms...")
    additional = get_additional_materials_terms()
    print(f"  Added {len(additional)} additional terms")

    # Phase 3: materials_engineering_vocab.csv (525-word domain vocabulary)
    print("\n[Phase 3] Loading materials engineering vocabulary...")
    eng_vocab = load_engineering_vocab()
    print(f"  Found {len(eng_vocab)} Japanese terms from vocabulary CSV")

    all_entries = yaml_terms + additional + eng_vocab

    # Generate CSV
    csv_path = PROJECT / "llm" / "mecab_materials.csv"
    n_unique = generate_mecab_csv(all_entries, csv_path)
    print(f"\nGenerated CSV: {csv_path} ({n_unique} unique terms)")

    # Compile to .dic
    dic_path = PROJECT / "llm" / "mecab_materials.dic"
    try:
        compile_mecab_dict(csv_path, dic_path)
    except Exception as e:
        print(f"WARNING: Could not compile .dic: {e}")
        print("CSV file is still usable for manual compilation.")
        return

    # Verify
    print("\n=== Verification ===")
    import MeCab
    import ipadic
    tagger_default = MeCab.Tagger(ipadic.MECAB_ARGS)
    tagger_custom = MeCab.Tagger(f"{ipadic.MECAB_ARGS} -u {dic_path}")

    test_queries = [
        "バルクモジュラスが高いL1₂構造の化合物",
        "生成エンタルピーが負の安定構造",
        "ニッケル基超合金のキュリー温度",
        "格子定数とせん断弾性率の関係",
        "準安定構造の単元素データ",
        "ステンレス鋼の応力腐食割れ",
        "マルテンサイト変態の活性化エネルギー",
        "金属間化合物のバンドギャップ",
        "第一原理計算による弾性テンソル",
        "材料インフォマティクスとベイズ推定",
    ]

    improved = 0
    for q in test_queries:
        d_result = tagger_default.parse(q)
        c_result = tagger_custom.parse(q)
        d_words = [tok.split("\t")[0] for tok in d_result.strip().split("\n") if "\t" in tok] if d_result else []
        c_words = [tok.split("\t")[0] for tok in c_result.strip().split("\n") if "\t" in tok] if c_result else []
        is_improved = len(c_words) < len(d_words)
        if is_improved:
            improved += 1
        marker = "improved" if is_improved else ("same" if len(c_words) == len(d_words) else "more")
        print(f"\nQuery: {q}")
        print(f"  Default ({len(d_words):2d} tok): {' / '.join(d_words)}")
        print(f"  Custom  ({len(c_words):2d} tok): {' / '.join(c_words)}  [{marker}]")
    print(f"\nImproved: {improved}/{len(test_queries)} queries")

    print("\n=== Summary ===")
    print(f"Total unique terms: {n_unique}")
    print(f"CSV: {csv_path}")
    print(f"DIC: {dic_path}")


if __name__ == "__main__":
    main()
