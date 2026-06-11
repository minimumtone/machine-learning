---
name: testing-mi-textbook
description: Test the MI textbook Streamlit app end-to-end. Use when verifying changes to mi_textbook_app.py, data files, or UI improvements.
---

# Testing the MI Textbook App

## Prerequisites

- Python 3.12 with dependencies: `streamlit`, `scikit-learn`, `plotly`, `pandas`, `numpy`, `matminer`
- Data files in `data/`: `steel_strength.csv` (312 rows), `superconductor_500.csv` (500 rows), `HEA_phases.csv` (1103 rows)

## Starting the App

```bash
cd /home/ubuntu/repos/machine-learning
pip install streamlit scikit-learn plotly pandas numpy matminer pymatgen 2>/dev/null
streamlit run mi_textbook_app.py --server.port 8501 --server.headless true &
# Wait ~10s for initial load (matminer imports are slow)
curl -s -o /dev/null -w "%{http_code}" http://localhost:8501  # expect 200
```

## Key Test Areas

### 1. Steel Data Leakage (Critical)

The steel dataset (`data/steel_strength.csv`) contains "tensile strength" and "elongation" columns that are co-measured outputs (NOT input features). These must be excluded from the feature set.

**Programmatic verification**:
```python
import pandas as pd
df = pd.read_csv('data/steel_strength.csv')
target_col = 'yield strength'
_response_cols = {'tensile strength', 'elongation'}
feature_cols = [c for c in df.columns if c != target_col]
feature_cols = [c for c in feature_cols if df[c].dtype in ['float64','int64','float32','int32'] and c not in _response_cols]
assert 'tensile strength' not in feature_cols
assert 'elongation' not in feature_cols
assert len(feature_cols) == 13  # composition elements only
```

**Pass criterion**: R² for linear regression with composition-only should be low (-0.2 to 0.5). If R² > 0.85, leakage is likely present.

### 2. Data Augmentation Leakage (Critical)

Augmentation must happen AFTER train/test split, and only on training data.

**Programmatic verification**:
```python
from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
# Training count should be ~80% of total (e.g., 249 for steel's 312)
# Augmented count should be training_count * multiplier
assert len(X_tr) == 249  # for steel
assert len(X_tr_aug) == 249 * aug_multiplier
```

**UI check**: Metrics should say "訓練データ数（元）" and "訓練データ数（増強後）", NOT "元データ数".

### 3. Section Navigation

The app has 9 sections accessible via sidebar radio buttons:
1. MIとは
2. データ探索
3. 次元削減 PCA
4. 回帰問題
5. 分類問題
6. 交差検証・汎化性能
7. 正則化・モデル選択
8. データ増強
9. まとめ＋レポート課題

### 4. PCA Biplot

Section 3 should show a biplot with:
- Scatter points (scores) colored by target variable
- Red arrows (loading vectors) showing feature directions
- Square aspect ratio (scaleanchor)
- Title containing "Biplot"

### 5. HEA Classification (Hume-Rothery)

Section 5 should show:
- Title: "Hume-Rothery 則の再現"
- Data: "高エントロピー合金 (HEA) 相分類 — Zenodo ACHIEF (1,103件)"
- VEC/δ/Δχ references
- SVM decision boundary demo + k-means clustering demo

## Tips

- The data table in the summary section renders as a `<canvas>` element (Streamlit's data editor), so column names aren't readable from HTML. Use programmatic verification for column checks.
- Streamlit takes 5-10s to switch sections (recomputes everything). Be patient.
- Japanese fonts rely on CSS override (`Yu Gothic, YuGothic, Meiryo`). On Linux servers without these fonts, Plotly will fall back to sans-serif.
- FFmpeg recording might not work in some environments. Fall back to screenshot-based testing.
- The sidebar has a dropdown for dataset selection (回帰用データ). Default is "鉄鋼（構造材料）".

## Devin Secrets Needed

None — this app uses only local CSV data files.
