---
name: testing-html-report
description: Test the report_for_students.html educational material for correct rendering, MathJax formulas, figures, and interactive elements. Use when verifying HTML report changes.
---

# Testing HTML Educational Report

## Overview
The file `report_for_students.html` is a static HTML educational report (~1600+ lines) about HEA lattice constant prediction. It uses MathJax for math rendering, CSS for styled boxes, and HTML5 `<details>` for expandable sections.

## How to Test

1. Open in browser via `file:///` protocol (no server needed)
2. Wait for MathJax to finish rendering (CDN loaded via script tag)

## Key Test Areas

### 1. Page Structure
- Title: "高エントロピー合金(HEA)の格子定数予測"
- Learning objectives (`.objectives`): Should have 7 items
- TOC: Should have ~20 items

### 2. MathJax Formulas
- Check `mjx-container` elements exist (expect 200+)
- No raw `$...$` patterns should remain in body text
- Key formulas: Vegard (a_Vegard = sum), Omega_sf definition, Kohn-Sham

### 3. Metrics (verify against latest code output)
- Independent test RMSE: check against `hea_lattice_xgboost.py` output
- Improvement percentages for BCC/FCC/overall
- These values may change if `INDEPENDENT_TEST` or `a_exp` values are modified

### 4. Figures
- All images in `<figure>` tags should load (check `naturalWidth > 0`)
- Figures are in `html_figures/` directory (relative path)
- Expect 17 figures

### 5. Interactive Elements
- `<details>` elements should be closed by default and expandable
- Expect 6 expandable sections
- First one: "具体例：CoCrFeNiの場合" with a_Vegard = 3.568

### 6. Crystal Structure Diagrams
- `.crystal-diagram` container with `.crystal-card` elements (2: BCC + FCC)
- `.crystal-ascii` blocks with monospace 3D wireframe art
- BCC: 配位数 8, FCC: 配位数 12

### 7. Pipeline Flowchart
- `.pipeline` container with `.pipeline-step` (5) and `.pipeline-arrow` (4)
- Steps: DFT → Ωsf計算 → 加法分解 → q最適化 → 独立検証

### 8. Glossary
- `.glossary` with `dt`/`dd` pairs (expect 15+ terms)
- Must include: HEA, DFT, RMSE, Vegard, BCC

### 9. Styled Boxes
- `.keypoint`: Green border-left (takeaway boxes)
- `.column-box`: Pink background (deeper dive)
- `.info-box`: Light blue background

## Common Issues
- MathJax CDN might be slow on first load — wait a few seconds
- Figures use relative paths (`html_figures/`), so the browser must be opened from the correct directory context
- If figures fail to load, check that `html_figures/*.png` files exist in the repo

## No Secrets Needed
This is a static HTML file with no authentication or API calls required.
