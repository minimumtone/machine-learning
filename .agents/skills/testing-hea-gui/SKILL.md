# Testing HEA Extrapolation Platform GUI

## Launch
```bash
pip install gradio plotly  # if not installed
python -m hea_extrapolation_platform gui --port 7860
```
Server starts in ~5 seconds. Navigate browser to http://localhost:7860

## Tab Structure (PR#125+)
The GUI uses nested Gradio tabs organized into 3 workflow phases:
- **1. 前処理 (Data Preparation)**: Config & Run, Data Summary
- **2. 解析 (Analysis)**: Dashboard, Results, OOD Map, FS Comparison
- **3. 後処理 (Post-processing)**: Literature Search, Report

CSV upload widget is exclusively in Config & Run tab.

## Running Analysis
1. Go to 1. 前処理 > Config & Run
2. Ensure Quick Mode checkbox is checked (reduces HPO grid)
3. Default settings: 200 samples, seeds 42/123/456, all 3 workflows enabled
4. Click "Run Analysis" button
5. Takes ~100 seconds for 702 runs
6. All tabs auto-refresh on completion via generator yields

## Verification Points
- Dashboard tab: KPIs (Total Runs, Best Feature Set, Best Total Score, OOD Samples) + Plotly charts
- Results tab: FS comparison table, physical interpretation, 702 run results table, parity plot
- Report tab: Full markdown experiment report with validity ranking
- No SIGSEGV (pandas 3.0 F-contiguous arrays handled by C-contiguous enforcement)

## Test Data
- Built-in sample data generator (default, no CSV needed)
- Real CSV: `hea_extrapolation_platform/data/HEA_ml_numeric_highconf.csv`

## CLI Alternative
```bash
python -m hea_extrapolation_platform run --quick --seeds 42 --skip-literature --skip-plots
```
Faster for regression testing (~100s), outputs to `results/` directory.
