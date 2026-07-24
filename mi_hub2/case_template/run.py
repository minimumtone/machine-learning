"""事例実行の定型スケルトン。全事例がこの形に従うとエージェント・人間双方が扱いやすい。"""
import argparse

from mi_hub import datastore as ds, tracking as tr

CASE_NAME = "case_template"          # 事例名(= experiment 名)


def main(dry_run: bool = False):
    rid = ds.new_run_id()
    with tr.track(CASE_NAME, run_id=rid, params={"dry_run": dry_run}):
        # --- ここに解析本体 ---
        import pandas as pd
        df = pd.DataFrame({"x": [1, 2, 3]})
        # ----------------------
        ds.save(df, CASE_NAME, run_id=rid, source="case", code_ver="v0")
        tr.log_table(df)
        tr.log_metrics({"n_rows": len(df)})
    print(f"done. mi_hub.run_id={rid}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    main(**vars(p.parse_args()))
