# Phase 2: OptiMat Alloys 導入メモ

1. `.env` に `OPENROUTER_API_KEY=...`(無料枠可)を記載。
2. `docker compose up -d` → `http://<GPUサーバ>:8000` で Chainlit UI。
3. コンテナ内 DB パス(`/app/database` は仮)は配布イメージのドキュメントで確認。
   `docker exec -it <ctr> find / -name "*.db" 2>/dev/null` 等で実パスを特定し、
   `volumes:` の右側を合わせる。
4. Jupyter(Phase 1 環境)側:

   ```bash
   export MI_HUB_OPTIMAT_DB=/data/optimat_db
   ```

   ```python
   from mi_hub import optimat_bridge as ob, datastore as ds
   snap = ob.snapshot()                     # {name: DataFrame}
   for name, df in snap.items():
       ds.save(df, f"optimat_{name}", source="optimat")
   ```

   以後、他の kind と同様に pygwalker / FLAML / Feast から参照できる。

運用の分業:
- 4元系以上・CALPHAD 未整備領域の当たり付け → OptiMat(U-MLIP)
- 有望組成の精査(相平衡・等温断面) → TC-Python(既存三元系パイプライン)
- 両者の結果は同じ datastore + MLflow に集約
