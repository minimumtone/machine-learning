# L12 Text-to-SQL — SQL資材一式（第22次SQLレビュー対応版）

- 生成元コミット: GIT_COMMIT ファイル参照（ブランチ devin/1788016904-sql-review-round22、第22次SQLレビュー対応済み）

## 一括検証（唯一の正式エントリポイント）

- `python scripts/verify_all.py` — package構造・Python構文・canonical 300問カタログ・expected JSON契約・generated SQL/manifest/source整合・provenance・main run一意性の静的検査に加え、DB接続時は 4 suite gold再実行・expected照合・ORDER BY total-order・semantic/vocabulary監査・scoring self-check・`FULL_DB_TEST=1` pytest まで実行し、全PASSで exit 0
- DB未構築の段階では `python scripts/verify_all.py --static-only`
- 本パッケージは verify_all.py の完全PASSを提出条件とする

## 依存関係（最初に読むこと）

- Python 3.10 以上が必要
- 実行時依存は psycopg と sqlglot の2つ（sqlglot は gold_compare.py の ORDER BY 意味論判定に必須。「psycopg のみ」ではない）。graph 系ヘルパは networkx、llm/ 設定読込は PyYAML、unit test は pytest を使用
- 検証済みバージョンを requirements-repro.txt に固定済み: `pip install -r requirements-repro.txt`
- MP 転用DBの再構築（scripts/build_mp_transfer_db.py）はデフォルトで同梱スナップショット db/mp_transfer_snapshot.json.gz（メタデータに SHA-256・APIエンドポイント・chemsys 一覧・件数を記録）から MP API なしで再構築できる（一時DB mp_transfer_build_tmp に構築→件数検証成功後に rename/swap する all-or-nothing 方式）。ライブ再取得は明示的な `--refresh-from-api` のみで実行され（追加で requests と MP_API_KEY が必要）、全 chemsys 成功・ページネーション完走・material_id 重複ゼロを検証してから新スナップショットを保存し、そこからDBを構築する
- unit test 実行: `python -m pytest -q -ra`（tests/ 同梱、155件。JOIN 生成器の回帰テスト・MPスナップショット整合テストを含む）。`-ra` で skipped の内訳（DB未接続時の test_db_integrity.py のスキップ）を必ず表示させ、passed 件数だけで「全テスト成功」と判断しないこと。DB接続を必須にする厳格モードは `FULL_DB_TEST=1 python -m pytest -q -ra`（DB未接続を skip ではなく failure にする）
- db/001_schema.sql → 002_reference_data.sql → 003_material_data.sql → 004_views.sql → 005_roles.sql → 006_integrity_checks.sql → 007_initialization_marker.sql の順で PostgreSQL 15 に適用（docker/docker-compose.yml で自動適用）
- 注意: 005_roles.sql は所有者ロール l12_user を名指しで参照するため、POSTGRES_USER はデフォルトの l12_user のまま使用すること
- 注意: 006_integrity_checks.sql は cross-row アサーションを `validate_fixture_integrity()` 関数として定義・実行する（再実行可能: `SELECT validate_fixture_integrity();`）。007_initialization_marker.sql は同関数の成功にゲートされてマーカーを作成し、第11次からは `schema_fingerprint` を強化し、public スキーマの列の正確な型（format_type）・NOT NULL・default・生成式・全制約（pg_get_constraintdef）・view（pg_get_viewdef）・trigger（pg_get_triggerdef）・trigger関数本体（pg_get_functiondef）の SHA-256 を記録する。あわせて `git_commit` 列を追加（ロード時に `PGOPTIONS="-c l12.git_commit=$(cat GIT_COMMIT)"` または docker compose の `GIT_COMMIT` 環境変数で伝播、未指定時は 'unknown'。docker/docker-compose.yml は `GIT_COMMIT=$(cat GIT_COMMIT) docker compose up -d` で自動伝播）。`SELECT version, schema_fingerprint, git_commit FROM schema_initialization_status;` で確認。マーカーは初期化完了マーカーであり、以後の非サポート書き込み後の整合性保証ではない
- 注意: 本DBは007完了後は不変（immutable）な検証用フィクスチャとして扱うこと。利用は読み取り専用ロール l12_reader で行うこと
- エネルギー規約: 材料データが使用する規約はパッケージ固有の `L12-FIXTURE-PBE-v1` の1件のみ。テスト専用の `L12-FIXTURE-DIVERGENCE-TEST-v1`（全純元素 delta_e を +0.05 eV/atom シフトした複製）を pure_element_reference に収録（reference_set 一致条件を欠いた JOIN の検出用。phase_stability からは参照されず fixture_source_reference_set にも登録しない）。化合物の生成エネルギーは実DB取り込み値ではなく本パッケージ生成器の合成値。material_entry.source_db は合成上の出所ラベルにすぎない。許容される (source_db, reference_set) の組は fixture_source_reference_set マップに宣言され、006 が全ロード行の組を検査する
- 物理整合（生成器とDDL/006の両方で保証）:
  - 弾性: youngs_modulus / poisson_ratio は K・G から導出（E=9KG/(3K+G), ν=(3K−2G)/(2(3K+G))）。elastic_tensor のスカラー弾性率と calculated_property（EAV）の弾性行は同一生成値からの意図的な非正規化重複であり、同値性に加えて第10次からは被覆（各 elastic_tensor 行に3種のミラーが揃って存在すること）を006が検査する
  - 電子: band_structure.cbm_energy = vbm_energy + band_gap、phase_stability.band_gap と band_structure の gap は整合（006検査）。band_gap_type が単一truthで is_direct_gap は生成列
  - 結晶: prototype 別 conventional cell 原子数（L12=4, B2=2, NaCl=8, D03/BiF3=16, NiAs=4）に基づく volume_per_atom、cubic は a=b=c・V=a³、hexagonal は a=b・γ=120°
- 構造幾何のNULLポリシー（第10次）: structure.lattice_a/b/c は「全NULLか全非NULL」（部分NULLはCHECKで拒否。純元素基底状態行は conventional cell が prototype 体系外のため lattice 3列がNULL）。volume_per_atom は全行 NOT NULL
- property_scope（第10次）: property_definition の単一値 applies_to を廃止し、多対多の property_scope(property_name, applies_to) に置換。calculated_property / measured_property / element_property への格納はトリガが scope 宣言を検査し、使用中 scope の削除・変更は拒否。component 形状（value_shape='component'）の property は tensor_component を保持できる calculated scope のみに制限
- カバレッジ検査（第10次006）: 各 material_entry に calculation が「正確に1件」（0件も重複も違反。gold SQL が calculation_type 無条件JOINでも行が増えない前提の明文化）、全 material_entry に phase_stability 行、全 phase_diagram_entry に親 phase_stability（LEFT JOINで親欠落も検出）＋ hull_distance 一致。第11次で phase_stability / structure の「正確に1行」被覆と全 property_definition の property_scope 被覆を dependent 検査より前に実行し、value_shape='component' の property の tensor_component 使用整合も set-level で再検査
- 安定性規約: stable = energy_above_hull <= 0.001（is_stable 生成列）、metastable = 0.001 < eah <= 0.05。is_on_hull は hull_distance <= 0.001 の生成列
- density_of_states.is_metallic は3値（NULL=未判定）。006 は非NULL行のみ `IS DISTINCT FROM (band_gap = 0)` でNULL-safeに検査。gold SQL の金属判定は `is_metallic = TRUE`
- db/transfer_schema.sql: 転用実験（別DB oqmd_transfer）用DDL。第10次で oqmd_formation_energies.entry_key を UNIQUE 化（1エントリ1行、JOIN増殖防止）。第11次でメインDBと契約を統一: gap_ev NOT NULL（>=0・有限）、oqmd_elements.atomic_number NOT NULL UNIQUE（1..118）、atomic_mass NOT NULL、UNIQUE が作る index と重複していた entry_key index を削除。db/transfer_integrity_checks.sql が組成合計=1 に加え「全エントリに組成行が存在すること」を検査（再実行可能）
- DROPガード（第10次）: scripts/build_transfer_db.py / build_obfuscated_transfer_db.py は DROP/再作成対象のDB名が `oqmd_transfer` プレフィックスであることを要求し、postgres / template0 / template1 / メインDB名を拒否する
- evaluation/gold_sql/: gold SQL 265件（本評価245件＋転用 q_transfer_001〜020 の20件。転用20件は oqmd_transfer 用で本DBでは実行対象外）。第10次で ORDER BY を全件 total order 化（scripts/audit_order_totality.py がDB実測で監査）。第11次で監査を強化: 最外層 LIMIT/OFFSET/FETCH を除去した候補集合全体で tie を検出（LIMIT境界の外側の tie も検出）、main/transfer/難読化の3 suite を対象化（ordered_queries=255 non_total_order=0）、DISTINCT・window関数・集合返却関数・volatile関数など SELECT 注入が安全でない構文は自動監査せず MANUAL REVIEW として失敗させる。これに伴い gold SQL 9件（q_cte_010 / q_expert_023 / 043 / 051 / 055 / 061 / q_medium_022 / q_transfer_006 / q_transfer_obf_006）に決定的タイブレーク列を追加
- evaluation/expected_results/: 各 gold SQL の正解実行結果JSON（columns + ordered + rows）。第10次から `ordered` は必須キー（欠落・非boolはMALFORMED。SQLからの推測はしない）。`ordered: true` は行順保持 sequence 比較、`ordered: false` は tolerance 対応 multiset 比較。第11次から両検証器が `ordered` フラグと SQL 実体の最外層 ORDER BY の一致を照合し、不一致は `order_contract_mismatch` として失敗させる。q_expert_003 / 022 / 041 は意図的な0行 negative control
- evaluation/gold_sql_mp/・expected_results_mp_transfer/: MP転用実験（q_mp_001〜015）の gold SQL と期待結果（第20次から gold SQL をファイル化し、run_gold_verification.py / audit_order_totality.py の検証対象に追加。MP_DSN 設定時に検証される）
- evaluation/gold_sql_obfuscated/・expected_results_obfuscated/: 難読化転用実験の gold SQL 20件＋期待結果（db/obfuscated_transfer_mapping.json と scripts/prepare_obfuscated_transfer.py で生成）
- evaluation/generated_sql/: LLM生成SQLログ（main/independent/cte/llm_only/prototype/transfer系/mp_transfer。各 manifest.json が参照する source_file / eval_file（main_eval_with_sql.json 等）は evaluation/ 直下に同梱）
- evaluation/main_evaluation_dataset.jsonl（第19次新設）: 本評価245問全件の自然言語質問データセット。各行は id / question / difficulty / gold_sql_path / expected_result_path（一部は semantic_contract: 自然言語語彙→スキーマ述語の対応）を持ち、gold SQL との対応を検査可能
- evaluation/transfer_evaluation_dataset.jsonl / transfer_obfuscated_evaluation_dataset.jsonl: 転用20問＋難読化20問の質問データセット（第19次で q_transfer_016/019/020 と難読化版の質問文を gold SQL の意味論に整合させた: 019/020 は「純元素基底状態基準へ再基準化した生成エネルギー（delta_e − Σ x_i·reference_delta_e_i）」を明示、016 は「OQMD登録の単元素構造候補数」へ修正。SQL・期待結果は不変）
- .env.example: db/005_roles.sql が参照する環境変数例を同梱

## 検証器（scripts/gold_compare.py + run_gold_verification.py）

- 列名比較: cursor description の実列名（alias含む）と expected `columns` の完全一致を要求
- 行比較: ordered=true は行順保持の sequence 比較、ordered=false は tolerance 対応の重複保持 multiset 比較（丸めキーのsortではなく isclose によるgreedy pairing）
- セル比較: 数値（bool除く int/float）のみ `math.isclose(rel_tol=1e-9, abs_tol=1e-8)`、str/None/bool は型込み厳密一致（TEXT '001' ≠ '1'）
- expected JSON schema 検査（columns=list[str], rows=list[list], ordered 必須bool, 行幅=列数）と orphan expected 検出
- gold SQL 実行は READ ONLY セッション＋ statement_timeout 30s
- DSN未設定によるsuiteスキップはデフォルトで失敗（明示的に許容する場合のみ `--allow-skip`）。出力カウンタに order_contract_mismatch / ordered_metadata_missing / skipped を含む
- scripts/check_expected_results.py も第11次から main / transfer / 難読化の3 suite を対象化（READ ONLY + 30s timeout、ordered 契約照合を含む）
- scripts/audit_semantics.py（第11次新設）: formula ↔ composition.atomic_fraction（許容誤差1e-8）、reduced_formula ↔ gcd既約化、prototype の formula_type ↔ 組成比を意味論監査（期待: entries_checked=1559 semantic_mismatch=0）

## 再検証手順（このZIPのみで完結）

1. メインDB構築: `cd docker && POSTGRES_PASSWORD=... GIT_COMMIT=$(cat ../GIT_COMMIT) docker compose up -d`（db/001→007 が自動適用。007マーカー行の存在を確認。GIT_COMMIT 未指定時は marker の git_commit='unknown'）
2. 転用DB構築: `POSTGRES_HOST=127.0.0.1 POSTGRES_PASSWORD=... python scripts/build_transfer_db.py`（コピー前にメインDBへ assert_valid_fixture（007マーカー＋fingerprint一致＋validate_fixture_integrity() の現在データ再実行）を要求し、コピー後に transfer_integrity_checks.sql を実行して validate_transfer_integrity() 関数・compute_transfer_schema_fingerprint() 関数と transfer_initialization_status マーカー（version='001'、schema_fingerprint＋git_commit を記録）を作成。GIT_COMMIT 環境変数または GIT_COMMIT ファイルでビルド由来 commit を伝播）
3. 難読化転用DB構築: `POSTGRES_HOST=127.0.0.1 POSTGRES_PASSWORD=... python scripts/build_obfuscated_transfer_db.py`（template 複製前に source transfer DB へ assert_valid_transfer を実行し fail-fast。識別子rename後に validate_transfer_integrity() を難読化識別子で再インストールし、コピーされたマーカー行を削除して rename 後スキーマ自身の fingerprint で再封印。マーカー表・validator/fingerprint 関数名は難読化対象外）
4. MP転用DB構築（任意・MP suite を検証する場合）: `POSTGRES_HOST=127.0.0.1 POSTGRES_PASSWORD=... python scripts/build_mp_transfer_db.py`（同梱スナップショットから構築。API キー不要）
5. gold SQL 全件検証（要 psycopg + sqlglot・LLM/APIキー不要）:
   `L12_DSN="postgresql://l12_user:...@127.0.0.1:5432/l12_materials" TRANSFER_DSN="postgresql://l12_user:...@127.0.0.1:5432/oqmd_transfer" OBF_TRANSFER_DSN="postgresql://l12_user:...@127.0.0.1:5432/oqmd_transfer_obfuscated" MP_DSN="postgresql://l12_user:...@127.0.0.1:5432/mp_transfer" python scripts/run_gold_verification.py`
   → 期待: `ok=300 stale=0 order_mismatch=0 order_contract_mismatch=0 column_mismatch=0 missing=0 malformed=0 ordered_metadata_missing=0 orphan=0 errors=0 skipped=0`（main+transfer 265件＋難読化20件＋MP 15件。第14次から接続直後に main=assert_valid_fixture / transfer・難読化=assert_valid_transfer / MP=assert_valid_mp_transfer（スナップショットSHA-256照合）を実行し、未初期化・schema/data drift・integrity未検証のDBはクエリ実行前に拒否）
6. （任意）expected_results 照合の別実装（3 suite）: `POSTGRES_HOST=127.0.0.1 POSTGRES_PASSWORD=... python scripts/check_expected_results.py` → 期待: `ok=285 ... order_contract_mismatch=0 ... skipped=0`
7. （任意）ORDER BY total order 監査（4 suite・LIMIT境界含む）: 手順5と同じ環境変数で `python scripts/audit_order_totality.py` → 期待: `ordered_queries=264 non_total_order=0 unmapped=0 manual=0 skipped=0`
8. （任意）意味論監査: `L12_DSN=... python scripts/audit_semantics.py` → 期待: `entries_checked=1559 semantic_mismatch=0`（第12次から material_entry 全件との coverage 契約付き: structure/prototype/formula_type 欠落は監査失敗として報告）
9. （任意）語彙監査（第13次刷新・第19次で全suite必須化・第21次でMP suite追加）: 手順5と同じ環境変数（＋MP_DSN）で `python scripts/audit_vocabulary.py` → 期待: `suites_audited=4 vocabulary_mismatch=0 unresolved=0 skipped_nontext=0 suites_missing=0`（デフォルトで4 suite全部が必須。TRANSFER_DSN / OBF_TRANSFER_DSN 未設定は失敗（main / MP はローカル既定 conninfo にフォールバック）。部分監査は明示的に `--allow-skip` を指定した場合のみ許可）（gold SQL の文字列リテラル（=、!=、IN、IS [NOT] DISTINCT FROM）を sqlglot の scope 解析でエイリアス・非修飾列を実テーブルに解決したうえで (table, column) 単位の実DISTINCT値と突合。LIKE / ILIKE パターンは実DBへの EXISTS probe で0件マッチを検出。NULLを含む監査対象列は NULLS_ALLOWED 宣言（純元素基底状態由来の strukturbericht / site_label / wyckoff_site 系）がない限り失敗。`--show-null-stats` で全監査列の non_null/null 件数を出力。現在 INTENTIONAL_ZERO_MATCH / INTENTIONAL_ZERO_MATCH_LIKE は空集合）
10. （任意）監査前提: main 側は scripts/fixture_guard.py の assert_valid_fixture（007マーカー＋fingerprint一致＋validate_fixture_integrity() の現在データ再実行）、transfer/難読化側は scripts/transfer_guard.py の assert_valid_transfer（第15次から transfer_initialization_status マーカー存在＋記録fingerprintと compute_transfer_schema_fingerprint() 再計算値の一致＋validate_transfer_integrity() の現在状態再実行の3点）を全監査・検証器・転用ビルダーが要求する

## 第12次SQLレビュー対応の要点

- element.category を全89元素で NOT NULL 化し、snake_case の統制語彙10種（alkali_metal / alkaline_earth_metal / transition_metal / post_transition_metal / metalloid / nonmetal / halogen / noble_gas / lanthanide / actinide）に CHECK 制約で固定。生成器（ingestion/generate_extended_data.py）が周期表規則から導出し、gold SQL のリテラル綴りと一致
- q_expert_014 の `e.category NOT IN ('transition_metal')` を NULL-safe な `IS DISTINCT FROM` に修正（category NOT NULL 化後も三値論理上安全な形）。期待結果は再実行で再生成（392行→218行）
- transfer: 全 oqmd_entries に formation-energy 行が「正確に1件」（entry_key UNIQUE ＋ 欠落0件検査を transfer_integrity_checks.sql に追加）。build_transfer_db.py はコピー前に phase_stability の reference_set が正確に {'L12-FIXTURE-PBE-v1'} であることを assert
- audit_semantics.py: material → structure → prototype を LEFT JOIN 化し、検査件数と material_entry 全件数の一致を要求（行欠落の静かな見逃しを排除）。formula_type 欠落は Python 例外ではなく監査失敗として報告
- prepare_obfuscated_transfer.py: 期待結果生成に失敗した場合は当該JSONを残さず ROLLBACK し、非零終了（失敗が成功風の生成物に化けない）
- 期待結果JSONの型方針: PostgreSQL 型OIDは記録しない。比較は列名完全一致＋型付き値比較（数値のみ isclose、str/None/bool は型込み厳密一致）＋重複保持セマンティクスで行う（意味論的比較。この方針は本文書とスクリプト docstring に明記）
- schema_fingerprint の範囲: 列の正確な型・NOT NULL・default・生成式・全制約・view・trigger・trigger関数本体を対象とし、index・ロール・GRANT/REVOKE・default privilege・ロール設定は対象外（スキーマ意味論のfingerprintであり権限監査ではない）
- ORDER BY 判定: sqlglot AST（dialect=postgres）で最外層 ORDER BY を判定し、従来の lexer 判定と照合。両者が食い違う場合は黙って片方を採らず例外として失敗（parse不能SQLのみ lexer にフォールバック）

## 第13次SQLレビュー対応の要点

- audit_vocabulary.py を (table, column) 単位に刷新: sqlglot scope 解析でエイリアス（e.category → element.category）・非修飾列を実テーブルに解決し、同名列（element / synthesis_method / defect_type / alloy_system の category 等）の語彙が混ざらない。LIKE / ILIKE の0件マッチ検出（EXISTS probe）、NULL統計報告（--show-null-stats）、NULLS_ALLOWED / INTENTIONAL_ZERO_MATCH(_LIKE) の根拠付き宣言制
- 新監査が q_expert_079 の死パターン（'%heat%' / '%thermal%'：格納語彙0件マッチ）を実際に検出 → 実在語彙に一致する '%high-temperature%' のみへ整理（結果集合不変・期待結果再検証済み）
- タクソノミ明文化: 本フィクスチャの transition_metal は d-block 定義（Sc–Zn・Y–Cd・Hf–Hg、Zn/Cd/Hg を含む）。q_expert_014 の質問文はDBカテゴリを明示参照し、評価データセットに semantic_contract（自然言語語彙→スキーマ述語の対応）フィールドを導入
- q_expert_038 の非正準 'Ni-Al' 条件を削除（chemical_system の正準形はアルファベット順 'Al-Ni' のみ）
- material_entry.reduced_formula を NOT NULL 化。audit_semantics.py は reduced_formula NULL をスキップせず監査失敗として報告
- scripts/fixture_guard.py 新設: 007マーカーの存在＋記録fingerprintと compute_schema_fingerprint() 再計算値の一致を監査・転用ビルドの前提として強制
- build_transfer_db.py は reference_set 単一性に加え reference_energy_set マスタ行の内容（method/functional/source/fit_name）が契約タプルと一致することを assert
- formation_enthalpy.reference_status は閉じた統制語彙（ok / missing_composition / element_count_mismatch / missing_composition_fraction / invalid_composition / missing_reference_for_set）であることを COMMENT ON VIEW とDDLに明記
- semantic audit の範囲明文化: prototype 監査は化学量論比（fraction multiset）のみで、Wyckoff/サイト占有は契約外（gold SQL は site_label でフィルタしない）
- 全スクリプトの接続文字列を psycopg.conninfo.make_conninfo による構築へ統一（空白・引用符を含む資格情報を正しくエスケープ）

## 第14次SQLレビュー対応の要点

- audit_vocabulary.py の unresolved 失敗化: CTE/subquery 越しの列参照を投影lineage（単純列投影・SELECT *・エイリアス）で基底テーブルまで追跡してから語彙を突合。基底テーブルへ追跡できない参照（計算列投影など）は unresolved としてデフォルト失敗。成功条件は vocabulary_mismatch=0 かつ unresolved=0（skipped_nontext は非text列への等値比較の明示スキップ）
- 保証範囲の分離: assert_initialized_fixture=スキーマ同一性のみ / validate_fixture_integrity()=現在データの行間不変条件（006）/ audit_semantics=化学・prototype意味論 / gold verification=出力回帰。新設 assert_valid_fixture はスキーマ同一性＋006再実行で、transfer ビルド前・両検証器・両監査（main）が使用（schema保存的な data drift の転用伝播を拒否。006で表現されない drift は fixture immutable 運用が前提）
- transfer/難読化の current-state validator + marker: transfer_integrity_checks.sql を validate_transfer_integrity() 関数へ統合し、成功後に transfer_initialization_status マーカーを書き込み。scripts/transfer_guard.py はマーカー確認に加え監査のたびに現在状態で validator を再実行（「実行したはず」ではなく「現在も通る」を保証）
- 007マーカーの immutable seal 化: 既存マーカーの fingerprint と現在値が不一致なら例外で再実行を拒否（再封印不可、ON CONFLICT DO NOTHING）。一致時の再実行は no-op。スキーマを変更した新 revision は DB の作り直しでのみ初期化可能

## 第15次SQLレビュー対応の要点

- transfer/難読化への schema identity 保証: db/transfer_integrity_checks.sql が compute_transfer_schema_fingerprint()（列の正確な型・NOT NULL・default・生成式・全制約・view・trigger・関数本体の SHA-256、マーカー表と自関数は対象外）を定義し、transfer_initialization_status マーカーに schema_fingerprint・git_commit を記録。assert_valid_transfer は記録fingerprintと再計算値の一致を必須とし、初期化後の ALTER TABLE / DROP CONSTRAINT 等を validator の検査対象外の変更でも拒否
- transfer マーカーの anti-reseal 化: 既存マーカーの fingerprint と現在スキーマが不一致なら integrity SQL の再実行は例外で停止（再封印拒否）。新しいスキーマは scripts/build_transfer_db.py による DB 再構築でのみ封印可能。難読化DBは rename 後スキーマを自身の fingerprint で再封印（通常 transfer の fingerprint は流用しない）
- validate_transfer_integrity() の DDL 非依存 cardinality 検査: entry ごとの形成エネルギー行 exactly-one・reference state の symbol ごと exactly-one・ratio の (entry, element, site) 自然キー重複ゼロを GROUP BY/HAVING で独立検査（UNIQUE 制約が事後削除されても現在状態で検出）
- build_obfuscated_transfer_db.py は template 複製前に source transfer へ assert_valid_transfer を実行し fail-fast
- NULL contract の対称化: transfer oqmd_entries の prototype_label / spacegroup_number / crystal_system / cell_volume_pa を NOT NULL 化（lattice_param_a は純元素で正当に NULL のため維持）、main element.atomic_mass を NOT NULL 化
- 保証範囲の明確化: guard が検出するのは「fingerprint 対象のスキーマドリフト」と「validator の行間不変条件に違反するデータドリフト」。不変条件に現れない値のみを変える schema 保存的データドリフトの完全検出には data fingerprint が必要だが本パッケージでは導入せず、immutable fixture 運用＋gold/semantic/vocabulary audits を補完層とする（「全 data drift を検出する」とは保証しない）

## 第16次SQLレビュー対応の要点

- build_transfer_db.py の fail-safe 順序: source fixture 検証（assert_valid_fixture・reference-set 契約・reference_energy_set マスタ内容）を既存 transfer DB の DROP より前に実行。source が壊れている場合は既存 transfer DB に一切触れない（難読化 builder と同一設計）
- transfer build の snapshot 一貫性: source 接続を REPEATABLE READ + READ ONLY に固定し、guard・契約検査・全コピー SELECT が単一 snapshot を共有（builder 実行中の並行書込があっても transfer DB は main DB のある一時点の正確なコピー）
- run_gold_verification.py / check_expected_results.py の検証接続も REPEATABLE READ + READ ONLY snapshot に固定: suite ごとに guard と全 gold query が同一 DB 状態を参照（285問すべてが同じ DB 状態に対して一致したことを保証。statement_timeout 30s は維持。クエリエラー時はその snapshot は破棄されるが run 自体が失敗扱い）
- graph/join_path_generator.py の generate_join_clause() に table→alias マップを導入: multi-hop path で既導入 alias を再利用（未導入 alias を ON 句が参照するバグを修正）
- graph/graph_builder.py の _SEMANTIC_JOINS から composition.element → element.symbol を削除: 現 DDL では物理 FK（REFERENCES element(symbol)）であり schema introspection が単一情報源（semantic join は物理 FK で表現できない関係のみ）
- 期待 JSON の expected_empty 契約化: expected_empty: true かつ rows 非空の自己矛盾 JSON は malformed として検証失敗
- fingerprint scope の明確化: schema fingerprint はリレーショナル意味論の fingerprint であり、インデックス・GRANT/REVOKE・ロール設定・セキュリティ構成は対象外。マーカーは DB owner 権限による削除→再封印までは防げない（accidental drift 検出が脅威モデル。owner tampering 証明にはパッケージ側期待 fingerprint との三者比較が必要だが本パッケージでは未導入）

## 第17次SQLレビュー対応の要点

- check_expected_results.py に errors bucket を追加: gold SQL の実行エラー（syntax error / division by zero / column does not exist / statement_timeout 等）は errors=N として集計され、1件でもあれば終了コード1で失敗（従来はメッセージ表示のみで成功終了し得た明確なバグの修正。run_gold_verification.py と同じ failure 契約に統一）
- SAVEPOINT による snapshot 維持: check_expected_results.py / run_gold_verification.py / audit_order_totality.py の各 gold/監査クエリは SAVEPOINT で囲み、SQL エラー時はそのクエリのみ ROLLBACK TO SAVEPOINT で巻き戻す。外側の REPEATABLE READ トランザクションは維持され、エラー後の残りクエリも同一 snapshot を参照（従来の conn.rollback() はトランザクションを終了させ、以降が別 snapshot になっていた）。audit_order_totality.py の接続も REPEATABLE READ + READ ONLY に統一
- prepare_mp_transfer.py の gold SQL 実行失敗を non-zero failure 化: 失敗クエリの期待結果ファイルは（既存があれば）削除し、正常形の期待 JSON を残さず、1件でも失敗があれば終了コード1（prepare_obfuscated_transfer.py と同一設計）
- generate_join_clause() の disconnected edge 例外化: 両端とも未 JOIN の edge には ValueError を送出（未導入 alias を参照する壊れた SQL を静かに生成しない）
- JOIN 生成器の unit test 拡充: multi-hop alias 再利用・逆方向 FK・3 hop チェーン・disconnected edge 例外・duplicate edge・alias 衝突の回帰テストを追加
- transfer build の destination cleanup: DROP 後の構築（schema load・copy・integrity check・marker 封印）のいずれかが失敗した場合、作りかけの transfer DB を DROP してから例外を再送出（marker なし DB は guard 付きツールに拒否されるが、紛らわしい半端な DB を残さない）
## 第18次SQLレビュー対応の要点

- build_obfuscated_transfer_db.py に destination cleanup を追加: template 複製後の rename・validator 再インストール・marker 再封印・最終 guard のいずれかが失敗した場合、部分的に難読化された DB を DROP DATABASE ... WITH (FORCE)（assert_safe_transfer_db ガード付き）で削除してから例外を再送出（通常 transfer builder と同一設計。半難読化 DB がその名前のまま残らない）
- source 検証後の TOCTOU 縮小: CREATE DATABASE ... WITH TEMPLATE 直後・rename 前に、template 複製された DB 自体へ assert_valid_transfer() を実行。複製 DB は source の marker/fingerprint をそのまま持つため、guard 済み source と同等に正常な実体が template 化されたことを複製実体そのものに対して確認する（guard 後〜template 作成前に source が書き換えられた場合はここで失敗し、cleanup で DB が削除される）
- prepare_mp_transfer.py の snapshot 固定: 接続を REPEATABLE READ + READ ONLY にし、各 gold SQL を SAVEPOINT（mp_gold）で囲む。失敗時は ROLLBACK TO SAVEPOINT のみで外側トランザクションを維持し、1回の生成 run の全期待結果が同一 DB 状態を参照（他の verification tool と同一の snapshot 規約）

## 第19次SQLレビュー対応の要点

- 依存宣言の是正: 「要 psycopg のみ」記述を廃止し、requirements-repro.txt（psycopg[binary]==3.3.4 / sqlglot==30.12.0 / networkx==3.4.2 / pytest==9.1.1。実際の検証環境のバージョンを固定）を同梱。run_gold_verification.py の docstring も sqlglot 必須を明記
- MP 転用の再現性: scripts/build_mp_transfer_db.py を同梱し、prepare_mp_transfer.py の壊れた import を解消。さらに mp_conninfo を軽量な scripts/db_conninfo.py へ移し、prepare_mp_transfer.py は LLM パイプライン依存（requests / openai 等）なしで import 可能（requirements-repro.txt のみで動作確認済み）
- 本評価245問の自然言語データセット evaluation/main_evaluation_dataset.jsonl を同梱（全 id の一意性・gold/expected パス実在を検査済み。q_expert_101〜115 の質問文は gold SQL ヘッダコメントの自然言語仕様から収録）
- q_transfer_019/020（と難読化版）の質問文を gold SQL の「純元素基底状態基準へ再基準化した生成エネルギー」に整合させ、q_transfer_016 を「OQMD登録の単元素構造候補数」へ修正（SQL・期待結果は不変）
- README が主張する JOIN 生成器 unit test（tests/test_join_path_generator.py 含む tests/ 一式）を同梱し、`python -m pytest -q` で実行可能化
- prepare_obfuscated_transfer.py を他の generator と同規約化: 接続直後に READ ONLY + REPEATABLE READ + statement_timeout 30s、assert_valid_transfer() を必須化、各 gold SQL を SAVEPOINT（obf_gold）で囲い失敗時も外側 snapshot を維持、失敗 query の stale expected を削除し非零終了（負例で exit 1・stale 不在を実測）
- audit_vocabulary.py の partial-success 禁止: デフォルトで3 suite 全部必須（DSN 未設定は suites_missing として失敗・exit 1）、部分監査は `--allow-skip` の明示指定のみ。各 suite 接続も READ ONLY + REPEATABLE READ + 30s timeout へ統一。audit_semantics.py も同様に snapshot 固定
- 科学的記述の修正: gs_spacegroup は「ground-state Hermann–Mauguin space-group symbol (e.g. Fm-3m)」（番号ではない）、polymorph_count は「OQMD 単元素構造エントリ（polymorph candidate）数」（「熱力学的安定な多形の数」ではない）へ修正し、難読化 prompt template を再生成。db/transfer_schema.sql にも同旨のコメントを追記（fingerprint はカタログ由来のため不変）
- db/005_roles.sql が参照する .env.example を同梱し broken reference を解消

## 第20次SQLレビュー対応の要点

- 安定性3分類の統一: パッケージ規約（stable = eah ≤ 0.001 / metastable = 0.001 < eah ≤ 0.05 / unstable = eah > 0.05）に q_expert_034/082/095/100 の gold SQL を整合（「不安定」を「非stable全部」と数えない）。q_expert_035 は質問文が二値 is_stable=false を明示する形に整理。期待結果は再生成（L12: stable=7 / metastable=182 / unstable=203）
- 明示的サイトラベルの使用: q_expert_016 は composition.site_label='B-site' ＋ element メタデータ（category='transition_metal', block='d', period_number=5）で4d遷移金属B-siteを判定、q_expert_093 は site_label='A-site' を使用（atomic_fraction からのサイト推定を廃止）
- very-hard 20問の question↔gold意味監査: q_vhard_003/006/007/012/013/017 は質問文に重み・閾値・スコア式を明文化、q_vhard_010 は根拠のない「4象限分類」文言を削除、q_vhard_011 は hull距離ランキングへ質問文を整合、q_vhard_014 は element メタデータ（period_number・block）による 3d vs 4d/5d 群間比較SQLへ改訂（件数・平均hull距離・stable割合・平均生成エネルギー）、q_vhard_015/019 と q_cte_012・q_transfer_008（難読化版含む）は「生成エネルギーが低い順」「再基準化生成エネルギーが低い順」の用語へ統一
- semantic contract のモデル可視化: llm/prompt_templates/sql_generation_prompt.md に安定性3分類閾値・chemical_system のアルファベット順正準形・element.category 統制語彙をスキーマ規則として明記（評価データの semantic_contract フィールドはメタデータであり、モデルに見える規則は共有プロンプトに存在する）
- MP 再現性: db/mp_transfer_snapshot.json.gz（SHA-256・件数・APIメタデータ付き）を同梱し、build_mp_transfer_db.py をスナップショット既定・`--refresh-from-api` 明示・全 chemsys 成功／ページネーション完走／material_id 重複ゼロ検証・一時DB構築→検証成功後 swap の all-or-nothing 方式へ全面改修
- MP suite の検証統合: gold SQL を evaluation/gold_sql_mp/ にファイル化し、run_gold_verification.py（MP_DSN・guard は scripts/mp_guard.py によるスナップショットSHA-256照合）と audit_order_totality.py の対象へ追加（ordered_queries=264 / ok=300）。q_mp_006/015 に決定的タイブレーク列を追加、q_mp_009/010/011/013 に一意タイブレーク（entry_id）を追加（q_mp_011 は entry_id を SELECT 列にも追加）
- pytest skip の可視化: 推奨コマンドを `python -m pytest -q -ra` とし、DB未接続時のスキップを常に表示。`FULL_DB_TEST=1` で skip を failure に変換する厳格モードを追加


## 第21次SQLレビュー対応の要点

- 質問文の暗黙閾値・規約の明文化: q_hard_003/015、q_expert_024、q_vhard_001/002 は gold SQL が使用する閾値・重みを質問文に明示。q_expert_039 の「極めて安定」文言を実装条件に整合。q_mp_008 はバンドギャップ百分率が「%」であることを明記。準安定・Ni–Al 系の質問は self-contained 化（外部文脈なしで gold SQL と一意対応）
- canonical クエリカタログ: evaluation/query_catalog.json を canonical 300 問（main 245 + transfer 20 + 難読化 20 + MP 15）で再構築し、全 id 一意性・suite 件数・gold/expected パス実在を assert。MP suite の expected_result_available を修正
- q_expert_101〜115 を 245 問本評価へ正式組込み（ablation・感度系は従来どおり標準 100 問 subset。同一質問テキスト）
- 指標名の是正: 過年度比較で使用してきた寛容採点値は「historical execution recall」と改名（strict mean recall / exact result-set match と明確に区別。scoring audit（evaluation/scoring_audit.json）が保存済み per-query スコアの完全再現を 4 データセット全部で自己検査）
- LLM-only ベースラインにも同一の semantic 規約 note を付与して再評価（プロンプト規約差による不公平比較を排除）
- provenance の埋め込み: 全評価 JSON・generated_sql manifest に dataset / gold / prompt の SHA-256 と git commit・モデル名を記録（どの質問文・gold・プロンプトで得た数値かを機械照合可能）
- MP 強化: mp_guard の文言・照合強化、prepare_mp_transfer.py の guard 必須化＋atomic replace、vocabulary audit へ MP suite を追加（suites_audited=4）
- save_snapshot の正準整列: --refresh-from-api 後も entry_id / (entry_id, element) / symbol の正準順で SHA-256 を記録（guard の照合順序と一致。逆順入力の回帰テスト追加、tests 155 件）
- 全 LLM 評価を新質問文で推論から再実行し、評価成果物（evaluation/*.json・generated_sql/）を全面更新（main 245 問 historical execution recall 平均 85.7% / LLM-only 77.9% / transfer 85.0% / 難読化 80.0% / MP 93.3% / 独立 81.5% / ablation 5run full 92.5%±0.4、Holm 補正後有意は no_fewshot・no_dict のみ）

## 第22次SQLレビュー対応の要点

- canonical main run の一本化: `evaluation/multiaxis_results.json` を唯一の main 推論結果とし、`main_eval_with_sql.json`・`generated_sql/main/`（245本＋manifest）・`generated_sql/llm_only/`（245本＋manifest）・`failure_analysis.json`・`scoring_audit.json` はすべて `scripts/derive_main_artifacts.py` が保存済みSQLから決定的に派生生成（LLM再実行なし）。派生成果物は `source_result_file` / `source_result_sha256` と `eval_file` を保持し、per-query recall の自己検査一致（mismatch=0）をゲートとする
- exact result-set match の一本化: `evaluation/metrics_strict.py::exact_result_set_match`（gold 列リスト完全一致＋行 multiset 一致＋ordered=true 時は行順 sequence 一致、大文字小文字は列名のみ非区別）を canonical 定義とし、main 245 問の canonical exact は 4.1%。旧 common-column 判定値は `common_column_exact_overlap`（診断専用）と改名して併記。列欠落・列過剰・列順・重複多重度・順序不一致の回帰テストを tests/ に追加
- モデル生成 SQL の実行安全化: 全 `scripts/eval_*.py` が共有の `scripts/eval_db.py` を使用 — connection-level READ ONLY＋REPEATABLE READ＋statement_timeout＋suite別 fixture guard（main=assert_valid_fixture / transfer系=assert_valid_transfer / MP=assert_valid_mp_transfer）を接続時に強制し、生成SQL 1 本ごとに SAVEPOINT/ROLLBACK で隔離（`no_guard` ablation は SQLGuard 静的検査のみを無効化し、DB書き込み保護は維持）
- provenance の是正: prompt hash キーを `prompt_template_file` / `prompt_template_sha256` に改名し、static template hash が few-shot・schema/JOIN listing・column hint 等の動的注入部を被覆しないことを `prompt_template_note` に明記（旧キーは互換エイリアスとして解決）。legacy の cte / prototype 結果には post-hoc 注記付き provenance を付与（推論時に記録されなかったことを明示）
- stale-resume の拒否: resume 機能を持つ評価スクリプト（llm_only / ablation / dict・fewshot 感度 / model_comparison / independent）は再開時に stored provenance（dataset / gold / prompt template / model / git commit）と現在値を比較し、不一致なら既定で拒否（明示的な `--force-stale-resume` でのみ警告付き続行）
- 評価コミットと配布コミット: 保存済み評価成果物の provenance `git_commit`（a608ec54…、第21次推論実行時のコード revision）と、配布パッケージの GIT_COMMIT（梱包時の revision）は意図的に異なり得る。verify_all.py はこの差を WARN として明示報告する（推論は第22次で再実行していないため、評価 revision は不変）
- パッケージ構築の一元化: `scripts/build_sql_package.py` が include リストから ZIP を構築（`__pycache__` / `.pytest_cache` / `*.pyc` / paper 依存スクリプトを除外、GIT_COMMIT を梱包時 revision で生成、梱包前に verify_all.py --static-only の PASS を必須化）
- artifact 整理: `ablation_significance_v2.json` は「paper 図生成専用の別手法統計。canonical は significance_recomputed.json」であることを `_meta.status` に明記。paper/ 依存スクリプト（verify_ssot / verify_paper_numbers / generate_figures / compute_all_figures）と `.pytest_cache` / `__pycache__` / `*.pyc` は SQL ZIP から除外
