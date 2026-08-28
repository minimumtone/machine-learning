# L1₂ Schema-Graph-Assisted Text-to-SQL

> **デリバリZIPをお持ちの方は [README_DELIVERY.md](README_DELIVERY.md) を先にお読みください。**
> 本READMEは完全リポジトリ（Docker・DB・テスト含む）向けです。

L1₂型金属間化合物探索のためのスキーマグラフ支援型Text-to-SQLシステム。

## Overview

自然言語クエリからPostgreSQLのSELECT文を自動生成し、L1₂型金属間化合物（Ni₃Al型γ'相候補、A₃B型規則化FCC化合物）を探索するシステムです。

ERスキーマをNetworkXグラフ化し、関連テーブル・カラム・JOIN経路を制約としてLLMに与えることで、不正JOIN・存在しないカラム生成・multi-hop query失敗を低減します。

```
Natural Language Query → 材料用語正規化 → 条件抽出 → テーブル・カラム推定
→ スキーマグラフJOIN経路探索 → 制約付きSQL生成 → SQL安全検査（SQLGuard 14種検証）
→ PostgreSQL実行 → 結果表示
```

## Quick Start

> **Note**: デリバリZIPを受け取った方は `README_DELIVERY.md` を参照してください。
> 以下はリポジトリのクローンを前提とした完全環境構築手順です。
> ZIPにはリポジトリの主要ファイルが同梱されています。詳細は `README_DELIVERY.md` を参照。

### 前提条件

- Python 3.11+
- Docker / Docker Compose
- OpenAI API key（Proposed手法の実行に必要。Rule-based fallbackはAPI key不要）

### 1. 依存パッケージのインストール

```bash
cd l12-schema-graph-text2sql
pip install -e ".[dev]"
```

### 2. PostgreSQL起動（スキーマ + データ自動投入）

```bash
cd docker
docker compose up -d
cd ..
```

これにより `db/001_schema.sql`（33テーブル）→ `db/002_reference_data.sql`（マスタ）→
`db/003_material_data.sql`（材料データ）→ `db/004_views.sql`（ビュー）→
`db/005_roles.sql`（読み取り専用ロール）→ `db/006_integrity_checks.sql`（整合性検査、再実行可能）→
`db/007_initialization_marker.sql`（初期化完了マーカー）が順に自動適用されます。

注意：`db/005_roles.sql` は所有者ロール `l12_user` を名指しで参照するため、`POSTGRES_USER` はデフォルトの `l12_user` から変更しないでください（変更するとロード時に明示的なエラーで停止します）。

注意：`db/006_integrity_checks.sql` が失敗したDB（001〜005のみ適用された途中状態）を検証用DBとして使用しないでください。006 はアサーションのみの再実行可能ファイルで、任意の時点で再検証に使えます。006 の全アサーション通過後に適用される `db/007_initialization_marker.sql` が `schema_initialization_status` テーブルに `version='007'` の行を作成するため、使用前に `SELECT 1 FROM schema_initialization_status WHERE version='007';` で初期化完了を確認できます。このマーカーは「初期化完了マーカー」であり、現在の整合性状態の保証ではありません（初期化後に書き換えれば壊せます）。本DBは初期化完了後は不変（immutable）な検証用フィクスチャとして扱い、初期化完了後のエンティティデータのINSERT/UPDATE/DELETEはサポートしません。利用は読み取り専用ロール `l12_reader` で行い、migration owner（`l12_user`）以外にwrite権限を与えないでください。また、propertyディクショナリ（`property_definition`）の変更と各propertyテーブルへの書き込みを並行して行うことは想定していません。

3値BOOLEANの扱い：`density_of_states.is_metallic` のみ意図的にNULL可（NULL=金属性未判定）です。gold SQL では「金属」を `is_metallic = TRUE`（判定済みのTRUEのみ）として扱い、NULL（未判定）は「金属」にも「非金属」にも含めない規約に統一しています。他のBOOLEAN列はすべて NOT NULL です（`phase_stability.is_stable` は `energy_above_hull NOT NULL` により生成列も常に2値）。

設計上の意図的な簡略化：`calculation` は (entry, calculation_type, method, functional) ごとに1件のみ保持します。カットオフ・k点メッシュ・擬ポテンシャル・U値などの数値パラメータ軸は本検証用DBでは持たず、汎用の計算アーカイブとしては UNIQUE が強すぎる点を明示しておきます。

エネルギー規約（reference_set）：`phase_stability.formation_energy_per_atom` は「その材料の `reference_set`（`reference_energy_set` マスタへのFK）が定める元素参照状態に対する生成エネルギー」です。純元素側の `pure_element_reference.delta_e` は OQMD の delta_e（生成エネルギー、eV/atom）であり、全DFTエネルギーでも参照エネルギー値そのものでもありません。`formation_enthalpy` ビューの `enthalpy_vs_element_ground_states` は同一 `reference_set` 内で `formation_energy - Σ xᵢ·delta_eᵢ` を計算し、「フィットされた参照状態基準」を「収録純元素基底状態基準」へ付け替えた値です（同一規約内では参照エネルギーが厳密に相殺するため二重補正にはなりません）。異なる `reference_set` 間の混用はビューのJOIN条件（`per.reference_set = ps.reference_set`）と006のset単位被覆検査で構造的に防がれます。なお本フィクスチャで材料（`phase_stability`）が使うエネルギー規約はパッケージ固有の共通規約 `L12-FIXTURE-PBE-v1` の1件のみです。これとは別に、`pure_element_reference` にはテスト専用規約 `L12-FIXTURE-DIVERGENCE-TEST-v1`（全元素の delta_e を +0.05 eV/atom シフトした複製。`fixture_source_reference_set` に未登録のため材料側からは使用不可）を収録しています。これは `reference_set` 条件を欠いたJOINが偶然正しい結果を返さないようにするための発散検出フィクスチャで、`tests/test_db_integrity.py` が実際に差が生じることを検証します。命名の根拠：化合物の生成エネルギーは実データベースからの取り込み値ではなく、本パッケージの生成器（`ingestion/generate_extended_data.py`。既知L1₂化合物はキュレーション値、その他は範囲内乱数）が合成した値であり、`pure_element_reference` に収録した OQMD DFT-PBE の純元素 delta_e（実データ）を元素参照状態として「宣言」したものです。変換式は存在せず（値の出所が合成であるため）、外部DB間のエネルギー補正も行っていません。したがって `OQMD-PBE` / `MP-PBE` などの実DB規約名を名乗ることは誤解を招くため、フィクスチャ固有名を採用しています。`material_entry.source_db`（OQMD / Materials Project / AFLOW）は合成上の出所ラベルにすぎず、エネルギー値の出所を意味しません。許容される (source_db, reference_set) の組は `fixture_source_reference_set` マップに宣言され、006が全ロード行の組がマップに存在することを検査します（マルチ規約データを載せる場合は `reference_energy_set` とマップに行を追加し、同じ機構がset別に機能します）。

数値・単一truth・不変条件（第6次レビュー対応）：(1) 物理量カラムには有限値CHECK（NaN / ±Infinity 拒否）を付与しています（生成エネルギー・E_hull・delta_e・組成分率・転用スキーマの delta_e / hull_distance など）。(2) `phase_diagram_entry.is_on_hull` は `hull_distance <= 0.001` から導出される生成列で、`phase_stability.is_stable` と同一の運用定義の単一truthです。(3) EAV 3表（calculated/measured/element property）の `value` は NOT NULL で、「値が未知」は行の不存在で表現します。(4) `property_definition.value_type` は本フィクスチャで実際に使用する `'float'` のみに限定しています（整数propertyは未使用のため。整数対応を追加する場合は小数値を拒否するtrigger検証が必要です）。(5) `property_definition.canonical_unit` のマスタ側UPDATEは、不整合な子行が存在する場合trigger（`prevent_invalid_canonical_unit_change`）で拒否されます。(6) `reference_energy_set` の規約フィールド（method/functional/source/fit_name）は、そのsetがロード済みエネルギーから参照された後はtrigger（`prevent_referenced_convention_change`）で変更不可です。

実験測定の未知条件の制限：`experimental_measurement` の `UNIQUE NULLS NOT DISTINCT` により、NULL の測定条件（reference/method/温度/圧力）は独立した測定を表しません。同一材料につき「条件未知の測定」は1件しか表現できず、独立した実測値を共存させるには実際の測定条件を記録する必要があります。

検証器と期待結果の比較方針（第9次レビュー対応）：`scripts/run_gold_verification.py` / `scripts/check_expected_results.py` は `scripts/gold_compare.py` の共通ポリシーで比較します。(1) 列名（alias含む）を `cur.description` と期待JSONの `columns` で完全一致比較、(2) 最外層に ORDER BY を持つクエリは行順を保持したsequence比較・持たないクエリは重複を保持したmultiset比較（set化しない）、(3) 数値同士のみ `math.isclose(rel_tol=1e-9, abs_tol=1e-8)` の許容誤差を適用し、文字列・boolean・NULLは型ごと厳密比較（数値風TEXTをfloat化しない・6桁丸めなし）、(4) 期待JSONのスキーマ（columns=文字列list / rows=listのlist / ordered=bool）を検査、(5) gold SQLを持たない孤児期待ファイルを失敗として検出。接続は READ ONLY + `statement_timeout=30s` を強制します。難読化転用suite（`gold_sql_obfuscated` × `expected_results_obfuscated`）は `OBF_TRANSFER_DSN` で同一検証器により独立検証できます。MP転用実験（q_mp_*）の期待結果は評価専用の `evaluation/expected_results_mp_transfer/` に分離しています。

第11次レビュー対応の追加検証：(1) 両検証器は期待JSONの `ordered` フラグをSQL実体の最外層 ORDER BY と照合し、不一致を `order_contract_mismatch` として失敗させます（メタデータの改変・SQL編集の片側だけの変更を検出）。`check_expected_results.py` は main / transfer / 難読化転用の3 suiteすべてを対象にします。(2) `scripts/audit_order_totality.py` は最外層の LIMIT/OFFSET/FETCH を除去した候補集合全体で ORDER BY キーの重複を監査し（LIMIT境界の外側のtieも検出）、3 suiteすべてを対象、SELECT注入が安全でない構文（DISTINCT・window関数・集合返却関数・volatile関数）は自動監査せず MANUAL REVIEW として失敗させます。(3) `scripts/audit_semantics.py` は formula ↔ composition.atomic_fraction（許容誤差1e-8）、reduced_formula ↔ gcd既約化、prototype の formula_type ↔ 組成比の3点を意味論監査します。(4) `db/007_initialization_marker.sql` のschema fingerprintは pg_catalog から列の正確な型（format_type）・NOT NULL・default・生成式・全制約（pg_get_constraintdef）・view（pg_get_viewdef）・trigger（pg_get_triggerdef）・関数本体（pg_get_functiondef）をSHA-256でハッシュ化し、markerに `git_commit`（ロード時に `PGOPTIONS="-c l12.git_commit=$(cat GIT_COMMIT)"` または docker-compose の `GIT_COMMIT` 環境変数で伝播、未指定時は 'unknown'）を記録します。

第12次レビュー対応の追加検証：(1) `element.category` は全89元素で必須（NOT NULL）とし、snake_case の統制語彙（alkali_metal / alkaline_earth_metal / transition_metal / post_transition_metal / metalloid / nonmetal / halogen / noble_gas / lanthanide / actinide）をCHECK制約で強制します。gold SQL（`q_expert_014`）は `IS DISTINCT FROM 'transition_metal'` によるNULL-safe比較です。(2) `scripts/audit_vocabulary.py` は全gold SQLをAST解析（sqlglot）して等値比較（= / <> / IN / NOT IN）の文字列リテラルを抽出し、DBの同名テキスト列の実DISTINCT値と突合します（gold側リテラルの綴りとデータ側語彙の不一致を実行前に検出。意図的な0件マッチは `INTENTIONAL_ZERO_MATCH` に根拠付きで宣言）。(3) `db/transfer_integrity_checks.sql` は全 `oqmd_entries` に形成エネルギー行がexactly one（UNIQUE＋欠落0件）であることを検査し、`scripts/build_transfer_db.py` はコピー前にソースDBの `phase_stability.reference_set` が正確に `{'L12-FIXTURE-PBE-v1'}` の1規約のみであることをassertします。(4) `scripts/audit_semantics.py` はLEFT JOINで全 `material_entry` を対象化し、検査件数と `material_entry` 行数の一致（coverage契約）を検証、structure/prototype/formula_type の欠落は監査失敗として明示報告します。(5) `scripts/prepare_obfuscated_transfer.py` は期待結果の生成に失敗したクエリについて正常形の期待JSONを残さず（既存ファイルも削除）、1件でも失敗があれば非零終了します。(6) 期待JSONの型方針：期待結果はPostgreSQLの結果型OIDを保存せず、意味論的な値の等価性（型を尊重したセル正規化＋数値のみ許容誤差）で比較する契約です。これは意図的な設計判断で、同一の意味値を返す別型（bigint vs numeric等）のSQLを正解として許容します。(7) `sql_is_ordered()` はsqlglot（postgres方言）のAST解析を一次判定とし、簡易lexerとの相互検査で不一致を例外化します（dollar-quoted文字列等lexer非対応構文への防御）。(8) 007のschema fingerprintは「スキーマ意味論のfingerprint」であり、インデックス・ロール・GRANT/REVOKE・デフォルト権限などの配備構成は対象外です（それらは 005_roles.sql と docker-compose が規定）。

第13次レビュー対応の追加検証：(1) `scripts/audit_vocabulary.py` を (table, column) 単位に刷新しました。sqlglot の scope 解析でエイリアス（`e.category` → `element.category`）と非修飾列を実テーブルに解決し、リテラルは「そのテーブルのその列」の実DISTINCT値とのみ突合します（同名列 `category` を持つ `element` / `synthesis_method` / `defect_type` / `alloy_system` の語彙が混ざりません）。`LIKE` / `ILIKE` パターンも実DBへの EXISTS probe で0件マッチを検出します（意図的0件は `INTENTIONAL_ZERO_MATCH_LIKE` に宣言）。等値比較対象列のNULLは、`NULLS_ALLOWED`（Strukturbericht記号のない純元素基底状態に由来する `structure.strukturbericht` / `prototype_definition.strukturbericht` / `composition.site_label` / transfer側 `oqmd_element_ratios.wyckoff_site` とその難読化対応列。各89件、根拠をコード内に明記）に宣言がない限り失敗として報告し、`--show-null-stats` で全監査列の non_null/null 件数を出力できます。(2) 元素タクソノミの明文化：本フィクスチャの `transition_metal` は d-block 定義（Sc–Zn・Y–Cd・Hf–Hg、Zn/Cd/Hg を含む）です。`q_expert_014` の質問文はDB上のカテゴリ（`element.category = 'transition_metal'`）を明示参照し、評価データセットに `semantic_contract`（自然言語語彙→スキーマ述語の明示対応）フィールドを導入しました。(3) `q_expert_038` の非正準リテラル `'Ni-Al'` 条件を削除しました（`chemical_system` の正準形は元素記号アルファベット順の `'Al-Ni'`。防御的な別表記受理は語彙監査と矛盾するため廃止）。(4) `material_entry.reduced_formula` を NOT NULL 化し、`audit_semantics.py` はNULLを（スキップではなく）監査失敗として報告します。(5) 監査・転用ビルドの前提検証：`scripts/fixture_guard.py` が version='007' マーカーの存在と、記録された schema fingerprint と `compute_schema_fingerprint()` の再計算値の一致を検査し、`audit_semantics.py` / `audit_vocabulary.py`（main suite）/ `build_transfer_db.py` は部分初期化・スキーマドリフトしたDBを拒否します。(6) `build_transfer_db.py` は reference_set 名の単一性に加え、`reference_energy_set` マスタ行の内容（method/functional/source/fit_name）が本フィクスチャの契約タプルと一致することをassertします。(7) `formation_enthalpy.reference_status` は閉じた統制語彙（ok / missing_composition / element_count_mismatch / missing_composition_fraction / invalid_composition / missing_reference_for_set）であることを `COMMENT ON VIEW` とDDLコメントに明記しました。(8) semantic audit の範囲明文化：prototype 監査は化学量論比（fraction multiset）のみを検証し、Wyckoff/サイト占有（どの元素がどのサイトか）は検証対象外です（gold SQL は `site_label` でフィルタしないため契約外。サイト占有まで保証する場合は prototype×site_label×多重度のマスタ表が必要）。(9) 接続文字列は全スクリプトで `psycopg.conninfo.make_conninfo` により構築し、空白・引用符を含む資格情報でも正しくエスケープされます。(10) 新設のLIKE/ILIKE監査が `q_expert_079` の死パターン（`'%heat%'` / `'%thermal%'`：格納語彙に0件マッチ）を実際に検出したため、実在語彙に一致する `'%high-temperature%'` のみに整理しました（結果集合は不変、期待結果も再検証済み）。

第14次レビュー対応の追加検証：(1) `scripts/audit_vocabulary.py` は「基底テーブルへ解決できなかった列参照」をデフォルト失敗にしました。CTE/subquery越しの列は投影lineage（単純列投影・`SELECT *`・エイリアス）で基底テーブルまで追跡してから語彙を突合し（CTEの外側で誤literalと比較しても検出）、計算列投影など安全に追跡できない参照は `unresolved` として失敗報告します。最終出力は `suites_audited= vocabulary_mismatch= unresolved= skipped_nontext=` で、成功条件は `vocabulary_mismatch=0` かつ `unresolved=0` です（`skipped_nontext` は非text列への等値比較で、実行時の型検査に委ねる明示スキップ）。(2) 保証範囲の分離：`fixture_guard.assert_initialized_fixture` はスキーマ同一性のみ（007マーカー＋fingerprint一致）、`validate_fixture_integrity()` は現在データの行間不変条件（006）、`audit_semantics.py` は化学・prototype意味論、gold verification はベンチマーク出力回帰をそれぞれ保証します。新設の `fixture_guard.assert_valid_fixture` はスキーマ同一性＋現在データへの006再実行を併せて検査し、`build_transfer_db.py`（転送前）・`run_gold_verification.py`・`check_expected_results.py`・`audit_semantics.py`・`audit_vocabulary.py`（main suite）が使用します（schema保存的なdata driftがtransferへ伝播する前に拒否。006で表現されない drift はfixture immutable運用が前提）。(3) `db/transfer_integrity_checks.sql` を再利用可能な `validate_transfer_integrity()` 関数に統合し、成功後に `transfer_initialization_status` マーカー（version='001'）を書き込みます。新設 `scripts/transfer_guard.assert_valid_transfer` はマーカーの存在確認に加え、監査のたびに現在状態でvalidatorを再実行します（「実行したはず」ではなく「現在も通る」を保証）。難読化DBでもマーカー表・validator関数名は難読化対象外とし、識別子rename後にvalidator本体を難読化識別子で再インストールして再実行します。(4) `db/007_initialization_marker.sql` のマーカーをimmutable seal化：既存markerのfingerprintと現在のfingerprintが不一致の場合は例外で停止し（再封印拒否、`ON CONFLICT DO NOTHING`）、スキーマを変更した新revisionはDBの作り直しでのみ初期化できます。fingerprint一致時の再実行はno-opです。

第15次レビュー対応の追加検証：(1) transfer / 難読化DBにもmainと同等のschema identity保証を導入しました。`db/transfer_integrity_checks.sql` が `compute_transfer_schema_fingerprint()`（列の正確な型・NOT NULL・default・生成式・全制約・view・trigger・関数本体をSHA-256、マーカー表と自関数は対象外）を定義し、`transfer_initialization_status` マーカーは `schema_fingerprint` と `git_commit` を記録します。`scripts/transfer_guard.assert_valid_transfer` はマーカー存在・記録fingerprintと再計算fingerprintの一致・現在データへの `validate_transfer_integrity()` 再実行の3点を検査し、初期化後の `ALTER TABLE` / `DROP CONSTRAINT` 等のスキーマドリフトをvalidatorの検査対象外の変更でも拒否します。(2) transferマーカーもanti-reseal化：既存マーカーのfingerprintと現在スキーマが不一致なら integrity SQL の再実行は例外で停止し（再封印拒否）、新しいスキーマは `scripts/build_transfer_db.py` によるDB再構築でのみ封印できます。難読化DBはrename後のスキーマを自身のfingerprintで再封印します（通常transferのfingerprintは流用しません）。(3) `validate_transfer_integrity()` はDDLのUNIQUE制約に依存しない独立のcardinality検査を持ちます：entryごとの形成エネルギー行exactly-one（GROUP BY/HAVING）、reference stateのsymbolごとexactly-one、ratioの (entry, element, site) 自然キー重複ゼロ。制約が事後削除されても現在状態の検査で検出されます。(4) `scripts/build_obfuscated_transfer_db.py` はtemplate複製の前にsource transfer DBへ `assert_valid_transfer()` を実行しfail-fastします。(5) NULL contractの対称化：transfer `oqmd_entries` の `prototype_label` / `spacegroup_number` / `crystal_system` / `cell_volume_pa` をmainのstructure表と同じくNOT NULL化（`lattice_param_a` は純元素で正当にNULL）、main `element.atomic_mass` をtransferと同じくNOT NULL化しました。(6) 保証範囲の明確化：guardが検出するのは「fingerprint対象のスキーマドリフト」と「validatorの行間不変条件に違反するデータドリフト」です。不変条件に現れない値のみを変えるschema保存的データドリフトの完全検出にはdata fingerprintが必要ですが、本パッケージでは導入せず、immutable fixture運用＋gold/semantic/vocabulary auditsを補完層とする方針です（「全data driftを検出する」とは保証しません）。

第16次レビュー対応の追加検証：(1) `scripts/build_transfer_db.py` をfail-safe順序に変更：source fixtureの検証（`assert_valid_fixture` と reference-set契約検査）を既存transfer DBのDROPより前に実行し、sourceが壊れている場合は既存transfer DBに一切触れません（難読化builderと同じ設計）。(2) transfer buildのsnapshot一貫性：source接続を REPEATABLE READ + READ ONLY に固定し、guard・契約検査・全コピーSELECTが単一snapshotを共有します（builder実行中の並行書込でも、transfer DBは「main DBのある一時点」の正確なコピーになります）。(3) `run_gold_verification.py` / `check_expected_results.py` の検証接続も REPEATABLE READ + READ ONLY のsnapshotに固定：suiteごとにguard実行と全gold query実行が同一のDB状態を参照し、「285問すべてが同じDB状態に対して一致した」ことを保証します（statement_timeout 30s は維持）。(4) `graph/join_path_generator.py` の `generate_join_clause()` にtable→aliasマップを導入し、multi-hop pathで既introduced aliasを再利用します（`calc2` のような未導入aliasをON句が参照するバグを修正。`generate_joins_for_tables()` と同じalias管理モデル）。(5) `graph/graph_builder.py` の `_SEMANTIC_JOINS` から `composition.element → element.symbol` を削除：現DDLでは物理FK（`REFERENCES element(symbol)`）でありschema introspectionが単一情報源です（semantic joinは物理FKで表現できない関係のみに限定）。(6) 期待JSONの `expected_empty` を契約化：`expected_empty: true` かつ `rows` 非空の自己矛盾JSONは malformed として検証失敗します。(7) fingerprint scopeの明確化（再掲）：schema fingerprintは「リレーショナル意味論のfingerprint」であり、インデックス・GRANT/REVOKE・ロール設定・セキュリティ構成は対象外です。またマーカーはDB owner権限による削除→再封印までは防げません（accidental drift検出が脅威モデル。owner tamperingまで証明する場合はパッケージ側期待fingerprintファイルとの三者比較が必要ですが本パッケージでは導入していません）。

意図的なnegative control：`q_expert_003`（3元系以上）・`q_expert_022`（非立方晶B2）・`q_expert_041`（GGA-PBE以外のfunctional）は本フィクスチャでは0行が正解の意図的な空結果クエリで、期待JSONに `expected_empty: true` と目的を明記しています。

弾性スカラーの意図的な非正規化重複：`elastic_tensor` のVRHスカラー（K/G/E）は `calculated_property`（EAV）にも複製されています。これはワイド表経由とEAV経由の両方のschema navigation問題を同一物理量に対して出題するためのベンチマーク上の意図的設計で、両者は生成器の単一値から書き込まれ、`validate_fixture_integrity()` が一致を検査します（フィクスチャはimmutable運用のため片側のみの事後更新は非サポート）。

NULL抜け穴の封鎖（第9次）：`phase_stability.band_gap`・`band_structure.cbm_energy / vbm_energy` を NOT NULL 化し、band gap整合検査がNULLで素通りしないようにしました。metallicity検査は `IS DISTINCT FROM` によるNULL-safe比較です。volume整合検査は、体積公式を持たない結晶系（cubic/hexagonal以外）が `conventional_cell_atoms` 付きprototypeに現れた時点で明示的に失敗します（公式追加までロード不可）。転用スキーマにも物理CHECK（原子番号1〜118・正の原子量・正の格子定数・正の原子あたり体積、いずれも有限値）を追加しています。

転用スキーマ（`db/transfer_schema.sql`）の安定性truth：転用評価DBでは `oqmd_formation_energies.on_hull` は `hull_distance <= 0.001` から導出される生成列であり、両者が矛盾する行は存在できません。転用gold SQLの安定判定は `on_hull = true`（すなわち `hull_distance <= 0.001`、本体スキーマと同一の運用定義）をtruthとします。

### 3. 環境変数の設定

```bash
cp .env.example .env
# .env を編集し OPENAI_API_KEY を設定
```

### 4. Text-to-SQL実行

```bash
python -c "
from llm.sql_generator import pipeline
result = pipeline('Niを含む安定なL1₂型化合物を形成エネルギーが低い順に出して')
print(result['sql'])
"
```

### 5. FastAPI起動（オプション）

```bash
uvicorn api.main:app --reload
# POST /query with {"query": "L1₂構造を持つ化合物を一覧にして"}
```

### 6. テスト実行

```bash
pytest tests/ -v
# 126テスト全パスを確認
```

### 7. 評価パイプライン実行（オプション）

```bash
# 100クエリ×5手法の完全評価（OpenAI API key必要、10-15分程度）
python scripts/run_full_evaluation.py
```

## Project Structure

```
l12-schema-graph-text2sql/
├── docker/              # Docker Compose設定（docker-compose.yml）
│   └── docker-compose.yml
├── db/                  # スキーマ定義・データ投入（001→006の順に自動適用）
│   ├── 001_schema.sql          # 33テーブルスキーマ（FK/UNIQUE/CHECK制約付き）
│   ├── 002_reference_data.sql  # マスタ・参照データ（元素・プロトタイプ・辞書等）
│   ├── 003_material_data.sql   # 材料エントリデータ（1,470化合物+89純元素）
│   ├── 004_views.sql           # 派生ビュー（formation_enthalpy）
│   ├── 005_roles.sql           # 読み取り専用ロール（l12_reader）
│   ├── 006_integrity_checks.sql # ロード後整合性検査（組成合計=1等、再実行可能）
│   ├── 007_initialization_marker.sql # 初期化完了マーカー
│   └── sample_queries.sql
├── ingestion/           # データ生成・正規化
│   ├── generate_extended_data.py  # 拡張データ生成
│   └── data_normalizer.py        # データ正規化
├── graph/               # Schema Graph構築・走査
│   ├── schema_parser.py        # FK関係抽出（information_schema）
│   ├── graph_builder.py        # NetworkXグラフ構築
│   ├── traversal_engine.py     # Steiner木近似走査
│   └── join_path_generator.py  # JOIN条件生成
├── llm/                 # LLM連携・条件抽出
│   ├── entity_extractor.py     # 材料用語抽出（元素、構造、安定性等）
│   ├── schema_linker.py        # テーブル・カラムマッピング
│   ├── condition_mapper.py     # SQL WHERE句生成
│   ├── sql_generator.py        # 制約付きSQL生成パイプライン
│   ├── few_shot_store.py       # Few-shot例の蓄積・検索
│   └── material_terms.yaml     # 材料用語辞書（L1₂, B2, γ'等）
├── safety/              # SQL安全検査（SQLGuard 14種検証）
│   ├── sql_validator.py      # 13種の個別検査 + 統合検証
│   ├── sql_guard.py          # ガードエントリポイント
│   └── allowed_schema.yaml   # 許可テーブル・カラム定義
├── evaluation/          # 評価パイプライン
│   ├── evaluation_dataset.jsonl # 100クエリ（Easy/Medium/Hard/VeryHard）
│   ├── gold_sql/        # 正解SQL 264件（本評価244件 + 転用20件）
│   ├── expected_results/ # 正解実行結果JSON（264件）
│   ├── metrics.py       # 評価指標（構文妥当率、実行精度等）
│   ├── run_proposed.py  # Proposed手法実行
│   ├── proposed_result.csv      # 代表ラン (= Run 2, 70.6%)
│   ├── proposed_result_run1.csv # Run 1 (72.7%)
│   ├── proposed_result_run2.csv # Run 2 (70.6%)
│   ├── proposed_result_run3.csv # Run 3 (69.4%)
│   └── baseline_result.csv     # ベースライン4手法結果
├── scripts/             # 評価・分析スクリプト
│   ├── run_full_evaluation.py      # 5手法完全評価
│   ├── run_proposed_only.py        # Proposed手法のみ再評価
│   ├── run_expert_evaluation.py    # 独立設計100件評価
│   ├── compute_paper_figures.py    # 論文数値JSON生成
│   └── validate_paper_numbers.py   # TeX数値検証
├── api/                 # FastAPI アプリケーション
│   └── main.py
├── tests/               # ユニットテスト（126件）
├── paper/               # LaTeX原稿
├── pyproject.toml       # Python依存パッケージ定義
└── .env.example         # 環境変数テンプレート
```

## Evaluation

100件の評価クエリ（Gold SQL参照テーブル数による再分類: Easy 27, Medium 28, Hard 22, Very Hard 23）で以下の5手法を比較:

| Method | LLMに渡す情報 | 構文妥当率 | 実行成功率 | 実行精度 | テーブル幻覚率 | JOIN幻覚 |
|--------|--------------|-----------|-----------|---------|--------------|---------|
| B1: LLM-only | 何も渡さない | 98% | 98% | 64.6% | 0% | 16件 |
| B2: Full Schema | 全テーブル一覧 | 94% | 94% | 68.7% | 0% | 18件 |
| B3: Rule-based | 辞書ルール（LLM不使用） | 100% | 100% | 52.8% | 0% | 0件 |
| B4: FK-list | FK関係リストのみ | 98% | 98% | 66.4% | 0% | 21件 |
| **P: Proposed** | **Steiner木で選んだサブグラフ** | **100%** | **100%** | **70.6%** (3回平均70.9%±1.7pp) | **0%** | **3件** |

### 3-run 統計の経緯

| ドラフト | ラン構成 | 平均±σ | 備考 |
|---|---|---|---|
| v1 | 69.3, 70.6, 69.4 | 69.8%±0.7pp | Run 1 (69.3%) のCSVが梱包ミスでRun 2の複製になっていた |
| v2 (現行) | 72.7, 70.6, 69.4 | 70.9%±1.7pp | 69.3ランのCSVは復元不可のため新規Run 1を独立再評価 |

v1の69.3%ランは生CSVが消失しており復元不可能。現3ランは全て独立実行（MD5一意確認済み）。
代表ラン = Run 2 (70.6%, 中央値ラン) を `proposed_result.csv` として使用。

### JOIN方向バグ修正の評価影響

v4でgraph層のJOIN方向バグ（`_edge_source`による逆方向走査時のカラム入れ替え）を修正。
結果CSVは修正前コードで生成されたものだが、評価100クエリへの影響はなし:
非対称カラム名のJOINは2件（`material_defect–element`, `application_domain`自己参照）のみで、
評価クエリでこれらのテーブルに触れるものは0件。残りは全て`entry_id=entry_id`型の対称JOIN。

### ベースラインCSVの注記

`baseline_result.csv` は `condition_mapper`/`entity_extractor` の辞書拡張
（elastic_tensor, thermal_property, magnetic_property対応）前のコードで生成。
辞書拡張後にB3 (Rule-based) を再ランすると52.8%から変動する可能性あり。

## Key Features

- **Schema Graph走査**: NetworkXによるFK関係のグラフ化、Steiner木近似による最小JOINパス探索
- **材料用語辞書**: L1₂, B2, γ', Cu₃Au型, CsCl型などの日英バイリンガル同義語辞書
- **制約付きSQL生成**: 許可テーブル・カラム・JOINのみ使用可能
- **SQLGuard 14種検証**: ブラックリスト、SELECT-only、複文検出、危険関数、テーブル/カラムホワイトリスト、JOIN整合性、LIMIT自動注入、CTE検査、型安全、トートロジー検出、サブクエリ深度制限、システムテーブル検出
- **ハイブリッドReranker**: 性能重視の3箇所再ランキング — SQL候補選択（GPT-5.5 LLM）、Few-shot例取得（Cross-Encoder ms-marco-MiniLM, ローカル<50ms）、Schema linkingテーブル並び替え（GPT-5.5 LLM）。84クエリA/Bテストで+4.9pp改善（81.5%→86.4%）
- **Rule-based fallback**: API keyなしでも動作する決定的SQL生成
- **B2対応**: CsCl型（B2）、NaCl型、NiAs型、BiF3型にも対応可能な設計

## Seed Data

デフォルト: 120件のL1₂型化合物mock data（既知11件を含む）:
Ni₃Al, Ni₃Ga, Ni₃Ge, Co₃Ti, Al₃Sc, Al₃Ti, Pt₃Al, Ir₃Nb, Co₃Al, Co₃W, Co₃Ta

OQMD拡張データ投入で最大1,470件（L12 392 + B2 636 + NaCl 355 + NiAs 74 + BiF3 13）。

## Environment Variables

| 変数名 | 説明 | デフォルト |
|--------|------|-----------|
| POSTGRES_USER | PostgreSQLユーザー | l12_user |
| POSTGRES_PASSWORD | PostgreSQLパスワード | l12_password |
| POSTGRES_DB | データベース名 | l12_materials |
| POSTGRES_HOST | ホスト | localhost |
| POSTGRES_PORT | ポート | 5432 |
| OPENAI_API_KEY | OpenAI APIキー | （要設定） |
| LLM_MODEL | 使用するLLMモデル | gpt-5.5 |
| RERANK_MODEL | Reranker用LLMモデル | gpt-5.5 |
| SQL_ROW_LIMIT | SQL結果の最大行数 | 100 |
| SQL_TIMEOUT_SECONDS | SQL実行タイムアウト | 10 |
