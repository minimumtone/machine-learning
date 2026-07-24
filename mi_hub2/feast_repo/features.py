"""Feast feature 定義(Phase 4)。

前提: datastore の "hea_features" を 1 ファイルに統合した parquet を
feast_repo/data/hea_features.parquet に置く(下のスクリプト参照)。
Feast の FileSource は event_timestamp 列を要求するため、
provenance の created_at をそのまま流用する。

準備:
    python -c "
    from mi_hub import datastore as ds
    import pandas as pd
    df = ds.load('hea_features')
    df['created_at'] = pd.to_datetime(df['created_at'])
    df.to_parquet('feast_repo/data/hea_features.parquet', index=False)"
    cd feast_repo && feast apply
"""
from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64

alloy = Entity(name="alloy_id", join_keys=["alloy_id"])

hea_source = FileSource(
    path="data/hea_features.parquet",
    timestamp_field="created_at",
)

hea_features = FeatureView(
    name="hea_features",
    entities=[alloy],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="vec", dtype=Float64),
        Field(name="delta_r", dtype=Float64),
        Field(name="dH_mix", dtype=Float64),
        Field(name="omega_sf_mean", dtype=Float64),
    ],
    source=hea_source,
)
