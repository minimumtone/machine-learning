"""Tests for graph.schema_parser (unit tests without DB)."""
from graph.schema_parser import ColumnMetadata, ForeignKeyMetadata, iter_columns


def test_column_metadata_creation():
    col = ColumnMetadata(
        table_name="material_entry",
        column_name="entry_id",
        data_type="text",
        is_nullable=False,
        is_primary_key=True,
    )
    assert col.table_name == "material_entry"
    assert col.is_primary_key is True


def test_foreign_key_metadata():
    fk = ForeignKeyMetadata(
        source_table="composition",
        source_column="entry_id",
        target_table="material_entry",
        target_column="entry_id",
    )
    assert fk.source_table == "composition"
    assert fk.target_column == "entry_id"


def test_iter_columns():
    cols = {
        "t1": [
            ColumnMetadata("t1", "a", "text", False, True),
            ColumnMetadata("t1", "b", "int", True, False),
        ],
        "t2": [
            ColumnMetadata("t2", "c", "text", False, True),
        ],
    }
    result = list(iter_columns(cols))
    assert len(result) == 3
