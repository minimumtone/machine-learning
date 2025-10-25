"""
Tests for SRS (Spaced Repetition System) utilities
"""

import pytest
import sys
from pathlib import Path
from datetime import date, timedelta
import tempfile
import json

sys.path.append(str(Path(__file__).parent.parent))

from utils.srs import SRSItem, SRSManager, create_scale_srs_items


class TestSRSItem:
    def test_create_srs_item(self):
        item = SRSItem(
            item_id='test_1',
            item_type='scale_pattern',
            content={'key': 'C', 'scale_type': 'major'}
        )
        
        assert item.item_id == 'test_1'
        assert item.item_type == 'scale_pattern'
        assert item.content['key'] == 'C'
        assert item.easiness == 2.5
        assert item.interval_days == 1
        assert item.repetitions == 0
    
    def test_srs_item_to_dict(self):
        item = SRSItem(
            item_id='test_1',
            item_type='scale_pattern',
            content={'key': 'C', 'scale_type': 'major'}
        )
        
        data = item.to_dict()
        assert data['item_id'] == 'test_1'
        assert data['item_type'] == 'scale_pattern'
        assert 'due_date' in data
    
    def test_srs_item_from_dict(self):
        data = {
            'item_id': 'test_1',
            'item_type': 'scale_pattern',
            'content': {'key': 'C', 'scale_type': 'major'},
            'easiness': 2.5,
            'interval_days': 1,
            'repetitions': 0,
            'due_date': date.today().isoformat()
        }
        
        item = SRSItem.from_dict(data)
        assert item.item_id == 'test_1'
        assert item.item_type == 'scale_pattern'


class TestSRSManager:
    def test_create_srs_manager(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = Path(tmpdir) / 'srs_test.json'
            manager = SRSManager(data_file)
            
            assert len(manager.items) == 0
    
    def test_add_item(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = Path(tmpdir) / 'srs_test.json'
            manager = SRSManager(data_file)
            
            item = SRSItem(
                item_id='test_1',
                item_type='scale_pattern',
                content={'key': 'C', 'scale_type': 'major'}
            )
            
            manager.add_item(item)
            assert len(manager.items) == 1
            assert 'test_1' in manager.items
    
    def test_update_item_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = Path(tmpdir) / 'srs_test.json'
            manager = SRSManager(data_file)
            
            item = SRSItem(
                item_id='test_1',
                item_type='scale_pattern',
                content={'key': 'C', 'scale_type': 'major'}
            )
            
            manager.add_item(item)
            manager.update_item('test_1', quality=4)
            
            updated_item = manager.get_item('test_1')
            assert updated_item.repetitions == 1
            assert updated_item.interval_days >= 1
    
    def test_update_item_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = Path(tmpdir) / 'srs_test.json'
            manager = SRSManager(data_file)
            
            item = SRSItem(
                item_id='test_1',
                item_type='scale_pattern',
                content={'key': 'C', 'scale_type': 'major'}
            )
            
            manager.add_item(item)
            manager.update_item('test_1', quality=2)
            
            updated_item = manager.get_item('test_1')
            assert updated_item.repetitions == 0
            assert updated_item.interval_days == 1
    
    def test_get_due_items(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = Path(tmpdir) / 'srs_test.json'
            manager = SRSManager(data_file)
            
            item1 = SRSItem(
                item_id='test_1',
                item_type='scale_pattern',
                content={'key': 'C', 'scale_type': 'major'},
                due_date=date.today()
            )
            
            item2 = SRSItem(
                item_id='test_2',
                item_type='scale_pattern',
                content={'key': 'D', 'scale_type': 'major'},
                due_date=date.today() + timedelta(days=5)
            )
            
            manager.add_item(item1)
            manager.add_item(item2)
            
            due_items = manager.get_due_items()
            assert len(due_items) == 1
            assert due_items[0].item_id == 'test_1'
    
    def test_get_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = Path(tmpdir) / 'srs_test.json'
            manager = SRSManager(data_file)
            
            item1 = SRSItem(
                item_id='test_1',
                item_type='scale_pattern',
                content={'key': 'C', 'scale_type': 'major'},
                due_date=date.today()
            )
            
            item2 = SRSItem(
                item_id='test_2',
                item_type='scale_pattern',
                content={'key': 'D', 'scale_type': 'major'},
                due_date=date.today() + timedelta(days=5)
            )
            
            manager.add_item(item1)
            manager.add_item(item2)
            
            stats = manager.get_stats()
            assert stats['total'] == 2
            assert stats['due'] == 1
            assert stats['upcoming'] == 1
    
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = Path(tmpdir) / 'srs_test.json'
            
            manager1 = SRSManager(data_file)
            item = SRSItem(
                item_id='test_1',
                item_type='scale_pattern',
                content={'key': 'C', 'scale_type': 'major'}
            )
            manager1.add_item(item)
            
            manager2 = SRSManager(data_file)
            assert len(manager2.items) == 1
            assert 'test_1' in manager2.items


class TestSRSHelpers:
    def test_create_scale_srs_items(self):
        keys = ['C', 'G', 'D']
        scale_types = ['major', 'dorian']
        
        items = create_scale_srs_items(keys, scale_types)
        
        assert len(items) == 6
        assert all(isinstance(item, SRSItem) for item in items)
        assert all(item.item_type == 'scale_pattern' for item in items)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
