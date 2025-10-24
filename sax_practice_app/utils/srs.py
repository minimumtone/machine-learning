"""
Spaced Repetition System (SRS) for practice items.
Based on SM-2 algorithm for optimal learning intervals.
"""

from datetime import datetime, timedelta, date
from typing import Dict, List, Optional
import json
from pathlib import Path


class SRSItem:
    def __init__(self, item_id: str, item_type: str, content: Dict, 
                 easiness: float = 2.5, interval_days: int = 1, 
                 repetitions: int = 0, due_date: Optional[date] = None):
        self.item_id = item_id
        self.item_type = item_type
        self.content = content
        self.easiness = easiness
        self.interval_days = interval_days
        self.repetitions = repetitions
        self.due_date = due_date or date.today()
    
    def to_dict(self) -> Dict:
        return {
            'item_id': self.item_id,
            'item_type': self.item_type,
            'content': self.content,
            'easiness': self.easiness,
            'interval_days': self.interval_days,
            'repetitions': self.repetitions,
            'due_date': self.due_date.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SRSItem':
        data['due_date'] = date.fromisoformat(data['due_date'])
        return cls(**data)


class SRSManager:
    def __init__(self, data_file: Optional[Path] = None):
        self.data_file = data_file or Path('data/srs_items.json')
        self.items: Dict[str, SRSItem] = {}
        self.load()
    
    def load(self):
        if self.data_file.exists():
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.items = {
                        item_id: SRSItem.from_dict(item_data)
                        for item_id, item_data in data.items()
                    }
            except Exception as e:
                print(f"Error loading SRS data: {e}")
                self.items = {}
    
    def save(self):
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.data_file, 'w', encoding='utf-8') as f:
            data = {
                item_id: item.to_dict()
                for item_id, item in self.items.items()
            }
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def add_item(self, item: SRSItem):
        self.items[item.item_id] = item
        self.save()
    
    def update_item(self, item_id: str, quality: int):
        if item_id not in self.items:
            return
        
        item = self.items[item_id]
        
        if quality < 3:
            item.repetitions = 0
            item.interval_days = 1
        else:
            if item.repetitions == 0:
                item.interval_days = 1
            elif item.repetitions == 1:
                item.interval_days = 6
            else:
                item.interval_days = int(item.interval_days * item.easiness)
            
            item.repetitions += 1
        
        item.easiness = max(1.3, item.easiness + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)))
        
        item.due_date = date.today() + timedelta(days=item.interval_days)
        
        self.save()
    
    def get_due_items(self, item_type: Optional[str] = None, limit: int = 10) -> List[SRSItem]:
        today = date.today()
        due_items = [
            item for item in self.items.values()
            if item.due_date <= today and (item_type is None or item.item_type == item_type)
        ]
        due_items.sort(key=lambda x: x.due_date)
        return due_items[:limit]
    
    def get_item(self, item_id: str) -> Optional[SRSItem]:
        return self.items.get(item_id)
    
    def get_all_items(self, item_type: Optional[str] = None) -> List[SRSItem]:
        if item_type is None:
            return list(self.items.values())
        return [item for item in self.items.values() if item.item_type == item_type]
    
    def get_stats(self) -> Dict:
        today = date.today()
        total = len(self.items)
        due = len([item for item in self.items.values() if item.due_date <= today])
        overdue = len([item for item in self.items.values() if item.due_date < today])
        
        return {
            'total': total,
            'due': due,
            'overdue': overdue,
            'upcoming': total - due,
        }


def create_scale_srs_items(keys: List[str], scale_types: List[str]) -> List[SRSItem]:
    items = []
    for key in keys:
        for scale_type in scale_types:
            item_id = f"scale_{key}_{scale_type}"
            content = {
                'key': key,
                'scale_type': scale_type,
                'pattern': 'ascending_descending',
                'octaves': 1,
            }
            items.append(SRSItem(item_id, 'scale_pattern', content))
    return items


def create_chord_progression_srs_items(keys: List[str], progressions: List[str]) -> List[SRSItem]:
    items = []
    for key in keys:
        for progression in progressions:
            item_id = f"progression_{key}_{progression}"
            content = {
                'key': key,
                'progression': progression,
            }
            items.append(SRSItem(item_id, 'chord_progression', content))
    return items


def create_theory_quiz_items(questions: List[Dict]) -> List[SRSItem]:
    items = []
    for i, question in enumerate(questions):
        item_id = f"theory_quiz_{i}"
        items.append(SRSItem(item_id, 'theory_quiz', question))
    return items
