"""
Database utilities for SAX practice app.
Simple JSON-based storage for practice records and user data.
"""

import json
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import List, Dict, Optional


class PracticeDatabase:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.user_file = self.data_dir / 'user_data.json'
        self.practice_file = self.data_dir / 'practice_history.json'
        self.srs_file = self.data_dir / 'srs_items.json'
    
    def load_user_data(self) -> Dict:
        if self.user_file.exists():
            try:
                with open(self.user_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading user data: {e}")
        
        return {
            'user_name': 'Player',
            'target_bpm': 120,
            'daily_minutes': 30,
            'swing_ratio': 0.67,
            'current_streak': 0,
            'total_practice_time': 0,
            'created_at': datetime.now().isoformat()
        }
    
    def save_user_data(self, data: Dict):
        with open(self.user_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load_practice_history(self) -> List[Dict]:
        if self.practice_file.exists():
            try:
                with open(self.practice_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading practice history: {e}")
        
        return []
    
    def save_practice_history(self, history: List[Dict]):
        with open(self.practice_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    
    def add_practice_record(self, record: Dict):
        history = self.load_practice_history()
        
        if 'timestamp' not in record:
            record['timestamp'] = datetime.now().isoformat()
        if 'date' not in record:
            record['date'] = str(date.today())
        
        history.append(record)
        self.save_practice_history(history)
    
    def get_records_by_date(self, date_str: str) -> List[Dict]:
        history = self.load_practice_history()
        return [r for r in history if r.get('date') == date_str]
    
    def get_records_by_category(self, category: str) -> List[Dict]:
        history = self.load_practice_history()
        return [r for r in history if r.get('category') == category]
    
    def get_recent_records(self, limit: int = 10) -> List[Dict]:
        history = self.load_practice_history()
        sorted_history = sorted(history, key=lambda x: x.get('timestamp', ''), reverse=True)
        return sorted_history[:limit]
    
    def calculate_streak(self) -> int:
        history = self.load_practice_history()
        
        if not history:
            return 0
        
        dates = sorted(set(r.get('date') for r in history if r.get('date')))
        
        if not dates:
            return 0
        
        today = date.today()
        streak = 0
        
        for i in range(len(dates)):
            check_date = today - timedelta(days=i)
            if str(check_date) in dates:
                streak += 1
            else:
                break
        
        return streak
    
    def export_to_csv(self, output_file: Path):
        import csv
        
        history = self.load_practice_history()
        
        if not history:
            return
        
        keys = set()
        for record in history:
            keys.update(record.keys())
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=sorted(keys))
            writer.writeheader()
            writer.writerows(history)
    
    def clear_all_data(self):
        if self.user_file.exists():
            self.user_file.unlink()
        if self.practice_file.exists():
            self.practice_file.unlink()
        if self.srs_file.exists():
            self.srs_file.unlink()


def initialize_database(data_dir: Path) -> PracticeDatabase:
    db = PracticeDatabase(data_dir)
    
    user_data = db.load_user_data()
    db.save_user_data(user_data)
    
    return db


if __name__ == '__main__':
    from datetime import timedelta
    
    data_dir = Path(__file__).parent.parent / 'data'
    db = initialize_database(data_dir)
    
    print("Database initialized successfully!")
    print(f"Data directory: {data_dir}")
    print(f"User file: {db.user_file}")
    print(f"Practice file: {db.practice_file}")
