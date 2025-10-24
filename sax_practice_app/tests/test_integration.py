"""
Integration tests for SAX practice app.
Tests the interaction between different modules and components.
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

from utils.theory import (
    generate_scale, generate_chord, get_guide_tones,
    transpose_progression, get_ii_v_i_progression
)
from utils.srs import SRSManager, SRSItem
from utils.midi import generate_sine_wave, generate_click_sound, note_to_frequency
from utils.database import PracticeDatabase


class TestScaleToSRSIntegration:
    """Test integration between scale generation and SRS system."""
    
    def test_scale_practice_workflow(self):
        """Test complete workflow: generate scale -> practice -> record in SRS."""
        with tempfile.TemporaryDirectory() as tmpdir:
            srs_file = Path(tmpdir) / 'srs.json'
            manager = SRSManager(srs_file)
            
            scale = generate_scale('C', 'major', octaves=1)
            assert len(scale) == 8
            
            item = SRSItem(
                item_id='scale_C_major',
                item_type='scale_pattern',
                content={
                    'key': 'C',
                    'scale_type': 'major',
                    'notes': scale
                }
            )
            
            manager.add_item(item)
            assert len(manager.items) == 1
            
            manager.update_item('scale_C_major', quality=4)
            updated = manager.get_item('scale_C_major')
            assert updated.repetitions == 1
            assert updated.easiness >= 2.5


class TestChordProgressionIntegration:
    """Test integration between chord progressions and theory."""
    
    def test_ii_v_i_all_keys(self):
        """Test ii-V-I progression generation in all 12 keys."""
        keys = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
        
        for key in keys:
            progression = get_ii_v_i_progression(key)
            assert len(progression) == 3
            
            assert 'min7' in progression[0]
            assert '7' in progression[1]
            assert 'maj7' in progression[2]
    
    def test_guide_tones_in_progression(self):
        """Test extracting guide tones from a progression."""
        progression = get_ii_v_i_progression('C')
        
        guide_tones = []
        for chord in progression:
            if 'min7' in chord:
                root = chord.replace('min7', '')
                chord_type = 'min7'
            elif 'maj7' in chord:
                root = chord.replace('maj7', '')
                chord_type = 'maj7'
            elif '7' in chord:
                root = chord.replace('7', '')
                chord_type = '7'
            else:
                continue
            
            tones = get_guide_tones(root, chord_type)
            guide_tones.append(tones)
        
        assert len(guide_tones) == 3
        for tones in guide_tones:
            assert len(tones) == 2


class TestPracticeDatabaseIntegration:
    """Test integration between practice records and database."""
    
    def test_practice_session_workflow(self):
        """Test complete practice session workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_dir = Path(tmpdir)
            db = PracticeDatabase(db_dir)
            
            db.add_practice_record({
                'category': 'scale',
                'duration_minutes': 15,
                'bpm': 120,
                'key': 'C',
                'notes': 'Practiced major scale in all patterns'
            })
            
            db.add_practice_record({
                'category': 'chord',
                'duration_minutes': 20,
                'bpm': 100,
                'key': 'F',
                'notes': 'ii-V-I progression practice'
            })
            
            today = str(datetime.now().date())
            records = db.get_records_by_date(today)
            assert len(records) == 2
            
            total_time = sum(r['duration_minutes'] for r in records)
            assert total_time == 35
    
    def test_streak_calculation(self):
        """Test practice streak calculation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_dir = Path(tmpdir)
            db = PracticeDatabase(db_dir)
            
            for i in range(5):
                date = datetime.now().date() - timedelta(days=i)
                db.add_practice_record({
                    'category': 'scale',
                    'duration_minutes': 10,
                    'date': date.isoformat()
                })
            
            streak = db.calculate_streak()
            assert streak >= 5


class TestAudioGenerationIntegration:
    """Test integration of audio generation with music theory."""
    
    def test_scale_audio_generation(self):
        """Test generating audio for a scale."""
        scale = generate_scale('C', 'major', octaves=1)
        
        for note in scale:
            freq = note_to_frequency(note, octave=4)
            assert freq > 0
            
            audio = generate_sine_wave(freq, duration=0.1)
            assert len(audio) > 0
    
    def test_chord_audio_generation(self):
        """Test generating audio for a chord."""
        chord = generate_chord('C', 'maj7')
        
        for note in chord:
            freq = note_to_frequency(note, octave=4)
            assert freq > 0
            
            audio = generate_sine_wave(freq, duration=0.5)
            assert len(audio) > 0
    
    def test_metronome_click_generation(self):
        """Test generating metronome clicks."""
        click = generate_click_sound(frequency=1000, duration=0.05)
        assert len(click) > 0


class TestSRSWithDatabaseIntegration:
    """Test integration between SRS and practice database."""
    
    def test_srs_practice_tracking(self):
        """Test tracking SRS items with practice database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            srs_file = Path(tmpdir) / 'srs.json'
            db_dir = Path(tmpdir)
            
            manager = SRSManager(srs_file)
            db = PracticeDatabase(db_dir)
            
            scales = [
                ('C', 'major'),
                ('G', 'major'),
                ('D', 'dorian'),
            ]
            
            for key, scale_type in scales:
                item = SRSItem(
                    item_id=f'scale_{key}_{scale_type}',
                    item_type='scale_pattern',
                    content={'key': key, 'scale_type': scale_type}
                )
                manager.add_item(item)
            
            for key, scale_type in scales:
                item_id = f'scale_{key}_{scale_type}'
                
                manager.update_item(item_id, quality=4)
                
                db.add_practice_record({
                    'category': 'scale',
                    'duration_minutes': 10,
                    'key': key,
                    'notes': f'{scale_type} scale practice'
                })
            
            stats = manager.get_stats()
            assert stats['total'] == 3
            
            records = db.get_records_by_category('scale')
            assert len(records) == 3


class TestTranspositionIntegration:
    """Test transposition across different components."""
    
    def test_transpose_progression_with_guide_tones(self):
        """Test transposing a progression and extracting guide tones."""
        original = ['C', 'F', 'G']
        
        for semitones in range(12):
            transposed = transpose_progression(original, semitones)
            assert len(transposed) == 3
            
            for chord in transposed:
                assert len(chord) > 0
    
    def test_scale_transposition(self):
        """Test generating scales in all 12 keys."""
        keys = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
        
        for key in keys:
            scale = generate_scale(key, 'major', octaves=1)
            assert len(scale) == 8
            assert scale[0] == key


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
