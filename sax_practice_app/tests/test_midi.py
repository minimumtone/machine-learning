"""
Tests for MIDI and audio generation utilities
"""

import pytest
import sys
from pathlib import Path
import numpy as np
import tempfile

sys.path.append(str(Path(__file__).parent.parent))

from utils.midi import (
    generate_sine_wave, generate_click_sound, save_wav,
    generate_metronome_clicks, note_to_frequency, generate_chord_tone
)


class TestAudioGeneration:
    def test_generate_sine_wave(self):
        wave = generate_sine_wave(440, 1.0, sample_rate=44100)
        
        assert isinstance(wave, np.ndarray)
        assert len(wave) == 44100
        assert wave.min() >= -0.5
        assert wave.max() <= 0.5
    
    def test_generate_click_sound(self):
        click = generate_click_sound(1000, 0.05, sample_rate=44100)
        
        assert isinstance(click, np.ndarray)
        assert len(click) == int(44100 * 0.05)
        assert click.min() >= -0.5
        assert click.max() <= 0.5
    
    def test_save_wav(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio = generate_sine_wave(440, 0.1, sample_rate=44100)
            output_file = Path(tmpdir) / 'test.wav'
            
            save_wav(output_file, audio, sample_rate=44100)
            
            assert output_file.exists()
            assert output_file.stat().st_size > 0
    
    def test_generate_metronome_clicks(self):
        clicks = generate_metronome_clicks(
            bpm=120,
            beats=4,
            subdivisions=1,
            accent_beats=[1],
            sample_rate=44100
        )
        
        assert isinstance(clicks, np.ndarray)
        assert len(clicks) > 0
        
        expected_duration = (4 * 60) / 120
        expected_samples = int(expected_duration * 44100)
        assert abs(len(clicks) - expected_samples) < 1000


class TestNoteFrequency:
    def test_note_to_frequency_a4(self):
        freq = note_to_frequency('A', 4)
        assert abs(freq - 440.0) < 0.1
    
    def test_note_to_frequency_c4(self):
        freq = note_to_frequency('C', 4)
        assert abs(freq - 261.63) < 1.0
    
    def test_note_to_frequency_c5(self):
        freq = note_to_frequency('C', 5)
        assert abs(freq - 523.25) < 1.0
    
    def test_note_to_frequency_with_sharps(self):
        freq_sharp = note_to_frequency('C#', 4)
        freq_flat = note_to_frequency('Db', 4)
        assert abs(freq_sharp - freq_flat) < 0.1


class TestChordGeneration:
    def test_generate_chord_tone(self):
        chord = generate_chord_tone(
            root_freq=261.63,
            chord_intervals=[0, 4, 7],
            duration=1.0,
            sample_rate=44100
        )
        
        assert isinstance(chord, np.ndarray)
        assert len(chord) == 44100
        assert chord.min() >= -1.0
        assert chord.max() <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
