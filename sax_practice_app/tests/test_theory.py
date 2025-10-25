"""
Tests for music theory utilities
"""

import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from utils.theory import (
    get_note_number, get_note_name, generate_scale, generate_scale_pattern,
    generate_chord, get_guide_tones, transpose_note, transpose_progression,
    get_interval_name, get_all_keys, get_ii_v_i_progression, roman_to_chord
)


class TestNoteOperations:
    def test_get_note_number(self):
        assert get_note_number('C') == 0
        assert get_note_number('D') == 2
        assert get_note_number('E') == 4
        assert get_note_number('G') == 7
        assert get_note_number('A') == 9
        assert get_note_number('B') == 11
    
    def test_get_note_number_with_sharps_flats(self):
        assert get_note_number('Db') == 1
        assert get_note_number('Eb') == 3
        assert get_note_number('Gb') == 6
        assert get_note_number('Ab') == 8
    
    def test_get_note_name(self):
        assert get_note_name(0) == 'C'
        assert get_note_name(2) == 'D'
        assert get_note_name(4) == 'E'
        assert get_note_name(7) == 'G'
    
    def test_get_note_name_with_sharps(self):
        assert get_note_name(1, use_sharps=True) == 'C#'
        assert get_note_name(6, use_sharps=True) == 'F#'
    
    def test_transpose_note(self):
        assert transpose_note('C', 2) == 'D'
        assert transpose_note('C', 7) == 'G'
        assert transpose_note('G', 5) == 'C'
        assert transpose_note('A', 3) == 'C'


class TestScales:
    def test_generate_major_scale(self):
        scale = generate_scale('C', 'major', octaves=1)
        assert len(scale) == 8
        assert scale[0] == 'C'
        assert scale[-1] == 'C'
    
    def test_generate_dorian_scale(self):
        scale = generate_scale('D', 'dorian', octaves=1)
        assert len(scale) == 8
        assert scale[0] == 'D'
    
    def test_generate_scale_two_octaves(self):
        scale = generate_scale('C', 'major', octaves=2)
        assert len(scale) == 15
    
    def test_generate_scale_pattern_ascending(self):
        pattern = generate_scale_pattern('C', 'major', 'ascending', octaves=1)
        assert pattern[0] == 'C'
        assert len(pattern) == 8
    
    def test_generate_scale_pattern_descending(self):
        pattern = generate_scale_pattern('C', 'major', 'descending', octaves=1)
        assert pattern[0] == 'C'
        assert pattern[-1] == 'C'
    
    def test_generate_scale_pattern_ascending_descending(self):
        pattern = generate_scale_pattern('C', 'major', 'ascending_descending', octaves=1)
        assert pattern[0] == 'C'
        assert pattern[-1] == 'C'
        assert len(pattern) == 15
    
    def test_all_keys(self):
        keys = get_all_keys()
        assert len(keys) == 12
        assert 'C' in keys
        assert 'G' in keys


class TestChords:
    def test_generate_major_chord(self):
        chord = generate_chord('C', 'maj')
        assert len(chord) == 3
        assert chord[0] == 'C'
        assert chord[1] == 'E'
        assert chord[2] == 'G'
    
    def test_generate_minor_chord(self):
        chord = generate_chord('C', 'min')
        assert len(chord) == 3
        assert chord[0] == 'C'
        assert chord[1] == 'Eb'
        assert chord[2] == 'G'
    
    def test_generate_maj7_chord(self):
        chord = generate_chord('C', 'maj7')
        assert len(chord) == 4
        assert chord[0] == 'C'
        assert chord[3] == 'B'
    
    def test_generate_min7_chord(self):
        chord = generate_chord('C', 'min7')
        assert len(chord) == 4
        assert chord[0] == 'C'
        assert chord[3] == 'Bb'
    
    def test_generate_dominant7_chord(self):
        chord = generate_chord('G', '7')
        assert len(chord) == 4
        assert chord[0] == 'G'
        assert chord[3] == 'F'
    
    def test_get_guide_tones_maj7(self):
        guide_tones = get_guide_tones('C', 'maj7')
        assert len(guide_tones) == 2
        assert 'G' in guide_tones
        assert 'B' in guide_tones
    
    def test_get_guide_tones_min7(self):
        guide_tones = get_guide_tones('D', 'min7')
        assert len(guide_tones) == 2
        assert 'A' in guide_tones
        assert 'C' in guide_tones


class TestProgressions:
    def test_transpose_progression(self):
        progression = ['C', 'D', 'G']
        transposed = transpose_progression(progression, 2)
        assert transposed[0] == 'D'
        assert transposed[1] == 'E'
        assert transposed[2] == 'A'
    
    def test_roman_to_chord(self):
        chord = roman_to_chord('I', 'C')
        assert chord == 'Cmaj7'
        
        chord = roman_to_chord('ii', 'C')
        assert chord == 'Dmin7'
        
        chord = roman_to_chord('V', 'C')
        assert chord == 'G7'
    
    def test_get_ii_v_i_progression(self):
        progression = get_ii_v_i_progression('C')
        assert len(progression) == 3
        assert progression[0] == 'Dmin7'
        assert progression[1] == 'G7'
        assert progression[2] == 'Cmaj7'
    
    def test_get_ii_v_i_all_keys(self):
        keys = get_all_keys()
        for key in keys:
            progression = get_ii_v_i_progression(key)
            assert len(progression) == 3


class TestIntervals:
    def test_get_interval_name(self):
        assert get_interval_name(0) == 'Unison'
        assert get_interval_name(1) == 'Minor 2nd'
        assert get_interval_name(2) == 'Major 2nd'
        assert get_interval_name(3) == 'Minor 3rd'
        assert get_interval_name(4) == 'Major 3rd'
        assert get_interval_name(5) == 'Perfect 4th'
        assert get_interval_name(7) == 'Perfect 5th'
        assert get_interval_name(12) == 'Unison'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
