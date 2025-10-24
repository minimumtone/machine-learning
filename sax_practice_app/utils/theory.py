"""
Music theory utilities for SAX practice app.
Handles scales, chords, intervals, and music theory calculations.
"""

from typing import List, Dict, Tuple, Optional
from enum import Enum
import yaml
from pathlib import Path


class Note(Enum):
    C = 0
    Db = 1
    D = 2
    Eb = 3
    E = 4
    F = 5
    Gb = 6
    G = 7
    Ab = 8
    A = 9
    Bb = 10
    B = 11


NOTE_NAMES = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
NOTE_NAMES_SHARP = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


SCALE_INTERVALS = {
    'major': [0, 2, 4, 5, 7, 9, 11],
    'natural_minor': [0, 2, 3, 5, 7, 8, 10],
    'harmonic_minor': [0, 2, 3, 5, 7, 8, 11],
    'melodic_minor': [0, 2, 3, 5, 7, 9, 11],
    'major_blues': [0, 2, 3, 4, 7, 9],
    'minor_blues': [0, 3, 5, 6, 7, 10],
    'bebop_major': [0, 2, 4, 5, 7, 8, 9, 11],
    'bebop_dominant': [0, 2, 4, 5, 7, 9, 10, 11],
    'whole_tone': [0, 2, 4, 6, 8, 10],
    'diminished_half_whole': [0, 1, 3, 4, 6, 7, 9, 10],
    'diminished_whole_half': [0, 2, 3, 5, 6, 8, 9, 11],
    'ionian': [0, 2, 4, 5, 7, 9, 11],
    'dorian': [0, 2, 3, 5, 7, 9, 10],
    'phrygian': [0, 1, 3, 5, 7, 8, 10],
    'lydian': [0, 2, 4, 6, 7, 9, 11],
    'mixolydian': [0, 2, 4, 5, 7, 9, 10],
    'aeolian': [0, 2, 3, 5, 7, 8, 10],
    'locrian': [0, 1, 3, 5, 6, 8, 10],
}


CHORD_INTERVALS = {
    'maj': [0, 4, 7],
    'min': [0, 3, 7],
    'dim': [0, 3, 6],
    'aug': [0, 4, 8],
    'maj7': [0, 4, 7, 11],
    'min7': [0, 3, 7, 10],
    '7': [0, 4, 7, 10],
    'min7b5': [0, 3, 6, 10],
    'dim7': [0, 3, 6, 9],
    'maj9': [0, 4, 7, 11, 14],
    'min9': [0, 3, 7, 10, 14],
    '9': [0, 4, 7, 10, 14],
    '7#11': [0, 4, 7, 10, 18],
    '7b9': [0, 4, 7, 10, 13],
}


CHORD_SCALE_MAP = {
    'maj7': ['ionian', 'lydian'],
    'min7': ['dorian', 'aeolian', 'phrygian'],
    '7': ['mixolydian', 'bebop_dominant', 'whole_tone'],
    'min7b5': ['locrian', 'diminished_half_whole'],
    'dim7': ['diminished_whole_half'],
    '7alt': ['diminished_half_whole'],
}


def get_note_number(note_name: str) -> int:
    note_name = note_name.replace('#', 'b')
    if note_name in NOTE_NAMES:
        return NOTE_NAMES.index(note_name)
    elif note_name in NOTE_NAMES_SHARP:
        return NOTE_NAMES_SHARP.index(note_name)
    raise ValueError(f"Invalid note name: {note_name}")


def get_note_name(note_number: int, use_sharps: bool = False) -> str:
    note_number = note_number % 12
    if use_sharps:
        return NOTE_NAMES_SHARP[note_number]
    return NOTE_NAMES[note_number]


def generate_scale(root: str, scale_type: str, octaves: int = 1, use_sharps: bool = False) -> List[str]:
    if scale_type not in SCALE_INTERVALS:
        raise ValueError(f"Unknown scale type: {scale_type}")
    
    root_num = get_note_number(root)
    intervals = SCALE_INTERVALS[scale_type]
    
    notes = []
    for octave in range(octaves):
        for interval in intervals:
            note_num = (root_num + interval + octave * 12) % 12
            notes.append(get_note_name(note_num, use_sharps))
    
    notes.append(get_note_name((root_num + octaves * 12) % 12, use_sharps))
    
    return notes


def generate_scale_pattern(root: str, scale_type: str, pattern: str = 'ascending', 
                          octaves: int = 1, use_sharps: bool = False) -> List[str]:
    base_scale = generate_scale(root, scale_type, octaves, use_sharps)
    
    if pattern == 'ascending':
        return base_scale
    elif pattern == 'descending':
        return list(reversed(base_scale))
    elif pattern == 'ascending_descending':
        return base_scale + list(reversed(base_scale[:-1]))
    elif pattern == 'thirds':
        result = []
        for i in range(len(base_scale) - 2):
            result.extend([base_scale[i], base_scale[i+2]])
        return result
    elif pattern == 'fourths':
        result = []
        for i in range(len(base_scale) - 3):
            result.extend([base_scale[i], base_scale[i+3]])
        return result
    elif pattern == '1235':
        result = []
        for i in range(len(base_scale) - 4):
            result.extend([base_scale[i], base_scale[i+1], base_scale[i+2], base_scale[i+4]])
        return result
    else:
        return base_scale


def generate_chord(root: str, chord_type: str, use_sharps: bool = False) -> List[str]:
    if chord_type not in CHORD_INTERVALS:
        raise ValueError(f"Unknown chord type: {chord_type}")
    
    root_num = get_note_number(root)
    intervals = CHORD_INTERVALS[chord_type]
    
    notes = []
    for interval in intervals:
        note_num = (root_num + interval) % 12
        notes.append(get_note_name(note_num, use_sharps))
    
    return notes


def get_guide_tones(root: str, chord_type: str, use_sharps: bool = False) -> List[str]:
    chord_notes = generate_chord(root, chord_type, use_sharps)
    
    if len(chord_notes) >= 4:
        return [chord_notes[2], chord_notes[3]]
    elif len(chord_notes) >= 3:
        return [chord_notes[2]]
    return []


def transpose_note(note: str, semitones: int, use_sharps: bool = False) -> str:
    note_num = get_note_number(note)
    new_note_num = (note_num + semitones) % 12
    return get_note_name(new_note_num, use_sharps)


def transpose_progression(progression: List[str], semitones: int, use_sharps: bool = False) -> List[str]:
    result = []
    for chord in progression:
        root = chord.rstrip('0123456789abdefgilmnorstuvy#b+-')
        suffix = chord[len(root):]
        new_root = transpose_note(root, semitones, use_sharps)
        result.append(new_root + suffix)
    return result


def get_interval_name(semitones: int) -> str:
    interval_names = {
        0: 'Unison',
        1: 'Minor 2nd',
        2: 'Major 2nd',
        3: 'Minor 3rd',
        4: 'Major 3rd',
        5: 'Perfect 4th',
        6: 'Tritone',
        7: 'Perfect 5th',
        8: 'Minor 6th',
        9: 'Major 6th',
        10: 'Minor 7th',
        11: 'Major 7th',
        12: 'Octave',
    }
    return interval_names.get(semitones % 12, f'{semitones} semitones')


def get_all_keys() -> List[str]:
    return NOTE_NAMES[:12]


def get_chord_scale_options(chord_type: str) -> List[str]:
    return CHORD_SCALE_MAP.get(chord_type, ['major'])


def roman_to_chord(roman: str, key: str) -> str:
    roman_map = {
        'I': (0, 'maj7'),
        'II': (2, 'min7'),
        'III': (4, 'min7'),
        'IV': (5, 'maj7'),
        'V': (7, '7'),
        'VI': (9, 'min7'),
        'VII': (11, 'min7b5'),
        'i': (0, 'min7'),
        'ii': (2, 'min7'),
        'iii': (4, 'min7'),
        'iv': (5, 'min7'),
        'v': (7, 'min7'),
        'vi': (9, 'min7'),
        'vii': (11, 'min7b5'),
    }
    
    if roman in roman_map:
        interval, chord_type = roman_map[roman]
        root = transpose_note(key, interval)
        return f"{root}{chord_type}"
    
    return roman


def get_ii_v_i_progression(key: str) -> List[str]:
    ii = roman_to_chord('ii', key)
    v = roman_to_chord('V', key)
    i = roman_to_chord('I', key)
    return [ii, v, i]
