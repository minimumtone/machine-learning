"""
MIDI and audio generation utilities for SAX practice app.
Generates backing tracks, click sounds, and practice audio.
"""

from typing import List, Optional, Tuple
import numpy as np
from pathlib import Path
import wave
import struct


def generate_sine_wave(frequency: float, duration: float, sample_rate: int = 44100, 
                       amplitude: float = 0.5) -> np.ndarray:
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = amplitude * np.sin(2 * np.pi * frequency * t)
    return wave


def generate_click_sound(frequency: int = 1000, duration: float = 0.05, 
                        sample_rate: int = 44100, amplitude: float = 0.5) -> np.ndarray:
    click = generate_sine_wave(frequency, duration, sample_rate, amplitude)
    
    envelope_length = int(sample_rate * duration * 0.3)
    envelope = np.ones_like(click)
    envelope[:envelope_length] = np.linspace(0, 1, envelope_length)
    envelope[-envelope_length:] = np.linspace(1, 0, envelope_length)
    
    return click * envelope


def save_wav(filename: Path, audio: np.ndarray, sample_rate: int = 44100):
    filename.parent.mkdir(parents=True, exist_ok=True)
    
    audio = np.clip(audio, -1.0, 1.0)
    audio_int = (audio * 32767).astype(np.int16)
    
    with wave.open(str(filename), 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int.tobytes())


def generate_metronome_clicks(bpm: int, beats: int, subdivisions: int = 1,
                              accent_beats: Optional[List[int]] = None,
                              sample_rate: int = 44100) -> np.ndarray:
    if accent_beats is None:
        accent_beats = [1]
    
    beat_duration = 60.0 / bpm
    click_duration = 0.05
    
    total_clicks = beats * subdivisions
    total_duration = beats * beat_duration
    
    audio = np.zeros(int(total_duration * sample_rate))
    
    for i in range(total_clicks):
        beat_num = (i // subdivisions) + 1
        
        if beat_num in accent_beats and i % subdivisions == 0:
            click = generate_click_sound(1200, click_duration, sample_rate, 0.7)
        else:
            click = generate_click_sound(800, click_duration, sample_rate, 0.4)
        
        click_start = int((i / subdivisions) * beat_duration * sample_rate)
        click_end = click_start + len(click)
        
        if click_end <= len(audio):
            audio[click_start:click_end] += click
    
    return audio


def generate_chord_tone(root_freq: float, chord_intervals: List[int], 
                       duration: float, sample_rate: int = 44100) -> np.ndarray:
    audio = np.zeros(int(duration * sample_rate))
    
    for interval in chord_intervals:
        freq = root_freq * (2 ** (interval / 12))
        wave = generate_sine_wave(freq, duration, sample_rate, 0.2)
        audio += wave
    
    envelope_length = int(sample_rate * 0.1)
    envelope = np.ones_like(audio)
    envelope[:envelope_length] = np.linspace(0, 1, envelope_length)
    envelope[-envelope_length:] = np.linspace(1, 0, envelope_length)
    
    return audio * envelope


def note_to_frequency(note_name: str, octave: int = 4) -> float:
    note_map = {
        'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
        'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
        'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
    }
    
    semitones_from_a4 = (octave - 4) * 12 + note_map[note_name] - 9
    frequency = 440.0 * (2 ** (semitones_from_a4 / 12))
    
    return frequency


def generate_backing_track(chord_progression: List[Tuple[str, str]], 
                          bpm: int, beats_per_chord: int = 4,
                          sample_rate: int = 44100) -> np.ndarray:
    from .theory import CHORD_INTERVALS
    
    beat_duration = 60.0 / bpm
    chord_duration = beats_per_chord * beat_duration
    
    total_duration = len(chord_progression) * chord_duration
    audio = np.zeros(int(total_duration * sample_rate))
    
    for i, (root, chord_type) in enumerate(chord_progression):
        if chord_type not in CHORD_INTERVALS:
            chord_type = 'maj7'
        
        intervals = CHORD_INTERVALS[chord_type]
        root_freq = note_to_frequency(root, 3)
        
        chord_audio = generate_chord_tone(root_freq, intervals, chord_duration, sample_rate)
        
        start_sample = int(i * chord_duration * sample_rate)
        end_sample = start_sample + len(chord_audio)
        
        if end_sample <= len(audio):
            audio[start_sample:end_sample] += chord_audio
    
    return audio


def apply_swing(audio: np.ndarray, swing_ratio: float = 0.67, 
                bpm: int = 120, sample_rate: int = 44100) -> np.ndarray:
    beat_duration = 60.0 / bpm
    eighth_note_samples = int((beat_duration / 2) * sample_rate)
    
    return audio


def generate_practice_audio_files(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    click_high = generate_click_sound(1200, 0.05, amplitude=0.7)
    save_wav(output_dir / 'click_high.wav', click_high)
    
    click_low = generate_click_sound(800, 0.05, amplitude=0.4)
    save_wav(output_dir / 'click_low.wav', click_low)
    
    stick = generate_click_sound(2000, 0.02, amplitude=0.6)
    save_wav(output_dir / 'stick.wav', stick)
    
    print(f"Generated audio files in {output_dir}")


if __name__ == '__main__':
    output_dir = Path(__file__).parent.parent / 'audio' / 'clicks'
    generate_practice_audio_files(output_dir)
