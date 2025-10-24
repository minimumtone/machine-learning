# SAX Practice App 🎷

A comprehensive web-based practice application for saxophone players learning JAZZ, built with Python and Streamlit.

## 📋 Overview

This application helps saxophone beginners and intermediate players learn JAZZ systematically through structured practice sessions, spaced repetition learning, and comprehensive music theory resources.

### Key Features

- **🎵 Metronome**: High-precision metronome with swing ratio, subdivisions, and customizable click patterns
- **🎼 Scale Practice**: Practice scales in all 12 keys with multiple patterns and automatic progression
- **🎹 Chord Progressions**: Learn ii-V-I and other essential jazz progressions with guide tones
- **👂 Ear Training**: Interval recognition and call & response exercises
- **📚 Music Theory**: Comprehensive theory reference covering scales, chords, and jazz concepts
- **📊 Statistics & Progress**: Track practice time, streaks, and achievement rates
- **⚙️ Settings**: Customize audio, display, and learning preferences
- **🔄 Spaced Repetition System (SRS)**: Optimal learning intervals based on SM-2 algorithm

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/minimumtone/machine-learning.git
cd machine-learning/sax_practice_app
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application

Start the Streamlit app:
```bash
streamlit run main.py
```

The app will open in your default web browser at `http://localhost:8501`

## 📁 Project Structure

```
sax_practice_app/
├── main.py                 # Main entry point
├── requirements.txt        # Python dependencies
├── README.md              # This file
│
├── pages/                 # Streamlit pages
│   ├── 1_メトロノーム.py    # Metronome page
│   ├── 2_スケール練習.py    # Scale practice page
│   ├── 3_コード進行.py      # Chord progression page
│   ├── 4_耳トレ.py         # Ear training page
│   ├── 5_楽典.py           # Music theory page
│   ├── 6_統計.py           # Statistics page
│   └── 7_設定.py           # Settings page
│
├── utils/                 # Utility modules
│   ├── __init__.py
│   ├── theory.py          # Music theory utilities
│   ├── srs.py             # Spaced repetition system
│   ├── midi.py            # Audio generation
│   └── database.py        # Data persistence
│
├── data/                  # Data files
│   ├── theory/
│   │   ├── scales.yml     # Scale definitions
│   │   └── progressions.yml # Chord progressions
│   └── user_data.json     # User data (auto-generated)
│
├── audio/                 # Audio files
│   └── clicks/            # Metronome click sounds
│       ├── click_high.wav
│       ├── click_low.wav
│       └── stick.wav
│
├── components/            # Custom components
│   └── metronome.html     # Tone.js metronome
│
└── tests/                 # Test suite
    ├── test_theory.py     # Theory module tests
    ├── test_srs.py        # SRS module tests
    ├── test_midi.py       # MIDI/audio tests
    └── test_integration.py # Integration tests
```

## 🎯 Features in Detail

### 1. Metronome (メトロノーム)

- **BPM Range**: 20-300 BPM
- **Time Signatures**: 2/4, 3/4, 4/4, 5/4, 6/8, 7/8, and more
- **Subdivisions**: Quarter notes, 8th notes, triplets, 16th notes
- **Swing Ratio**: Adjustable from 50:50 to 75:25
- **Click Patterns**: Accent on beat 1, 2&4 only, or custom patterns
- **Visual Feedback**: Flashing beat indicator

### 2. Scale Practice (スケール練習)

**Supported Scales:**
- Major (Ionian)
- Natural/Harmonic/Melodic Minor
- Church Modes (Dorian, Phrygian, Lydian, Mixolydian, Aeolian, Locrian)
- Blues Scales (Major/Minor)
- Bebop Scales (Major/Dominant)
- Whole Tone
- Diminished (Half-Whole/Whole-Half)

**Practice Patterns:**
- Ascending/Descending
- Ascending & Descending
- Thirds
- Fourths
- 1-2-3-5 pattern

**Features:**
- Practice in all 12 keys
- Adjustable BPM with metronome
- 1-2 octave range
- Progress tracking per key
- SRS integration for optimal review

### 3. Chord Progressions (コード進行)

**Progressions:**
- ii-V-I (Major/Minor)
- Blues (12-bar, variations)
- Rhythm Changes
- Common Standards (Autumn Leaves, All The Things You Are, etc.)

**Features:**
- Transpose to any key
- Guide tone display (3rd & 7th)
- Voice leading visualization
- Backing track playback (Swing/Latin/Bossa/Ballad)
- ii-V-I Master Class with exercises

### 4. Ear Training (耳トレ)

**Exercises:**
- **Interval Recognition**: Identify intervals by ear (2nd through octave)
- **Call & Response**: Listen and repeat melodic phrases
- **Chord Quality**: Identify major, minor, dominant, etc.

**Features:**
- Adjustable difficulty levels
- Immediate feedback
- Progress tracking
- SRS-based review

### 5. Music Theory (楽典)

**Topics Covered:**
- **Basics**: Note names, intervals, key signatures
- **Scales**: Construction and application
- **Chords**: Triads, 7th chords, tensions
- **Progressions**: Functional harmony, common progressions
- **Jazz Concepts**: Swing feel, guide tones, voice leading, approach notes
- **Glossary**: Essential jazz terminology

### 6. Statistics (統計)

**Metrics:**
- Total practice time
- Current streak (consecutive days)
- BPM progress over time
- Practice distribution by category
- Achievement rates

**Visualizations:**
- Calendar heatmap
- Progress curves
- Category breakdown charts
- Goal tracking

### 7. Settings (設定)

**Customization Options:**
- **Profile**: Name, instrument, experience level
- **Practice Goals**: Daily/weekly/long-term targets
- **Audio**: Metronome sound, volume, swing ratio
- **Display**: Theme, language, font size, notation style
- **Learning**: SRS strength, difficulty level, practice mode

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_theory.py -v

# Run with coverage
pytest tests/ --cov=utils --cov-report=html
```

**Test Coverage:**
- 55 tests total
- Unit tests for all utility modules
- Integration tests for cross-module functionality
- 100% pass rate

## 🎼 Music Theory Implementation

### Scales

The app supports 17 different scale types with accurate interval patterns:

```python
SCALE_INTERVALS = {
    'major': [0, 2, 4, 5, 7, 9, 11],
    'dorian': [0, 2, 3, 5, 7, 9, 10],
    'bebop_dominant': [0, 2, 4, 5, 7, 9, 10, 11],
    # ... and more
}
```

### Chord Generation

Chords are generated with proper intervals:

```python
CHORD_INTERVALS = {
    'maj7': [0, 4, 7, 11],
    'min7': [0, 3, 7, 10],
    '7': [0, 4, 7, 10],
    # ... and more
}
```

### Transposition

All scales and progressions can be transposed to any of the 12 keys automatically.

## 🔄 Spaced Repetition System (SRS)

The app uses the SM-2 algorithm for optimal learning intervals:

- **Quality Ratings**: 0-5 (0 = complete failure, 5 = perfect recall)
- **Interval Calculation**: Based on previous performance
- **Easiness Factor**: Adjusts difficulty based on success rate
- **Due Date Tracking**: Items become due for review automatically

## 📊 Data Storage

User data is stored locally in JSON format:

- **User Profile**: `data/user_data.json`
- **Practice History**: `data/practice_history.json`
- **SRS Items**: `data/srs_items.json`
- **Settings**: `data/settings.json`

## 🎨 UI/UX Design

- **Responsive Layout**: Works on desktop and mobile
- **Intuitive Navigation**: Sidebar with quick access to all features
- **Visual Feedback**: Real-time updates and animations
- **Accessibility**: Keyboard shortcuts, high contrast options
- **Internationalization**: Japanese and English support

## 🔧 Technical Stack

- **Frontend**: Streamlit 1.38+
- **Audio**: Tone.js (WebAudio API) for high-precision metronome
- **Data Processing**: NumPy for audio generation
- **Data Storage**: JSON files (SQLite optional)
- **Testing**: pytest
- **Music Theory**: Custom implementation with YAML configuration

## 📈 Performance

- **Metronome Jitter**: < ±5ms (using WebAudio)
- **Page Load Time**: < 2.5s (local)
- **Audio Generation**: Real-time for clicks, pre-generated for backing tracks
- **Data Persistence**: Automatic save on changes

## 🛠️ Development

### Adding New Scales

1. Add scale definition to `data/theory/scales.yml`:
```yaml
new_scale:
  name_en: "New Scale"
  name_ja: "新しいスケール"
  intervals: [0, 2, 3, 5, 7, 8, 10]
  description: "Description of the scale"
  difficulty: 3
```

2. The scale will automatically appear in the scale practice page.

### Adding New Progressions

1. Add progression to `data/theory/progressions.yml`:
```yaml
new_progression:
  name: "New Progression"
  chords: ["Cmaj7", "Am7", "Dm7", "G7"]
  description: "Description"
  difficulty: 2
```

2. The progression will automatically appear in the chord progression page.

### Extending the SRS System

```python
from utils.srs import SRSManager, SRSItem

# Create new SRS item
item = SRSItem(
    item_id='unique_id',
    item_type='scale_pattern',
    content={'key': 'C', 'scale_type': 'major'}
)

# Add to manager
manager = SRSManager()
manager.add_item(item)

# Update after practice
manager.update_item('unique_id', quality=4)
```

## 🐛 Troubleshooting

### Audio Not Playing

1. Check browser audio permissions
2. Verify audio files exist in `audio/clicks/`
3. Try different metronome sound in settings

### Metronome Timing Issues

1. Close other browser tabs
2. Reduce buffer size in audio settings
3. Use Chrome/Edge for best WebAudio support

### Data Not Saving

1. Check write permissions for `data/` directory
2. Verify JSON files are not corrupted
3. Try "Reset All Data" in settings

## 📝 License

This project is part of the machine-learning repository. See the main repository for license information.

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit a pull request

## 📧 Contact

For questions or feedback, please open an issue in the repository.

## 🙏 Acknowledgments

- **Tone.js**: High-precision audio timing
- **Streamlit**: Rapid web app development
- **SM-2 Algorithm**: Spaced repetition system
- **Jazz Community**: Theory and practice insights

## 🗺️ Roadmap

### v0.2 (Planned)
- [ ] Real-time pitch detection with microphone input
- [ ] MIDI keyboard support
- [ ] MusicXML import/export
- [ ] Cloud sync for practice data
- [ ] Mobile app (PWA)

### v0.3 (Future)
- [ ] AI-powered practice recommendations
- [ ] Video lessons integration
- [ ] Multiplayer practice sessions
- [ ] Advanced statistics and analytics
- [ ] Custom exercise builder

---

**Version**: 0.1  
**Last Updated**: 2025-10-24  
**Status**: Active Development
