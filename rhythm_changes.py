#!/usr/bin/env python3
"""
Jazz Piano Accompaniment with Mood-Based Melody
================================================
Plays a stride/boogie-woogie accompaniment *and* an improvisatory melody over
Rhythm Changes, 12-bar Blues, or Coltrane Changes.

Mood (happiness / sadness / anger / fear / surprise / disgust / contempt / neutral)
shapes the melody in real time and can be changed on the fly by sending the
mood name as a plain-text TCP message (default port 3000), e.g. from a
PureData [netsend] object.

Run with --help to see all options.  Press Ctrl+C to stop.

Requirements: python-rtmidi  (pip install python-rtmidi)
"""

import argparse
import random
import re
import socket
import sys
import threading
import time
import rtmidi

# ──────────────────────────────────────────────────────────────────────────────
# Defaults (overridden by command-line arguments)
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_BPM       = 130
DEFAULT_LOOPS     = 0
DEFAULT_PORT      = 0
DEFAULT_CHANNEL   = 0
DEFAULT_PROGRAM   = 0
DEFAULT_VEL_BASS  = 82
DEFAULT_VEL_CHORD = 64
DEFAULT_NOTE_FILL = 0.88
DEFAULT_KEY             = 'Bb'
DEFAULT_STYLE           = 'stride'
DEFAULT_CHANGES         = 'rhythm'
DEFAULT_MELODY_CHANNEL  = 1      # 0-indexed → MIDI channel 2
DEFAULT_MELODY_PROGRAM  = 66     # GM 67 = Tenor Sax
DEFAULT_NET_PORT        = 3000   # TCP port for PureData netsend
DEFAULT_MOOD            = 'happiness'
DEFAULT_BASS_VARIATION  = 0.0    # 0.0 = no variation, 1.0 = maximum
DEFAULT_MOOD_TEMPO      = 0.0    # 0.0 = mood has no tempo effect, 1.0 = full effect

# Module-level names kept for use by helper functions; set in main() from args.
BPM       = DEFAULT_BPM
LOOPS     = DEFAULT_LOOPS
PORT      = DEFAULT_PORT
CHANNEL   = DEFAULT_CHANNEL
PROGRAM   = DEFAULT_PROGRAM
VEL_BASS  = DEFAULT_VEL_BASS
VEL_CHORD = DEFAULT_VEL_CHORD
NOTE_FILL = DEFAULT_NOTE_FILL
KEY       = DEFAULT_KEY
TRANSPOSE = 0          # semitone offset from changes' ref key; computed in main()
STYLE          = DEFAULT_STYLE
CHANGES        = DEFAULT_CHANGES
MELODY_CHANNEL = DEFAULT_MELODY_CHANNEL
MELODY_PROGRAM = DEFAULT_MELODY_PROGRAM
NET_PORT       = DEFAULT_NET_PORT
MOOD           = DEFAULT_MOOD
BASS_VARIATION = DEFAULT_BASS_VARIATION
MOOD_TEMPO     = DEFAULT_MOOD_TEMPO

# Natural ("home") key for each changes type; used to compute TRANSPOSE.
CHANGES_REF_KEY: dict[str, str] = {
    'rhythm':   'Bb',
    'blues':    'Bb',
    'coltrane': 'B',
}

# Accepted key names and their semitone values (C = 0)
KEY_SEMITONES: dict[str, int] = {
    'C': 0,
    'C#': 1, 'Db': 1,
    'D': 2,
    'D#': 3, 'Eb': 3,
    'E': 4,
    'F': 5,
    'F#': 6, 'Gb': 6,
    'G': 7,
    'G#': 8, 'Ab': 8,
    'A': 9,
    'A#': 10, 'Bb': 10,
    'B': 11,
}

# ──────────────────────────────────────────────────────────────────────────────
# Shared state  (accompaniment thread writes, melody thread reads)
# ──────────────────────────────────────────────────────────────────────────────

_current_chord = None           # set by play_chorus before each chord slot
_mood_lock     = threading.Lock()
_stop_event    = threading.Event()
_play_event    = threading.Event()
_play_event.set()               # playing by default; cleared on 'face_off'

# ──────────────────────────────────────────────────────────────────────────────
# Mood → melody parameters
# ──────────────────────────────────────────────────────────────────────────────
#
# octave_lo / octave_hi : melody register  (MIDI note = (oct+1)*12 + pitch_class)
# density               : probability 0–1 of playing a note (vs rest) each step
# scale                 : 'major' | 'minor' | 'chromatic'
# chord_w / scale_w / tension_w : relative note-pool weights
# vel_base / vel_spread : velocity = base + randint(-spread, spread)
# step_beats            : melody step in quarter beats (0.25 = ♬, 0.5 = ♪, 1.0 = ♩)
# fill                  : note-on fraction of the step duration
# accomp_dissonance     : probability 0–1 of adding a chromatic colour tone to the
#                         accompaniment chord voicing on any given beat

SCALE_INTERVALS: dict[str, list] = {
    'major':    [0, 2, 4, 5, 7, 9, 11],
    'minor':    [0, 2, 3, 5, 7, 8, 10],
    'chromatic': list(range(12)),
}

MOOD_PARAMS: dict[str, dict] = {
    'happiness': dict(
        octave_lo=4, octave_hi=6, density=0.75, scale='major',
        chord_w=0.55, scale_w=0.38, tension_w=0.07,
        vel_base=90, vel_spread=18, step_beats=0.5, fill=0.80,
        accomp_dissonance=0.00,
    ),
    'sadness': dict(
        octave_lo=3, octave_hi=5, density=0.40, scale='minor',
        chord_w=0.38, scale_w=0.50, tension_w=0.12,   # extra blue notes
        vel_base=74, vel_spread=14, step_beats=1.25, fill=0.95,
        accomp_dissonance=0.10,
    ),
    'anger': dict(
        octave_lo=4, octave_hi=6, density=0.88, scale='chromatic',
        chord_w=0.28, scale_w=0.28, tension_w=0.44,
        vel_base=108, vel_spread=14, step_beats=0.25, fill=0.65,
        accomp_dissonance=0.50,
    ),
    'fear': dict(
        octave_lo=3, octave_hi=6, density=0.30, scale='chromatic',
        chord_w=0.25, scale_w=0.35, tension_w=0.40,
        vel_base=48, vel_spread=38, step_beats=0.5, fill=0.55,
        accomp_dissonance=0.45,
    ),
    'surprise': dict(
        octave_lo=3, octave_hi=7, density=0.60, scale='major',
        chord_w=0.45, scale_w=0.25, tension_w=0.30,   # more unexpected colour
        vel_base=82, vel_spread=42, step_beats=0.5, fill=0.72,
        accomp_dissonance=0.20,
    ),
    'disgust': dict(
        octave_lo=3, octave_hi=5, density=0.38, scale='chromatic',
        chord_w=0.18, scale_w=0.28, tension_w=0.54,
        vel_base=76, vel_spread=12, step_beats=0.75, fill=0.60,
        accomp_dissonance=0.55,
    ),
    'contempt': dict(
        octave_lo=4, octave_hi=5, density=0.28, scale='major',
        chord_w=0.55, scale_w=0.40, tension_w=0.05,
        vel_base=62, vel_spread=6, step_beats=1.0, fill=0.38,
        accomp_dissonance=0.00,
    ),
    'neutral': dict(
        octave_lo=4, octave_hi=5, density=0.50, scale='major',
        chord_w=0.50, scale_w=0.45, tension_w=0.05,
        vel_base=80, vel_spread=12, step_beats=0.75, fill=0.78,
        accomp_dissonance=0.00,
    ),
}

# BPM multiplier per mood at MOOD_TEMPO = 1.0.
# Values reflect conventional associations between emotion and musical tempo.
_MOOD_BPM_FACTORS: dict[str, float] = {
    'happiness': 1.08,   # bright, energetic
    'sadness':   0.80,   # heavy, slow
    'anger':     1.18,   # driven, fast
    'fear':      1.06,   # nervous energy, slight rush
    'surprise':  1.10,   # sudden burst
    'disgust':   0.88,   # reluctant, dragging
    'contempt':  0.93,   # detached, unhurried
    'neutral':   1.00,   # baseline — no change
}

# ──────────────────────────────────────────────────────────────────────────────
# MIDI helpers
# ──────────────────────────────────────────────────────────────────────────────

def midi_note(name: str, octave: int) -> int:
    """Return MIDI pitch number.  midi_note('C', 4) → 60 (middle C)."""
    semis = {
        'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
        'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
        'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11,
    }
    return semis[name] + (octave + 1) * 12


def open_midi_output() -> rtmidi.MidiOut:
    midiout = rtmidi.MidiOut()
    ports = midiout.get_ports()
    if not ports:
        sys.exit("Error: no MIDI output ports found. Connect a device and retry.")
    print("Available MIDI output ports:")
    for i, p in enumerate(ports):
        print(f"  [{i}] {p}")
    midiout.open_port(PORT)
    print(f"\nOpened port {PORT}: {ports[PORT]}\n")
    return midiout


def note_on(midiout: rtmidi.MidiOut, pitch: int, velocity: int) -> None:
    midiout.send_message([0x90 | CHANNEL, pitch, velocity])


def note_off(midiout: rtmidi.MidiOut, pitch: int) -> None:
    midiout.send_message([0x80 | CHANNEL, pitch, 0])


def all_notes_off(midiout: rtmidi.MidiOut) -> None:
    """Send MIDI CC 123 (All Notes Off) as a safety cleanup."""
    midiout.send_message([0xB0 | CHANNEL, 123, 0])


def program_change(midiout: rtmidi.MidiOut, program: int) -> None:
    midiout.send_message([0xC0 | CHANNEL, program])


def transpose_chord(chord_data: tuple, semitones: int) -> tuple:
    """Return a new chord tuple shifted by the given number of semitones."""
    bass, tones = chord_data
    return (bass + semitones, [t + semitones for t in tones])


# ──────────────────────────────────────────────────────────────────────────────
# Chord voicings
# ──────────────────────────────────────────────────────────────────────────────
#
# Each chord is a tuple: (bass_pitch, [chord_tone_pitches])
#   Bass  : octave 2  (e.g. Bb2 = 46) — left hand
#   Tones : octave 3–4                 — right hand

def chord(bass_name: str, bass_oct: int, *tone_pairs) -> tuple:
    """Build chord tuple from (name, octave) pairs for the chord tones."""
    bass  = midi_note(bass_name, bass_oct)
    tones = [midi_note(tn, to) for tn, to in tone_pairs]
    return (bass, tones)


# ── A-section chords (Bb major) ───────────────────────────────────────────────

Bbmaj7 = chord('Bb', 2, ('D', 4), ('F', 4), ('A', 4))      # 3-5-7
Gm7    = chord('G',  2, ('Bb', 3), ('D', 4), ('F', 4))     # b3-5-b7
Cm7    = chord('C',  3, ('Eb', 4), ('G', 4), ('Bb', 4))    # b3-5-b7
F7     = chord('F',  2, ('A', 3),  ('C', 4), ('Eb', 4))    # 3-5-b7
Bb7    = chord('Bb', 2, ('D', 4),  ('F', 4), ('Ab', 4))    # 3-5-b7  (→ Eb)
Ebmaj7 = chord('Eb', 2, ('G', 3),  ('Bb', 3), ('D', 4))    # 3-5-7
Edim7  = chord('E',  2, ('G', 3),  ('Bb', 3), ('Db', 4))   # b3-b5-bb7  (passing)
BbF    = chord('F',  2, ('Bb', 3), ('D', 4),  ('F', 4))    # Bb/F  (pedal point)

# ── Bridge chords (secondary dominants, cycle of 5ths) ───────────────────────

D7  = chord('D',  2, ('F#', 3), ('A', 3), ('C', 4))        # V7/iii  → G7
G7  = chord('G',  2, ('B',  3), ('D', 4), ('F', 4))        # V7/ii   → C7
C7  = chord('C',  3, ('E',  4), ('G', 4), ('Bb', 4))       # V7/V    → F7
F7b = chord('F',  2, ('A',  3), ('C', 4), ('Eb', 4))       # V7/I    → Bb  (same as F7)

# ── Blues chords (Bb blues) ───────────────────────────────────────────────────────────────────────────

Eb7   = chord('Eb', 2, ('G',  3), ('Bb', 3), ('Db', 4))    # 3-5-b7

# ── Coltrane changes chords (Giant Steps in B) ─────────────────────────────────────

Bmaj7 = chord('B',  2, ('Eb', 4), ('Gb', 4), ('Bb', 4))    # 3-5-7  (D#-F#-A#)
Gmaj7 = chord('G',  2, ('B',  3), ('D',  4), ('Gb', 4))    # 3-5-7  (B-D-F#)
Am7   = chord('A',  2, ('C',  4), ('E',  4), ('G',  4))    # b3-5-b7
Fshm7 = chord('Gb', 2, ('A',  3), ('Db', 4), ('E',  4))    # b3-5-b7 (F#m7)
B7    = chord('B',  2, ('Eb', 4), ('Gb', 4), ('A',  4))    # 3-5-b7
Emaj7 = chord('E',  2, ('Ab', 3), ('B',  3), ('Eb', 4))    # 3-5-7
Cshm7 = chord('Db', 3, ('E',  4), ('Ab', 4), ('B',  4))    # b3-5-b7 (C#m7)
Fsh7  = chord('Gb', 2, ('Bb', 3), ('Db', 4), ('E',  4))    # 3-5-b7  (F#7)
Fm7   = chord('F',  2, ('Ab', 3), ('C',  4), ('Eb', 4))    # b3-5-b7

# ──────────────────────────────────────────────────────────────────────────────
# Chord progression — AABA (32 bars)
# ──────────────────────────────────────────────────────────────────────────────
# Each bar is a list of two half-note chord slots (2 quarter beats each).

A = [                                       # 8 bars
    [Bbmaj7, Gm7   ],   # 1   Bb  | Gm7
    [Cm7,    F7    ],   # 2   Cm7 | F7
    [Bbmaj7, Bb7   ],   # 3   Bb  | Bb7
    [Ebmaj7, Edim7 ],   # 4   Eb  | Edim7
    [BbF,    F7    ],   # 5   Bb/F| F7
    [Bbmaj7, Gm7   ],   # 6   Bb  | Gm7
    [Cm7,    F7    ],   # 7   Cm7 | F7
    [Bbmaj7, F7    ],   # 8   Bb  | F7   ← turnaround
]

A_end = [                                   # final A: land on tonic
    [Bbmaj7, Gm7   ],
    [Cm7,    F7    ],
    [Bbmaj7, Bb7   ],
    [Ebmaj7, Edim7 ],
    [BbF,    F7    ],
    [Bbmaj7, Gm7   ],
    [Cm7,    F7    ],
    [Bbmaj7, Bbmaj7],   # 8   hold tonic (no turnaround)
]

B = [                                       # bridge: 8 bars, each V7 lasts 2 bars
    [D7,  D7 ],   # 1 ┐ D7
    [D7,  D7 ],   # 2 ┘
    [G7,  G7 ],   # 3 ┐ G7
    [G7,  G7 ],   # 4 ┘
    [C7,  C7 ],   # 5 ┐ C7
    [C7,  C7 ],   # 6 ┘
    [F7b, F7b],   # 7 ┐ F7
    [F7b, F7b],   # 8 ┘
]

# ── Blues progression (12 bars, Bb) ────────────────────────────────────────────────────────────

BLUES = [
    [Bb7, Bb7],   #  1
    [Bb7, Bb7],   #  2
    [Bb7, Bb7],   #  3
    [Bb7, Bb7],   #  4
    [Eb7, Eb7],   #  5
    [Eb7, Eb7],   #  6
    [Bb7, Bb7],   #  7
    [Bb7, Bb7],   #  8
    [F7,  F7 ],   #  9
    [Eb7, Eb7],   # 10
    [Bb7, Bb7],   # 11
    [Bb7, F7 ],   # 12  turnaround
]

BLUES_end = BLUES[:-1] + [[Bb7, Bb7]]  # final chorus: hold tonic, no turnaround

# ── Coltrane changes progression (Giant Steps, 16 bars in B) ───────────────────────

COLTRANE = [
    [Bmaj7,  D7     ],  #  1
    [Gmaj7,  Bb7    ],  #  2
    [Ebmaj7, Ebmaj7 ],  #  3  whole bar Ebmaj7
    [Am7,    D7     ],  #  4
    [Gmaj7,  Bb7    ],  #  5
    [Ebmaj7, Ebmaj7 ],  #  6  whole bar Ebmaj7
    [Fshm7,  B7     ],  #  7
    [Emaj7,  Emaj7  ],  #  8  whole bar Emaj7
    [Cshm7,  Fsh7   ],  #  9
    [Bmaj7,  Bmaj7  ],  # 10  whole bar Bmaj7
    [Fm7,    Bb7    ],  # 11
    [Ebmaj7, Ebmaj7 ],  # 12  whole bar Ebmaj7
    [Am7,    D7     ],  # 13
    [Gmaj7,  Gmaj7  ],  # 14  whole bar Gmaj7
    [Cshm7,  Fsh7   ],  # 15
    [Bmaj7,  Bb7    ],  # 16  turnaround → top
]

COLTRANE_end = COLTRANE[:-1] + [[Bmaj7, Bmaj7]]  # final chorus: hold tonic

# ── Section labels per changes type ────────────────────────────────────────────────────────

SECTION_LABELS = {
    'rhythm':   {0: 'A1', 8: 'A2', 16: 'B', 24: 'A3'},
    'blues':    {0: 'Blues (12 bars)'},
    'coltrane': {0: 'Coltrane Changes / Giant Steps (16 bars)'},
}


def build_chorus(final: bool = False) -> list:
    if CHANGES == 'blues':
        raw = BLUES_end if final else BLUES
    elif CHANGES == 'coltrane':
        raw = COLTRANE_end if final else COLTRANE
    else:
        raw = A + A + B + (A_end if final else A)
    if TRANSPOSE == 0:
        return raw
    return [
        [transpose_chord(slot, TRANSPOSE) for slot in bar]
        for bar in raw
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Playback
# ──────────────────────────────────────────────────────────────────────────────

def _mood_beat(base_beat: float) -> float:
    """Return beat duration (seconds) adjusted for the current mood tempo scaling."""
    factor = _MOOD_BPM_FACTORS.get(MOOD, 1.0)
    effective = 1.0 + (factor - 1.0) * MOOD_TEMPO
    return base_beat / max(effective, 0.1)


# Semitone offsets from root used for accompaniment colour tones:
# b9, #9 (Hendrix chord), b5/#11 (tritone), b13
_COLOUR_INTERVALS = [1, 3, 6, 8]


def _colour_tone(bass: int, tones: list[int]) -> int | None:
    """
    Return a random chromatic colour/tension pitch in the chord's register.
    Picks one of b9, #9, b5/#11, b13 above the bass root and places it
    within the octave range spanned by the existing chord tones.
    Returns None if no suitable pitch can be placed.
    """
    if not tones:
        return None
    root_pc = bass % 12
    pc      = (root_pc + random.choice(_COLOUR_INTERVALS)) % 12
    lo, hi  = min(tones), max(tones) + 12
    start   = (lo // 12) * 12 + pc
    if start < lo:
        start += 12
    candidates = [start + 12 * k for k in range(3) if lo <= start + 12 * k <= hi]
    return random.choice(candidates) if candidates else None


def _play_bass_note(midiout: rtmidi.MidiOut, pitch: int, beat: float,
                    vel: int, *, alt: int | None = None) -> None:
    """
    Play one bass note lasting `beat` seconds, with optional variation.

    When BASS_VARIATION > 0 a random check fires; if it passes, either:
      • subdivide the beat into two eighth notes — main pitch then a chromatic
        passing note one semitone up or down (played ~18 velocity softer), or
      • substitute the root with `alt` for the full beat (e.g. the fifth or
        the sixth), giving the bass line a more varied, walking quality.
    Total elapsed time is always exactly `beat` seconds.
    """
    ring = beat * NOTE_FILL
    gap  = beat * (1.0 - NOTE_FILL)

    if BASS_VARIATION == 0.0 or random.random() > BASS_VARIATION:
        note_on(midiout, pitch, vel)
        time.sleep(ring)
        note_off(midiout, pitch)
        time.sleep(gap)
        return

    if alt is not None and random.random() < 0.5:
        # Substitute: alternate chord tone for the full beat
        note_on(midiout, alt, vel)
        time.sleep(ring)
        note_off(midiout, alt)
        time.sleep(gap)
    else:
        # Subdivide: main pitch on first eighth, chromatic passing on second
        half = beat / 2.0
        h_ring = half * NOTE_FILL
        h_gap  = half * (1.0 - NOTE_FILL)
        passing = pitch + random.choice([-1, 1])
        note_on(midiout, pitch, vel)
        time.sleep(h_ring)
        note_off(midiout, pitch)
        time.sleep(h_gap)
        note_on(midiout, passing, max(40, vel - 18))
        time.sleep(h_ring)
        note_off(midiout, passing)
        time.sleep(h_gap)


def _hv(vel: int, spread: int = 7) -> int:
    """Return vel nudged by a small random offset to humanise the accompaniment."""
    return max(1, min(127, vel + random.randint(-spread, spread)))


def play_slot_stride(midiout: rtmidi.MidiOut, chord_data: tuple, beat: float) -> None:
    """Stride style: bass on beat 1/3, chord voicing on beat 2/4."""
    beat  = _mood_beat(beat)
    bass, tones = chord_data
    fifth = bass + 7   # perfect fifth — alternate bass tone for variation
    ring  = beat * NOTE_FILL
    gap   = beat * (1.0 - NOTE_FILL)

    # Beat 1 / 3 — bass (with optional variation)
    _play_bass_note(midiout, bass, beat, _hv(VEL_BASS), alt=fifth)

    # Beat 2 / 4 — chord (+ optional colour/tension tone for dissonant moods)
    colour = (_colour_tone(bass, tones)
              if random.random() < MOOD_PARAMS[MOOD]['accomp_dissonance']
              else None)
    for t in tones:
        note_on(midiout, t, _hv(VEL_CHORD))
    if colour is not None:
        note_on(midiout, colour, max(1, _hv(VEL_CHORD - 14)))
    time.sleep(ring)
    for t in tones:
        note_off(midiout, t)
    if colour is not None:
        note_off(midiout, colour)
    time.sleep(gap)


def play_slot_boogie(midiout: rtmidi.MidiOut, chord_data: tuple, beat: float) -> None:
    """
    Boogie-woogie style: walking left-hand bass on every beat, right-hand
    chord stab on beat 2/4.

    Per half-bar slot (2 beats):
      Beat 1: bass root (octave 2)  — variation may swap to sixth or add passing note
      Beat 2: bass fifth (root + 7, octave 2) + chord stab (right hand)
    """
    beat  = _mood_beat(beat)
    bass, tones = chord_data
    fifth = bass + 7      # perfect fifth above the root
    sixth = bass + 9      # major sixth — idiomatic boogie walking-bass colour
    ring = beat * NOTE_FILL
    gap  = beat * (1.0 - NOTE_FILL)

    # Beat 1 — root in bass (with optional variation; alt = sixth for boogie colour)
    _play_bass_note(midiout, bass, beat, _hv(VEL_BASS), alt=sixth)

    # Beat 2 — fifth in bass + chord stab (+ optional colour/tension tone)
    colour = (_colour_tone(bass, tones)
              if random.random() < MOOD_PARAMS[MOOD]['accomp_dissonance']
              else None)
    note_on(midiout, fifth, _hv(VEL_BASS))
    for t in tones:
        note_on(midiout, t, _hv(VEL_CHORD))
    if colour is not None:
        note_on(midiout, colour, max(1, _hv(VEL_CHORD - 14)))
    time.sleep(ring)
    note_off(midiout, fifth)
    for t in tones:
        note_off(midiout, t)
    if colour is not None:
        note_off(midiout, colour)
    time.sleep(gap)


def play_slot(midiout: rtmidi.MidiOut, chord_data: tuple, beat: float) -> None:
    """Dispatch to the appropriate style handler."""
    if STYLE == 'boogie-woogie':
        play_slot_boogie(midiout, chord_data, beat)
    else:
        play_slot_stride(midiout, chord_data, beat)


def play_chorus(midiout: rtmidi.MidiOut, beat: float,
                chorus_num: int, final: bool = False) -> None:
    global _current_chord
    print(f"Chorus {chorus_num}:")
    bars = build_chorus(final)
    labels = SECTION_LABELS.get(CHANGES, {})
    _face_paused = False
    for bar_idx, bar in enumerate(bars):
        label = labels.get(bar_idx)
        if label:
            print(f"  {label}")
        for slot in bar:
            # Block here if mood_detector has signalled no face
            if not _play_event.is_set():
                if not _face_paused:
                    midiout.send_message([0xB0 | CHANNEL,        123, 0])
                    midiout.send_message([0xB0 | MELODY_CHANNEL, 123, 0])
                    print("[play] No face \u2014 paused.", flush=True)
                    _face_paused = True
                while not _play_event.is_set():
                    if _stop_event.is_set():
                        return
                    _play_event.wait(timeout=0.5)
                print("[play] Face detected \u2014 resuming.", flush=True)
                _face_paused = False
            with _mood_lock:
                _current_chord = slot
            play_slot(midiout, slot, beat)
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Melody thread
# ──────────────────────────────────────────────────────────────────────────────

def _melody_candidates(chord_data: tuple, params: dict) -> list:
    """Return a weighted pool of MIDI pitches for the current chord & mood."""
    bass, tones = chord_data
    root_pc     = bass % 12
    chord_pcs   = frozenset(t % 12 for t in [bass] + list(tones))
    scale_pcs   = frozenset((root_pc + i) % 12
                            for i in SCALE_INTERVALS[params['scale']])
    tension_pcs = frozenset(range(12)) - scale_pcs

    lo = (params['octave_lo'] + 1) * 12
    hi = (params['octave_hi'] + 1) * 12 + 11

    chord_notes   = [p for p in range(lo, hi + 1) if p % 12 in chord_pcs]
    scale_only    = [p for p in range(lo, hi + 1) if p % 12 in (scale_pcs - chord_pcs)]
    tension_notes = [p for p in range(lo, hi + 1) if p % 12 in tension_pcs]

    cw = max(1, round(params['chord_w']   * 20))
    sw = max(1, round(params['scale_w']   * 20))
    tw = max(1, round(params['tension_w'] * 20))
    return chord_notes * cw + scale_only * sw + tension_notes * tw


def _approach_note(chord_data: tuple, prev_note: int) -> int:
    """
    Return a chromatic approach pitch 1 semitone below the nearest chord tone
    to prev_note.  Used as the lead-in note when the chord changes (#4 / #7).
    """
    _, tones = chord_data
    target   = min(tones, key=lambda t: abs(t - prev_note)) if tones else prev_note
    approach = target - 1
    # Keep it within an octave of the previous note
    while approach < prev_note - 12:
        approach += 12
    while approach > prev_note + 12:
        approach -= 12
    return approach


def _build_phrase(chord_data: tuple, params: dict, prev_note: int | None,
                  phrase_len: int, *, step_beats: float = 1.0,
                  approach_target: int | None = None) -> list[int]:
    """
    Pre-plan a short melodic phrase with arc-shaped contour.

    - Stepwise / proximity + direction weighting for natural lines.
    - approach_target pins the first note for chord-change resolution (#4 / #7).
    - step_beats < 0.75 clamps the pool to within an octave of the current
      note at each step, preventing wide leaps on fast sub-beats (#8).
    """
    pool = _melody_candidates(chord_data, params)
    if not pool:
        return []

    ascending_first = random.random() < 0.6
    peak_idx = max(1, random.randint(max(1, phrase_len // 3),
                                     max(1, 2 * phrase_len // 3)))

    # First note: approach target > proximity bias > free choice
    if approach_target is not None:
        current = min(pool, key=lambda p: abs(p - approach_target))
    elif prev_note is not None:
        weights = [max(1, 12 - abs(p - prev_note)) for p in pool]
        current = random.choices(pool, weights=weights, k=1)[0]
    else:
        current = random.choice(pool)

    phrase = [current]
    for i in range(1, phrase_len):
        going_up = ascending_first if i <= peak_idx else not ascending_first

        # On fast notes clamp the candidate pool to within one octave (#8)
        active_pool = (
            [p for p in pool if abs(p - current) <= 12] or pool
        ) if step_beats < 0.75 else pool

        prox_w = [max(1, 7 - abs(p - current)) for p in active_pool]
        dir_w  = [3 if (going_up and p > current) or (not going_up and p < current)
                  else 1
                  for p in active_pool]
        combined_w = [pr * d for pr, d in zip(prox_w, dir_w)]
        current = random.choices(active_pool, weights=combined_w, k=1)[0]
        phrase.append(current)

    return phrase


def melody_thread_func(midiout: rtmidi.MidiOut, beat: float) -> None:
    """Continuously generate an improvisatory melody shaped by MOOD."""
    active_note     = None
    prev_note       = None          # last played pitch — seeds each new phrase
    prev_chord_data = None          # chord-change detection (#4 / #7)
    phrase: list[int] = []
    phrase_idx      = 0
    motif: list[int] = []           # last completed phrase — motivic recall (#5)
    rest_steps      = 0             # forced grouped-rest counter (#6)
    consec_rests    = 0             # consecutive rest-votes counter (#6)

    while not _stop_event.is_set():
        # Pause if face not detected
        if not _play_event.is_set():
            if active_note is not None:
                midiout.send_message([0x80 | MELODY_CHANNEL, active_note, 0])
                prev_note   = active_note
                active_note = None
            phrase = []; phrase_idx = 0
            rest_steps = 0; consec_rests = 0
            _play_event.wait(timeout=0.5)
            continue

        with _mood_lock:
            chord_data = _current_chord
            mood       = MOOD

        if chord_data is None:
            _stop_event.wait(beat * 0.25)
            continue

        params = MOOD_PARAMS[mood]
        step   = _mood_beat(beat) * params['step_beats']
        ring   = step * params['fill']
        gap    = step * (1.0 - params['fill'])

        # Release any note still ringing from the previous step
        if active_note is not None:
            midiout.send_message([0x80 | MELODY_CHANNEL, active_note, 0])
            prev_note   = active_note
            active_note = None

        # Detect chord change; compute approach note for phrase start (#4 / #7)
        chord_changed = (chord_data != prev_chord_data)
        if chord_changed:
            prev_chord_data = chord_data
            approach_target = (_approach_note(chord_data, prev_note)
                               if prev_note is not None else None)
            phrase = []; phrase_idx = 0   # fresh phrase on every new chord
        else:
            approach_target = None

        # Enforce grouped rests (#6): skip this step if in forced-rest window
        if rest_steps > 0:
            rest_steps -= 1
            _stop_event.wait(step)
            continue

        if random.random() < params['density']:
            consec_rests = 0   # reset rest counter on a note vote

            # Build a new phrase when the current one is exhausted
            if phrase_idx >= len(phrase):
                phrase_len  = random.randint(4, 8)
                sb          = params['step_beats']

                # Motivic repetition (#5): 25 % chance to replay last phrase
                if motif and random.random() < 0.25:
                    pool = _melody_candidates(chord_data, params)
                    if pool and prev_note is not None:
                        closest_start = min(pool, key=lambda p: abs(p - prev_note))
                        shift      = closest_start - motif[0]
                        transposed = [m + shift for m in motif]
                        lo, hi     = min(pool), max(pool)
                        phrase = (transposed
                                  if all(lo <= n <= hi for n in transposed)
                                  else _build_phrase(chord_data, params, prev_note,
                                                     phrase_len, step_beats=sb,
                                                     approach_target=approach_target))
                    else:
                        phrase = _build_phrase(chord_data, params, prev_note,
                                               phrase_len, step_beats=sb,
                                               approach_target=approach_target)
                else:
                    phrase = _build_phrase(chord_data, params, prev_note,
                                           phrase_len, step_beats=sb,
                                           approach_target=approach_target)
                phrase_idx = 0

            if phrase and phrase_idx < len(phrase):
                note = phrase[phrase_idx]
                phrase_idx += 1
                # Save phrase as motif when fully played (#5)
                if phrase_idx >= len(phrase):
                    motif = list(phrase)
                vel = max(1, min(127,
                    params['vel_base'] + random.randint(-params['vel_spread'],
                                                         params['vel_spread'])))
                midiout.send_message([0x90 | MELODY_CHANNEL, note, vel])
                active_note = note
                if _stop_event.wait(ring):
                    break
                midiout.send_message([0x80 | MELODY_CHANNEL, active_note, 0])
                active_note = None
                if gap > 0:
                    _stop_event.wait(gap)
            else:
                _stop_event.wait(step)
        else:
            # Rest vote: discard phrase; after 2 consecutive votes force a
            # grouped rest of 1–3 steps (#6)
            phrase = []; phrase_idx = 0
            consec_rests += 1
            if consec_rests >= 2:
                rest_steps   = random.randint(1, 3)
                consec_rests = 0
            _stop_event.wait(step)

    if active_note is not None:
        midiout.send_message([0x80 | MELODY_CHANNEL, active_note, 0])


# ──────────────────────────────────────────────────────────────────────────────
# Network listener  (PureData netsend → mood changes)
# ──────────────────────────────────────────────────────────────────────────────

_MOOD_ALIASES: dict[str, str] = {
    'happiness': 'happiness', 'happy':      'happiness',
    'sadness':   'sadness',   'sad':        'sadness',
    'anger':     'anger',     'angry':      'anger',
    'fear':      'fear',      'fearful':    'fear',
    'surprise':  'surprise',  'surprised':  'surprise',
    'disgust':   'disgust',   'disgusted':  'disgust',
    'contempt':  'contempt',
    'neutral':   'neutral',
}


def network_listener_func(port: int) -> None:
    """TCP server: accept PureData netsend connections and update MOOD."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.bind(('0.0.0.0', port))
    except OSError as exc:
        print(f"[net] Warning: could not bind on port {port}: {exc}")
        return
    server.listen(5)
    server.settimeout(1.0)
    print(f"[net] Listening for mood messages on TCP port {port}")

    def handle_client(conn: socket.socket) -> None:
        global MOOD
        buf = ''
        with conn:
            conn.settimeout(1.0)
            while not _stop_event.is_set():
                try:
                    chunk = conn.recv(256)
                    if not chunk:
                        break
                    buf += chunk.decode('ascii', errors='ignore')
                    tokens = re.split(r'[;\n\r\x00]+', buf)
                    buf = tokens[-1]
                    for token in tokens[:-1]:
                        word = token.strip().lower()
                        if word in _MOOD_ALIASES:
                            with _mood_lock:
                                MOOD = _MOOD_ALIASES[word]
                            factor = _MOOD_BPM_FACTORS.get(MOOD, 1.0)
                            eff_bpm = round(BPM * (1.0 + (factor - 1.0) * MOOD_TEMPO))
                            tempo_note = f"  (tempo → {eff_bpm} BPM)" if MOOD_TEMPO > 0 else ""
                            print(f"[net] Mood \u2192 {MOOD}{tempo_note}")
                        elif word == 'face_on':
                            _play_event.set()
                            print("[net] Face detected \u2192 resuming", flush=True)
                        elif word == 'face_off':
                            _play_event.clear()
                            print("[net] No face \u2192 pausing", flush=True)
                except socket.timeout:
                    continue
                except OSError:
                    break

    while not _stop_event.is_set():
        try:
            conn, _addr = server.accept()
            t = threading.Thread(target=handle_client, args=(conn,),
                                 daemon=True, name='net-client')
            t.start()
        except socket.timeout:
            continue
        except OSError:
            break
    server.close()


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    valid_keys = ', '.join(KEY_SEMITONES.keys())
    p = argparse.ArgumentParser(
        description="Jazz piano accompaniment + mood-driven melody over "
                    "Rhythm / Blues / Coltrane Changes."
    )
    p.add_argument(
        "--bpm", type=int, default=DEFAULT_BPM, metavar="BPM",
        help=f"Tempo in beats per minute (default: {DEFAULT_BPM})",
    )
    p.add_argument(
        "--loops", type=int, default=DEFAULT_LOOPS, metavar="N",
        help=f"Number of AABA choruses to play; 0 = loop forever (default: {DEFAULT_LOOPS})",
    )
    p.add_argument(
        "--port", type=int, default=DEFAULT_PORT, metavar="N",
        help=f"MIDI output port index (default: {DEFAULT_PORT})",
    )
    p.add_argument(
        "--channel", type=int, default=DEFAULT_CHANNEL, metavar="N",
        choices=range(0, 16),
        help=f"MIDI channel 0–15 (default: {DEFAULT_CHANNEL}, i.e. channel 1)",
    )
    p.add_argument(
        "--program", type=int, default=DEFAULT_PROGRAM, metavar="N",
        choices=range(0, 128),
        help=f"GM program number 0–127 (default: {DEFAULT_PROGRAM}, Acoustic Grand Piano)",
    )
    p.add_argument(
        "--vel-bass", type=int, default=DEFAULT_VEL_BASS, metavar="N",
        help=f"Left-hand bass velocity 0–127 (default: {DEFAULT_VEL_BASS})",
    )
    p.add_argument(
        "--vel-chord", type=int, default=DEFAULT_VEL_CHORD, metavar="N",
        help=f"Right-hand chord velocity 0–127 (default: {DEFAULT_VEL_CHORD})",
    )
    p.add_argument(
        "--note-fill", type=float, default=DEFAULT_NOTE_FILL, metavar="F",
        help=f"Note duration as fraction of one beat, 0.0–1.0 (default: {DEFAULT_NOTE_FILL})",
    )
    p.add_argument(
        "--key", type=str, default=None, metavar="KEY",
        help=f"Key to play in. Defaults to the natural key of the chosen changes "
             f"(Bb for rhythm/blues, B for coltrane). Accepted values: {valid_keys}",
    )
    p.add_argument(
        "--style", type=str, default=DEFAULT_STYLE, metavar="STYLE",
        help="Accompaniment style: 'stride' (or 's') / 'boogie-woogie' (or 'b') "
             f"(default: {DEFAULT_STYLE})",
    )
    p.add_argument(
        "--changes", type=str, default=DEFAULT_CHANGES, metavar="CHANGES",
        help="Chord changes: 'rhythm' (or 'r') / 'blues' (or 'bl') / 'coltrane' (or 'c') "
             f"(default: {DEFAULT_CHANGES})",
    )
    p.add_argument(
        "--melody-channel", type=int, default=DEFAULT_MELODY_CHANNEL, metavar="N",
        choices=range(0, 16),
        help=f"MIDI channel for melody, 0\u201315 "
             f"(default: {DEFAULT_MELODY_CHANNEL} \u2192 ch {DEFAULT_MELODY_CHANNEL + 1})",
    )
    p.add_argument(
        "--melody-program", type=int, default=DEFAULT_MELODY_PROGRAM, metavar="N",
        choices=range(0, 128),
        help=f"GM program for melody, 0\u2013127 "
             f"(default: {DEFAULT_MELODY_PROGRAM} = Tenor Sax)",
    )
    p.add_argument(
        "--net-port", type=int, default=DEFAULT_NET_PORT, metavar="PORT",
        help=f"TCP port for PureData netsend mood control "
             f"(default: {DEFAULT_NET_PORT})",
    )
    p.add_argument(
        "--mood", type=str, default=DEFAULT_MOOD, metavar="MOOD",
        help=f"Starting mood: happiness | sadness | anger | fear | "
             f"surprise | disgust | contempt | neutral  (default: {DEFAULT_MOOD})",
    )
    p.add_argument(
        "--bass-variation", type=float, default=DEFAULT_BASS_VARIATION, metavar="F",
        help=f"Bass-line variation amount 0.0–1.0 (default: {DEFAULT_BASS_VARIATION}). "
             f"Controls the probability per bass beat of either adding a chromatic "
             f"passing note (beat subdivided into two eighths) or substituting the "
             f"root with an alternate chord tone (fifth in stride, sixth in boogie).",
    )
    p.add_argument(
        "--mood-tempo", type=float, default=DEFAULT_MOOD_TEMPO, metavar="F",
        help=f"Scale of mood\u2019s effect on tempo, 0.0\u20131.0 (default: {DEFAULT_MOOD_TEMPO}). "
             f"At 0.0 mood never changes BPM; at 1.0 the full per-mood multiplier "
             f"applies (e.g. anger \u00d71.18, sadness \u00d70.80, neutral \u00d71.00).",
    )
    args = p.parse_args()
    # Validation
    if args.bpm < 20 or args.bpm > 300:
        p.error("--bpm must be between 20 and 300")
    if args.loops < 0:
        p.error("--loops must be 0 or positive")
    if not (0 <= args.vel_bass <= 127):
        p.error("--vel-bass must be between 0 and 127")
    if not (0 <= args.vel_chord <= 127):
        p.error("--vel-chord must be between 0 and 127")
    if not (0.0 < args.note_fill <= 1.0):
        p.error("--note-fill must be between 0.0 (exclusive) and 1.0 (inclusive)")
    if not (0.0 <= args.bass_variation <= 1.0):
        p.error("--bass-variation must be between 0.0 and 1.0")
    if not (0.0 <= args.mood_tempo <= 1.0):
        p.error("--mood-tempo must be between 0.0 and 1.0")
    if args.key is not None and args.key not in KEY_SEMITONES:
        p.error(f"--key '{args.key}' is not recognised. Valid values: {valid_keys}")
    # Normalise style aliases
    style_map = {'s': 'stride', 'stride': 'stride', 'b': 'boogie-woogie', 'boogie-woogie': 'boogie-woogie'}
    if args.style.lower() not in style_map:
        p.error("--style must be 'stride' (or 's') or 'boogie-woogie' (or 'b')")
    args.style = style_map[args.style.lower()]
    # Normalise changes aliases
    changes_map = {
        'r': 'rhythm', 'rhythm': 'rhythm',
        'bl': 'blues', 'blues': 'blues',
        'c': 'coltrane', 'coltrane': 'coltrane',
    }
    if args.changes.lower() not in changes_map:
        p.error("--changes must be 'rhythm' (or 'r'), 'blues' (or 'bl'), or 'coltrane' (or 'c')")
    args.changes = changes_map[args.changes.lower()]
    valid_moods = set(MOOD_PARAMS.keys())
    if args.mood.lower() not in valid_moods:
        p.error(f"--mood '{args.mood}' not recognised. "
                f"Valid: {', '.join(sorted(valid_moods))}")
    args.mood = args.mood.lower()
    return args


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    global BPM, LOOPS, PORT, CHANNEL, PROGRAM, VEL_BASS, VEL_CHORD, NOTE_FILL
    global KEY, TRANSPOSE, STYLE, CHANGES, BASS_VARIATION, MOOD_TEMPO
    global MELODY_CHANNEL, MELODY_PROGRAM, NET_PORT, MOOD

    args = parse_args()
    BPM            = args.bpm
    LOOPS          = args.loops
    PORT           = args.port
    CHANNEL        = args.channel
    PROGRAM        = args.program
    VEL_BASS       = args.vel_bass
    VEL_CHORD      = args.vel_chord
    NOTE_FILL      = args.note_fill
    STYLE          = args.style
    CHANGES        = args.changes
    MELODY_CHANNEL = args.melody_channel
    MELODY_PROGRAM = args.melody_program
    NET_PORT       = args.net_port
    MOOD           = args.mood
    BASS_VARIATION = args.bass_variation
    MOOD_TEMPO     = args.mood_tempo
    ref_key  = CHANGES_REF_KEY[CHANGES]
    KEY      = args.key if args.key is not None else ref_key
    TRANSPOSE = (KEY_SEMITONES[KEY] - KEY_SEMITONES[ref_key]) % 12
    if TRANSPOSE > 6:
        TRANSPOSE -= 12

    beat    = 60.0 / BPM
    midiout = open_midi_output()
    program_change(midiout, PROGRAM)
    midiout.send_message([0xC0 | MELODY_CHANNEL, MELODY_PROGRAM])

    changes_label = {
        'rhythm': 'Rhythm Changes', 'blues': 'Blues', 'coltrane': 'Coltrane Changes'
    }[CHANGES]
    loop_desc = f"{LOOPS} chorus{'es' if LOOPS != 1 else ''}" if LOOPS else "\u221e (Ctrl+C to stop)"
    eff_bpm = round(BPM * (1.0 + (_MOOD_BPM_FACTORS.get(MOOD, 1.0) - 1.0) * MOOD_TEMPO))
    bpm_str = f"{BPM} BPM" if eff_bpm == BPM else f"{BPM} BPM \u2192 {eff_bpm} BPM ({MOOD})"
    print(f"{changes_label} \u2014 {KEY} \u2014 {bpm_str} \u2014 {STYLE} \u2014 {loop_desc}")
    print(f"Mood: {MOOD}  |  melody ch: {MELODY_CHANNEL + 1}  |  accompaniment ch: {CHANNEL + 1}\n")

    net_thread = threading.Thread(
        target=network_listener_func, args=(NET_PORT,),
        daemon=True, name='net-listener',
    )
    net_thread.start()

    mel_thread = threading.Thread(
        target=melody_thread_func, args=(midiout, beat),
        daemon=True, name='melody',
    )
    mel_thread.start()

    chorus = 0
    try:
        while True:
            chorus += 1
            is_final = LOOPS > 0 and chorus == LOOPS
            play_chorus(midiout, beat, chorus, final=is_final)
            if is_final:
                break
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        _stop_event.set()
        mel_thread.join(timeout=2.0)
        midiout.send_message([0xB0 | CHANNEL,        123, 0])
        midiout.send_message([0xB0 | MELODY_CHANNEL, 123, 0])
        del midiout


if __name__ == '__main__':
    main()
