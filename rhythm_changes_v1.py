#!/usr/bin/env python3
"""
Rhythm Changes Piano Accompaniment
===================================
Plays a stride-style jazz piano accompaniment over Rhythm Changes
(32-bar AABA form) at medium tempo. Defaults to Bb major.

  • Left hand  : bass note on beat 1 and beat 3
  • Right hand : chord voicing on beat 2 and beat 4

All configuration values can be set via command-line arguments.
Run with --help to see available options.
Press Ctrl+C at any time to stop.

Requirements: python-rtmidi  (pip install python-rtmidi)
"""

import argparse
import sys
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
DEFAULT_KEY       = 'Bb'

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
TRANSPOSE = 0          # semitone offset from Bb; computed in main()

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

SECTION_LABELS = {0: 'A1', 8: 'A2', 16: 'B', 24: 'A3'}


def build_chorus(final: bool = False) -> list:
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

def play_slot(midiout: rtmidi.MidiOut, chord_data: tuple, beat: float) -> None:
    """
    Play one half-note chord slot (2 quarter beats) in stride style:
      beat 1 (or 3) → bass note
      beat 2 (or 4) → chord voicing
    """
    bass, tones = chord_data
    ring = beat * NOTE_FILL
    gap  = beat * (1.0 - NOTE_FILL)

    # Beat 1 / 3 — bass
    note_on(midiout, bass, VEL_BASS)
    time.sleep(ring)
    note_off(midiout, bass)
    time.sleep(gap)

    # Beat 2 / 4 — chord
    for t in tones:
        note_on(midiout, t, VEL_CHORD)
    time.sleep(ring)
    for t in tones:
        note_off(midiout, t)
    time.sleep(gap)


def play_chorus(midiout: rtmidi.MidiOut, beat: float,
                chorus_num: int, final: bool = False) -> None:
    print(f"Chorus {chorus_num}:")
    bars = build_chorus(final)
    for bar_idx, bar in enumerate(bars):
        label = SECTION_LABELS.get(bar_idx)
        if label:
            print(f"  {label}")
        for slot in bar:
            play_slot(midiout, slot, beat)
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    valid_keys = ', '.join(KEY_SEMITONES.keys())
    p = argparse.ArgumentParser(
        description="Stride-style piano accompaniment over Rhythm Changes."
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
        "--key", type=str, default=DEFAULT_KEY, metavar="KEY",
        help=f"Key to play in (default: {DEFAULT_KEY}). Accepted values: {valid_keys}",
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
    if args.key not in KEY_SEMITONES:
        p.error(f"--key '{args.key}' is not recognised. Valid values: {valid_keys}")
    return args


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    global BPM, LOOPS, PORT, CHANNEL, PROGRAM, VEL_BASS, VEL_CHORD, NOTE_FILL
    global KEY, TRANSPOSE

    args = parse_args()
    BPM       = args.bpm
    LOOPS     = args.loops
    PORT      = args.port
    CHANNEL   = args.channel
    PROGRAM   = args.program
    VEL_BASS  = args.vel_bass
    KEY       = args.key
    # Transposition offset relative to Bb (10 semitones from C)
    TRANSPOSE = (KEY_SEMITONES[KEY] - KEY_SEMITONES['Bb']) % 12
    # Prefer downward shift when it is smaller (e.g. +10 → -2)
    if TRANSPOSE > 6:
        TRANSPOSE -= 12
    VEL_CHORD = args.vel_chord
    NOTE_FILL = args.note_fill

    beat   = 60.0 / BPM
    midiout = open_midi_output()
    program_change(midiout, PROGRAM)

    loop_desc = f"{LOOPS} chorus{'es' if LOOPS != 1 else ''}" if LOOPS else "∞ (Ctrl+C to stop)"
    print(f"Rhythm Changes — {KEY} major — {BPM} BPM — {loop_desc}\n")

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
        all_notes_off(midiout)
        del midiout


if __name__ == '__main__':
    main()
