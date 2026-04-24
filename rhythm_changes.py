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
    ),
    'sadness': dict(
        octave_lo=3, octave_hi=5, density=0.40, scale='minor',
        chord_w=0.38, scale_w=0.54, tension_w=0.08,
        vel_base=52, vel_spread=12, step_beats=1.25, fill=0.95,
    ),
    'anger': dict(
        octave_lo=4, octave_hi=6, density=0.88, scale='chromatic',
        chord_w=0.28, scale_w=0.28, tension_w=0.44,
        vel_base=108, vel_spread=14, step_beats=0.25, fill=0.65,
    ),
    'fear': dict(
        octave_lo=3, octave_hi=6, density=0.30, scale='chromatic',
        chord_w=0.25, scale_w=0.35, tension_w=0.40,
        vel_base=48, vel_spread=38, step_beats=0.5, fill=0.55,
    ),
    'surprise': dict(
        octave_lo=3, octave_hi=7, density=0.60, scale='major',
        chord_w=0.45, scale_w=0.30, tension_w=0.25,
        vel_base=82, vel_spread=42, step_beats=0.5, fill=0.72,
    ),
    'disgust': dict(
        octave_lo=3, octave_hi=5, density=0.38, scale='chromatic',
        chord_w=0.18, scale_w=0.28, tension_w=0.54,
        vel_base=58, vel_spread=10, step_beats=0.75, fill=0.60,
    ),
    'contempt': dict(
        octave_lo=4, octave_hi=5, density=0.28, scale='major',
        chord_w=0.55, scale_w=0.40, tension_w=0.05,
        vel_base=62, vel_spread=6, step_beats=1.0, fill=0.38,
    ),
    'neutral': dict(
        octave_lo=4, octave_hi=5, density=0.50, scale='major',
        chord_w=0.50, scale_w=0.45, tension_w=0.05,
        vel_base=68, vel_spread=10, step_beats=0.75, fill=0.78,
    ),
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

def play_slot_stride(midiout: rtmidi.MidiOut, chord_data: tuple, beat: float) -> None:
    """Stride style: bass on beat 1/3, chord voicing on beat 2/4."""
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


def play_slot_boogie(midiout: rtmidi.MidiOut, chord_data: tuple, beat: float) -> None:
    """
    Boogie-woogie style: walking left-hand bass on every beat, right-hand
    chord stab on beat 2/4.

    Per half-bar slot (2 beats):
      Beat 1: bass root (octave 2)
      Beat 2: bass fifth (root + 7, octave 2) + chord stab (right hand)
    """
    bass, tones = chord_data
    fifth = bass + 7      # perfect fifth above the root
    ring = beat * NOTE_FILL
    gap  = beat * (1.0 - NOTE_FILL)

    # Beat 1 — root in bass
    note_on(midiout, bass, VEL_BASS)
    time.sleep(ring)
    note_off(midiout, bass)
    time.sleep(gap)

    # Beat 2 — fifth in bass + chord stab
    note_on(midiout, fifth, VEL_BASS)
    for t in tones:
        note_on(midiout, t, VEL_CHORD)
    time.sleep(ring)
    note_off(midiout, fifth)
    for t in tones:
        note_off(midiout, t)
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
    for bar_idx, bar in enumerate(bars):
        label = labels.get(bar_idx)
        if label:
            print(f"  {label}")
        for slot in bar:
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


def melody_thread_func(midiout: rtmidi.MidiOut, beat: float) -> None:
    """Continuously generate an improvisatory melody shaped by MOOD."""
    active_note = None

    while not _stop_event.is_set():
        with _mood_lock:
            chord_data = _current_chord
            mood       = MOOD

        if chord_data is None:
            _stop_event.wait(beat * 0.25)
            continue

        params = MOOD_PARAMS[mood]
        step   = beat * params['step_beats']
        ring   = step * params['fill']
        gap    = step * (1.0 - params['fill'])

        # Release any note still ringing from the previous step
        if active_note is not None:
            midiout.send_message([0x80 | MELODY_CHANNEL, active_note, 0])
            active_note = None

        if random.random() < params['density']:
            pool = _melody_candidates(chord_data, params)
            if pool:
                note = random.choice(pool)
                vel  = max(1, min(127,
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
                            print(f"[net] Mood \u2192 {MOOD}")
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
    global KEY, TRANSPOSE, STYLE, CHANGES
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
    print(f"{changes_label} \u2014 {KEY} \u2014 {BPM} BPM \u2014 {STYLE} \u2014 {loop_desc}")
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
