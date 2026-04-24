import rtmidi
import time
import random

midiout = rtmidi.MidiOut()
ports = midiout.get_ports()

if not ports:
    print("No MIDI output ports found.")
    exit(1)

print("Available MIDI ports:")
for i, port in enumerate(ports):
    print(f"  {i}: {port}")

midiout.open_port(0)
print(f"\nUsing port: {ports[0]}")
print("Playing random notes. Press Ctrl+C to stop.\n")

CHANNEL = 0       # MIDI channel 0 (channel 1)
VELOCITY = 100
NOTE_DURATION = 0.5  # seconds the note is held

try:
    current_note = None
    while True:
        # Turn off previous note
        if current_note is not None:
            midiout.send_message([0x80 | CHANNEL, current_note, 0])

        # Pick and play a random note (MIDI range 36–84, C2–C6)
        current_note = random.randint(36, 84)
        midiout.send_message([0x90 | CHANNEL, current_note, VELOCITY])
        print(f"Note ON:  {current_note}")

        time.sleep(NOTE_DURATION)

        midiout.send_message([0x80 | CHANNEL, current_note, 0])
        print(f"Note OFF: {current_note}")

        time.sleep(1.0 - NOTE_DURATION)

except KeyboardInterrupt:
    if current_note is not None:
        midiout.send_message([0x80 | CHANNEL, current_note, 0])
    print("\nStopped.")
finally:
    del midiout
