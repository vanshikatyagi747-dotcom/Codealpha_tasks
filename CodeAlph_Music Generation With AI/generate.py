import os
import numpy as np
from music21 import converter, instrument, note, chord
from tensorflow.keras.utils import to_categorical

def get_notes_from_midi(folder_path):
    notes = []
    for file in os.listdir(folder_path):
        if file.endswith(".mid") or file.endswith(".midi"):
            print(f"Parsing {file}...")
            midi = converter.parse(os.path.join(folder_path, file))
            try: 
                # Try to get instrument parts
                parts = instrument.partitionByInstrument(midi)
                notes_to_parse = parts.parts[0].recurse()
            except: 
                # Fallback if no parts exist
                notes_to_parse = midi.flat.notes
            
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))
    return notes

# 1. Load notes
notes = get_notes_from_midi("midi_songs")
n_vocab = len(set(notes))

# 2. Map notes to integers
pitchnames = sorted(set(notes))
note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

# 3. Create input sequences (lookback of 100 notes)
sequence_length = 100
network_input = []
network_output = []

for i in range(len(notes) - sequence_length):
    seq_in = notes[i:i + sequence_length]
    seq_out = notes[i + sequence_length]
    network_input.append([note_to_int[char] for char in seq_in])
    network_output.append(note_to_int[seq_out])

# 4. Reshape and normalize for LSTM
n_patterns = len(network_input)
X = np.reshape(network_input, (n_patterns, sequence_length, 1)) / float(n_vocab)
y = to_categorical(network_output)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation

# Build the network
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(256))
model.add(Dropout(0.3))
model.add(Dense(n_vocab))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

# Train the model (Keep epochs low for a fast test, increase to 50+ for better music)
print("Training the model... This might take a few minutes.")
model.fit(X, y, epochs=10, batch_size=64)

from music21 import stream

# 1. Pick a random seed sequence to start the music
start = np.random.randint(0, len(network_input)-1)
int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
pattern = network_input[start]
prediction_output = []

# 2. Generate 200 new notes
print("Generating new music sequence...")
for note_index in range(200):
    prediction_input = np.reshape(pattern, (1, len(pattern), 1)) / float(n_vocab)
    prediction = model.predict(prediction_input, verbose=0)
    
    index = np.argmax(prediction)
    result = int_to_note[index]
    prediction_output.append(result)
    
    pattern.append(index)
    pattern = pattern[1:]

# 3. Convert sequence back to MIDI structures
offset = 0
output_notes = []

for pattern in prediction_output:
    # Pattern is a chord
    if ('.' in pattern) or pattern.isdigit():
        notes_in_chord = pattern.split('.')
        chord_notes = []
        for current_note in notes_in_chord:
            new_note = note.Note(int(current_note))
            new_note.storedInstrument = instrument.Piano()
            chord_notes.append(new_note)
        new_chord = chord.Chord(chord_notes)
        new_chord.offset = offset
        output_notes.append(new_chord)
    # Pattern is a single note
    else:
        new_note = note.Note(pattern)
        new_note.offset = offset
        new_note.storedInstrument = instrument.Piano()
        output_notes.append(new_note)
    
    # Increase offset so notes don't stack on top of each other
    offset += 0.5

# 4. Save to file
midi_stream = stream.Stream(output_notes)
midi_stream.write('midi', fp='output_song.mid')
print("Finished! Saved music to 'output_song.mid'")
