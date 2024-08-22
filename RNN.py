import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.optimizers import Adam

# Use a smaller sample text directly in the script
text = """
Shall I compare thee to a summer's day?
Thou art more lovely and more temperate:
Rough winds do shake the darling buds of May,
And summer's lease hath all too short a date:
Sometime too hot the eye of heaven shines,
And often is his gold complexion dimmed;
And every fair from fair sometime declines,
By chance or nature's changing course untrimmed;
But thy eternal summer shall not fade
Nor lose possession of that fair thou owest;
Nor shall Death brag thou wanderest in his shade,
When in eternal lines to time thou growest:
So long as men can breathe or eyes can see,
So long lives this, and this gives life to thee.
""".lower()

# Create a mapping of unique characters to indices and vice versa
chars = sorted(list(set(text)))
char_indices = {c: i for i, c in enumerate(chars)}
indices_char = {i: c for i, c in enumerate(chars)}

# Cut the text into sequences
maxlen = 40  # Length of each sequence
step = 3  # Step size to move the window for each sequence
sequences = []
next_chars = []

for i in range(0, len(text) - maxlen, step):
    sequences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])

print(f"Number of sequences: {len(sequences)}")

# Vectorize the sequences
X = np.zeros((len(sequences), maxlen), dtype=np.int32)
y = np.zeros((len(sequences), len(chars)), dtype=bool)  # Use bool instead of np.bool

for i, seq in enumerate(sequences):
    X[i] = [char_indices[char] for char in seq]
    y[i, char_indices[next_chars[i]]] = True  # Use True instead of 1

# Build the LSTM model
model = Sequential()
model.add(Embedding(input_dim=len(chars), output_dim=128, input_length=maxlen))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(len(chars), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001))

# Train the model
model.fit(X, y, batch_size=128, epochs=20)

# Text generation function
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(seed_text, length=400, temperature=1.0):
    generated_text = seed_text
    for i in range(length):
        sampled = np.zeros((1, maxlen))
        for t, char in enumerate(generated_text[-maxlen:]):
            sampled[0, t] = char_indices[char]

        preds = model.predict(sampled, verbose=0)[0]
        next_index = sample(preds, temperature)
        next_char = indices_char[next_index]

        generated_text += next_char

    return generated_text

# Generate text
seed_text = "shall i compare thee to a summer's day"
print("Generated text:", generate_text(seed_text, length=400, temperature=0.5))
