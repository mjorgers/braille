import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns

# Braille Unicode mapping for the 26 English letters
braille_mapping = {
    'a': '⠁', 'b': '⠃', 'c': '⠉', 'd': '⠙', 'e': '⠑', 'f': '⠋', 'g': '⠛',
    'h': '⠓', 'i': '⠊', 'j': '⠚', 'k': '⠅', 'l': '⠇', 'm': '⠍', 'n': '⠝',
    'o': '⠕', 'p': '⠏', 'q': '⠟', 'r': '⠗', 's': '⠎', 't': '⠞', 'u': '⠥',
    'v': '⠧', 'w': '⠺', 'x': '⠭', 'y': '⠽', 'z': '⠵'
}

def load_dataset(dataset_dir, braille_mapping, image_size=(90, 90), test_size=0.2):
    X = []  # Image data
    y = []  # Labels
    label_map = {letter: idx for idx, letter in enumerate(braille_mapping)}

    for letter in braille_mapping:
        for filename in os.listdir(dataset_dir):
            if filename.startswith(letter) and filename.endswith('.jpg'):
                image_path = os.path.join(dataset_dir, filename)
                image = load_img(image_path, target_size=image_size, color_mode='grayscale')
                image_array = img_to_array(image)
                X.append(image_array)
                y.append(label_map[letter])

    # Convert to numpy arrays and normalize the pixel values
    X = np.array(X) / 255.0
    y = to_categorical(y, num_classes=len(braille_mapping))

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    return (X_train, y_train), (X_test, y_test)

# Now let's load the dataset
(X_train, y_train), (X_test, y_test) = load_dataset("dataset", braille_mapping)

def create_cnn(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

# Instantiate the model
model = create_cnn((90, 90, 1), len(braille_mapping))

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Define an early stopping callback with a patience of a few epochs.
early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=5,               # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True, # Restores model weights from the epoch with the lowest validation loss.
    verbose=1
)

# Train the model with the EarlyStopping callback
history = model.fit(
    X_train, y_train,
    epochs=50,
    validation_data=(X_test, y_test),
    batch_size=32,
    callbacks=[early_stopping_callback]  # Add the EarlyStopping callback
)
model.summary()
model.evaluate(X_test, y_test)

# Assuming `history.history['loss']` and `history.history['val_loss']` are populated
time = np.arange(1, len(history.history['loss']) + 1)

plt.figure(figsize=(10, 5))  # Optionally set a figure size
plt.plot(time, history.history['loss'], label='Loss')
plt.plot(time, history.history['val_loss'], label='Validation loss')
plt.title('Loss fitting history')
plt.xlabel('Epoch')  # Label the x-axis
plt.ylabel('Loss')  # Label the y-axis
plt.legend()  # Ensure the legend is shown
plt.tight_layout()  # Ensure the plot is nicely formatted

# Save the figure
plt.savefig('loss_fitting_history.png')  # Saves the figure to a file.
plt.show()


model.save('model.keras')