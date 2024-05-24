from PIL import Image, ImageDraw, ImageFont
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
import numpy as np
import os
import shutil

# Braille Unicode mapping for the 26 English letters
braille_mapping = {
    'a': '⠁', 'b': '⠃', 'c': '⠉', 'd': '⠙', 'e': '⠑', 'f': '⠋', 'g': '⠛',
    'h': '⠓', 'i': '⠊', 'j': '⠚', 'k': '⠅', 'l': '⠇', 'm': '⠍', 'n': '⠝',
    'o': '⠕', 'p': '⠏', 'q': '⠟', 'r': '⠗', 's': '⠎', 't': '⠞', 'u': '⠥',
    'v': '⠧', 'w': '⠺', 'x': '⠭', 'y': '⠽', 'z': '⠵'
}

def create_braille_image(braille_char, file_number, directory):
    """
    Create an image with the braille character and save it to the given directory.
    """
    image_size = (100, 100)
    font_size = 120

    img = Image.new('RGB', image_size, color='black')
    draw = ImageDraw.Draw(img)

    # Provide the path to a .ttf font file that includes Braille characters
    font_path = "/home/jorge/Documents/2023-2024/ia_org/practicafinal/BRAILLE1.ttf"
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()
        print("Using default font because the specified ttf font with Braille could not be loaded.")

    text_width, text_height = draw.textsize(letter, font=font)
    position = ((image_size[0] - text_width) // 2, (image_size[1] - text_height) // 2)
    draw.text(position, letter, font=font, fill="white")

    # Create the full path for the image
    image_path = os.path.join(directory, f'{file_number}.jpg')
    img.save(image_path)

    print(f"Image saved as '{image_path}'")

def create_folder_for_word(word, script_dir):
    """
    Create a folder for the given word in the script directory and ensure it is empty.
    """
    folder_path = os.path.join(script_dir, "output")

    # Create the folder if it does not exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        # If the folder exists, ensure it is empty
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove files and links
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove directories
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    
    return folder_path

def predict_word_from_braille_images(image_directory, model_braille, alphabet):
    images = os.listdir(image_directory)
    predicted_string_list = []

    # Custom sorting function to sort filenames numerically by converting the file number to an integer
    def sort_key(filename):
        # Assuming filename format is 'number.jpg'
        file_number = int(os.path.splitext(filename)[0])
        return file_number

    for img_name in sorted(images, key=sort_key):
        print(img_name)
        img_path = os.path.join(image_directory, img_name)
        if os.path.isfile(img_path):
            # Load the image as grayscale
            img = image.load_img(img_path, color_mode="grayscale", target_size=(90, 90))
            img_array = image.img_to_array(img)
            # Reshape the image array in case of discrepancy
            img_array = np.reshape(img_array, (90, 90, 1))
            # Normalize the image array
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            predictions = model_braille.predict(img_array)
            predicted_class_index = np.argmax(predictions)
            predicted_character = alphabet[predicted_class_index]
            predicted_string_list.append(predicted_character)
    
    return ''.join(predicted_string_list)

# Main program
if __name__ == "__main__":
    user_input_word = input("Please enter a word: ").lower()
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the script
    folder_path = create_folder_for_word(user_input_word, script_dir)
    
    for index, letter in enumerate(user_input_word):
        if letter.isalpha() and letter in braille_mapping:
            braille_character = braille_mapping[letter]
            create_braille_image(braille_character, index + 1, folder_path)
        else:
            print(f"Skipping non-alphabetical character: '{letter}'")

    # Load the Braille classification model
    model_braille = load_model('model.keras')
    
    # Alphabet used for predictions
    alphabet = 'abcdefghijklmnopqrstuvwxyz'

    # Predict the word from the Braille images created
    predicted_string = predict_word_from_braille_images(folder_path, model_braille, alphabet)
    print("Predicted string:", predicted_string)