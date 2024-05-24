from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import os
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter
import random
from tqdm import tqdm

FONT_PATH = "BRAILLE1.ttf"

# Braille Unicode mapping for the 26 English letters
braille_mapping = {
    'a': '⠁', 'b': '⠃', 'c': '⠉', 'd': '⠙', 'e': '⠑', 'f': '⠋', 'g': '⠛',
    'h': '⠓', 'i': '⠊', 'j': '⠚', 'k': '⠅', 'l': '⠇', 'm': '⠍', 'n': '⠝',
    'o': '⠕', 'p': '⠏', 'q': '⠟', 'r': '⠗', 's': '⠎', 't': '⠞', 'u': '⠥',
    'v': '⠧', 'w': '⠺', 'x': '⠭', 'y': '⠽', 'z': '⠵'
}


def random_translation(image, max_translation):
    # Generate random distances to move along the x and y axes
    dx = random.randint(-max_translation, max_translation)
    dy = random.randint(-max_translation, max_translation)

    # Convert to numpy for displacement
    numpy_image = np.array(image)
    translated_image = np.roll(numpy_image, shift=(dy, dx), axis=(0, 1))

    # Pad black color for areas outside original boundaries
    if dy > 0:
        translated_image[:dy, :] = 0
    elif dy < 0:
        translated_image[dy:, :] = 0
    if dx > 0:
        translated_image[:, :dx] = 0
    elif dx < 0:
        translated_image[:, dx:] = 0

    return translated_image


# Function to add Gaussian noise to an image
def add_gaussian_noise(image, mean=0, std=1):
    row, col, ch = image.shape
    gauss = np.random.normal(mean, std, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy_image = image + gauss
    return np.clip(noisy_image, 0, 255).astype('uint8')


# Function to add Speckle noise to an image
def add_speckle_noise(image):
    row, col, ch = image.shape
    gauss = np.random.randn(row, col, ch)
    gauss = gauss.reshape(row, col, ch)
    noisy_image = image + image * gauss
    return np.clip(noisy_image, 0, 255).astype('uint8')


# Function to apply an elastic transformation to an image
def add_elastic_transform(image, alpha, sigma):
    assert len(image.shape) == 3, "Image should be a 3D numpy array"
    random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij")
    indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1)), np.reshape(z, (-1, 1))

    distored_image = map_coordinates(image, indices, order=1, mode='reflect')
    return distored_image.reshape(image.shape)


def create_braille_image(letter, rotation, directory, noise_type='None'):
    image_size = (90, 90)
    font_size = 110

    # Create base image with a black background
    img = Image.new('RGB', image_size, color='black')
    draw = ImageDraw.Draw(img)

    try:
        # Before loading, check if the font file exists
        if not os.path.exists(FONT_PATH):
            raise IOError("Font file not found.")

        font = ImageFont.truetype(FONT_PATH, font_size)
    except IOError:
        raise IOError("Font file could not be loaded. Please check the path.")

    # Draw the Braille character in white on the image
    text_width, text_height = draw.textsize(letter, font=font)
    position = ((image_size[0] - text_width) // 2, (image_size[1] - text_height) // 2)
    draw.text(position, letter, font=font, fill="white")

    # Convert to numpy array for noise operations
    numpy_image = np.array(img)

    # Add noise if a type is specified
    if noise_type == 'speckle':
        numpy_image = add_speckle_noise(numpy_image)
    elif noise_type == 'gaussian':
        numpy_image = add_gaussian_noise(numpy_image, mean=0, std=30)
    elif noise_type == 'elastic':
        numpy_image = add_elastic_transform(numpy_image, alpha=100, sigma=4)
    elif noise_type == 'translation':
        numpy_image = random_translation(numpy_image, max_translation=15)

    # Convert back to PIL Image
    img_with_noise = Image.fromarray(numpy_image)

    # Rotate the image as specified by rotation parameter
    if rotation == 0:
        img_with_noise.save(os.path.join(directory, f"{letter}_{noise_type}.jpg"))
    else:
        img_rotated = img_with_noise.rotate(rotation, fillcolor='black')
        rotation_suffix = 'l' if rotation > 0 else 'r'
        img_rotated.save(os.path.join(directory, f"{letter}_{noise_type}_{abs(rotation)}{rotation_suffix}.jpg"))


def create_dataset():
    directory = "dataset"
    if not os.path.exists(directory):
        os.makedirs(directory)
    noises = ['speckle', 'gaussian', 'elastic']
    # Calculate total iterations for progress bar
    total_iterations = len(braille_mapping) * len(range(-30, 35, 5)) * len(noises)
    with tqdm(total=total_iterations, desc="Creating images", unit="image") as pbar:
        for letter, braille_char in braille_mapping.items():
            for rotation in range(-20, 25, 5):
                for noise in noises:
                    create_braille_image(letter, rotation, directory, noise)
                    pbar.update(1)


# Main program
if __name__ == "__main__":
    # Make sure the font file is available
    if not os.path.exists(FONT_PATH):
        print("Font file not found. Please specify the correct path.")
    else:
        create_dataset()
