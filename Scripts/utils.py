import numpy as np
import random 
from PIL import Image


def scramble_image(img_grey, block_size=70):
    """
    Scramble an image by dividing it into blocks and shuffling them.
    
    Args:
        image_path (str): Path to the image to scramble.
        block_size (int): Size of each block to divide the image into.
    
    Returns:
        scrambled_img (Image): Scrambled PIL image.
    """
    img_array = np.array(img_grey)
    # Get image dimensions
    h, w = img_array.shape
    # Ensure the dimensions are divisible by block_size
    h_blocks = h // block_size
    w_blocks = w // block_size
    img_cropped = img_array[:h_blocks * block_size, :w_blocks * block_size]
    
    # Divide image into blocks
    blocks = [
        img_cropped[i:i+block_size, j:j+block_size]
        for i in range(0, h_blocks * block_size, block_size)
        for j in range(0, w_blocks * block_size, block_size)
    ]
    
    # Shuffle the blocks
    random.shuffle(blocks)
    
    # Reassemble the image
    scrambled_array = np.vstack([
        np.hstack(blocks[i:i + w_blocks])
        for i in range(0, len(blocks), w_blocks)
    ])
    
    # Convert back to a PIL Image
    scrambled_img = Image.fromarray(scrambled_array)
    return scrambled_img

def logiF(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))

def inverse_logistic(y, L, k, x0):
    return x0 - (1/k) * np.log(L/y - 1)