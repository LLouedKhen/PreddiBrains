import numpy as np
import random 
from PIL import Image
from psychopy import visual, event


# FACE 

def bais_mapping(x) : 
    if  x > 0.5:
        emo= 'happy'
    elif x == 0.5:
        emo = 'neutral'
    elif x < 0.5:
        emo = 'angry'
    else:
        emo = 'null'
    return emo

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


# CUBE 

def camera(aspect, zf=10, zn=0.1, scale=2):
    scale = 1 / scale
    return np.array([
        [scale, 0, 0, 0],
        [0, scale/aspect, 0, 0],
        [0, 0, (zf+zn)/(zn-zf), (2*zf*zn)/(zn-zf)],
        [0, 0, -1, 0]
    ])

def transmat(x, y, z):
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [-x, -y, -z, 1]
    ])

def rotmat(phi, axis):
    if axis == 'x':
        return np.array([[1, 0, 0, 0], [0, np.cos(phi), np.sin(phi), 0], [0, -np.sin(phi), np.cos(phi), 0], [0, 0, 0, 1]])
    elif axis == 'y':
        return np.array([[np.cos(phi), 0, -np.sin(phi), 0], [0, 1, 0, 0], [np.sin(phi), 0, np.cos(phi), 0], [0, 0, 0, 1]])
    elif axis == 'z':
        return np.array([[np.cos(phi), np.sin(phi), 0, 0], [-np.sin(phi), np.cos(phi), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

def draw_cube_intro(p2, message, win):
    vup = np.array([
    [1, 1, 1, 1], [1, 1, -1, 1], [1, -1, 1, 1], [1, -1, -1, 1],
    [-1, 1, 1, 1], [-1, 1, -1, 1], [-1, -1, 1, 1], [-1, -1, -1, 1]
    ])

    vdn = vup.copy()
    vdn[:, 2] *= -1

    rx, ry = win.size
    aspect = rx / ry
    cam = camera(aspect)
    phi = np.pi / 2.6
    theta = np.arctan(np.sqrt(1.6))

    edges = [
    [0, 1], [0, 2], [1, 3], [2, 3],
    [0, 4], [1, 5], [2, 6], [3, 7],
    [4, 5], [4, 6], [5, 7], [6, 7]
    ]

    d = 2

    m4up = rotmat(phi, 'z') @ rotmat(theta, 'x') @ transmat(0, 0, d)
    m4dn = rotmat(phi, 'z') @ rotmat(-theta, 'x') @ transmat(0, 0, d)

    cam = camera(aspect)

    vup_proj = vup @ m4up @ np.linalg.inv(cam)
    vdn_proj = vdn @ m4dn @ np.linalg.inv(cam)
    vclip = p2 * vup_proj + (1 - p2) * vdn_proj
    vclip[:, :3] /= vclip[:, 3:]  # Perspective divide
    vscreen = vclip[:, :2] * 0.7  # Scale up to desired size
    for start, end in edges:
        visual.Line(win, start=vscreen[start], end=vscreen[end], lineColor='black', lineWidth=6).draw()
    message.draw()
    win.flip()
    event.waitKeys()
    
def draw_cube(p2, win):
    vup = np.array([
    [1, 1, 1, 1], [1, 1, -1, 1], [1, -1, 1, 1], [1, -1, -1, 1],
    [-1, 1, 1, 1], [-1, 1, -1, 1], [-1, -1, 1, 1], [-1, -1, -1, 1]
    ])

    vdn = vup.copy()
    vdn[:, 2] *= -1
    phi = np.pi / 2.6
    theta = np.arctan(np.sqrt(1.6))

    edges = [
    [0, 1], [0, 2], [1, 3], [2, 3],
    [0, 4], [1, 5], [2, 6], [3, 7],
    [4, 5], [4, 6], [5, 7], [6, 7]
    ]
    d = 2


    rx, ry = win.size
    aspect = rx / ry
    cam = camera(aspect)
    
    m4up = rotmat(phi, 'z') @ rotmat(theta, 'x') @ transmat(0, 0, d)
    m4dn = rotmat(phi, 'z') @ rotmat(-theta, 'x') @ transmat(0, 0, d)

    vup_proj = vup @ m4up @ np.linalg.inv(cam)
    vdn_proj = vdn @ m4dn @ np.linalg.inv(cam)
    vclip = p2 * vup_proj + (1 - p2) * vdn_proj
    vclip[:, :3] /= vclip[:, 3:]  # Perspective divide
    vscreen = vclip[:, :2] * 0.7  # Scale up to desired size
    for start, end in edges:
        visual.Line(win, start=vscreen[start], end=vscreen[end], lineColor='black', lineWidth=6).draw()


def calculate_edge_length(vertices):
    return np.linalg.norm(vertices[0] - vertices[1])  # Assuming consistent edge length

def generate_random_lines_in_cube_area(num_lines=12, length=0.15):
    lines = []
    for _ in range(num_lines):
        start = np.random.uniform(-0.35, 0.35, size=2)
        angle = np.random.uniform(0, 2 * np.pi)
        end = start + length * np.array([np.cos(angle), np.sin(angle)])
        if np.all(np.abs(end) <= 0.35):
            lines.append((start, end))
    return lines
