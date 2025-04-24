#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 10:39:16 2025

@author: sysadmin
"""


from psychopy import visual, event, core
import math
import numpy as np
import pandas as pd
import random
import os
from psychopy.hardware import keyboard
from psychopy_visionscience.noise import NoiseStim
from psychopy.visual import GratingStim
from psychopy import logging

# Enable detailed logging
logging.console.setLevel(logging.WARNING)

outPath = '/Users/sysadmin/Documents/PreddiBrains/Output/Imaging/NC'
os.chdir(outPath)

subjNum = input("Enter main participant identifier: ") 
subjPath = os.path.join(outPath, 'PBF' + subjNum)
os.chdir(subjPath)
resFile = 'NCI_' + subjNum + '.csv'
while os.path.isfile(os.path.join(subjPath, resFile)):
    print("Error! This subject file exists already.")
    subjNum  = input("Please re-enter main participant identifier: ")    
    print(subjNum)
    
subjData = pd.read_csv('PBNC'+subjNum +'_IntakeData.csv')
lang = subjData.iloc[5,1]



# win = visual.Window([1200, 900], pos = (2,2), color=(0, 0, 0), units='pix')
win = visual.Window([1512,982], [0, 0],useFBO=True, monitor="testMonitor", units="pix")

n1 = []
n2 = []
#winner used
for nn in range(1152):
    n1.append(NoiseStim(
    win=win, name='noise1', units='pix',
    mask=None, ori=0.0, pos=(0, 0), size=(win.size[0]/4, win.size[1]/3),
    opacity=2, blendmode='avg', contrast=1.5,
    texRes=512, noiseType='Binary', noiseElementSize=7,  # Adjusted size
    noiseBaseSf=12.0/512, noiseFilterLower=3/512, noiseFilterUpper=10.0/512.0,
    interpolate=False, depth=-1.0
    ))

for nnn in range(1152):
    n2.append(NoiseStim(
    win=win, name='noise3', units='pix',
    mask=None, ori=0.0, pos=(0, 0), size=(win.size[0]/4, win.size[1]/3),
    opacity=2, blendmode='avg', contrast=1.7,
    texRes=512, noiseType='Binary', noiseElementSize=10,  # Adjusted size
    noiseBaseSf=12.0/512, noiseFilterLower=3/512, noiseFilterUpper=10.0/512.0,
    interpolate=False, depth=-1.0
    ))


if lang == 'E':      
    instrText = visual.TextStim(win, text = 'In this task, you will be presented with a series of noise masks. In between the noise masks, we may show you an image of a cube. We will then ask you if you saw a cube. If yes, press the rightmost button. If no, press the leftmost button. You will then be asked if you saw the cube from above (press right), or from below (press left). You must provide an answer even if you did not see the cube. You will then be asked how sure you are of your response on a scale from 1-10. Navigate through the scale with left and right buttons, and confirm your selection with either of the two middle buttons. Press any key to continue.', 
    font='', pos=(0, 0), depth=0, rgb=None, color= 'black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
    ori=0.0, height = 32, antialias=True, bold=False, italic=False,  anchorVert='center', anchorHoriz='center',
    fontFiles=(), wrapWidth=None, flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None)  
    
    
    Quest1 = visual.TextStim(win, text = 'Did you see a cube?', 
    font='', pos=(0, 0), depth=0, rgb=None, color= 'black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
    ori=0.0, height = 32, antialias=True, bold=False, italic=False,  anchorVert='center', anchorHoriz='center',
    fontFiles=(), wrapWidth=None, flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None)  
    
    Quest2 = visual.TextStim(win, text = 'Frome below or from above?', 
    font='', pos=(0, 0), depth=0, rgb=None, color= 'black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
    ori=0.0, height = 32, antialias=True, bold=False, italic=False, anchorVert='center', anchorHoriz='center',
    fontFiles=(), wrapWidth=None, flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None)  
    
    thanksText = visual.TextStim(win=win, text='Thank you', font='', pos=(0, 0),
    depth=0, rgb=None, color='black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
    ori=0.0, height = 32, antialias=True, bold=True, italic=False, alignHoriz='center', alignVert='center',
    fontFiles=(), wrapWidth=None, flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None)
    
    confQuestion= visual.TextStim(win, text = 'How sure are you of your last answer?', 
    font='', pos=(0, 200), depth=0, rgb=None, color= 'black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
    ori=0.0, height=32, antialias=True, bold=False, italic=False, alignHoriz='center', alignVert='center',
    fontFiles=(), wrapWidth=None, flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None) 
    
    startText = visual.TextStim(win=win, text='Press any key to begin', font='', pos=(0, 0),
    depth=0, rgb=None, color='black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
    ori=0.0, height = 32, antialias=True, bold=True, italic=False,  anchorVert='center', anchorHoriz='center',
    fontFiles=(), wrapWidth=None, flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None)
    
elif lang == 'F':
    instrText = visual.TextStim(win, text = "Dans cette tâche, vous verrez une série de masques bruités. Entre ces masques, nous pourrons vous présenter un cube. Nous vous demanderons ensuite si vous avez vu un cube. Si oui, appuyez sur le bouton le plus à droite. Si non, appuyez sur le bouton le plus à gauche. Ensuite, nous vous demanderons si vous avez vu le cube d’en haut (appuyez à droite) ou d’en bas (appuyez à gauche). Vous devez fournir une réponse, même si vous n’avez pas vu le cube. Enfin, nous vous demanderons à quel point vous êtes sûr(e) de votre dernière réponse sur une échelle de 1 à 10. Naviguez l’échelle avec les boutons gauche et droite, puis  valider votre sélection avec l’un des boutons du milieu. Appuyez sur une touche pour continuer.", 
    font='', pos=(0, 0), depth=0, rgb=None, color= 'black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
    ori=0.0, height = 32, antialias=True, bold=False, italic=False,  anchorVert='center', anchorHoriz='center',
    fontFiles=(), wrapWidth=None, flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None)  
    
    
    Quest1 = visual.TextStim(win, text = 'Avez-vous vu un cube?', 
    font='', pos=(0, 0), depth=0, rgb=None, color= 'black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
    ori=0.0, height = 32, antialias=True, bold=False, italic=False,  anchorVert='center', anchorHoriz='center',
    fontFiles=(), wrapWidth=None, flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None)  
    
    Quest2 = visual.TextStim(win, text = "D'en haut ou d'en bas?", 
    font='', pos=(0, 0), depth=0, rgb=None, color= 'black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
    ori=0.0, height = 32, antialias=True, bold=False, italic=False, anchorVert='center', anchorHoriz='center',
    fontFiles=(), wrapWidth=None, flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None)  
    
    thanksText = visual.TextStim(win=win, text='Merci', font='', pos=(0, 0),
    depth=0, rgb=None, color='black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
    ori=0.0, height = 32, antialias=True, bold=True, italic=False, alignHoriz='center', alignVert='center',
    fontFiles=(), wrapWidth=None, flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None)
    
    confQuestion= visual.TextStim(win, text = 'A quel degré êtes-vous sur de votre dernière réponse?', 
    font='', pos=(0, 200), depth=0, rgb=None, color= 'black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
    ori=0.0, height=32, antialias=True, bold=False, italic=False, alignHoriz='center', alignVert='center',
    fontFiles=(), wrapWidth=None, flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None) 
    
    startText = visual.TextStim(win=win, text='Appuyez sur un bouton pour commencer', font='', pos=(0, 0),
    depth=0, rgb=None, color='black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
    ori=0.0, height = 32, antialias=True, bold=True, italic=False,  anchorVert='center', anchorHoriz='center',
    fontFiles=(), wrapWidth=None, flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None)


FixationText = visual.TextStim(win=win, text='+', font='', pos=(0, 0),
depth=0, rgb=None, color='black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
ori=0.0, height = 32, antialias=True, bold=True, italic=False,  anchorVert='center', anchorHoriz='center',
fontFiles=(), wrapWidth=None, flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None)

scannerWait = visual.TextStim(win, text = 'Please wait for scanner...', 
font='', pos=(0, 0), depth=0, rgb=None, color= 'black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
ori=0.0, height=32, antialias=True, bold=False, italic=False, alignHoriz='center', alignVert='center',
fontFiles=(), wrapWidth=None, flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None)  

confScale = visual.Slider(win,
    ticks=list(range(1, 11)),             # `low=1, high=10` -> ticks from 1 to 10
    labels=None,                           # No specific labels
    startValue=random.randint(1, 10),      # `markerStart`
    granularity=1,                         # Ensures only integer values (discrete steps)
    color='black',                     # `textColor`
    pos= (0, -0.3), markerColor='Black')
# Set up custom key responses for moving the marker and accepting the response
cleftKeys = '1'
crightKeys = '4'
cacceptKeys = ['2', '3']

bk = visual.rect.Rect(win, width = win.size[0]/4, height=win.size[1]/3, lineWidth=1, lineColor=False, fillColor='white', colorSpace='rgb', pos=(0, 0), size=None, anchor=None, ori=0.0, opacity=None, contrast=1.0, depth=0, interpolate=True, draggable=False, name=None, autoLog=None, autoDraw=False, color=None, lineColorSpace=None, fillColorSpace=None, lineRGB=False, fillRGB=False)

# Define the camera matrix function
def camera(aspect, zf=10, zn=0.1, scale=2):
    # Inverse the scale for the camera matrix
    scale = 1/scale
    # Define the camera projection matrix
    cmat = np.array([
        [scale, 0, 0, 0],
        [0, scale/aspect, 0, 0],
        [0, 0, (zf+zn)/(zn-zf), (2*zf*zn)/(zn-zf)],
        [0, 0, -1, 0]
    ])
    return cmat

# Define the translation matrix function
def transmat(x, y, z):
    tmat = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [-x, -y, -z, 1]
    ])
    return tmat

# Define the rotation matrix function
def rotmat(phi, axis='z'):
    # Rotation matrix around the X-axis
    if axis == 'x' or axis == 'X':
        rmat = np.array([
            [1, 0, 0, 0],
            [0, np.cos(phi), np.sin(phi), 0],
            [0, -np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1]
        ])
    # Rotation matrix around the Y-axis
    elif axis == 'y' or axis == 'Y':
        rmat = np.array([
            [np.cos(phi), 0, -np.sin(phi), 0],
            [0, 1, 0, 0],
            [np.sin(phi), 0, np.cos(phi), 0],
            [0, 0, 0, 1]
        ])
    # Rotation matrix around the Z-axis
    else:  # axis == 'z'
        rmat = np.array([
            [np.cos(phi), np.sin(phi), 0, 0],
            [-np.sin(phi), np.cos(phi), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    return rmat

# Convert the 3D points from clip space to screen space
def clip2screen(vclip, rx, ry):
    # Perform perspective division
    vclip[:, :3] /= vclip[:, 3:]

    # Scale and offset the points to screen coordinates
    vclip[:, :2] *= np.array([rx/4, -ry/4])
    vclip[:, :2] += np.array([rx/4, ry/4])

    # Clamp the values to the screen boundaries
    vclip[:, 0] = np.clip(vclip[:, 0], 0, rx - 1)
    vclip[:, 1] = np.clip(vclip[:, 1], 0, ry - 1)

    return vclip[:, :2]

# Create a PsychoPy window
# n1 = []
# n2 = []
# n3 = []
# n4 = []
response1 = []
response2 = []
stimPres = []
q1Pres = []
r1Ons = []
q2Pres = []
r2Ons = []
confQOns = []
confROns = []
confRT = []
confRating = []


# for i in range(72):
#     n1.append(noise1) 
#     n2.append(noise2)

noise = n1 + n2 
#Watch out for screen refresh rate!!!
dur = np.hstack([np.tile(0.0167,54), np.tile(0.200,54)])
# dur = np.tile(0.016,270)
cond1C = np.ones(54)
cond2C = np.zeros(54)
condC = np.hstack([cond1C, cond2C])

cond1P = np.ones(90)
cond2P = np.zeros(18)
condP = np.hstack([cond1P, cond2P])

# Generate interpolation parameter array `p`
p = np.arange(-0.9, 1, 0.2)
p = np.insert(p, 5, 0)
p = np.delete(p,[4,6])
percept = np.array([-1, -1, -1, -1, 0, 1, 1, 1, 1])
pss= np.tile(p, 5)
Percepts = np.tile(percept, 0)
x = np.tile(np.nan, 18)
y = np.tile(np.nan, 18)
ps = np.hstack([pss, x])
Percept =  np.hstack([Percepts, y])

dat1 = pd.concat([pd.Series(Percept), pd.Series(ps), pd.Series(condP)], axis = 1)
dat2 = pd.concat([pd.Series(condC), pd.Series(dur)], axis = 1)
dat3 = pd.concat([dat1, dat2], axis = 1)

# gratingType = condC
random.Random(9).shuffle(ps)
random.Random(9).shuffle(Percept)
random.Random(8).shuffle(condC)
random.Random(9).shuffle(condP)
random.Random(8).shuffle(dur)

dat11 = pd.concat([pd.Series(Percept), pd.Series(ps), pd.Series(condP)], axis = 1)
dat22 = pd.concat([pd.Series(condC), pd.Series(dur)], axis = 1)
dat33 = pd.concat([dat1, dat2], axis = 1)


random.Random(5).shuffle(noise)

# random.Random(7).shuffle(gratingType)

dused = []
nused = []
nsize = []
val = []
condition = []
present = []
gused = []


# Define the vertices of a cube
vup = np.array([
    [1, 1, 1, 1],
    [1, 1, -1, 1],
    [1, -1, 1, 1],
    [1, -1, -1, 1],
    [-1, 1, 1, 1],
    [-1, 1, -1, 1],
    [-1, -1, 1, 1],
    [-1, -1, -1, 1]
])
# Create a mirrored version of the cube along the Z-axis
vdn = vup.copy()
vdn[:, 2] = -vdn[:, 2]

# Define the lines that make up the edges of the cube
lines = [
    [0, 1], [0, 2], [1, 3], [2, 3],
    [0, 4], [1, 5], [2, 6], [3, 7],
    [4, 5], [4, 6], [5, 7], [6, 7]
]

# Get the screen dimensions
rx, ry = win.size

# Apply the camera and transformation matrices
cam = camera(rx/ry, 10, 0.1, 2)
phi = np.pi/2.6  # Rotation angle for Z-axis
theta = np.arctan(np.sqrt(1.6))  # Rotation angle for X-axis
d = 2  # Translation distance

# Combine rotation and translation matrices for "up" and "down" cubes
m4up = rotmat(phi, 'z') @ rotmat(theta, 'x') @ transmat(0, 0, d)
m4dn = rotmat(phi, 'z') @ rotmat(-theta, 'x') @ transmat(0, 0, d)

# Transform the vertices of the cubes
for i in range(vup.shape[0]):
    vup[i, :] = vup[i, :] @ m4up @ np.linalg.inv(cam)
    vdn[i, :] = vdn[i, :] @ m4dn @ np.linalg.inv(cam)
    
# Apply transformations to vertices
vup_transformed = vup @ m4up @ np.linalg.inv(cam)
vdn_transformed = vdn @ m4dn @ np.linalg.inv(cam)

# Define center of the window and scaling factors
center_x, center_y = win.size / 2
scaling_factor = rx / 4  # Adjust scaling factor as needed
sc = 1.5  # Scaling factor for the cube


# Original vertices of the cube (bounding box)
vscreen_original = np.array([
    [-100, -100],
    [100, -100],
    [-100, 100],
    [100, 100],
    [-150, -150],
    [150, -150],
    [-150, 150],
    [150, 150]
])

# Calculate the bounding box of the cube
x_min, x_max = vscreen_original[:, 0].min(), vscreen_original[:, 0].max()
y_min, y_max = vscreen_original[:, 1].min(), vscreen_original[:, 1].max()

# Function to calculate edge length (assuming all edges are the same length)
def calculate_edge_length(vertices):
    return np.linalg.norm(vertices[0] - vertices[1])  # Assuming consistent edge length

# Original edge length of the cube
edge_length = calculate_edge_length(vscreen_original)

# Function to generate fixed-length lines that stay within the cube's area
def generate_fixed_length_lines_within_cube(num_lines, edge_length):
    lines = []
    while len(lines) < num_lines:
        # Random starting point within the bounding box
        start = np.random.uniform([x_min, y_min], [x_max, y_max])
        
        # Generate a random angle for the line
        valid = False
        while not valid:
            angle = np.random.uniform(0, 2 * np.pi)
            end = start + edge_length * np.array([np.cos(angle), np.sin(angle)])
            
            # Check if the end point is within bounds
            if (x_min <= end[0] <= x_max) and (y_min <= end[1] <= y_max):
                valid = True
        
        # Append the valid line
        lines.append((start, end))
    return lines

def generate_constrained_random_lines(num_lines=10, magnitude=100, x_range=(-200, 200), y_range=(-200, 200)):
    lines = []
    for _ in range(num_lines):
        # Randomly choose a start point within the area
        start_x = random.randint(*x_range)
        start_y = random.randint(*y_range)
        # Generate a random angle for the line
        angle = random.uniform(0, 2 * math.pi)
        # Calculate the end point based on the magnitude and angle
        end_x = start_x + magnitude * math.cos(angle)
        end_y = start_y + magnitude * math.sin(angle)
        # Ensure the end point stays within bounds
        if x_range[0] <= end_x <= x_range[1] and y_range[0] <= end_y <= y_range[1]:
            lines.append(((start_x, start_y), (end_x, end_y)))
    return lines

# # Generate 12 random lines
# num_lines = 12
# random_lines = generate_fixed_length_lines_within_cube(num_lines, edge_length)


startText = visual.TextStim(win=win, text='Press any key to begin', font='', pos=(0, 0),
depth=0, rgb=None, color='black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
ori=0.0, height = 32, antialias=True, bold=True, italic=False, anchorVert='center', anchorHoriz='center',
fontFiles=(), wrapWidth=None, flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None)

frC = 12 # for my Mac, ca 60 Hz refresh rate. Adjust to approx. 100-150 ms
frUC = 1 # for my Mac, 60 Hz refresh rate. Adjust to approx. 16-24 ms

instrText.draw()
win.flip()
event.waitKeys()
startText.draw()
win.flip()
event.waitKeys()

if lang == 'E': 
    firstText = visual.TextStim(win=win, text='This is a Necker Cube. Press any key to continue.', font='', pos=(0, -300),
    depth=0, rgb=None, color='black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
    ori=0.0, height = 32, antialias=True, bold=True, italic=False, anchorVert='center', anchorHoriz='center',
    fontFiles=(), wrapWidth=None, flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None)
    
    secondText = visual.TextStim(win=win, text='This is a Necker Cube viewed from below. Press any key to continue.', font='', pos=(0, -300),
    depth=0, rgb=None, color='black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
    ori=0.0, height = 32, antialias=True, bold=True, italic=False, anchorVert='center', anchorHoriz='center',
    fontFiles=(), wrapWidth=None, flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None)
    
    thirdText = visual.TextStim(win=win, text='This is a Necker Cube viewed from above. Press any key to continue.', font='', pos=(0, -300),
    depth=0, rgb=None, color='black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
    ori=0.0, height = 32, antialias=True, bold=True, italic=False, anchorVert='center', anchorHoriz='center',
    fontFiles=(), wrapWidth=None, flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None)

elif lang == 'F': 
    
    firstText = visual.TextStim(win=win, text='Ceci est le cube Necker. Appuyez sur une touche pour continuer.', font='', pos=(0, -300),
    depth=0, rgb=None, color='black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
    ori=0.0, height = 32, antialias=True, bold=True, italic=False, anchorVert='center', anchorHoriz='center',
    fontFiles=(), wrapWidth=None, flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None)
    
    secondText = visual.TextStim(win=win, text="Ceci est le cube Necker vu d'en bas. Appuyez sur une touche pour continuer.", font='', pos=(0, -300),
    depth=0, rgb=None, color='black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
    ori=0.0, height = 32, antialias=True, bold=True, italic=False, anchorVert='center', anchorHoriz='center',
    fontFiles=(), wrapWidth=None, flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None)
    
    thirdText = visual.TextStim(win=win, text="Ceci est le cube Necker vu d'en haut. Appuyez sur une touche pour continuer.", font='', pos=(0, -300),
    depth=0, rgb=None, color='black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
    ori=0.0, height = 32, antialias=True, bold=True, italic=False, anchorVert='center', anchorHoriz='center',
    fontFiles=(), wrapWidth=None, flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None)


#ambiguous Cube 
p2 = 0.5 + 0 / 2  # Interpolation factor

# Interpolate between `vup` and `vdn`, apply scaling
vclip = p2 * vup + (1 - p2) * vdn
vclip[:, :3] *= sc

# Convert 3D points to 2D screen coordinates
vscreen = clip2screen(vclip, win.size[0]/2, win.size[1]/2)
cube_center = np.mean(vscreen, axis=0)
#random lines
for line in lines:
    start, end = line
# Draw each line segment of the cube
    visual.Line(win, start=vscreen[line[0]] - cube_center, end=vscreen[line[1]] - cube_center, lineColor=(-1, -1, -1), lineWidth=4).draw()

firstText.draw()
win.flip()
event.waitKeys()


#Below Bias Cube 
p2 = 0.5 + -1 / 2  # Interpolation factor

# Interpolate between `vup` and `vdn`, apply scaling
vclip = p2 * vup + (1 - p2) * vdn
vclip[:, :3] *= sc

# Convert 3D points to 2D screen coordinates
vscreen = clip2screen(vclip, win.size[0]/2, win.size[1]/2)
cube_center = np.mean(vscreen, axis=0)
#random lines
for line in lines:
    start, end = line
# Draw each line segment of the cube
    visual.Line(win, start=vscreen[line[0]] - cube_center, end=vscreen[line[1]] - cube_center, lineColor=(-1, -1, -1), lineWidth=4).draw()


secondText.draw()
win.flip()
event.waitKeys()

#Below Bias Cube 
p2 = 0.5 + 1 / 2  # Interpolation factor

# Interpolate between `vup` and `vdn`, apply scaling
vclip = p2 * vup + (1 - p2) * vdn
vclip[:, :3] *= sc

# Convert 3D points to 2D screen coordinates
vscreen = clip2screen(vclip, win.size[0]/2, win.size[1]/2)
cube_center = np.mean(vscreen, axis=0)
#random lines
for line in lines:
    start, end = line
# Draw each line segment of the cube
    visual.Line(win, start=vscreen[line[0]] - cube_center, end=vscreen[line[1]] - cube_center, lineColor=(-1, -1, -1), lineWidth=4).draw()

thirdText.draw()
win.flip()
event.waitKeys()

clock = core.Clock()
startText.draw()
win.flip()
event.waitKeys()

scannerWait.draw()
win.flip()


logFile = os.path.join(outPath, "outputFaces" + subjNum + ".txt")
with open(logFile, "a") as f:
    trigger = event.waitKeys(keyList = ['5'], clearEvents=True, timeStamped=True) 
    jitter = random.uniform(1,2)
    #block key 5 to keep console clear
    #keyboard.block_key("5")

    trigTime = trigger[0][1]
    startExp = clock.getTime()
    subjData.loc[6,1] = trigTime
    subjData.loc[7,1] = startExp
    print('Trigger at ' + str(trigTime), file=f)
    print('Trigger at ' + str( trigTime))
    print('Start at ' + str(startExp), file=f)
    print('Start at ' + str(startExp))


    # Iterate over each parameter in `allP`
    #for thisp in ps:
    for im in range(len(condC)):
        thisp = ps[im]
        print('Bias in trial ' + str(im) +' is ' + str(thisp), file=f)
        print('Bias in trial ' + str(im) +' is ' + str(thisp))
        print('Condition in trial ' + str(im) +' is ' + str(condC[im]), file=f)
        print('Condition in trial ' + str(im) +' is ' + str(condC[im]))
        print('Presence in trial ' + str(im) +' is ' + str(condP[im]), file=f)
        print('Presence in trial ' + str(im) +' is ' + str(condP[im]))
        p2 = 0.5 + thisp / 2  # Interpolation factor
    
        # Interpolate between `vup` and `vdn`, apply scaling
        vclip = p2 * vup + (1 - p2) * vdn
        vclip[:, :3] *= sc
        
        # Convert 3D points to 2D screen coordinates
        vscreen = clip2screen(vclip, win.size[0]/2, win.size[1]/2)
        cube_center = np.mean(vscreen, axis=0)
        #random lines
        random_lines = generate_constrained_random_lines()
        
        
        if condC[im] == 1:    
            n = random.sample(noise, 4)
            for nm in range(4): 
                n[nm].draw() 
                random_lines = generate_constrained_random_lines()
            # Draw the random lines
                for start, end in random_lines:
                    visual.Line(
                    win,
                    start=start,
                    end=end,
                    lineColor=(-1, -1, -1),
                    lineWidth=6
                    ).draw()
                win.flip()
                core.wait(0.066)
            
            
            if condP[im] == 1:
                # stim.draw()
                # Draw lines for the cube
                for _ in range(frC):
                    bk.draw()
                    for line in lines:
                        start, end = line
                    # Draw each line segment of the cube
                        visual.Line(win, start=vscreen[line[0]] - cube_center, end=vscreen[line[1]] - cube_center, lineColor=(-1, -1, -1), lineWidth=4).draw()
                    win.flip()
                    if _ == 0:
                        sP = clock.getTime()
                        stimPres.append(clock.getTime())
                        print('Stimulus presented at ' + str(sP), file=f)
                        print('Stimulus presented at  ' + str(sP))
                # core.wait(0.1515)
                # if gratingType[im] == 1:
            elif condP[im]== 0:
                for _ in range(frC):
                    bk.draw()
                    win.flip()
                    if _ == 0:
                        sP = clock.getTime()
                        stimPres.append(clock.getTime())
                        print('Stimulus presented at ' + str(sP), file=f)
                        print('Stimulus presented at ' + str(sP))
                # core.wait(0.1515)
            n = random.sample(noise, 4)
            for nm in range(4): 
                n[nm].draw() 
                random_lines = generate_constrained_random_lines()
            # Draw the random lines
                for start, end in random_lines:
                    visual.Line(
                    win,
                    start=start,
                    end=end,
                    lineColor=(-1, -1, -1),
                    lineWidth=6
                    ).draw()
                win.flip()
                core.wait(0.066)
            FixationText.draw()
            win.flip()
            core.wait(4.5 + jitter)
            Quest1.draw()
            win.flip()
            q1P = clock.getTime()
            q1Pres.append(q1P)
            
        elif condC[im] == 0:
            n = random.sample(noise, 4)
            for nm in range(4): 
                n[nm].draw() 
                random_lines = generate_constrained_random_lines()
            # Draw the random lines
                for start, end in random_lines:
                    visual.Line(
                    win,
                    start=start,
                    end=end,
                    lineColor=(-1, -1, -1),
                    lineWidth=6
                    ).draw()
                win.flip()
                core.wait(0.066)
    
            if condP[im] == 1:
                # stim.draw()
                # Draw lines for the cube
                for _ in range(frUC):
                    bk.draw()
                    for line in lines:
                        start, end = line
                    # Draw each line segment of the cube
                        visual.Line(win, start=vscreen[line[0]] - cube_center, end=vscreen[line[1]] - cube_center, lineColor=(-1, -1, -1), lineWidth=4).draw()
                    win.flip()
                    if _ == 0:
                        sP = clock.getTime()
                        stimPres.append(clock.getTime())
                        print('Stimulus presented at ' + str(sP), file=f)
                        print('Stimulus presented at ' + str(sP))
                # core.wait(0.0166)
                # if gratingType[im] == 1:
            elif condP[im] == 0:
                for _ in range(frC):
                    bk.draw()
                    win.flip()
                    if _ == 0:
                        sP = clock.getTime()
                        stimPres.append(clock.getTime())
                        print('Stimulus presented at ' + str(sP), file=f)
                        print('Stimulus presented at ' + str(sP))
                # core.wait(0.0166)
            n = random.sample(noise, 4)
            for nm in range(4): 
                n[nm].draw() 
                random_lines = generate_constrained_random_lines()
            # Draw the random lines
                for start, end in random_lines:
                    visual.Line(
                    win,
                    start=start,
                    end=end,
                    lineColor=(-1, -1, -1),
                    lineWidth=6
                    ).draw()
                win.flip()
                core.wait(0.066)   
            FixationText.draw()
            win.flip()
            core.wait(4.5 + jitter)
            Quest1.draw()
            win.flip()
            q1Pres.append(clock.getTime())
            
        resp1 = event.waitKeys(keyList = ['1', '4'], clearEvents=True, timeStamped=True)
        press1 = resp1[0][0]
        response1.append(press1)
        r1O = clock.getTime()
        r1Ons.append(clock.getTime())
        print('Response ' + str(press1) + ' given at ' + str(r1O), file=f)
        print('Response ' + str(press1) + ' given at ' + str(r1O))
        FixationText.draw()
        win.flip()
        core.wait(2)
        Quest2.draw()
        win.flip()
        q2P = clock.getTime()
        q2Pres.append(q2P)
        resp2 = event.waitKeys(keyList = ['1', '4'], clearEvents=True, timeStamped=True)
        press2 = resp2[0][0]
        response2.append(press2) 
        r2O = clock.getTime()
        r2Ons.append(clock.getTime())
        print('Response ' + str(press2) + ' given at ' + str(r2O), file=f)
        print('Response ' + str(press2) + ' given at ' + str(r2O))
        FixationText.draw()
        win.flip()
        core.wait(3)
        # nused.append(n.getNoiseType())
        confQO = []
        while confScale.getRating() is None:          # Keep looping until a value is selected
            confScale.draw()
            confQuestion.draw()
            win.flip()
            confQO.append(clock.getTime())
            ckeys = event.getKeys()                     
            if cleftKeys in ckeys:                  # Move left
                confScale.markerPos = max(confScale.markerPos - 1, 1)
            elif crightKeys in ckeys:               # Move right
                confScale.markerPos = min(confScale.markerPos + 1, 10)
            elif any(key in ckeys for key in cacceptKeys):  # Accept response
                rating = confScale.getMarkerPos()  # Retrieve the rating when Enter is pressed
                c2 = clock.getTime()
                
                
                print(f"Rating accepted: {rating}")
                t1 = confQO[0]
                confQOns.append(t1)
                confROns.append(c2)
                break       
        
        rating = confScale.getMarkerPos()
        confRT.append(t1-c2)
        print('Subject reported a confidence of ' + str(rating) + ' at ' + str(c2), file = f)
        print('Subject reported a confidence of ' + str(rating) + ' at ' + str(c2))
        confRating.append(rating)
        # nused.append(n.name)
        # # if condC[im] == 1:
        # #     dused.append(0.10)
        # # else:
        # dused.append(d)
        # if 'noise' in n.name:
        #     nsize.append(n.noiseElementSize)
        val.append(thisp)
        condition.append(condC[im])
        present.append(condP[im])
        
        print('Duration in trial ' + str(im) +' is ' + str(d), file=f)
        print('Duration in trial ' + str(im) +' is ' + str(d))
        print('Percept in trial ' + str(im) +' is ' + str(percept[im]), file=f)
        print('Percept in trial ' + str(im) +' is ' + str(percept[im]))

        
        FixationText.draw()
        win.flip()
        core.wait(4.5 + jitter)


allRes =pd.concat([pd.Series(stimPres), pd.Series(q1Pres), pd.Series(response1), pd.Series(r1Ons), pd.Series(q2Pres), pd.Series(response2), pd.Series(r2Ons), pd.Series(confQOns),pd.Series(confROns), pd.Series(confRT),pd.Series(confRating), pd.Series(response1), pd.Series(response2), pd.Series(val), pd.Series(condition), pd.Series(present)], axis = 1)
allRes.columns =['StimOnset', 'Q1Onset', 'Seen', 'Resp1Onset', 'Q2Onset', 'Emotion','Resp2Onset', 'ConfQOnset', 'ConfROnset', 'ConfRT', 'ConfRating','Seen', 'Percept','Bias', 'Condition', 'StimPresent']

endExp = clock.getTime()
subjData.loc[8,1] = endExp
subjData.to_csv('PBNC' +subjNum +'_IntakeData.csv')
expDur = endExp - startExp

print('Experiment lasted ' + str(expDur))



em = []
correct1 = []
correct2 = []


for i in range(len(allRes)):
    if  allRes.iloc[i,2] > 0:
        em.append('Up')
    elif allRes.iloc[i,2] == 0:
        em.append('Amb')
    elif allRes.iloc[i,2] < 0:
        em.append('Down')
    else:
        em.append('Null')

emC = []  
for i in range(len(allRes)):
    if '4' in allRes.iloc[i,1]:
        emC.append('Up')
    elif '1' in allRes.iloc[i,1]:
        emC.append('Down')
        

for i in range(len(allRes)):
    if emC[i] in em[i]:
        correct2.append(1)
    elif allRes.iloc[i,2] == 0:
        correct2.append(np.nan)
    else:
        correct2.append(0)
        
for i in range(len(allRes)):
    if '4' in allRes.iloc[i,0] and allRes.iloc[i,4] == 0:
        correct1.append(1)
    elif '4' in allRes.iloc[i,0] and allRes.iloc[i,4] == 1:
        correct1.append(0)
    elif '1' in allRes.iloc[i,0] and allRes.iloc[i,4] == 0:
            correct1.append(0)
    elif '1' in allRes.iloc[i,0] and allRes.iloc[i,4] == 1:
            correct1.append(-1)     
        
allRes = pd.concat([allRes, pd.Series(emC), pd.Series(correct1),pd.Series(correct2)], axis = 1)
allRes.columns =['StimOnset', 'Q1Onset', 'Seen', 'Resp1Onset', 'Q2Onset', 'Emotion','Resp2Onset', 'ConfQOnset', 'ConfROnset', 'ConfRT', 'ConfRating','Seen', 'Percept','Bias', 'Condition', 'StimPresent', 'PerceptC', 'CorrectSeen', 'CorrectPercept']

allRes.to_csv('NCI_' + subjNum + '.csv')


thanksText.draw()
win.flip()
core.wait(2)
win.close()

score = []

        
allpVals = np.unique(allRes.Bias.values)
allResults = allRes.dropna()

for i in allpVals:
    thisB = allResults.loc[allResults['Bias'] == i]
    count = sum(thisB['selection'] == 1)
    subProb = count/len(thisB)
    score.append(subProb)
    
import matplotlib.pyplot as plt
plt.plot(allpVals, score)  
    
    # Wait for a key press before proceeding to the next frame
    #event.waitKeys()

# Close the PsychoPy window

from scipy.optimize import curve_fit

def logiF(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))

def inverse_logistic(y, L, k, x0):
    return x0 - (1/k) * np.log(L/y - 1)

# data for fit
x_data = allpVals
y_data = score

ass = [1.0, 1.0, 0.5]

# Fit the curve
popt, pcov = curve_fit(logiF, x_data, y_data, p0=ass)

y_values = np.array([0.1, 0.25, 0.5, 0.75, 0.9])

# Compute corresponding x-values using the fitted parameters
x_values = inverse_logistic(y_values, *popt)

# Print the results
for y, x in zip(y_values, x_values):
    print(f"x corresponding to y = {y}: {x}")

# Plotting
x_fit = np.arange(-1, 1.1, 0.1)  # Adjust this to match your data
y_fit = logiF(x_fit, *popt)

plt.scatter(x_data, y_data, label='Data')
plt.plot(x_fit, y_fit, color='red', label='Fitted logistic curve')
plt.scatter(x_values, y_values, color='green', label='Specific y-values')
plt.legend()
plt.savefig(os.path.join(subjPath, 'NC' + subjNum + '_DataFit.png'))  # save the figure to file
plt.show()
plt.close()
   
fitVals = pd.Series(x_values)
fitVals.to_csv(os.path.join(subjPath, 'NC' + subjNum + '_DataFit.csv'))

win.close()

core.quit()



