#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 10:50:39 2025

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
subjPath = os.path.join(outPath, 'PBNC' + subjNum)
os.chdir(subjPath)
resFile = 'NCI_' + subjNum + '.csv'
while os.path.isfile(os.path.join(subjPath, resFile)):
    print("Error! This subject file exists already.")
    subjNum  = input("Please re-enter main participant identifier: ")    
    print(subjNum)
    
subjData = pd.read_csv('PBNC'+subjNum +'_IntakeData.csv')
lang = subjData.iloc[5,1]


#test screen resolution WARNING
# win = visual.Window([1200, 900], pos = (2,2), color=(0, 0, 0), units='pix')
#win = visual.Window([1512,982], [0, 0],useFBO=True, monitor="testMonitor", units="norm") #my laptop
win = visual.Window([1920,1080], [0, 0],useFBO=True, monitor="testMonitor", units="norm") #FCBG
#win = visual.Window(fullscr=True, monitor="testMonitor", units="norm")


n1 = []
n2 = []
stim_size = (0.833, 1.25)
element_size1 = 0.032  # or 0.0039 for finer noise
element_size2 = 0.016

#winner used
for nn in range(1152):
    n1.append(NoiseStim(
    win=win, name='noise1', units='norm',
    mask=None, ori=0.0, pos=(0, 0), size=((win.size[0]/3) / win.size[0] * 2.5, (win.size[1]/2) / win.size[1] * 2.5),
    opacity=2, blendmode='avg', contrast=1.5,
    texRes=512, noiseType='Binary', noiseElementSize=element_size1,  # Adjusted size
    noiseBaseSf=12.0/512, noiseFilterLower=3/512, noiseFilterUpper=10.0/512.0,
    interpolate=False, depth=-1.0
    ))

for nnn in range(1152):
    n2.append(NoiseStim(
    win=win, name='noise3', units='norm',
    mask=None, ori=0.0, pos=(0, 0), size=((win.size[0]/3) / win.size[0] * 2.5, (win.size[1]/2) / win.size[1] * 2.5),
    opacity=2, blendmode='avg', contrast=1.7,
    texRes=512, noiseType='Binary', noiseElementSize=element_size2,  # Adjusted size
    noiseBaseSf=12.0/512, noiseFilterLower=3/512, noiseFilterUpper=10.0/512.0,
    interpolate=False, depth=-1.0
    ))


if lang == 'E':      
    instrText = visual.TextStim(win, text = 'In this task, you will be presented with a series of noise masks. In between the noise masks, we may show you an image of a cube. We will then ask you if you saw a cube. If yes, press the rightmost button. If no, press the leftmost button. You will then be asked if you saw the cube from above (press right), or from below (press left). You must provide an answer even if you did not see the cube. You will then be asked how sure you are of your response on a scale from 1-10. Navigate through the scale with left and right buttons, and confirm your selection with either of the two middle buttons. Press any key to continue.', 
    font='Arial', height=0.07, wrapWidth=1.5, depth=0, rgb=None, color='black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
    ori=0.0,  antialias=True, bold=True, italic=False,  anchorVert='center', anchorHoriz='center',
    fontFiles=(), flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None)    
    
    Quest1 = visual.TextStim(win, text = 'Did you see a cube?', 
    font='Arial', height=0.07, wrapWidth=1.5, depth=0, rgb=None, color='black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
    ori=0.0,  antialias=True, bold=True, italic=False,  anchorVert='center', anchorHoriz='center',
    fontFiles=(), flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None)
    
    Quest2 = visual.TextStim(win, text = 'Frome below or from above?', 
    font='Arial',
    height=0.07, wrapWidth=1.5, depth=0, rgb=None, color='black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
    ori=0.0,  antialias=True, bold=True, italic=False,  anchorVert='center', anchorHoriz='center',
    fontFiles=(), flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None)
    
    thanksText = visual.TextStim(win=win, text='Thank you', font='Arial',
    height=0.07, wrapWidth=1.5, depth=0, rgb=None, color='black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
    ori=0.0,  antialias=True, bold=True, italic=False,  anchorVert='center', anchorHoriz='center',
    fontFiles=(), flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None)
    
    confQuestion= visual.TextStim(win, text = 'How sure are you of your last answer?', 
    font='Arial', height=0.07, wrapWidth=1.5, depth=0, rgb=None, color='black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
    ori=0.0,  antialias=True, bold=True, italic=False,  anchorVert='center', anchorHoriz='center',
    fontFiles=(), flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None, pos=(0, 0.4))
    
    startText = visual.TextStim(win=win, text='Press any key to begin',  font='Arial',
    height=0.07, wrapWidth=1.5, depth=0, rgb=None, color='black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
    ori=0.0,  antialias=True, bold=True, italic=False,  anchorVert='center', anchorHoriz='center',
    fontFiles=(), flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None)
    
elif lang == 'F':
    instrText = visual.TextStim(win, text = "Dans cette tâche, vous verrez une série de masques bruités. Entre ces masques, nous pourrons vous présenter un cube. Nous vous demanderons ensuite si vous avez vu un cube. Si oui, appuyez sur le bouton le plus à droite. Si non, appuyez sur le bouton le plus à gauche. Ensuite, nous vous demanderons si vous avez vu le cube d’en haut (appuyez à droite) ou d’en bas (appuyez à gauche). Vous devez fournir une réponse, même si vous n’avez pas vu le cube. Enfin, nous vous demanderons à quel point vous êtes sûr(e) de votre dernière réponse sur une échelle de 1 à 10. Naviguez l’échelle avec les boutons gauche et droite, puis  valider votre sélection avec l’un des boutons du milieu. Appuyez sur une touche pour continuer.", 
    font='Arial', height=0.07, wrapWidth=1.5, depth=0, rgb=None, color='black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
    ori=0.0,  antialias=True, bold=True, italic=False,  anchorVert='center', anchorHoriz='center',
    fontFiles=(), flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None)
    
    Quest1 = visual.TextStim(win, text = 'Avez-vous vu un cube?', 
    font='Arial', height=0.07, wrapWidth=1.5, depth=0, rgb=None, color='black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
    ori=0.0,  antialias=True, bold=True, italic=False,  anchorVert='center', anchorHoriz='center',
    fontFiles=(), flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None)
    
    Quest2 = visual.TextStim(win, text = "D'en haut ou d'en bas?", 
    font='Arial', height=0.07, wrapWidth=1.5, depth=0, rgb=None, color='black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
    ori=0.0,  antialias=True, bold=True, italic=False,  anchorVert='center', anchorHoriz='center',
    fontFiles=(), flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None)
    
    thanksText = visual.TextStim(win=win, text='Merci', font='Arial', height=0.07, wrapWidth=1.5, depth=0, rgb=None, color='black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
    ori=0.0,  antialias=True, bold=True, italic=False,  anchorVert='center', anchorHoriz='center',
    fontFiles=(), flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None)
    
    confQuestion= visual.TextStim(win, text = 'A quel degré êtes-vous sur de votre dernière réponse?', 
    font='Arial', height=0.07, wrapWidth=1.5, depth=0, rgb=None, color='black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
    ori=0.0,  antialias=True, bold=True, italic=False,  anchorVert='center', anchorHoriz='center',
    fontFiles=(), flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None, pos=(0, 0.4))
        
    startText = visual.TextStim(win=win, text='Appuyez sur un bouton pour commencer', 
    font='Arial', height=0.07, wrapWidth=1.5, depth=0, rgb=None, color='black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
    ori=0.0,  antialias=True, bold=True, italic=False,  anchorVert='center', anchorHoriz='center',
    fontFiles=(), flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None)
    

FixationText = visual.TextStim(win=win, text='+', font='Arial', height=0.1, wrapWidth=1.5, depth=0, rgb=None, color='black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
ori=0.0,  antialias=True, bold=True, italic=False,  anchorVert='center', anchorHoriz='center',
fontFiles=(), flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None)

scannerWait = visual.TextStim(win, text = 'Please wait for scanner...', 
font='Arial', height=0.07, wrapWidth=1.5, depth=0, rgb=None, color='black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
ori=0.0,  antialias=True, bold=True, italic=False,  anchorVert='center', anchorHoriz='center',
fontFiles=(), flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None)

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

# bk = visual.rect.Rect(win, units='pix', width = win.size[0]/4, height=win.size[1]/3, lineWidth=1, lineColor=False, fillColor='white', colorSpace='rgb', pos=(0, 0), size=None, anchor=None, ori=0.0, opacity=None, contrast=1.0, depth=0, interpolate=True, draggable=False, name=None, autoLog=None, autoDraw=False, color=None, lineColorSpace=None, fillColorSpace=None, lineRGB=False, fillRGB=False)

bk_width_norm = (win.size[0]/3) / win.size[0] * 2.5  # scale from pixels to [-1, 1]
bk_height_norm = (win.size[1]/2) / win.size[1] * 2.5


bk = visual.Rect(
    win,
    width=bk_width_norm,
    height=bk_height_norm,
    units='norm',
    fillColor='white',
    lineColor=None,
    pos=(0, 0)
)

# Define cube vertices (homogeneous coordinates)
vup = np.array([
    [1, 1, 1, 1], [1, 1, -1, 1], [1, -1, 1, 1], [1, -1, -1, 1],
    [-1, 1, 1, 1], [-1, 1, -1, 1], [-1, -1, 1, 1], [-1, -1, -1, 1]
])
vdn = vup.copy()
vdn[:, 2] *= -1  # Mirror Z

# Cube edge list
edges = [
    [0, 1], [0, 2], [1, 3], [2, 3],
    [0, 4], [1, 5], [2, 6], [3, 7],
    [4, 5], [4, 6], [5, 7], [6, 7]
]

# Projection helpers
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

# Scene parameters
rx, ry = win.size
aspect = rx / ry
cam = camera(aspect)
phi = np.pi / 2.6
theta = np.arctan(np.sqrt(1.6))
d = 2
m4up = rotmat(phi, 'z') @ rotmat(theta, 'x') @ transmat(0, 0, d)
m4dn = rotmat(phi, 'z') @ rotmat(-theta, 'x') @ transmat(0, 0, d)

# Function to render cube with given interpolation and text

def draw_cube_intro(p2, message):
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
    
def draw_cube(p2):
    vup_proj = vup @ m4up @ np.linalg.inv(cam)
    vdn_proj = vdn @ m4dn @ np.linalg.inv(cam)
    vclip = p2 * vup_proj + (1 - p2) * vdn_proj
    vclip[:, :3] /= vclip[:, 3:]  # Perspective divide
    vscreen = vclip[:, :2] * 0.7  # Scale up to desired size
    for start, end in edges:
        visual.Line(win, start=vscreen[start], end=vscreen[end], lineColor='black', lineWidth=6).draw()


    
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
# dur = np.hstack([np.tile(0.0167,54), np.tile(0.200,54)])
# # dur = np.tile(0.016,270)
# cond1C = np.zeros(54)
# cond2C = np.ones(54)
dur = np.hstack([np.tile(0.0167,66), np.tile(0.200,66)])
# dur = np.tile(0.016,270)
cond1C = np.zeros(65)
cond2C = np.ones(65)
condC = np.hstack([cond1C, cond2C])

cond1P = np.ones(108)
cond2P = np.zeros(22)
condP = np.hstack([cond1P, cond2P])

# Generate interpolation parameter array `p`
p = np.arange(-0.9, 1, 0.2)
p = np.insert(p, 5, 0)
p = np.delete(p,[4,6])
# percept = np.array([-1, -1, -1, -1, 0, 1, 1, 1, 1])
percept = np.array([-1, -1, -1, -1, 0, 1, 1, 1, 1])
pss= np.tile(p, 12)
Percepts = np.tile(percept, 12)
x = np.tile(np.nan, 22)
y = np.tile(np.nan, 22)
ps = np.hstack([pss, x])
Percept =  np.hstack([Percepts, y])

# dat1 = pd.concat([pd.Series(Percept), pd.Series(ps), pd.Series(condP)], axis = 1)
# dat2 = pd.concat([pd.Series(condC), pd.Series(dur)], axis = 1)
# dat3 = pd.concat([dat1, dat2], axis = 1)

# gratingType = condC
random.Random(9).shuffle(ps)
random.Random(9).shuffle(Percept)
random.Random(8).shuffle(condC)
random.Random(9).shuffle(condP)
random.Random(8).shuffle(dur)

dat11 = pd.concat([pd.Series(Percept), pd.Series(ps), pd.Series(condP)], axis = 1)
dat22 = pd.concat([pd.Series(condC), pd.Series(dur)], axis = 1)
dat33 = pd.concat([dat11, dat22], axis = 1)
dat33.columns = ['Percept', 'Bias', 'Present', 'ConsciousCondition', 'Duration']



random.Random(5).shuffle(noise)

# random.Random(7).shuffle(gratingType)

dused = []
nused = []
nsize = []
val = []
condition = []
present = []
gused = []


# Function to calculate edge length (assuming all edges are the same length)
def calculate_edge_length(vertices):
    return np.linalg.norm(vertices[0] - vertices[1])  # Assuming consistent edge length

# Random line generator in cube's 2D space
def generate_random_lines_in_cube_area(num_lines=12, length=0.15):
    lines = []
    for _ in range(num_lines):
        start = np.random.uniform(-0.35, 0.35, size=2)
        angle = np.random.uniform(0, 2 * np.pi)
        end = start + length * np.array([np.cos(angle), np.sin(angle)])
        if np.all(np.abs(end) <= 0.35):
            lines.append((start, end))
    return lines

# # Generate 12 random lines
# num_lines = 12
# random_lines = generate_fixed_length_lines_within_cube(num_lines, edge_length)


startText = visual.TextStim(win=win, text='Press any key to begin', font='Arial', pos=(0, 0),
depth=0, rgb=None, color='black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
ori=0.0, height = 0.07, antialias=True, bold=True, italic=False, anchorVert='center', anchorHoriz='center',
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
    firstText = visual.TextStim(win=win, text='This is a Necker Cube. Press any key to continue.', font='', pos=(0, -0.6),
    depth=0, rgb=None, color='black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
    ori=0.0, height = 0.05, antialias=True, bold=True, italic=False, anchorVert='center', anchorHoriz='center',
    fontFiles=(), flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None)
    
    secondText = visual.TextStim(win=win, text='This is a Necker Cube viewed from below. Press any key to continue.', font='', pos=(0, -0.6),
    depth=0, rgb=None, color='black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
    ori=0.0, height = 0.05, antialias=True, bold=True, italic=False, anchorVert='center', anchorHoriz='center',
    fontFiles=(), flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None)
    
    thirdText = visual.TextStim(win=win, text='This is a Necker Cube viewed from above. Press any key to continue.', font='', pos=(0, -0.6),
    depth=0, rgb=None, color='black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
    ori=0.0, height = 0.05, antialias=True, bold=True, italic=False, anchorVert='center', anchorHoriz='center',
    fontFiles=(), flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None)

elif lang == 'F': 
    
    firstText = visual.TextStim(win=win, text='Ceci est le cube Necker. Appuyez sur une touche pour continuer.', font='', pos=(0, -0.6),
    depth=0, rgb=None, color='black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
    ori=0.0, height = 0.05, antialias=True, bold=True, italic=False, anchorVert='center', anchorHoriz='center',
    fontFiles=(), flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None)
    
    secondText = visual.TextStim(win=win, text="Ceci est le cube Necker vu d'en bas. Appuyez sur une touche pour continuer.", font='', pos=(0, -0.6),
    depth=0, rgb=None, color='black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
    ori=0.0, height = 0.05, antialias=True, bold=True, italic=False, anchorVert='center', anchorHoriz='center',
    fontFiles=(), flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None)
    
    thirdText = visual.TextStim(win=win, text="Ceci est le cube Necker vu d'en haut. Appuyez sur une touche pour continuer.", font='', pos=(0, -0.6),
    depth=0, rgb=None, color='black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
    ori=0.0, height = 0.05, antialias=True, bold=True, italic=False, anchorVert='center', anchorHoriz='center',
    fontFiles=(), flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None)


#ambiguous Cube 
draw_cube_intro(0.5, firstText)
win.flip()
#event.waitKeys()

#Below Bias Cube 
draw_cube_intro(0.0, secondText)
win.flip()
#event.waitKeys()

#Above Bias Cube 
draw_cube_intro(1, thirdText)
win.flip()
#event.waitKeys()



clock = core.Clock()
startText.draw()
win.flip()
event.waitKeys()

scannerWait.draw()
win.flip()


logFile = os.path.join(outPath, "outputNC" + subjNum + ".txt")
with open(logFile, "a") as f:
    trigger = event.waitKeys(keyList = ['5'], clearEvents=True, timeStamped=True) 
    event.clearEvents()  #  Clean buffer
    jitter = random.uniform(1,2)
    #block key 5 to keep console clear
    #keyboard.block_key("5")

    trigTime = clock.getTime()
    startExp = clock.getTime()
    subjData.loc[6,1] = trigTime
    subjData.loc[7,1] = startExp
    print('Trigger at ' + str(trigTime), file=f)
    print('Trigger at ' + str( trigTime))
    print('Start at ' + str(startExp), file=f)
    print('Start at ' + str(startExp))
    
        # Determine number of trials
    n_trials = len(condC)
    
    # Determine indices where condP == 1
    condP_1_indices = [i for i in range(n_trials) if condP[i] == 1 and condC[i] in [0, 1]]
    
    # Split them by condC value
    condP1_condC0 = [i for i in condP_1_indices if condC[i] == 0]
    condP1_condC1 = [i for i in condP_1_indices if condC[i] == 1]
    
    # Determine how many to sample from each (half of the total condP==1 trials)
    n_conf_total = len(condP_1_indices) // 2
    n_each = n_conf_total // 2
    
    # Randomly sample the trials
    random.seed(42)  # for reproducibility
    conf_trials_condC0 = random.sample(condP1_condC0, n_each)
    conf_trials_condC1 = random.sample(condP1_condC1, n_each)
    
    # Create a boolean array of whether to ask the confidence question
    ask_conf = [False] * n_trials
    for idx in conf_trials_condC0 + conf_trials_condC1:
        ask_conf[idx] = True




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
        #random lines
        random_lines = generate_random_lines_in_cube_area()
        
        
        if condC[im] == 1:    
            n = random.sample(noise, 4)
            for nm in range(4): 
                n[nm].draw() 
                random_lines = generate_random_lines_in_cube_area()
            # Draw the random lines
                for start, end in random_lines:
                    visual.Line(
                    win,
                    start=start,
                    end=end,
                    lineColor=(-1, -1, -1),
                    lineWidth=6,
                    units='norm'  # <- this is the key fix
                    ).draw()
                win.flip()
                core.wait(0.066)
            
            
            if condP[im] == 1:
                # stim.draw()
                # Draw lines for the cube
                for _ in range(frC):
                    bk.draw()
                    draw_cube(p2)                      
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
                random_lines = generate_random_lines_in_cube_area()
            # Draw the random lines
                for start, end in random_lines:
                    visual.Line(
                    win,
                    start=start,
                    end=end,
                    lineColor=(-1, -1, -1),
                    lineWidth=6,
                    units='norm'  # <- this is the key fix
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
                random_lines = generate_random_lines_in_cube_area()
            # Draw the random lines
                for start, end in random_lines:
                    visual.Line(
                    win,
                    start=start,
                    end=end,
                    lineColor=(-1, -1, -1),
                    lineWidth=6,
                    units='norm'  # <- this is the key fix
                    ).draw()
                win.flip()
                core.wait(0.066)
    
            if condP[im] == 1:
                # stim.draw()
                # Draw lines for the cube
                for _ in range(frUC):
                    bk.draw()
                    draw_cube(p2)  
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
                random_lines = generate_random_lines_in_cube_area()
            # Draw the random lines
                for start, end in random_lines:
                    visual.Line(
                    win,
                    start=start,
                    end=end,
                    lineColor=(-1, -1, -1),
                    lineWidth=6,
                    units='norm'  # <- this is the key fix
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
        if ask_conf[im]:
    # confidence question logic
            confQO = []
            while confScale.getRating() is None:
                confScale.draw()
                confQuestion.draw()
                win.flip()
                confQO.append(clock.getTime())
                ckeys = event.getKeys(keyList=['1', '2', '3', '4'])
                if cleftKeys in ckeys:
                    confScale.markerPos = max(confScale.markerPos - 1, 1)
                elif crightKeys in ckeys:
                    confScale.markerPos = min(confScale.markerPos + 1, 10)
                elif any(key in ckeys for key in cacceptKeys):
                    rating = confScale.getMarkerPos()
                    c2 = clock.getTime()
                    print(f"Rating accepted: {rating}")
                    t1 = confQO[0]
                    confQOns.append(t1)
                    confROns.append(c2)
                    break
        
            rating = confScale.getMarkerPos()
            confRT.append(t1 - c2)
            print('Subject reported a confidence of ' + str(rating) + ' at ' + str(c2), file=f)
            print('Subject reported a confidence of ' + str(rating) + ' at ' + str(c2))
            confRating.append(rating)
        else:
            # Append NAs for trials without confidence
            confQOns.append(np.nan)
            confROns.append(np.nan)
            confRT.append(np.nan)
            confRating.append(np.nan)

        # while confScale.getRating() is None:          # Keep looping until a value is selected
        #     confScale.draw()
        #     confQuestion.draw()
        #     win.flip()
        #     confQO.append(clock.getTime())
        #     ckeys = event.getKeys()                     
        #     if cleftKeys in ckeys:                  # Move left
        #         confScale.markerPos = max(confScale.markerPos - 1, 1)
        #     elif crightKeys in ckeys:               # Move right
        #         confScale.markerPos = min(confScale.markerPos + 1, 10)
        #     elif any(key in ckeys for key in cacceptKeys):  # Accept response
        #         rating = confScale.getMarkerPos()  # Retrieve the rating when Enter is pressed
        #         c2 = clock.getTime()
                
                
        #         print(f"Rating accepted: {rating}")
        #         t1 = confQO[0]
        #         confQOns.append(t1)
        #         confROns.append(c2)
        #         break       
        
        # rating = confScale.getMarkerPos()
        # confRT.append(t1-c2)
        # print('Subject reported a confidence of ' + str(rating) + ' at ' + str(c2), file = f)
        # print('Subject reported a confidence of ' + str(rating) + ' at ' + str(c2))
        # confRating.append(rating)
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
        
        print('Duration in trial ' + str(im) +' is ' + str(dur[im]), file=f)
        print('Duration in trial ' + str(im) +' is ' + str(dur[im]))
        print('Percept in trial ' + str(im) +' is ' + str(Percept[im]), file=f)
        print('Percept in trial ' + str(im) +' is ' + str(Percept[im]))

        
        FixationText.draw()
        win.flip()
        core.wait(4.5 + jitter)

####  Fix column names/values below!!!!
allRes =pd.concat([pd.Series(stimPres), pd.Series(q1Pres), pd.Series(response1), pd.Series(r1Ons), pd.Series(q2Pres), pd.Series(response2), pd.Series(r2Ons), pd.Series(confQOns),pd.Series(confROns), pd.Series(confRT),pd.Series(confRating),  pd.Series(val), pd.Series(condition), pd.Series(present)], axis = 1)
allRes.columns =['StimOnset', 'Q1Onset', 'Seen', 'Resp1Onset', 'Q2Onset', 'Percept','Resp2Onset', 'ConfQOnset', 'ConfROnset', 'ConfRT', 'ConfRating','Bias', 'Condition', 'StimPresent']

endExp = clock.getTime()
subjData.loc[8,1] = endExp
subjData.to_csv('PBNC' +subjNum +'_IntakeData.csv')
expDur = endExp - startExp

print('Experiment lasted ' + str(expDur))



orientation = []
correct1 = []
correct2 = []


for i in range(len(allRes)):
    if  allRes.Bias.iloc[i] > 0:
        orientation.append('Up')
    elif allRes.Bias.iloc[i] == 0:
        orientation.append('Amb')
    elif allRes.Bias.iloc[i] < 0:
        orientation.append('Down')
    else:
        orientation.append('Null')

orientationC = []  
for i in range(len(allRes)):
    if '4' in allRes.Percept.iloc[i]:
        orientationC.append('Up')
    elif '1' in allRes.Percept.iloc[i]:
       orientationC.append('Down')
        

for i in range(len(allRes)):
    if orientationC[i] in orientation[i]:
        correct2.append(1)
    elif allRes.Bias.iloc[i] == 0:
        correct2.append(np.nan)
    else:
        correct2.append(0)
        
for i in range(len(allRes)):
    if '4' in allRes.Seen.iloc[i] and allRes.StimPresent.iloc[i] == 0:
        correct1.append(1)
    elif '4' in allRes.Seen.iloc[i] and allRes.StimPresent.iloc[i] == 1:
        correct1.append(0)
    elif '1' in allRes.Seen.iloc[i] and allRes.StimPresent.iloc[i] == 0:
            correct1.append(0)
    elif '1' in allRes.Seen.iloc[i] and allRes.StimPresent.iloc[i] == 1:
            correct1.append(-1)     
        
allRes = pd.concat([allRes, pd.Series(orientationC), pd.Series(correct1),pd.Series(correct2)], axis = 1)
allRes.columns =['StimOnset', 'Q1Onset', 'Seen', 'Resp1Onset', 'Q2Onset', 'Percept','Resp2Onset', 'ConfQOnset', 'ConfROnset', 'ConfRT', 'ConfRating','Bias', 'Condition', 'StimPresent', 'PerceptC', 'CorrectSeen', 'CorrectPercept']

allRes.to_csv('NCI_' + subjNum + '.csv')


thanksText.draw()
win.flip()
core.wait(2)
win.close()

score = []

        
allpVals = np.unique(allRes.Bias.values)
allpVals = [x for x in allpVals if not np.isnan(x)]
allResults = allRes[allRes.StimPresent == 1]
allResults = allResults[allResults.Condition == 1]

#check with Barbara if there should be a catch here
for i in allpVals:
    thisB = allResults.loc[allResults['Bias'] == i]
    count = (thisB['Percept'] == 1).sum()
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

#### work on this ####
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
#x_fit = np.arange(-1, 1.1, 0.1)  # Adjust this to match your data
x_fit = np.linspace(min(x_data), max(x_data), 200)
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



