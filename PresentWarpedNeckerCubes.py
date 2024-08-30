#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 11:34:58 2024

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


stimPath ='/Users/sysadmin/Documents/PreddiBrains/Stimuli' 
img = os.path.join(stimPath, 'NeckerCube.png')
path = '/Users/sysadmin/Documents/PreddiBrains/'
outPath = '/Users/sysadmin/Documents/PreddiBrains/Output/NC/Pilot'
os.chdir(outPath)

def subjInfo(outPath):
    subjNum = input("Enter main participant identifier: ") 
    if os.path.isdir(subjNum):
      subjPath = os.path.join(outPath, subjNum)  
    else: 
      os.mkdir(subjNum)
      subjPath = os.path.join(outPath, subjNum)
    return subjNum, subjPath

subjNum, subjPath = subjInfo(outPath)
os.chdir(subjPath)
if os.path.isfile('NC' + subjNum + '_pilotData.csv'):
    print('Warning! This subject data exists already! Please enter another subject number.')
    subjInfo(outPath)

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
win = visual.Window([800, 600], color=(0, 0, 0), units='pix')

Instr = visual.TextStim(win, text = 'In the following experiment, you will be shown cubes. Please indicate if you see the cube from above, by pressing 4, or from below, by pressing 1. Press any key to begin.', 
font='', pos=(0, 0), depth=0, rgb=None, color= 'black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
ori=0.0, height=None, antialias=True, bold=False, italic=False, alignHoriz='center', alignVert='center',
fontFiles=(), wrapWidth=None, flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None)  

Quest1 = visual.TextStim(win, text = 'Did you see an object?', 
font='', pos=(0, 0), depth=0, rgb=None, color= 'black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
ori=0.0, height=None, antialias=True, bold=False, italic=False, alignHoriz='center', alignVert='center',
fontFiles=(), wrapWidth=None, flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None)  

Quest2 = visual.TextStim(win, text = 'Did you see it from above or from below?', 
font='', pos=(0, 0), depth=0, rgb=None, color= 'black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
ori=0.0, height=None, antialias=True, bold=False, italic=False, alignHoriz='center', alignVert='center',
fontFiles=(), wrapWidth=None, flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None)  

Quest3= visual.TextStim(win, text = 'How sure are you of your response?', 
font='', pos=(0, 0), depth=0, rgb=None, color= 'black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
ori=0.0, height=None, antialias=True, bold=False, italic=False, alignHoriz='center', alignVert='center',
fontFiles=(), wrapWidth=None, flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None) 

FixationText = visual.TextStim(win=win, text='+', font='', pos=(0, 0),
depth=0, rgb=None, color='black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
ori=0.0, height=None, antialias=True, bold=True, italic=False, alignHoriz='center', alignVert='center',
fontFiles=(), wrapWidth=None, flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None)

noise1 = NoiseStim(
                win=win, name='noise',units='pix',
                noiseImage=img, mask=None,
                ori=1.0, pos=(0, 0), size=(512, 512), sf=None, phase=0,
                color=[1,1,1], colorSpace='rgb', opacity=1, blendmode='add', contrast=1.0,
                texRes=512, filter='None', imageComponent='Phase',
                noiseType='Gabor', noiseElementSize=4, noiseBaseSf=32.0/512,
                noiseBW=1.0, noiseBWO=30, noiseFractalPower=-1,noiseFilterLower=3/512, noiseFilterUpper=8.0/512.0,
                noiseFilterOrder=3.0, noiseClip=3.0, interpolate=False, depth=-1.0)

sbias = []
rpercept = []
response = []
rt = []
RT = []
correct = []
selection = []
# confidence = []
# confidenceRT = []
t0 = [] #stim appears
t1 = [] #AFCq
t2 = [] #AFCr
# t3 = [] #confScale
# t4 = [] #confResponse

logFile = os.path.join(subjPath, "outputNCFull" + subjNum + ".txt")
with open(logFile, "a") as f:
    
    clock = core.Clock()
    Instr.draw()
    win.flip()
    startExp = clock.getTime()
    print('Experiment Started at', startExp)
    event.waitKeys()
    FixationText.draw()
    core.wait(1)
    win.flip()
    
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
    sc = 2 # Scaling factor for the cube
    
    # Generate interpolation parameter array `p`
    p = np.arange(-1,1.1,0.1)
    p[10] = 0
    percept = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    ps= np.tile(p, 10)
    Percept = np.tile(percept, 10)
    random.Random(4).shuffle(ps)
    random.Random(4).shuffle(Percept)
    
    trial = 0
    # Iterate over each parameter in `allP`
    for thisp in ps:
        trial +=1
        print("Trial Number", trial)
        print("Outer loop p:", thisp)
        
        p2 = 0.5 + thisp / 2  # Interpolation factor
    
        # Interpolate between `vup` and `vdn`, apply scaling
        vclip = p2 * vup + (1 - p2) * vdn
        vclip[:, :3] *= sc
        
        # Convert 3D points to 2D screen coordinates
        vscreen = clip2screen(vclip, win.size[0]/2, win.size[1]/2)
        cube_center = np.mean(vscreen, axis=0)
    
        # Draw lines for the cube
        for line in lines:
           start, end = line
          
           # Draw each line segment of the cube
           visual.Line(win, start=vscreen[line[0]] - cube_center, end=vscreen[line[1]] - cube_center, lineColor=(-1, -1, -1), lineWidth=8).draw()
    
        # Display the updated window
        win.flip()
        thisT0 = clock.getTime()
        print("Cube  ", thisp, " presented at time ", thisT0)
        t0.append(thisT0)
        core.wait(1)
        noise1.draw()
        win.flip()
        core.wait(0.133)
    
        # Quest1.draw()
        # win.flip()
        # event.waitKeys(['1', '4'])
        FixationText.draw()
        win.flip()
        core.wait(1)
        Quest2.draw()
        win.flip()
        thisT1 = clock.getTime()
        t1.append(thisT1)
        resp = event.waitKeys(keyList = ['1', '4'], clearEvents=True, timeStamped=True)
        thisT2 = clock.getTime()
        t2.append(thisT2)
        RT.append(thisT2 - thisT1)
        press = resp[0][0]
        response.append(press)
        if '1' in press:
            selection.append(-1)
            print("subject selected ", press, " at time ", thisT2)
        elif '4' in press:
            selection.append(1)
            print("subject selected ", press, " at time ", thisT2)
            
        thisRT = resp[0][1]
        rt.append(thisRT)
        sbias.append(thisp)
        rpercept.append(Percept[trial-1])
        if selection[trial-1] == rpercept[trial-1]:
            correct.append(1)
            print("Congruent Selection.")
        else:
            correct.append(0)
            print("Incongruent Selection.")
            
            
        # confScale = visual.RatingScale(win, low=1, high=10, markerStart= random.randint(1,10), choices =None,scale= None, acceptPreText =None, 
        # showValue = None, showAccept = None, labels=None, 
        # leftKeys='1', rightKeys = '4', acceptKeys=['2','3'], textColor = 'black')
        # confScale.reset()
        # while confScale.noResponse:
        #     Quest3.draw()
        #     confScale.draw()
        #     win.flip()    
        # rating = confScale.getRating()
        # decision_time = confScale.getRT()
        # thisT3 = clock.getTime()
        # t3.append(thisT3)
        # confidence.append(rating)
        # confidenceRT.append(decision_time)
        
        FixationText.draw()
        win.flip()
        core.wait(1)
        event.clearEvents()
    
    endExp = clock.getTime()
    print('Experiment ended at ', endExp)
    print('Experiment took ', endExp - startExp)
    allResults = pd.concat([pd.Series(response), pd.Series(selection), pd.Series(RT), pd.Series(rpercept), pd.Series(sbias), pd.Series(t0), pd.Series(t1), pd.Series(t2)], axis=1)
    allResults.columns =['Response', 'selection','RT', 'Percept', 'Bias', 't0', 't1', 't2']
    allResults.to_csv(os.path.join(subjPath, 'NC' + subjNum + '_pilotData.csv'))
    endExp = clock.getTime()
        
    win.close()

score = []

        
allpVals = np.unique(allResults.Bias.values)
allResults = allResults.dropna()

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
plt.show()
plt.savefig(os.path.join(subjPath, 'NC' + subjNum + '_DataFit.png'))  # save the figure to file
plt.close()
   
fitVals = pd.Series(x_values)
fitVals.to_csv(os.path.join(subjPath, 'NC' + subjNum + '_DataFit.csv'))

win.close()

core.quit()

