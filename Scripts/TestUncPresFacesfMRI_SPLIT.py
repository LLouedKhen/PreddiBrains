#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 15:58:48 2025

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
import glob
from PIL import Image
from psychopy import prefs
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from memory_profiler import profile

from utils import scramble_image, logiF, inverse_logistic

prefs.general['windowType'] ='pygame'

MR_settings = { 
    'TR': 2.4, # duration (sec) per volume CHECK
    'volumes': 375, # number of whole-brain 3D volumes / frames
    'sync': '5', # character to use as the sync timing event; assumed to come at start of a volume
    'skip': 0, # number of volumes lacking a sync pulse at start of scan (dummies)
    'sound': True    # in test mode: play a tone as a reminder of scanner noise
    }


# /!\ Change the path to your local path 
path = '/Users/sysadmin/Documents/PreddiBrains/'
#path = '/Users/barbaragrosjean/Desktop/CHUV/PreddiBrains/'

stimPath = path + 'Stimuli/Faces' 
outPath = path + 'Output/Imaging/FA'

############################
###### Set the Subject #####
############################

subjNum = input("Enter main participant identifier: ") 
subjPath = os.path.join(outPath, 'PBF' + subjNum)
os.chdir(subjPath)
resFile = 'FaceI_' + subjNum + '.csv'
while os.path.isfile(os.path.join(subjPath, resFile)):
    print("Error! This subject file exists already.")
    subjNum  = input("Please re-enter main participant identifier: ")    
    print(subjNum)
    

subjData = pd.read_csv('PBF'+subjNum +'_IntakeData.csv').drop(columns = 'Unnamed: 0')
lang = subjData.Language.loc[0]

############################
###### Set the Window ###### 
############################

# Set the display parameters
width_norm = 1.6  # corresponds to 16:10 aspect
height_norm = 1.0

# Create window with norm units
#win = visual.Window([1920, 1080],[0, 0], units="norm", fullscr=False) 
win = visual.Window(size=[1920, 1080], screen=1, units="norm", fullscr=True)#FCBG 

# Get actual window size in pixels1
win_width, win_height = win.size
win_aspect = win_width / win_height

# Your image size in pixels
img_width = 736
img_height = 1080
img_aspect = img_width / img_height

# Decide how tall (in norm units) the image should appear (max = 2 vertically)
norm_height = 1.5  # fill most of vertical space, but leave margins
norm_width = norm_height * img_aspect / win_aspect

# Use this size for all your ImageStims
norm_image_size = [norm_width, norm_height]

###########################
###### Set the Noise ###### 
###########################

n1 = []
n2 = []
for nn in range(1152):
    n1.append(NoiseStim(
    win=win, name='noise', units='pix',
    mask=None, ori=0.0, pos=(0, 0), size=(736, 1080),
    opacity=1, blendmode='avg', contrast=1.7,
    texRes=512, noiseType='Binary', noiseElementSize=7,  # Adjusted size
    noiseBaseSf=8.0/512, noiseFilterLower=3/512, noiseFilterUpper=8.0/512.0,
    interpolate=False, depth=-1.0
))

for nnn in range(1152):
    n2.append(NoiseStim(
    win=win, name='noise', units='pix',
    mask=None, ori=0.0, pos=(0, 0), size=(736, 1080),
    opacity=1, blendmode='avg', contrast=1.7,
    texRes=512, noiseType='Uniform', noiseElementSize=14,  # Adjusted size
    noiseBaseSf=8.0/512, noiseFilterLower=3/512, noiseFilterUpper=8.0/512.0,
    interpolate=False, depth=-1.0))

noise = n1 + n2

######################################
#### Set the Stimulus and Percept ####
######################################

p = []
g = []
r = []

# Define stims 
stims = []
genders = ['Males', 'Females']
whichG = np.random.choice(genders, 1)[0]
genderpath = stimPath + '/' + whichG
os.chdir(genderpath)
Imgs = glob.glob(f'*{whichG[:1]}*.png') 

Ids = []
for i in Imgs:
        isp = i.split('_')
        Ids.append(isp[0]) 

uIds = list(set(Ids)) 
sIds = random.sample(uIds,1) # Took only 1 randomly

for i in Imgs:
        if sIds[0] in i:
            thisIm = os.path.join(genderpath, i)
            stims.append(thisIm)

accept= ['H30','H60', 'H80', 'H50', 'H20', 'H40', 'H70']
stims = [i for i in stims if any(allow in i for allow in accept)]

# Define Percept 
percept = []    
for m in stims:
    if 'M' in m:
        g.append(1)
    else:
        g.append(0)
    if 'A100' in m:
        continue
    elif 'H100' in m:
        continue
    elif 'H80' in m:
        p.append(0.8)
        percept.append('H')
    elif 'H70' in m:
        p.append(0.7)
        percept.append('A')
    elif 'H60' in m:
        p.append(0.6)
        percept.append('H')
    elif 'H50' in m:
        p.append(0.5)
        percept.append('N')
    elif 'H40' in m:
        p.append(0.4)
        percept.append('A')
    elif 'H30' in m:
        p.append(0.3)
        percept.append('A')
    elif 'H20' in m:
        p.append(0.2)
        percept.append('A')

##########################
###### Set the Text ######
##########################

if lang == 'E':      
    instrText = visual.TextStim(win, text = 'In this task, you will be presented with a series of noise masks. In between the noise masks, we may show you an image of a face. We will then ask you if you saw a face. If yes, press the rightmost button. If no, press the leftmost button. You will then be asked if the face you saw was happy (press right), or angry(press left). You must provide an answer even if you did not see the face. You will then be asked how sure you are of your response on a scale from 0-10. Navigate through the scale with left and right buttons, and confirm your selection with either of the two middle buttons. Press any key to continue.', 
    font='', pos=(0, 0), depth=0, rgb=None, color= 'black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
    ori=0.0, height = 0.07, antialias=True, bold=False, italic=False,  anchorVert='center', anchorHoriz='center',
    fontFiles=(), wrapWidth=None, flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None)  
    
    Quest1 = visual.TextStim(win, text = 'Did you see a face?', 
    font='', pos=(0, 0), depth=0, rgb=None, color= 'black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
    ori=0.0, height = 0.07, antialias=True, bold=False, italic=False,  anchorVert='center', anchorHoriz='center',
    fontFiles=(), wrapWidth=None, flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None)  
    
    Quest2 = visual.TextStim(win, text = 'Angry or happy?', 
    font='', pos=(0, 0), depth=0, rgb=None, color= 'black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
    ori=0.0, height = 0.07, antialias=True, bold=False, italic=False, anchorVert='center', anchorHoriz='center',
    fontFiles=(), wrapWidth=None, flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None)  
    
    thanksText = visual.TextStim(win=win, text='Thank you', font='', pos=(0, 0),
    depth=0, rgb=None, color='black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
    ori=0.0, height = 0.07, antialias=True, bold=True, italic=False, alignHoriz='center', alignVert='center',
    fontFiles=(), wrapWidth=None, flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None)
    
    confQuestion= visual.TextStim(win, text = 'How sure are you of your last answer?', 
    font='', pos=(0, 0.4), depth=0, rgb=None, color= 'black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
    ori=0.0, height=0.07, antialias=True, bold=False, italic=False, alignHoriz='center', alignVert='center',
    fontFiles=(), wrapWidth=None, flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None) 
    
    startText = visual.TextStim(win=win, text='Press any key to begin', font='', pos=(0, 0),
    depth=0, rgb=None, color='black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
    ori=0.0, height = 0.07, antialias=True, bold=True, italic=False,  anchorVert='center', anchorHoriz='center',
    fontFiles=(), wrapWidth=None, flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None)

    scannerWait = visual.TextStim(win, text = 'Please wait for scanner...', 
    font='', pos=(0, 0), depth=0, rgb=None, color= 'black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
    ori=0.0, height=0.07, antialias=True, bold=False, italic=False, alignHoriz='center', alignVert='center',
    fontFiles=(), wrapWidth=None, flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None)  
    
    restText = visual.TextStim(win=win, text='Rest...', font='Arial', pos=(0, 0),
    depth=0, rgb=None, color='black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
    ori=0.0, height = 0.07, antialias=True, bold=True, italic=False, anchorVert='center', anchorHoriz='center',
    fontFiles=(), wrapWidth=None, flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None)

    
elif lang == 'F':
    instrText = visual.TextStim(win, text = "Dans cette tâche, vous verrez une série de masques bruités. Entre ces masques, nous pourrons vous présenter un visage. Nous vous demanderons ensuite si vous avez vu un visage. Si oui, appuyez sur le bouton le plus à droite. Si non, appuyez sur le bouton le plus à gauche. Ensuite, nous vous demanderons si le visage etait heureux (appuyez à droite) ou fâché (appuyez à gauche). Vous devez fournir une réponse, même si vous n’avez pas vu le visage. Enfin, nous vous demanderons à quel point vous êtes sûr(e) de votre dernière réponse sur une échelle de 0 à 10. Naviguez l’échelle avec les boutons gauche et droite, puis  valider votre sélection avec l’un des boutons du milieu. Appuyez sur une touche pour continuer.", 
    font='', pos=(0, 0), depth=0, rgb=None, color= 'black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
    ori=0.0, height = 0.07, antialias=True, bold=False, italic=False,  anchorVert='center', anchorHoriz='center',
    fontFiles=(), wrapWidth=None, flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None)  
    
    Quest1 = visual.TextStim(win, text = 'Avez-vous vu un visage?', 
    font='', pos=(0, 0), depth=0, rgb=None, color= 'black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
    ori=0.0, height = 0.07, antialias=True, bold=False, italic=False,  anchorVert='center', anchorHoriz='center',
    fontFiles=(), wrapWidth=None, flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None)  
    
    Quest2 = visual.TextStim(win, text = 'Faché ou heureux?', 
    font='', pos=(0, 0), depth=0, rgb=None, color= 'black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
    ori=0.0, height = 0.07, antialias=True, bold=False, italic=False, anchorVert='center', anchorHoriz='center',
    fontFiles=(), wrapWidth=None, flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None)  
    
    thanksText = visual.TextStim(win=win, text='Merci', font='', pos=(0, 0),
    depth=0, rgb=None, color='black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
    ori=0.0, height = 0.07, antialias=True, bold=True, italic=False, alignHoriz='center', alignVert='center',
    fontFiles=(), wrapWidth=None, flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None)
    
    confQuestion= visual.TextStim(win, text = 'A quel degré êtes-vous sur de votre dernière réponse?', 
    font='', pos=(0, 0.4), depth=0, rgb=None, color= 'black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
    ori=0.0, height=0.07, antialias=True, bold=False, italic=False, alignHoriz='center', alignVert='center',
    fontFiles=(), wrapWidth=None, flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None) 
    
    startText = visual.TextStim(win=win, text='Appuyez sur un bouton pour commencer', font='', pos=(0, 0),
    depth=0, rgb=None, color='black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
    ori=0.0, height = 0.07, antialias=True, bold=True, italic=False,  anchorVert='center', anchorHoriz='center',
    fontFiles=(), wrapWidth=None, flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None)

    scannerWait = visual.TextStim(win, text = "S'il vous plait, attendez le scanner ...", 
    font='', pos=(0, 0), depth=0, rgb=None, color= 'black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
    ori=0.0, height=0.07, antialias=True, bold=False, italic=False, alignHoriz='center', alignVert='center',
    fontFiles=(), wrapWidth=None, flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None)  
    
    restText = visual.TextStim(win=win, text='Repos...', font='Arial', pos=(0, 0),
    depth=0, rgb=None, color='black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
    ori=0.0, height = 0.07, antialias=True, bold=True, italic=False, anchorVert='center', anchorHoriz='center',
    fontFiles=(), wrapWidth=None, flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None)

FixationText = visual.TextStim(win=win, text='+', font='', pos=(0, 0),
depth=0, rgb=None, color='black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
ori=0.0, height = 0.07, antialias=True, bold=True, italic=False,  anchorVert='center', anchorHoriz='center',
fontFiles=(), wrapWidth=None, flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None)

# Define stimulus list 
stimAll = []
stimScr = []
sName = []
imIDs = []

for r in range(6):
    for s in stims:
        img_color = Image.open(s)
        img_grey = img_color.convert('L')
        scrambled_image = scramble_image(img_grey)
        stimScr.append(visual.ImageStim(win, image=scrambled_image, pos = [0,0], size=norm_image_size, opacity = 0.5))
        stimAll.append(visual.ImageStim(win, image=img_grey, pos = [0,0], size=norm_image_size))
        sName.append(s.split('_')[1][:-4])
        imIDs.append(s)

#############################################
###### Set the Duration and Resolution ###### 
#############################################
dur = np.hstack([np.tile(0.200, 51), np.tile(0.016, 51)])
frC = 12 # for my Mac, ca 60 Hz refresh rate. Adjust to approx. 100-150 ms
frUC = 1 # for my Mac, 60 Hz refresh rate. Adjust to approx. 16-24 ms

###############################
###### Set the Condition ###### 
###############################

condC = np.hstack([np.ones(51), np.zeros(51)])
condP = np.hstack([np.ones(84), np.zeros(18)])

sNames = sName + sName + list(np.tile(np.NaN, 18)) 
imIDs = imIDs + imIDs + list(np.tile(np.NaN, 18))
allStims = stimAll + stimAll + list(np.tile(np.NaN, 18))
allStimscr = stimScr + stimScr + list(random.sample(stimScr, 18))
p = p * 12 + list(np.tile(np.nan, 18))
g = g * 12 + list(np.tile(np.nan, 18))
percept = percept * 12 + list(np.tile(np.nan, 18))

# Randomize 
seed = random.randint(1,5)
random.Random(seed).shuffle(p)
random.Random(seed).shuffle(g)
random.Random(seed).shuffle(percept)
random.Random(seed).shuffle(allStims)
random.Random(seed).shuffle(allStimscr)
random.Random(seed).shuffle(condP)
random.Random(seed).shuffle(sNames)
random.Random(seed).shuffle(imIDs)
random.Random(9).shuffle(condC)
random.Random(9).shuffle(dur)
random.Random(8).shuffle(noise)

dat11 = pd.concat([pd.Series(percept), pd.Series(p), pd.Series(condP)], axis = 1)
dat22 = pd.concat([pd.Series(condC), pd.Series(dur)], axis = 1)
dat33 = pd.concat([dat11, dat22], axis = 1)
dat33.columns = ['Percept', 'Bias', 'Present', 'ConsciousCondition', 'Duration']
dat33.to_csv('StimulusPresentationFace_' + subjNum +'.csv')

random.Random(5).shuffle(noise)

####################################
###### Set the Confidence Bar ###### 
####################################

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

# Determine which trials to ask the confidence question
n_trials = len(condC)

# Get indices where condP == 1 and split by condC
condP1_indices = [i for i in range(n_trials) if condP[i] == 1]
condP1_C0 = [i for i in condP1_indices if condC[i] == 0]
condP1_C1 = [i for i in condP1_indices if condC[i] == 1]

# Total confidence questions to ask: half of all condP == 1 trials
n_conf_total = len(condP1_indices) // 2
n_each = n_conf_total // 2

# Randomly sample trials from each condC group
random.seed(42)
ask_conf_trials_C0 = random.sample(condP1_C0, n_each)
ask_conf_trials_C1 = random.sample(condP1_C1, n_each)

# Build ask_conf list
ask_conf = [False] * n_trials
for idx in ask_conf_trials_C0 + ask_conf_trials_C1:
    ask_conf[idx] = True

#############################
###### Set the Storage ###### 
############################# 

response1 = []
response2 = []

dused = []
nused = []
nsize = []
val = []
condition = []
present = []
bias = []
gender = []
Percepts = []
Id = []


stimPres = []
q1Pres = []
r1Ons = []
r1RT = []
q2Pres = []
r2Ons = []
r2RT = []
confQOns = []
confROns = []
confRating = []
confRT = []

#_____________________ Starting Task _____________________
clock = core.Clock()
instrText.draw()
win.flip()
event.waitKeys()
scannerWait.draw()
win.flip()

for block in range(2):
    b = block
    logFile = os.path.join(subjPath, "outputNC" + subjNum + "Block" + str(b) + ".txt")
    scannerWait.draw()
    win.flip()
    with open(logFile, "a") as f:
        trigger = event.waitKeys(keyList = ['5'], clearEvents=True, timeStamped=True) 
        jitter = random.uniform(1,2)
    
        trigTime = clock.getTime()
        startExp = clock.getTime()
        if b == 0:
            subjData.Trigger1.loc[0] = trigTime
            subjData.StartTime1.loc[0] = startExp
        else:
            subjData.Trigger2.loc[0] = trigTime
            subjData.StartTime2.loc[0] = startExp
        print('Trigger at ' + str(startExp), file=f)
        print('Trigger at ' + str(startExp))
        print('Start at ' + str(startExp), file=f)
        print('Start at ' + str(startExp))
    
    
        if b == 0:
            tRange = np.arange(0,len(condC)/2)
        elif b == 1:
            tRange = np.arange(len(condC)/2, len(condC))
        for im in tRange:
            im = int(im)
            thisP = p[im]
            print('Bias in trial ' + str(im) +' is ' + str(thisP), file=f)
            print('Bias in trial ' + str(im) +' is ' + str(thisP))
            print('Condition in trial ' + str(im) +' is ' + str(condC[im]), file=f)
            print('Condition in trial ' + str(im) +' is ' + str(condC[im]))
            print('Presence in trial ' + str(im) +' is ' + str(condP[im]), file=f)
            print('Presence in trial ' + str(im) +' is ' + str(condP[im]))
            
            # Stim presentation
            if condC[im] == 1: # Conscious condition
                stim = allStims[im]
                sScr = allStimscr[im]
                d = dur[im]
                n = random.sample(noise, 4)
                for nm in range(4): 
                    n[nm].draw() 
                    sScr.draw()
                    win.flip()
                    core.wait(0.066)
                
                if condP[im] == 1: # stim present
                    for _ in range(frC):
                        stim.draw()
                        win.flip()
                        if _ == 0:
                            sP = clock.getTime()
                            stimPres.append(clock.getTime())
                            print('Stimulus presented at ' + str(sP), file=f)
                            print('Stimulus presented at  ' + str(sP))
    
                elif condP[im] == 0: # stim not present
                    for _ in range(frC):
                        win.flip()
                        if _ == 0:
                            sP = clock.getTime()
                            stimPres.append(clock.getTime())
                            print('Stimulus presented at ' + str(sP), file=f)
                            print('Stimulus presented at ' + str(sP))
        
                n = random.sample(noise, 4)
                for nm in range(4): 
                    n[nm].draw() 
                    sScr.draw()
                    win.flip()
                    core.wait(0.066)
    
                FixationText.draw()
                win.flip()
    
                core.wait(4.5 + jitter)
                Quest1.draw()
                win.flip()
                q1P = clock.getTime()
                q1Pres.append(q1P)
                
            elif condC[im] == 0: # Not conscious condition
                stim = allStims[im]
                sScr = allStimscr[im]
    
                d = dur[im]
                n = random.sample(noise, 4)
                for nm in range(4): 
                    sScr.draw()
                    n[nm].draw() 
                    win.flip()
                    core.wait(0.066)
        
                if condP[im] == 1: # Stim present
                    for _ in range(frUC):
                        stim.draw()
                        win.flip()
                        if _ == 0:
                            sP = clock.getTime()
                            stimPres.append(clock.getTime())
                            print('Stimulus presented at ' + str(sP), file=f)
                            print('Stimulus presented at ' + str(sP))
    
                elif condP[im] == 0: # stim not present 
                    for _ in range(frUC):
                        win.flip()
                        if _ == 0:
                            sP = clock.getTime()
                            stimPres.append(clock.getTime())
                            print('Stimulus presented at ' + str(sP), file=f)
                            print('Stimulus presented at ' + str(sP))
      
                n = random.sample(noise, 4)
                for nm in range(4): 
                    n[nm].draw() 
                    sScr.draw()
                    win.flip()
                    core.wait(0.066)
                FixationText.draw()
                win.flip()
                core.wait(4 + jitter)
                Quest1.draw()
                win.flip()
                q1Pres.append(clock.getTime())
                
                
            # Take the answer 
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
            core.wait(2)
    
            # Ask for confidence
            if ask_conf[im]:  # Only ask on selected trials
                confQO = []
                while confScale.getRating() is None:
                    confScale.draw()
                    confQuestion.draw()
                    win.flip()
                    confQO.append(clock.getTime())
                    ckeys = event.getKeys()
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
                        confRT.append(t1 - c2)
                        break
            
                rating = confScale.getMarkerPos()
                #confRT.append(t1 - c2)
                print('Subject reported a confidence of ' + str(rating) + ' at ' + str(c2), file=f)
                print('Subject reported a confidence of ' + str(rating) + ' at ' + str(c2))
                confRating.append(rating)
            else:
                print('Subject confidence not asked')
                confQOns.append('Not asked')
                confROns.append('Not asked')
                confRT.append('Not asked')
                confRating.append('Not asked')
    
            # Save 
            dused.append(d)
            val.append(sNames[im])
            Percepts.append(percept[im])
            bias.append(p[im])
            gender.append(g[im])
            condition.append(condC[im])
            present.append(condP[im])
            Id.append(imIDs[im])
            
            print('Duration in trial ' + str(im) +' is ' + str(d), file=f)
            print('Duration in trial ' + str(im) +' is ' + str(d))
            print('Percept in trial ' + str(im) +' is ' + str(percept[im]), file=f)
            print('Percept in trial ' + str(im) +' is ' + str(percept[im]))
            print('Stim Name in trial ' + str(im) +' is ' + str(sNames[im]), file=f)
            print('Stim Name in trial ' + str(im) +' is ' + str(sNames[im]))
            print('Stim Id in trial ' + str(im) +' is ' + str(imIDs[im]), file=f)
            print('Stim Id in trial ' + str(im) +' is ' + str(imIDs[im]))
            
            FixationText.draw()
            win.flip()
            core.wait(3.5 + jitter)
    
        # Save in csv
        allRes =pd.concat([pd.Series(stimPres), pd.Series(q1Pres), pd.Series(response1), pd.Series(r1Ons), pd.Series(q2Pres), pd.Series(response2), pd.Series(r2Ons), pd.Series(confQOns),pd.Series(confROns), pd.Series(confRT),pd.Series(confRating), pd.Series(bias), pd.Series(condition),pd.Series(present), pd.Series(dur), pd.Series(val), pd.Series(Id)], axis = 1)
        allRes.columns =['StimOnset', 'Q1Onset', 'Seen', 'Resp1Onset', 'Q2Onset', 'Emotion','Resp2Onset', 'ConfQOnset', 'ConfROnset', 'ConfRT', 'ConfRating', 'Bias', 'Condition', 'StimPresent', 'Duration','Sname', 'StimID']
        
        endExp = clock.getTime()
        if b == 0:
            subjData.EndTime1.loc[0] = endExp
        else: 
            subjData.EndTime2.loc[0]  = endExp
            
    subjData.to_csv('PBF' +subjNum +'_IntakeData.csv')
    expDur = endExp - startExp
    print('Experiment lasted ' + str(expDur))

    #_____________________ End of the Task _____________________
    
    # Compute and Save scores
    correct1 = []
    correct2 = []
    
    H = ['H60', 'H70', 'H80']
    A = ['H20', 'H30', 'H40']
    
    em = []
    for i in range(len(allRes)):
        if  allRes.Bias.iloc[i] > 0.5:
            em.append('happy')
        elif allRes.Bias.iloc[i] == 0.5:
            em.append('neutral')
        elif allRes.Bias.iloc[i] < 0.5:
            em.append('angry')
        else:
            em.append('null')
    
    emC = []  
    for i in range(len(allRes)):
        emotion = allRes.Emotion.iloc[i] 
        if '4' in emotion:
            emC.append('happy')
        elif '1' in emotion:
            emC.append('angry')
        else:
            emC.append(np.nan)
    
            
    for i in range(len(allRes)):
        if em[i] in emC[i]:
            correct2.append(1)
        else:
            correct2.append(0)
            
    for i in range(len(allRes)):
        if '4' in allRes.Seen.iloc[i] and allRes.StimPresent.iloc[i] == 1:
            correct1.append(0)
        elif '4' in allRes.Seen.iloc[i] and allRes.StimPresent.iloc[i] == 0:
            correct1.append(-1)
        elif '1' in allRes.Seen.iloc[i] and allRes.StimPresent.iloc[i] == 0:
            correct1.append(0)
        elif '1' in allRes.Seen.iloc[i] and allRes.StimPresent.iloc[i] == 1:
            correct1.append(1)
    
            
    os.chdir(subjPath)     
    
    allRes = pd.concat([allRes, pd.Series(emC), pd.Series(correct1), pd.Series(correct2), pd.Series(gender)], axis = 1)
    allRes = allRes.rename(columns = {0 :'PerceptC', 1: 'CorrectSeen', 2: 'CorrectPercept', 3:'Gender'})
    
    allRes.to_csv('FaceI_' + subjNum + 'Block' + str(b) + '.csv')
    
    if b == 0:
        restText.draw()
        win.flip()

thanksText.draw()
win.flip()
core.wait(2)
win.close()

score= []
    
# sol form leyla mail 
allpVals = np.unique(allRes.Bias.values)
allpVals = [x for x in allpVals if not np.isnan(x)]
allResults = allRes[allRes.StimPresent == 1]
allResults = allResults[allResults.Condition == 1]

for i in allpVals:
    thisB = allResults.loc[allResults['Bias'] == i]
    count = sum(thisB['Emotion'].astype(str) =='1') # to be sure to have str and not float
    subProb = count/len(thisB)
    score.append(subProb)

# logfit 
ass = [1.0, 1.0, 0.5]

# Fit the curve
popt, pcov = curve_fit(logiF, allpVals, score, p0=ass)
# y_values = np.arange(0, 11)*0.1
y_values = allpVals
x_values = inverse_logistic(y_values, *popt)

# Print the results
for y, x in zip(y_values, x_values):
    print(f"x corresponding to y = {y}: {x}")

# Plotting
# x_fit = np.arange(0, 1.1, 0.1)  # Adjust this to match your data
x_fit = allpVals  # Adjust this to match your data
y_fit = logiF(x_fit, *popt)

fig, ax = plt.subplots()

ax.scatter(allpVals, score, label='Data')
ax.plot(x_fit, y_fit, color='red', label='Fitted logistic curve')
ax.scatter(x_values, y_values, color='green', label='Specific y-values')
ax.legend()
plt.show()
fig.savefig(os.path.join(subjPath, 'PBF' + subjNum + '_DataFit.png'))  # save the figure to file
plt.close()
   
fitVals = pd.Series(x_values)
fitVals.to_csv(os.path.join(subjPath, 'PBF' + subjNum + '_DataFit.csv'))

