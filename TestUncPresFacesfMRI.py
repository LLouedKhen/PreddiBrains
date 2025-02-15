#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 11:02:33 2024

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

prefs.general['windowType'] ='pygame'
from memory_profiler import profile


MR_settings = { 
    'TR': 2.4, # duration (sec) per volume CHECK
    'volumes': 375, # number of whole-brain 3D volumes / frames
    'sync': '5', # character to use as the sync timing event; assumed to come at start of a volume
    'skip': 0, # number of volumes lacking a sync pulse at start of scan (dummies)
    'sound': True    # in test mode: play a tone as a reminder of scanner noise
    }



stimPath ='/Users/sysadmin/Documents/PreddiBrains/Stimuli/Faces' 
mStimPath = '/Users/sysadmin/Documents/PreddiBrains/Stimuli/Faces/Males' 
fStimPath = '/Users/sysadmin/Documents/PreddiBrains/Stimuli/Faces/Females' 
path = '/Users/sysadmin/Documents/PreddiBrains/'
outPath = '/Users/sysadmin/Documents/PreddiBrains/Output/Imaging/FA'


subjNum = input("Enter main participant identifier: ") 
subjPath = os.path.join(outPath, 'PBF' + subjNum)
os.chdir(subjPath)
resFile = 'FaceI_' + subjNum + '.csv'
while os.path.isfile(os.path.join(subjPath, resFile)):
    print("Error! This subject file exists already.")
    subjNum  = input("Please re-enter main participant identifier: ")    
    print(subjNum)
    

subjData = pd.read_csv('PBF'+subjNum +'_IntakeData.csv')
lang = subjData.iloc[5,1]

win = visual.Window([1512,982], [0, 0], monitor="testMonitor", units="pix")

n1 = []
n2 = []
#winner used
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


p = []
g = []
r = []

imgs = []
percept = []

os.chdir(mStimPath)
mImgs = glob.glob('*M*.png')
mIds = []
fIds = []
for i in mImgs:
    isp = i.split('_')
    mIds.append(isp[0])

umIds = list(set(mIds))
smIds = random.sample(umIds,1)
    
os.chdir(fStimPath)
fImgs = glob.glob('*F*.png')

for i in fImgs:
    isp = i.split('_')
    fIds.append(isp[0])

ufIds = list(set(fIds))
sfIds = random.sample(ufIds,1)

whichG = random.randint(1,2)

if whichG == 1:
    subjData.loc[6,1] = 'M'
    for i in mImgs:
        #if smIds[0] in i or smIds[1] in i:
        if smIds[0] in i:
            thisIm = os.path.join(mStimPath, i)
            imgs.append(thisIm)
elif whichG == 2:
    subjData.loc[6,1] = 'F'
    for j in fImgs:
        #if sfIds[0] in j or sfIds[1] in j:
        if sfIds[0] in j:
            thisIm = os.path.join(fStimPath, j)
            imgs.append(thisIm)
        
accept= ['H30','H60', 'H80', 'H50', 'H20', 'H40', 'H70']
imgs = [i for i in imgs if any(allow in i for allow in accept)]

        
for m in imgs:
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


if lang == 'E':      
    instrText = visual.TextStim(win, text = 'In this task, you will be presented with a series of noise masks. In between the noise masks, we may show you an image of a face. We will then ask you if you saw a face. If yes, press the rightmost button. If no, press the leftmost button. You will then be asked if the face you saw was happy (press right), or angry(press left). You must provide an answer even if you did not see the face. You will then be asked how sure you are of your response on a scale from 1-10. Navigate through the scale with left and right buttons, and confirm your selection with either of the two middle buttons. Press any key to continue.', 
    font='', pos=(0, 0), depth=0, rgb=None, color= 'black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
    ori=0.0, height = 32, antialias=True, bold=False, italic=False,  anchorVert='center', anchorHoriz='center',
    fontFiles=(), wrapWidth=None, flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None)  
    
    
    
    Quest1 = visual.TextStim(win, text = 'Did you see a face?', 
    font='', pos=(0, 0), depth=0, rgb=None, color= 'black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
    ori=0.0, height = 32, antialias=True, bold=False, italic=False,  anchorVert='center', anchorHoriz='center',
    fontFiles=(), wrapWidth=None, flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None)  
    
    Quest2 = visual.TextStim(win, text = 'Angry or happy?', 
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
    instrText = visual.TextStim(win, text = "Dans cette tâche, vous verrez une série de masques bruités. Entre ces masques, nous pourrons vous présenter un visage. Nous vous demanderons ensuite si vous avez vu un visgae. Si oui, appuyez sur le bouton le plus à droite. Si non, appuyez sur le bouton le plus à gauche. Ensuite, nous vous demanderons si le visage etait heureux (appuyez à droite) ou fâché (appuyez à gauche). Vous devez fournir une réponse, même si vous n’avez pas vu le visage. Enfin, nous vous demanderons à quel point vous êtes sûr(e) de votre dernière réponse sur une échelle de 1 à 10. Naviguez l’échelle avec les boutons gauche et droite, puis  valider votre sélection avec l’un des boutons du milieu. Appuyez sur une touche pour continuer.", 
    font='', pos=(0, 0), depth=0, rgb=None, color= 'black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
    ori=0.0, height = 32, antialias=True, bold=False, italic=False,  anchorVert='center', anchorHoriz='center',
    fontFiles=(), wrapWidth=None, flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None)  

    
    Quest1 = visual.TextStim(win, text = 'Avez-vous vu un visage?', 
    font='', pos=(0, 0), depth=0, rgb=None, color= 'black', colorSpace='rgb', opacity=1.0, contrast=1.0, units='', 
    ori=0.0, height = 32, antialias=True, bold=False, italic=False,  anchorVert='center', anchorHoriz='center',
    fontFiles=(), wrapWidth=None, flipHoriz=False, flipVert=False, languageStyle='LTR', name=None, autoLog=None)  
    
    Quest2 = visual.TextStim(win, text = 'Faché ou heureux?', 
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

# fStimPath = '/Users/sysadmin/Documents/PreddiBrains/Stimuli/Faces/Females' 
# os.chdir(fStimPath)
# stims = glob.glob("*H*.png")
stims = imgs
stimAll = []
stimScr = []
sName = []
imIDs = []

for r in range(10):
    for s in stims:
# img_color = Image.open('/Users/sysadmin/Documents/PreddiBrains/Stimuli/Faces/Females/WF003_H50A50.png')
        img_color = Image.open(s)
        img_grey = img_color.convert('L')
        scrambled_image = scramble_image(img_grey)
        stimScr.append(visual.ImageStim(win, image=scrambled_image, pos = [0,0], size = [736/2, 1080/2], opacity = 0.5))
        stimAll.append(visual.ImageStim(win, image=img_grey, pos = [0,0], size = [736/2, 1080/2]))
        sName.append(s.split('_')[1][:-4])
        imIDs.append(s)
        

response1 = []
response2 = []

noise = n1 + n2
dur = np.hstack([np.tile(0.200, 42), np.tile(0.016, 42)])

frC = 12 # for my Mac, ca 60 Hz refresh rate. Adjust to approx. 100-150 ms
frUC = 1 # for my Mac, 60 Hz refresh rate. Adjust to approx. 16-24 ms

cond1C = np.ones(42)
cond2C = np.zeros(42)
condC = np.hstack([cond1C, cond2C])
cond1P = np.ones(70)
cond2P = np.zeros(14)
condP = np.hstack([cond1P, cond2P])

sNames = sName + sName + list(np.tile(np.nan, 14))
imIDs = imIDs + imIDs + list(np.tile(np.nan, 14))
allStims = stimAll + stimAll + list(np.tile(np.nan, 14))
allStimscr = stimScr + stimScr + list(random.sample(stimScr, 14));

p = p * 5 + list(np.tile(np.nan, 14))
g = g * 5 + list(np.tile(np.nan, 14))
percept = percept * 5 + list(np.tile(np.nan, 14))


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


clock = core.Clock()
instrText.draw()
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
    subjData.loc[7,1] = trigTime
    subjData.loc[8,1] = startExp
    print('Trigger at ' + str(startExp), file=f)
    print('Trigger at ' + str(startExp))
    print('Start at ' + str(startExp), file=f)
    print('Start at ' + str(startExp))


    for im in range(len(condC)): 
        thisP = p[im]
        print('Bias in trial ' + str(im) +' is ' + str(thisP), file=f)
        print('Bias in trial ' + str(im) +' is ' + str(thisP))
        print('Condition in trial ' + str(im) +' is ' + str(condC[im]), file=f)
        print('Condition in trial ' + str(im) +' is ' + str(condC[im]))
        print('Presence in trial ' + str(im) +' is ' + str(condP[im]), file=f)
        print('Presence in trial ' + str(im) +' is ' + str(condP[im]))
        
        if condC[im] == 1:
            stim = allStims[im]
            sScr = allStimscr[im]
            # n = noise[im00]
            d = dur[im]
            n = random.sample(noise, 4)
            for nm in range(4): 
                n[nm].draw() 
                sScr.draw()
                win.flip()
                core.wait(0.066)
            # n.draw() 
            # win.flip()
            # core.wait(0.120)
            if condP[im] == 1:
                for _ in range(frC):
                    stim.draw()
                    win.flip()
                    if _ == 0:
                        sP = clock.getTime()
                        stimPres.append(clock.getTime())
                        print('Stimulus presented at ' + str(sP), file=f)
                        print('Stimulus presented at  ' + str(sP))
                #core.wait(0.080)
            elif condP[im] == 0:
                for _ in range(frC):
                    win.flip()
                    if _ == 0:
                        sP = clock.getTime()
                        stimPres.append(clock.getTime())
                        print('Stimulus presented at ' + str(sP), file=f)
                        print('Stimulus presented at ' + str(sP))
                # core.wait(0.080)      
            n = random.sample(noise, 4)
            for nm in range(4): 
                n[nm].draw() 
                sScr.draw()
                win.flip()
                core.wait(0.066)
            # n.draw() 
            FixationText.draw()
            win.flip()
            # core.wait(0.120)
            core.wait(4.5 + jitter)
            Quest1.draw()
            win.flip()
            q1P = clock.getTime()
            q1Pres.append(q1P)
            
        elif condC[im] == 0:
            stim = allStims[im]
            sScr = allStimscr[im]
            # n = noise[im]
            d = dur[im]
            n = random.sample(noise, 4)
            for nm in range(4): 
                sScr.draw()
                n[nm].draw() 
                win.flip()
                core.wait(0.066)
            # n.draw() 
            # win.flip()
            # core.wait(0.120)
            if condP[im] == 1:
                for _ in range(frUC):
                    stim.draw()
                    win.flip()
                    if _ == 0:
                        sP = clock.getTime()
                        stimPres.append(clock.getTime())
                        print('Stimulus presented at ' + str(sP), file=f)
                        print('Stimulus presented at ' + str(sP))
                #core.wait(d)
            elif condP[im] == 0:
                for _ in range(frUC):
                    win.flip()
                    if _ == 0:
                        sP = clock.getTime()
                        stimPres.append(clock.getTime())
                        print('Stimulus presented at ' + str(sP), file=f)
                        print('Stimulus presented at ' + str(sP))
                # core.wait(d)     
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
        dused.append(d)
        # nsize.append(n.noiseElementSize)
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

allRes =pd.concat([pd.Series(stimPres), pd.Series(q1Pres), pd.Series(response1), pd.Series(r1Ons), pd.Series(q2Pres), pd.Series(response2), pd.Series(r2Ons), pd.Series(confQOns),pd.Series(confROns), pd.Series(confRT),pd.Series(confRating), pd.Series(bias), pd.Series(condition),pd.Series(present), pd.Series(dur), pd.Series(val), pd.Series(Id)], axis = 1)
allRes.columns =['StimOnset', 'Q1Onset', 'Seen', 'Resp1Onset', 'Q2Onset', 'Emotion','Resp2Onset', 'ConfQOnset', 'ConfROnset', 'ConfRT', 'ConfRating', 'Bias', 'Condition', 'StimPresent', 'Duration','Sname', 'StimID']
endExp = clock.getTime()
subjData.loc[9,1] = endExp

subjData.to_csv('PBF' +subjNum +'_IntakeData.csv')

expDur = endExp - startExp
print('Experiment lasted ' + str(expDur))


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
allRes.columns =['StimOnset', 'Q1Onset', 'Seen', 'Resp1Onset', 'Q2Onset', 'Emotion','Resp2Onset', 'ConfQOnset', 'ConfROnset', 'ConfRT', 'ConfRating','Bias', 'Condition', 'StimPresent', 'Duration','Sname', 'StimID', 'PerceptC', 'CorrectSeen', 'CorrectPercept', 'Gender']
allRes.to_csv('FaceI_' + subjNum + '.csv')


thanksText.draw()
win.flip()
core.wait(2)
win.close()
    
#unblock trigger key
#keyboard.unblock_key('5')

score= []
     
allpVals = np.unique(allRes.Bias.values)
allRes = allRes.dropna()


import matplotlib.pyplot as plt
for i in allpVals:
    thisB = allRes.loc[allRes['Bias'] == i]
    count = sum(thisB['selection'] == 'h')
    subProb = count/len(thisB)
    score.append(subProb)
    

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

y_values = np.array([0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

# Compute corresponding x-values using the fitted parameters
x_values = inverse_logistic(y_values, *popt)

# Print the results
for y, x in zip(y_values, x_values):
    print(f"x corresponding to y = {y}: {x}")

# Plotting
x_fit = np.arange(0, 1.1, 0.1)  # Adjust this to match your data
y_fit = logiF(x_fit, *popt)

plt.scatter(x_data, y_data, label='Data')
plt.plot(x_fit, y_fit, color='red', label='Fitted logistic curve')
plt.scatter(x_values, y_values, color='green', label='Specific y-values')
plt.legend()
plt.show()
plt.savefig(os.path.join(subjPath, 'PBF' + subjNum + '_DataFit.png'))  # save the figure to file
plt.close()
   
fitVals = pd.Series(x_values)
fitVals.to_csv(os.path.join(subjPath, 'PBF' + subjNum + '_DataFit.csv'))

