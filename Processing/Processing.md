# Mini Guide - How to run the analysis in python

Author : @barbaragrosjean

*The goal of this folder is to run the preprocessing and the first level analysis using python pipline based on fsl.*

## Before starting 
Important step before starting is to get you a proper environnement to run the files. 

### A. Set up your environnement
The required library are : 
- fsl 
- nibabel
- argparse
- shutil
- subprocess
I highly recommand you to set up a proper environnement for the project.

### B. How to use the functions
For the step *Data preparation* and *Preprocessing* you call the file preproc.py in the terminal with the following command line : \

`python preproc.py --subj 001 --sess 01`

Your environnement should be set in advance and activate in the terminal. 

`conda activate *your_env_name*`

For the steps *First Level analysis* for now it's a jupyter notebook.

*Note:* You can call the function for all subject using `--subj 'all' --sess 'all'`.

## 1. Data preparation
The DICOM data are in a zip file (1 zip file per subject per session) in the folder /data/raw. The fMRI sequence used is a multiECHO sequence and all the following step will take it into account. To convert the DICOM file into niftii format the function DICOM2Nifiti that use mainly dcm2niix is used. A selection of the file is then down by the function extract_usfull_file. \
Those two functions are called in the main of the file preproc.py \
At the end of this step you should have a folder called /raw/PB_*** for each subject and each session where you can find your raw data (2 run, 3 echos per session per subject). The zip folder is moved into an /Archive folder.

## 2. Preprocessing 
The preprocessing is specific for multi echo fMRI and it is devided into 8 steps, all the steps are called sequentially in the function prepocessing called in the main of the file preproc.py. Until the combination of the echo each step is run by echo but the transformations are computed on the first echo and then apply to all the echo as recommanded for mulit Echo preprocessing.

**Steps to preprocess multi-echo images:**
- Slice timing correction - using `slicetimer`
- Motion correciton - using `mcflirt`
- Field map correction - using `fsl_prepare_fieldmap`
- Combine echo using the function combine_echo called in the main of preproc.py. it mostly rely on `tedana` (info here : https://tedana.readthedocs.io/en/stable/).
- Skull stripping T1 and MNI - using `optiBET.sh` \
From *Lutkenhoff ES, Rosenberg M, Chiang J, Zhang K, Pickard JD, Owen AM, Monti MM. (2014) Optimized Brain Extraction for Pathological Brains (optiBET). PLoS ONE 9(12): e115551. doi:10.1371/journal.pone.0115551* because bet did not worked.
- T1 co-Registration - using `flirt`
- Normalization - using `flirt`
- 8mm-smoothing using `fslmaths`

*NOTE: All the functions mention are supposed to be called in the main prepoc.py*

Once that all your preprocessing steps run you should have the following structre for each subject and session in /data.

<pre> data/ 
├── AAL3/ 
│   ├── MNI.nii 
│   ├── MNI_optiBET_brain_mask.nii.gz 
│   └── MNI_optiBET_brain.nii.gz 
├── PB_***/ 
|   ├── anat/ 
│   ├── fieldmap/ 
│   ├── func/ 
│   └── preproc/ 
├── raw/ 
└── trash/ </pre>


*NOTE* : 
- You can throw away the folder trash at any point.
- The AAL3 folder is used to normalize the data you can download MNI file on internet.
- the preprocess images are in /preproc under the name sm_func_MNI_PB___run__.nii.gz

### 2. First Level analysis - subject level GLM
For now the behaivoral data are in data/PB__sess__/behav. 

#### A. Design matrix 
From Leyla design matrix we have the following design matrix: 

*PUT THE IMAGE HERE*

That you can find in the folder /Processing.

You can run the first level analysis using the first_level_analysis.ipynb file that is a jupyter notebook to vizualize the area of activation.



