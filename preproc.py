## Preprocessing Pipline -- 1

import os 
import subprocess
import nibabel as nib
import argparse
import itertools
import shutil
import pandas as pd
import json

from fsl.wrappers import mcflirt, fslmaths, fslmerge

def buildArgsParser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter,
        epilog="")
    
    p._optionals.title = "Generic options"

    # subject
    p.add_argument('--subj', nargs='+', dest='subj', help="Subject index.") # 000 

    # session
    p.add_argument('--sess', nargs='+', dest='sess', help="Session folder name.") # 02 or ' ' 

    # datapath
    p.add_argument('--data_path', default='/Users/barbaragrosjean/Desktop/CHUV/PreddiBrains/data',
                   dest='data_path', help="Data folder path. ['%(default)s']")
    
    return p

def DICOM2Nifti(data_path:str, subj:str, sess:str):

    subj_path = data_path + '/raw/' + subj + sess

    if not os.path.exists(subj_path):
        os.makedirs(subj_path)

    # unzip the folder PB_subj_sess
    from zipfile import ZipFile
    ZipFile(subj_path + '.zip').extractall(subj_path)

    # get the name of the DICOM folder
    dicom_folder = os.listdir(subj_path)

    if len(dicom_folder) != 1 : 
        print("Warning multiple dicom folder!")
    else :
        # convert the dicom folder to nifti 
        command = ["dcm2niix", "-z", "y", "-o", subj_path, "-f", "%p_%s",subj_path + '/' + dicom_folder[0]]
        try:
            subprocess.run(command, check=True)
            print("Conversion done! :)")
        except subprocess.CalledProcessError as e:
            print(f'ERROR : {subj}, {command} ', e)

        # remove the .zip and the DICOM
        shutil.rmtree(subj_path + '/' + dicom_folder[0])
        os.remove(subj_path + '.zip')

        # remove the files that do not match the 4D size
        nifti_files = [file for file in os.listdir(subj_path) if file[-6:] == 'nii.gz']

        for nifti_file in nifti_files : 
            # identify the run files
            filepath = subj_path + '/' + nifti_file
            img = nib.load(filepath)
            if len(img.shape) >= 4 : 
                if img.shape[3] < 5 : 

                    if not os.path.isdir(subj_path.replace('raw', 'trash')):
                        os.makedirs(subj_path.replace('raw', 'trash'))

                    # move to trash folder nii.gz + json associated
                    shutil.move(filepath, filepath.replace('raw', 'trash'))
                    shutil.move(filepath[:-6] + 'json', filepath.replace('raw', 'trash')[:-6] + 'json')
            else :
                continue

def extract_usfull_file(data_path:str, subj:str, sess:str,  nb_run=2, nb_echo=3):
    subj_folder = data_path + '/' + subj + sess
    subj_raw_path = data_path + '/raw/' + subj + sess
    trash_folder = data_path + '/trash/' + subj + sess

    if not os.path.exists(subj_folder) : 
        os.makedirs(subj_folder + '/func')
        os.makedirs(subj_folder + '/anat')
    
    # Select the runs files
    files = [f for f in os.listdir(subj_raw_path) if f[-6:]=='nii.gz']

    # Sorte files
    for file in files:
        img = nib.load(subj_raw_path + '/' +file)

        # Runs
        if len(img.shape) >3 :
            for i in range(nb_run):
                for j in range(nb_echo) : 
                    if 'run'+ str(i+1) in file :
                        if 'e' + str(j+1) in file :
                            destination_path = subj_folder + '/func/raw_' + subj + sess + '_run' + str(i+1) + '_e' + str(j+1)
                            niigz_file = os.path.join(subj_raw_path, file)
                            json_file = os.path.join(subj_raw_path, file[:-6]+'json')

                            shutil.move(niigz_file, destination_path + '.nii.gz')
                            shutil.move(json_file, destination_path + '.json')
        # t1
        if 't1_mprage' in file : 
            destination_path = subj_folder + '/anat/T1_' + subj + sess
            niigz_file = os.path.join(subj_raw_path, file)
            json_file = os.path.join(subj_raw_path, file[:-6]+'json')

            shutil.move(niigz_file, destination_path + '.nii.gz')
            shutil.move(json_file, destination_path + '.json')

        # not PA and not grey fieldmap
        if not '_PA_' in file : 
            if not 'gre_field_mapping' in file :
                file_path = os.path.join(subj_raw_path, file)
                destination= os.path.join(trash_folder, file)
                shutil.move(file_path, destination)

def slice_timing_corr(data_path:str, subj:str, sess:str, nb_echo = 3, nb_run = 2): 
    output_path = data_path + '/' + subj + sess + '/preproc'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for r in range(nb_run) :
        for e in range(nb_echo) : 
            output_file = output_path + '/st_' + subj + sess + '_run' + str(r+1) + '_e' + str(e+1)
            input_file = data_path + '/' + subj + sess + '/func/raw_' + subj + sess + '_run' + str(r+1) + '_e' + str(e+1)+'.nii.gz'
            if not os.path.exists(output_file):
                try :
                    command = ['slicetimer', '-i', input_file, '-o', output_file,  "-r", "2", "--odd"]
                    subprocess.call(command)
                    print(f"Slice Timing Correction done, run {r+1}, echo {e+1}! :)")

                except subprocess.CalledProcessError as err:
                    print(f'ERROR : {subj}, {r+1}', err)

            else : 
                print('Slice Timing correction already done')

def motion_corr(data_path:str, subj:str, sess:str, nb_echo = 3, nb_run = 2): 
    output_path = data_path+ '/' + subj + sess + '/preproc' 
    for r in range(nb_run) :
        for e in range(nb_echo) : 
            output_file = output_path + '/st_mc_' + subj + sess + '_run' + str(r+1) + '_e' + str(e+1)
            input_file = output_path + '/st_' + subj + sess + '_run' + str(r+1) + '_e' + str(e+1)+'.nii.gz'
            if not os.path.exists(output_file):

                try :
                    mcflirt(infile=input_file, refvol=0, o=output_file, plots=True, mats=True, dof=6)
                    print(f"Motion Correction done, run {r+1}, echo {e+1}! :)")

                except :
                    print(f'ERROR : {subj}, run {r+1} mcfilrt')

            else : 
                print('Motion correction already done')

def combine_echo(data_path:str, subj:str, sess:str, nb_run=2, nb_echo=3):
    preproc_path= data_path +'/preproc/' + subj + sess 
    subjpath=data_path + '/' + subj + sess

    # combine runs
    for i in range(nb_run):
        outputdir_run = preproc_path + '/tedana' + '/run' + str(i+1)

        if not os.path.isdir(outputdir_run):
            os.makedirs(outputdir_run)

        time = []
        for e in  range(nb_echo) :
            jsonfile_path = subjpath + '/func/'+ f'raw_{subj}_run{i+1}_e{e+1}.json'
            with open(jsonfile_path, 'r') as jfile:
                json_file = json.load(jfile)
                time.append(json_file["EchoTime"]*1000)

        run = [preproc_path + f'/st_mc_{subj}_run{i+1}_e{e+1}' for e in range(nb_echo)]

        command = f'tedana -d {" ".join(run)} -e {" ".join(map(str, time))} --out-dir {outputdir_run}'
        try : 
            subprocess.call(command, shell=True)  
            print('Tedana done! :)')
        except subprocess.CalledProcessError as err:
            print(f'ERROR : {subj}, {command}', err)

        # move the output that we are interested in 
        file = preproc_path +'/tedana/run' + str(i+1) + '/desc-optcom_bold.nii.gz'
        destination_file = preproc_path + '/run' + str(i+1) + '/tedana_' +subj + sess + '_run' + str(i+1) + '.nii.gz'
        shutil.move(file, destination_file)

    # rest goes to trash
    if not os.path.isdir(data_path + '/trash') : 
        os.mkdir(data_path + '/trash')

    shutil.move(preproc_path +'/tedana', data_path + '/trash')

def normalize(data_path:str, subj:str, sess:str, nb_run=2) : 
    preproc_path = data_path +'/preproc/' + subj + sess 
    anat_file = data_path + '/' + subj + sess + '/anat/T1_' + subj + sess

    for i in range(nb_run) :
        # Compute mean volume 
        mean_func = preproc_path + '/run' + str(i+1) + f'/mean_{subj}{sess}_run{str(i)}.nii.gz'
        input_file = preproc_path + '/run' + str(i+1) + f'/tedana_{subj}{sess}_run{str(i+1)}.nii.gz'

        if not os.path.exists(mean_func):
            command = ['fslmaths', input_file, '-Tmean', mean_func]
            try : 
                subprocess.run(command)
                print('fslmaths command done! :)')
            except subprocess.CalledProcessError as err:
                print(f'ERROR : {subj}, {command}', err)

        output_mean_reg_T1 = preproc_path + '/run' + str(i+1) + f'/mean_func_in_T1_{subj}{sess}_run{i+1}.nii.gz'
        mat_func_to_T1 = preproc_path + '/run' + str(i+1)+ f'/func_to_T1_{subj}{sess}_run{i+1}.mat'

        if not os.path.exists(output_mean_reg_T1) : 
            flirt_command = ['flirt', '-in', mean_func, '-ref', anat_file,
                                '-out', output_mean_reg_T1, '-omat', mat_func_to_T1,
                                '-dof', '6']
            try :
                subprocess.run(flirt_command)
                print('Flirt command done! :)')
            except subprocess.CalledProcessError as err:
                print(f'ERROR : {subj}, {command}', err)


if __name__ == "__main__":

    parser = buildArgsParser()
    args = parser.parse_args()

    subj_list = [subj for subj in args.subj]
    sess_list = [sess for sess in args.sess]

    data_path = args.data_path

    if "all" in subj_list:
        subjects = [s for s in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, s)) if "sub-TIMESwp11s" in s]
    else:
        subjects = ['PB_' + subj for subj in subj_list]

    if "all" in sess_list:
        sessions = ["", "_02"]
    else:
        sessions = ['_02' if sess == '2' else '' for sess in sess_list]
 
    for subj, sess in itertools.product(subjects, sessions):
        print(subj, sess)
        print("Dicom to nifti conversion ...")
        #DICOM2Nifti(data_path, subj, sess)   

        print("Slice timing correction processing ...")
        slice_timing_corr(data_path, subj, sess)
         
        print("Motion correction processing ...") 
        motion_corr(data_path, subj, sess)

        print("Tedana processing ...")
        combine_echo(data_path, subj, sess)   

        print("Normalization processing ...") #to T1
        normalize(data_path, subj, sess)

        #print("to MNI")

        #print("Smoothing processing ...")
   

   


