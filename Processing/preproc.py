## Preprocessing Pipline ##

import os 
import subprocess
import nibabel as nib
import argparse
import itertools
import shutil
import json
from tqdm import trange


from fsl.wrappers import mcflirt

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
        if os.path.exists(subj_path + '/' + dicom_folder[0]) : 
            shutil.rmtree(subj_path + '/' + dicom_folder[0])
       
        if os.path.exists(subj_path + '.zip') : 
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
            niigz_file_in = os.path.join(subj_raw_path, file)
            for i in range(nb_run):
                for j in range(nb_echo) : 
                    niigz_file_out = subj_folder + '/func/raw_' + subj + sess + '_run' + str(i+1) + '_e' + str(j+1)

                    if os.path.exists(niigz_file_in) : 
                        if 'run'+ str(i+1) in file :
                            if 'e' + str(j+1) in file :
                                niigz_file_in = os.path.join(subj_raw_path, file)
                                json_file = os.path.join(subj_raw_path, file[:-6]+'json')

                                shutil.move(niigz_file_in, niigz_file_out + '.nii.gz')
                                shutil.move(json_file, niigz_file_out + '.json')
        # t1
        if 't1_mprage' in file : 
            niigz_file = os.path.join(subj_raw_path, file)
            if os.path.exists(niigz_file) : 
                t1_out = subj_folder + '/anat/T1_' + subj + sess
                json_file = os.path.join(subj_raw_path, file[:-6]+'json')

                shutil.move(niigz_file, t1_out + '.nii.gz')
                shutil.move(json_file, t1_out + '.json')

        # field map 

        if not os.path.isdir(subj_folder + '/fieldmap') :
            os.makedirs(subj_folder + '/fieldmap')

        if 'gre_field_mapping' in file :
            destination= os.path.join(subj_folder + '/fieldmap/', file)
            json_file = os.path.join(subj_raw_path, file[:-6]+'json')
            shutil.move(file_path, destination)
            shutil.move(json_file, destination[:-6]+'json')

        # not PA 
        if not '_PA_' in file : 
            file_path = os.path.join(subj_raw_path, file)
            if os.path.exists(file_path) : 
                destination= os.path.join(trash_folder, file)
                json_file = os.path.join(subj_raw_path, file[:-6]+'json')
                shutil.move(file_path, destination)
                shutil.move(json_file, destination[:-6]+'json')

def preprocessing(data_path:str, subj:str, sess:str, nb_run=2, nb_echo=3) : 
    output_path = data_path+ '/' + subj + sess + '/preproc' 
    subj_path =  data_path+ '/' + subj + sess
    trash_path = data_path+ '/trash/' +  subj + sess 

    if not os.path.isdir(output_path) : 
        os.makedirs(output_path)
            
    for r in trange(nb_run, desc="preprocess runs") :
        for e in trange(nb_echo, desc= "preprocess echos") : 
            input_file = subj_path + '/func/raw_' + subj + sess + '_run' + str(r+1) + '_e' + str(e+1)+'.nii.gz'
            
            # Motion correction 
            print('Motion correction')
            output_file = output_path + '/mc_' + subj + sess + '_run' + str(r+1) + '_e' + str(e+1) + '.nii.gz'

            if not os.path.isfile(output_file):
                try :
                    mcflirt(infile=input_file, refvol=0, o=output_file, plots=True, mats=True, dof=6)
                    print(f"Motion Correction done, run {r+1}, echo {e+1}! :)")

                except :
                    print(f'ERROR : {subj}, run {r+1} mcfilrt')

            else : 
                print('Motion correction already done')

            # Trash 
            if os.path.isdir(output_file+ '.mat') : 
                destination= trash_path + '/mc_' + subj + sess + '_run' + str(r+1) + '_e' + str(e+1) + '.nii.gz.mat'
                shutil.move(output_file + '.mat', destination)

            if os.path.isfile(output_file+ '.par') : 
                destination= trash_path + '/mc_' + subj + sess + '_run' + str(r+1) + '_e' + str(e+1) + '.nii.gz.par'
                shutil.move(output_file+ '.par', destination)

            # Slice timing correction 
            print('Slice Timing Correction')
            input_file = output_file
            output_file = output_path + '/mc_st_' + subj + sess + '_run' + str(r+1) + '_e' + str(e+1) + '.nii.gz'
            
            if not os.path.isfile(output_file):
                try :
                    command = ['slicetimer', '-i', input_file, '-o', output_file,  "-r", "2", "--odd"]
                    subprocess.call(command)
                    print(f"Slice Timing Correction done, run {r+1}, echo {e+1}! :)")

                except subprocess.CalledProcessError as err:
                    print(f'ERROR : {subj}, {r+1}', err)

            else : 
                print('Slice Timing correction already done')

            
            # Skull Stripping 
            input_file = output_file
            output_file = output_path + '/mc_st_bet_' + subj + sess + '_run' + str(r+1) + '_e' + str(e+1) + '.nii.gz'
            mean_file = output_path + '/mean_epi_' + subj + sess + '_run' + str(r+1) + '_e' + str(e+1) + '.nii.gz'
            mean_file_bet = output_path + '/mean_epi_bet_' + subj + sess + '_run' + str(r+1) + '_e' + str(e+1) + '.nii.gz'
            
            if not os.path.isfile(output_file):
                try :
                    # compute mean image
                    command = ['fslmaths', input_file, '-Tmean', mean_file]
                    subprocess.call(command)

                    # use bet on mean imagg
                    command = ['bet', mean_file, mean_file_bet ,'-f', '0.3' ,'-m']
                    subprocess.call(command)

                    # apply the mask mean file bet to the 4D image
                    command = ['fslmaths', input_file, '-mas', mean_file_bet, output_file]
                    subprocess.call(command)
                    
                    print(f"Skull stripping done, run {r+1}, echo {e+1}! :)")

                except subprocess.CalledProcessError as err:
                    print(f'ERROR : {subj}, {r+1}', err)

            else : 
                print('Skull stripping already done')

            # trash
            if os.path.isfile(mean_file) : 
                destination= trash_path + '/mean_epi_' + subj + sess + '_run' + str(r+1) + '_e' + str(e+1) + '.nii.gz'
                shutil.move(mean_file, destination)

            if os.path.isfile(mean_file_bet) : 
                destination= trash_path + '/mean_epi_bet_' + subj + sess + '_run' + str(r+1) + '_e' + str(e+1) + '.nii.gz'
                shutil.move(mean_file_bet, destination)
                shutil.move(mean_file_bet[:-7] + '_mask.nii.gz', destination[:-7] + '_mask.nii.gz')


            # Field map correciton 
            print('Field map correction')
            input_file = output_file
            output_file = output_path + '/fm_bet_st_mc_' + subj + sess + '_run' + str(r+1) + '_e' + str(e+1)

            fieldmap_path = subj_path + '/fieldmap/'

            if r == 0 : 
                name = 'run1_5_e2'
                phi_name = 'run1_6_e2'
            if r == 1 :
                name = 'run2_13_e2'
                phi_name = 'run2_14_e2'

            magnitude_file = fieldmap_path + f'gre_field_mapping_{name}.nii.gz'
            magnitude_bet = fieldmap_path + f'bet_gre_field_mapping_run{r+1}.nii.gz'
            phase_file = fieldmap_path + f'gre_field_mapping_{phi_name}_ph.nii.gz'
            field_map = fieldmap_path + f'field_map_run{r+1}.nii.gz'

            # got TE : 
            jsonfile_path = fieldmap_path + f'gre_field_mapping_{name}.json'
            with open(jsonfile_path, 'r') as jfile:
                json_file = json.load(jfile)
                TE = str(json_file["EchoTime"]*1000)
            
            if not os.path.exists(output_file):
                try :
                    # skull strip the magnitude brain 
                    command = ['bet',  magnitude_file, magnitude_bet, '-f', '0.35', '-m']
                    subprocess.call(command)

                    # prep field map 
                    command= ['fsl_prepare_fieldmap', 'SIEMENS',phase_file , magnitude_bet,  field_map, TE]
                    subprocess.call(command)

                    # apply distrortion
                    json_path = f'/Users/barbaragrosjean/Desktop/CHUV/PreddiBrains/data/PB_001/func/raw_PB_001_run{r+1}_e{e+1}.json'
                    with open(json_path, 'r') as jfile:
                        json_file = json.load(jfile)
                        EffectiveEchoSpacing = str(json_file["EffectiveEchoSpacing"])

                    command = ['fugue', '-i', input_file, f'--dwell={EffectiveEchoSpacing}', f'--loadfmap={field_map}', f'--mask={magnitude_bet[:-7] + '_mask.nii.gz'}', '-u', output_file]
                    subprocess.call(command)

                    print(f"Field map correction done, run {r+1}, echo {e+1}! :)")                    

                except subprocess.CalledProcessError as err:
                    print(f'ERROR : {subj}, {r+1}', err)

            else : 
                print('Field map correction already done')

            # trash TODO

def combine_echo(data_path:str, subj:str, sess:str, nb_run=2, nb_echo=3) : 
    output_path = data_path+ '/' + subj + sess + '/preproc' 
    subj_path =  data_path+ '/' + subj + sess
    trash_path = data_path+ '/trash/' +  subj + sess 

    for r in range(nb_run) : 
        print('Combining Echo')
        # Tedana
        destination_file = output_path + '/tedana_' +subj + sess + '_run' + str(r+1) + '.nii.gz'

        if not os.path.exists(destination_file) : 
            outputdir_run = output_path + '/tedana' + '/run' + str(r+1)

            if not os.path.isdir(outputdir_run):
                os.makedirs(outputdir_run)

            time = []
            for e in  range(nb_echo) :
                jsonfile_path = subj_path + '/func/'+ f'raw_{subj}_run{r+1}_e{e+1}.json'
                with open(jsonfile_path, 'r') as jfile:
                    json_file = json.load(jfile)
                    time.append(json_file["EchoTime"]*1000)

            run = [output_path + f'/fm_bet_st_mc_{subj}_run{r+1}_e{e+1}.nii.gz' for e in range(nb_echo)]

            command = f'tedana -d {" ".join(run)} -e {" ".join(map(str, time))} --out-dir {outputdir_run}'
            try : 
                subprocess.call(command, shell=True)  
                print('Tedana done! :)')
                
                # move the output that we are interested in 
                file = output_path +'/tedana/run' + str(r+1) + '/desc-optcom_bold.nii.gz'
                
                if os.path.exists(file) : 
                    shutil.move(file, destination_file)

            except subprocess.CalledProcessError as err:
                print(f'ERROR : {subj}) #, {command}', err)

        # rest goes to trash
        if not os.path.isdir(trash_path) : 
            os.makedirs(trash_path)

        if os.path.isdir(output_path +'/tedana') : 
            shutil.move(output_path +'/tedana', trash_path + '_run' + str(r) )

def registration_to_T1(data_path: str, subj: str, sess: str, nb_run=2):
    output_path = os.path.join(data_path, subj + sess, 'preproc')
    anat_file = os.path.join(data_path, subj + sess, 'anat', 'T1_' + subj + sess)
    anat_file_bet = os.path.join(data_path, subj + sess, 'anat', 'bet_T1_' + subj + sess)

    # Skull strip the anatomical image
    if not os.path.exists(anat_file_bet + '.nii.gz'):
        command = ['bet', anat_file, anat_file_bet, '-f', '0.5', '-m']
        subprocess.call(command)

    for r in trange(nb_run, desc="Registering runs to T1"):
        print(f'Registering run {r+1} to T1')

        input_func = os.path.join(output_path, f'tedana_{subj}{sess}_run{r+1}.nii.gz')
        mean_func = os.path.join(output_path, f'mean_{subj}{sess}_run{r+1}.nii.gz')

        # Step 1: Compute mean functional image (used for registration)
        if not os.path.exists(mean_func):
            command = ['fslmaths', input_func, '-Tmean', mean_func]
            try:
                subprocess.run(command, check=True)
                print('fslmaths (mean func) done.')
            except subprocess.CalledProcessError as err:
                print(f'ERROR during mean func computation: {subj}, {command}', err)

        # Step 2: Compute transformation matrix from mean_func to T1
        mat_func_to_T1 = os.path.join(output_path, f'func_to_T1_{subj}{sess}_run{r+1}.mat')
        if not os.path.exists(mat_func_to_T1):
            flirt_command = [
                'flirt', '-in', mean_func, '-ref', anat_file_bet,
                '-omat', mat_func_to_T1, '-dof', '6'
            ]
            try:
                subprocess.run(flirt_command, check=True)
                print('FLIRT: Computed transform matrix (mean func → T1).')
            except subprocess.CalledProcessError as err:
                print(f'ERROR during FLIRT matrix computation: {subj}, {flirt_command}', err)

        # Step 3: Apply transformation to the whole functional run
        func_in_T1 = os.path.join(output_path, f'func_in_T1_{subj}{sess}_run{r+1}.nii.gz')
        if not os.path.exists(func_in_T1):
            applyxfm_command = [
                'flirt', '-in', input_func, '-ref', anat_file_bet,
                '-applyxfm', '-init', mat_func_to_T1, '-out', func_in_T1
            ]
            try:
                subprocess.run(applyxfm_command, check=True)
                print('FLIRT: Applied transform to full run → T1 space.')
            except subprocess.CalledProcessError as err:
                print(f'ERROR applying transform: {subj}, {applyxfm_command}', err)
        else:
            print(f'Run {r+1} already transformed to T1.')

def smoothing(data_path:str, subj:str, sess:str, nb_run =2, smooth = '4mm') : 

    sigma_map = {'4mm' : 1.70, '5mm' : 2.12, '6mm': 2.55, '8mm': 3.40}
    sigma = sigma_map[smooth]

    output_path = data_path+ '/' + subj + sess + '/preproc' 

    for r in trange(nb_run, desc="smoothing runs") :
        input_file = os.path.join(output_path, f'func_in_MNI_PB_001_run{str(r+1)}.nii.gz')
        output_file = os.path.join(output_path, f'sm_func_in_MNI_PB_001_run{str(r+1)}.nii.gz')

        if not os.path.isfile(input_file) :
            try : 
                command = ['fslmaths', input_file,  '-s', sigma, output_file]
                subprocess.call(command)
                print('Smoothing  done! :)')

            except subprocess.CalledProcessError as err:
                print(f'ERROR : {subj}, {command}', err)

        else : 
            print('Smoothing already done')

def concat_runs(data_path:str, subj:str, sess:str, nb_run =2, smooth = '4mm') : 
    # TODO
    return 0


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
    
        DICOM2Nifti(data_path, subj, sess)   

        extract_usfull_file(data_path, subj, sess)   
 
        preprocessing(data_path, subj, sess)

        combine_echo(data_path, subj, sess)   

        registration(data_path, subj, sess)

        smoothing(data_path, subj, sess)
   

   


