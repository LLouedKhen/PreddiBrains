## Preprocessing Pipline for multi Echo fMRI ##
## Author @barbaragrosjean ##


import os 
import subprocess
import nibabel as nib
import argparse
import itertools
import shutil
import json

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

    if os.path.exists('/Users/barbaragrosjean/Desktop/CHUV/PreddiBrains/data/' + subj + sess) :
        return 'Already extracted'
    
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

    if os.path.exists('/Users/barbaragrosjean/Desktop/CHUV/PreddiBrains/data/' + subj + sess) :
        return 'Already extracted'

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

        if not os.path.isdir(subj_folder + '/fmap') :
            os.makedirs(subj_folder + '/fmap')

        if 'gre_field_mapping' in file :
            file_path = os.path.join(subj_raw_path, file)
            destination= os.path.join(subj_folder + '/fmap', file)
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

def preprocessing(data_path:str, subj:str, sess:str, nb_run=2, nb_echo=3, smooth='8mm') : 
    output_path = data_path+ '/' + subj + sess + '/preproc' 
    subj_path =  data_path+ '/' + subj + sess
    trash_path = data_path+ '/trash/' +  subj + sess  

    if not os.path.isdir(output_path) : 
        os.makedirs(output_path)


    for r in range(nb_run) :
        print( f'___________________ RUN {r+1} ___________________')
        for e in range(nb_echo) : 
            print( f'___________________ ECHO {e+1} ___________________')

            input_file = subj_path + '/func/raw_' + subj + sess + '_run' + str(r+1) + '_e' + str(e+1)+'.nii.gz'
            
            ############################
            # 1. Slice timing correction 
            ############################
            print('########### Slice Timing correction ##########')
            output_file = output_path + '/st_' + subj + sess + '_run' + str(r+1) + '_e' + str(e+1) + '.nii.gz'
            
            if not os.path.isfile(output_file):

                # got the TR
                json_path = subj_path + '/func/raw_' + subj +sess + '_run' + str(r+1) + '_e1.json'
                with open(json_path, 'r') as jfile:
                    json_file = json.load(jfile)
                    TR = str(json_file["RepetitionTime"])

                try :
                    command = ['slicetimer', '-i', input_file, '-o', output_file,  "-r", str(TR), "--odd"]
                    subprocess.call(command)
                    print(f"Done.")

                except subprocess.CalledProcessError as err:
                    print(f'ERROR : ', err)

            else : 
                print('Already done.')
            

            ######################
            # 2. Motion correction
            ######################
         
            print('########### Motion correction ##########')
            input_file = output_file
            output_file = output_path + '/st_mc_' + subj + sess + '_run' + str(r+1) + '_e' + str(e+1) + '.nii.gz'
            mean_file_e1 = output_path + '/mean_' + subj + sess + '_run' + str(r+1) + '_e1.nii.gz'
            mat_files_e1 = output_path + '/st_mc_'+subj +sess + '_run'+ str(r+1) +'_e1.nii.gz.mat' 

            if e == 0 and not os.path.isfile(mean_file_e1) : 
                print('-> Save the mean volume from e1')
                command = ['fslmaths', input_file, '-Tmean', mean_file_e1]
                try:
                    subprocess.run(command, check=True)
                    print('Done.')
                except subprocess.CalledProcessError as err:
                    print(f'ERROR: ', err)

            if e ==0 and not os.path.exists(mat_files_e1):
                print('-> Compute motion correction transformation')
                try :
                    mcflirt(infile=input_file, refvol=mean_file_e1, o=output_file, plots=True, mats=True, dof=6)

                    # Change the name of the files to be compatible with applyxfm4D
                    files = sorted(os.listdir(mat_files_e1))

                    for f in files:
                        if f.startswith("MAT_"):
                            num = int(f.split("_")[1])
                            new_name = f"MAT_{num:05d}"
                            os.rename(os.path.join(mat_files_e1, f), os.path.join(mat_files_e1, new_name))

                    print(f"Done.")

                except :
                    print(f'ERROR: mcfilrt')

            if not os.path.isfile(output_file) :  
                print('-> Apply transformation')
                command = ['applyxfm4D', input_file,  mean_file_e1, output_file, mat_files_e1]
                try : 
                    subprocess.call(command)
                    print(f"Done.")

                except subprocess.CalledProcessError as err:
                    print(f'ERROR: ', err)
                
            else : 
                print('Already done.')


            ##########################
            # 3. Field map correciton 
            ##########################
         
            print('########### Field map correciton  ##########')
            input_file = output_file
            output_file = output_path + '/st_mc_fm_' + subj + sess + '_run' + str(r+1) + '_e' + str(e+1) + '.nii.gz'

            fieldmap_path = subj_path + '/fmap/'

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
                    if not os.path.isfile(field_map) : 
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

                    command = ['fugue', '-i', input_file, f'--dwell={EffectiveEchoSpacing}', f'--loadfmap={field_map}', 
                               f'--mask={magnitude_bet[:-7]}_mask.nii.gz', '-u', output_file]
                    
                    subprocess.call(command)

                    print(f"Done.")                    

                except subprocess.CalledProcessError as err:
                    print(f'ERROR: ', err)

            else : 
                print('Already done.')

    
        #################
        # 4. Combine Echo
        #################
        print('########### Combine Echo - Tedana ##########')
        destination_file = output_path + '/tedana_' +subj + sess + '_run' + str(r+1) + '.nii.gz'

        if not os.path.exists(destination_file) : 
            outputdir_run = output_path + '/tedana' + '/run' + str(r+1)

            if not os.path.isdir(outputdir_run):
                os.makedirs(outputdir_run)

            # get echo time
            time = []
            for e in  range(nb_echo) :
                jsonfile_path = subj_path + '/func/'+ f'raw_{subj}_run{r+1}_e{e+1}.json'
                with open(jsonfile_path, 'r') as jfile:
                    json_file = json.load(jfile)
                    time.append(json_file["EchoTime"]*1000)

            run = [output_path + f'/st_mc_fm_{subj}{sess}_run{r+1}_e{e+1}.nii.gz' for e in range(nb_echo)]

            command = f'tedana -d {" ".join(run)} -e {" ".join(map(str, time))} --out-dir {outputdir_run}'
            try : 
                subprocess.call(command, shell=True)  
                print('Done.')
                
                # move the output that we are interested in 
                file = output_path +'/tedana/run' + str(r+1) + '/desc-optcom_bold.nii.gz'
                
                if os.path.exists(file) : 
                    shutil.move(file, destination_file)

            except subprocess.CalledProcessError as err:
                print(f'ERROR: ', err)
        else:
            print('Already done.')

        ##########################
        # 5. Co-Registration to T1
        ##########################
        print('########### Co-Registration ##########')
        input_file=destination_file
        output_file = output_path + '/func_in_T1_' +subj + sess + 'run' + str(r+1) + '.nii.gz'

        # Anat prep - Skull strip the anatomical image
        print('-> Preparing anatomical image T1 ...')
        anat_file = os.path.join(data_path, subj + sess, 'anat', 'T1_' + subj + sess)
        anat_file_bet = os.path.join(data_path, subj + sess, 'anat', 'T1_' + subj + sess + '_optiBET_brain.nii.gz')
        optiBET_file = '/Users/barbaragrosjean/Desktop/CHUV/PreddiBrains/Processing/optiBET.sh'
        shutil.copy(optiBET_file, subj_path + '/anat/optiBET.sh')
        global_path = os.getcwd()

        if not os.path.exists(anat_file_bet):
            os.chdir(os.path.join(data_path, subj + sess, 'anat'))
            command = ['sh', 'optiBET.sh', anat_file]
            subprocess.run(command)

            # remove optibet file
            if os.path.isfile(output_path + '/optiBET.sh') : 
                os.remove(output_path + '/optiBET.sh')

            # get back to the right folder
            os.chdir(global_path)
        else : 
            print('Already done.')
        
        # Compute mean image after tedana
        print('-> Compute mean functional combine image')
        mean_func = os.path.join(output_path, f'mean_func_{subj}{sess}_run{r+1}.nii.gz')
       
        if not os.path.exists(mean_func):
            command = ['fslmaths', input_file, '-Tmean', mean_func]
            try:
                subprocess.run(command, check=True)
                print('Done.')
            except subprocess.CalledProcessError as err:
                print(f'ERROR: ', err)
        else : 
            print('Already done.')

        # Compute transform 
        print('Compute and apply the transform matrix Func - T1')
        if not os.path.exists(output_file):
            print('-> Compute transform Func - T1')
            mat_func_to_T1 = os.path.join(output_path, f'func_to_T1_{subj}{sess}_run{r+1}.mat')
            if not os.path.isfile(mat_func_to_T1) : 
                flirt_command = ['flirt', '-in',mean_func, '-ref', anat_file_bet, '-omat', mat_func_to_T1, '-dof', '6']

                try:
                    subprocess.run(flirt_command, check=True)
                    print('Done.')
                except subprocess.CalledProcessError as err:
                    print(f'ERROR:', err)        
            
            print('-> Apply transform Func - T1')
            try:
                #apply_command = ['flirt', '-in', input_file, '-ref', anat_file_bet,'-applyxfm', '-init', mat_func_to_T1, '-out', output_file]
                apply_command = ['applyxfm4D', input_file, anat_file_bet, output_file, mat_func_to_T1, '-singlematrix']
                 
                subprocess.run(apply_command, check=True)
                print(f'Done.')
            except subprocess.CalledProcessError as err:
                print(f'ERROR:', err)
        else:
            print(f'Already done.')

        ###################
        # 6. Normalization
        ###################
        print('########### Normalization ##########')
        input_file=output_file
        output_file = output_path + '/func_in_MNI' + subj + sess + '_run' + str(r+1) + '.nii.gz'

        print('-> Preparing MNI template ...')
        MNI_file = os.path.join(data_path, 'AAL3/MNI.nii')
        MNI_file_bet = os.path.join(data_path, 'AAL3/MNI_optiBET_brain.nii.gz')
        MNI_file_mask = os.path.join(data_path, 'AAL3/MNI_optiBET_brain_mask.nii.gz')

        optiBET_file = '/Users/barbaragrosjean/Desktop/CHUV/PreddiBrains/Processing/optiBET.sh'
        shutil.copy(optiBET_file, data_path + '/AAL3/optiBET.sh')
        global_path = os.getcwd()

        if not os.path.exists(anat_file_bet):
            os.chdir(os.path.join(data_path, '/AAL3'))
            command = ['sh', 'optiBET.sh', MNI_file]
            subprocess.run(command)

            # Remove optibet file
            if os.path.isfile(output_path + '/optiBET.sh') : 
                os.remove(output_path + '/optiBET.sh')

            # get back to the right folder
            os.chdir(global_path)
        else : 
            print('Already done.')

        if not os.path.isfile(output_file) : 
            print('-> Compute transform T1 - MNI')
            mat_T1_to_MNI = os.path.join(output_path, f'T1_to_MNI_{subj}{sess}.mat')
            mat_T1_to_MNI_2 = os.path.join(output_path, f'T1_to_MNI_2_{subj}{sess}.mat')
            T1_2_MNI_warp = os.path.join(output_path, f'T1_to_MNI_warp_{subj}{sess}.mat')
            if not os.path.exists(mat_T1_to_MNI):
                flirt_command = ['flirt', '-in', anat_file_bet, '-ref', MNI_file_bet,
                                '-omat', mat_T1_to_MNI, '-dof', '6']

                fnirt_command  =[ 'fnirt', '--in=', anat_file, '--aff=', mat_T1_to_MNI,
                                 '--ref=', MNI_file, '--refmask=', MNI_file_mask,
                                 '--iout=', mat_T1_to_MNI_2,'--cout=', T1_2_MNI_warp]
                try:
                    subprocess.run(flirt_command, check=True)
                    subprocess.run(fnirt_command, check=True)

                    print('Done.')
                except subprocess.CalledProcessError as err:
                    print(f'ERROR: ', err)

            print('-> Apply transform T1 - MNI to functional volumes')
            #applyxfm_command = ['flirt', '-in', input_file, '-ref', MNI_file_bet,'-applyxfm', '-init', mat_T1_to_MNI, '-out', output_file]
            applywarp_command = ['applywarp', '--in=', input_file, '--ref=', MNI_file, '--warp=', 
                                 T1_2_MNI_warp, '--premat=', mat_T1_to_MNI, '--out=', output_file]
            try:
                subprocess.run(applywarp_command, check=True)
                print('Done.')
            except subprocess.CalledProcessError as err:
                print(f'ERROR: ', err)
        else:
            print(f'Already Done.')

        ###############
        # 6. Smoothing
        ################
        print(f'########### Smoothing - {smooth} ##########')

        sigma_map = {'4mm' : 1.70, '5mm' : 2.12, '6mm': 2.55, '8mm': 3.40}
        sigma = sigma_map[smooth]

        output_path = data_path+ '/' + subj + sess + '/preproc' 

        for r in range(nb_run) :
            input_file = output_file
            output_file = os.path.join(output_path, f'sm_func_MNI_{subj}{sess}_run{r+1}.nii.gz')

            if not os.path.isfile(input_file) :
                try : 
                    command = ['fslmaths', input_file,  '-s', sigma, output_file]
                    subprocess.call(command)
                    print('Done.')

                except subprocess.CalledProcessError as err:
                    print(f'ERROR: ', err)

            else : 
                print('Already done.')


    # Rest goes to trash
    if not os.path.isdir(trash_path) : 
        os.makedirs(trash_path)

    if os.path.isdir(output_path +'/tedana') : 
        shutil.move(output_path +'/tedana', trash_path)

    if os.path.isfile(output_path + '/optiBET.sh') : 
        os.remove(output_path + '/optiBET.sh')

    for r in range(nb_run) :
        if os.path.isfile(output_path + f'/mean_{subj}{sess}_run{r+1}_e1.nii.gz'):
            shutil.move(output_file + '.mat', trash_path + f'/mean_{subj}{sess}_run{r+1}_e1.nii.gz')

        if os.path.isdir(output_path + f'/st_mc_{subj}{sess}_run{r+1}_e1.nii.gz.mat') : 
                destination= trash_path + '/mc_' + subj + sess + '_run' + str(r+1) + '_e1.nii.gz.mat'
                shutil.move(output_file + '.mat', destination)

        for e in range(nb_echo) :
            if os.path.isfile(output_file+ '.par') : 
                destination= trash_path + '/mc_' + subj + sess + '_run' + str(r+1) + '_e' + str(e+1) + '.nii.gz.par'
                shutil.move(output_file+ '.par', destination)

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

            run = [output_path + f'/fm_st_mc_{subj}{sess}_run{r+1}_e{e+1}.nii.gz' for e in range(nb_echo)]

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
            shutil.move(output_path +'/tedana', trash_path)



def smoothing(data_path:str, subj:str, sess:str, nb_run =2, smooth = '8mm') : 

    sigma_map = {'4mm' : 1.70, '5mm' : 2.12, '6mm': 2.55, '8mm': 3.40}
    sigma = sigma_map[smooth]

    output_path = data_path+ '/' + subj + sess + '/preproc' 

    for r in range(nb_run) :
        input_file = os.path.join(output_path, f'tedana_{subj}{sess}_run{r+1}.nii.gz')
        output_file = os.path.join(output_path, f'sm_func_MNI_{subj}{sess}_run{r+1}.nii.gz')

        if not os.path.isfile(input_file) :
            try : 
                command = ['fslmaths', input_file,  '-s', sigma, output_file]
                subprocess.call(command)
                print('Smoothing  done! :)')

            except subprocess.CalledProcessError as err:
                print(f'ERROR : {subj}, {command}', err)

        else : 
            print('Smoothing already done')

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

        #combine_echo(data_path, subj, sess)

        #smoothing(data_path, subj, sess)
   

   


