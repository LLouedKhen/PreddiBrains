#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 10:13:01 2023

@author: sysadmin
"""

import os
import datetime as datetime
import re
import pandas as pd


def validate(subjDOB):
    while 1:
        try:
            datetime.datetime.strptime(subjDOB, '%Y-%m-%d')
        except ValueError:
            subjDOB= input('Incorrect date format. Please re-enter subject birthdate in YYYY-MM-DD format: ')
        else:
            break
        

 #Experiment Datapath
outputPath = '/Users/sysadmin/Documents/PreddiBrains/Output/Imaging/NC'
   
os.chdir(outputPath)

#What day are we today?
StudyDate = datetime.datetime.now()


#Enter subject number
subjectNum = input("Enter main participant identifier: ") 
while len(subjectNum) != 3:
    print("Error! Must enter 3 characters!")
    subjectNum  = input("Please re-enter main participant identifier: ")    
    print(subjectNum)
while os.path.exists(str('PBNC_' + subjectNum)):
    print("Error! This subject directory exists already.")
    subjectNum  = input("Please re-enter main participant identifier: ")    
    print(subjectNum)
        

#Enter subject birthdate
subjDOB= input('Enter subject birthdate in YYYY-MM-DD format: ')
validate(subjDOB)
now = datetime.datetime.strptime(str(StudyDate.date()), '%Y-%m-%d')
dob = datetime.datetime.strptime(subjDOB, '%Y-%m-%d')
Age = (now - dob)
Age = Age.days/365.2425
print (Age)

#Enter subject gender
subjGender= input("Enter main participant gender (M/F/O): ") 
while not re.match("^[MFmfOo]*$", subjGender):
    print ("Error! Only letters M, F allowed!")
    subjGender= input("Please re-enter main participant gender (M/F): ") 
    
#Enter language
language= input("Enter experiment language (E/F): ") 
while not re.match("^[EF]*$", language):
    print ("Error! Only letters E, F allowed!")
    subjGender= input("Please re-enter experiment language (E/F): ") 


# #Enter Random Sequence of Stimuli 
# if int(subjectNum) % 2 == 0:
#     StimSeq = 1;
# elif int(subjectNum) % 2 != 0:
#     StimSeq = 2;

subjectNum = 'PBNC' + subjectNum


if not os.path.exists(subjectNum)  :
    os.makedirs(subjectNum)
subjPath = os.path.join(outputPath, subjectNum)
os.chdir(subjPath)



intakeData = [StudyDate, subjectNum, subjDOB, Age, subjGender,language, subjPath]

intakeData = pd.DataFrame(intakeData)
intakeData.to_csv(subjectNum + '_IntakeData.csv')

#return (subjectNum, subjPath, RatScaleOr)