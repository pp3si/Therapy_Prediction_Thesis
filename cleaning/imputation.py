import pandas as pd
import numpy as np
import CAPS
from CAPS import ProgressBar
import os.path
import random
from scipy.linalg import svd

# for specifying dtypes
dtypes_ = { "PatientID": "str",
            "TherapistID": "str",
            "MainTherapist": "str",
            "SessionID": "str",
            "StartDate": "str",
            "EndDate": "str",
            "NumOfAttended": "int",
            "Crisis": "bool",
            "notedate": "str",
            "DateDifference": "int",
            "Date of Birth": "str",
            "age": "float",
            "Female": "bool",
            "RacialMinority": "bool",
            "International Student": "bool",
            "SexOrientationMinority": "bool",
            "MarriedMale": "bool",
            "NursingOrLaw": "bool",
            "Disabilities": "int",
            "Have you been diagnosed with an autism-spectrum disorder or Asperger's Syndrome?": "int",
            "Homeless": "bool",
            "ROTC": "bool",
            "Military Service": "bool",
            "Military Stress": "bool",
            "First Generation": "bool",
            "Financial Stress Now": "int",
            "Financial Stress Past": "int",
            "AgnosticOrAtheist": "bool",
            "ReligiousMinority": "bool",
            "Religion Importance": "int",
            "Prior Counseling": "bool",
            "Prior Meds": "bool",
            "Prior Hospitalization (How many)": "int", # not actual count of times
            "Prior Hospitalization (Recent)": "bool",
            "Need to Reduce D&A (How many)": "int",
            "Need to Reduce D&A (Recent)": "bool",
            "Others Concern Alcohol (How many)": "int",
            "Others Concern Alcohol (Recent)": "bool",
            "Prior D&A Treatment (How many)": "int",
            "Prior D&A Treatment (Recent)": "bool",
            "Self-Injury (How many)": "int",
            "Self-Injury (Recent)": "bool",
            "Considered Suicide (How many)": 'int',
            "Considered Suicide (Recent)": "bool",
            "Suicide Attempt (How many)": "int",
            "Suicide Attempt (Recent)": "bool",
            "Considered Harming (How many)": "int",
            "Considered Harming (Recent)": "bool",
            "Harmed Another (How many)": "int",
            "Harmed Another (Recent)": "bool",
            "Unwanted Sexual Exp. (How many)": "int",
            "Unwanted Sexual Exp. (Recent)": "bool",
            "Harassment/Abuse (How many)": 'int',
            "Harassment/Abuse (Recent)": "bool",
            "PTSD Experience (How many)": "int",
            "PTSD Experience (Recent)": "bool",
            "ChildhoodTrauma": "bool",
            "Trauma": "int",
            "Family Support": "int",
            "Social Support": "int",
            "In the past week, about how many nights has it taken you more than half an hour to fall asleep?": 'int',
            "In the past week, about how many nights have you woken during the night AND needed more than half an hour to fall back to sleep?": 'int',
            "In the past two weeks, have you taken a substance to help with sleep? Please consider prescription medication, over the counter medication and supplements, and substances such as alcohol and marijuana.": "bool",
            "Academics": "int",
            "Social life/relationships": "int",
            "Emotional well-being": "int",
            "Freshman / First-year": "bool",
            "Sophomore": "bool",
            "Junior": "bool",
            "Senior": "bool",
            "Graduate / professional degree student": "bool",
            "Confusion about religious beliefs or values": "int",
            "Gender, ethnic, or racial discrimination": "int",
            "Perfectionism": "int",
            "Physical health problems (headaches, GI trouble)": "int",
            "Sexual concerns": "int",
            "Sexual orientation or identity": "int",
            "Pornography": "int",
            "I get sad or angry when I think of my family": "int",
            "I am shy around others": "int",
            "There are many things I am afraid of": "int",
            "My heart races for no good reason": "int",
            "I feel out of control when I eat": "int",
            "I enjoy my classes": "int",
            "I feel that my family loves me": "int",
            "I feel disconnected from myself": "int",
            "I don't enjoy being around people as much as I used to": "int",
            "I feel isolated and alone": "int",
            "My family gets on my nerves": "int",
            "I lose touch with reality": "int",
            "I think about food more than I would like to": "int",
            "I am anxious that I might have a panic attack while in public": "int",
            "I feel confident that I can succeed academically": "int",
            "I become anxious when I have to speak in front of audiences": "int",
            "I have sleep difficulties": "int",
            "My thoughts are racing": "int",
            "I am satisfied with my body shape": "int",
            "I feel worthless": "int",
            "My family is basically a happy one": "int",
            "I am dissatisfied with my weight": "int",
            "I feel helpless": "int",
            "I use drugs more than I should": "int",
            "I eat too much": "int",
            "I drink alcohol frequently": "int",
            "I have spells of terror or panic": "int",
            "I am enthusiastic about life": "int",
            "When I drink alcohol I can't remember what happened": "int",
            "I feel tense": "int",
            "When I start eating I can't stop": "int",
            "I have difficulty controlling my temper": "int",
            "I am easily frightened or startled": "int",
            "I diet frequently": "int",
            "I make friends easily": "int",
            "I sometimes feel like breaking or smashing things": "int",
            "I have unwanted thoughts I can't control": "int",
            "There is a history of abuse in my family": "int",
            "I experience nightmares or flashbacks": "int",
            "I feel sad all the time": "int",
            "I am concerned that other people do not like me": "int",
            "I wish my family got along better": "int",
            "I get angry easily": "int",
            "I feel uncomfortable around people I don't know": "int",
            "I feel irritable": "int",
            "I have thoughts of ending my life": "int",
            "I feel self conscious around others": "int",
            "I purge to control my weight": "int",
            "I drink more than I should": "int",
            "I enjoy getting drunk": "int",
            "I am not able to concentrate as well as usual": "int",
            "I am afraid I may lose control and act violently": "int",
            "It's hard to stay motivated for my classes": "int",
            "I feel comfortable around other people": "int",
            "I like myself": "int",
            "I have done something I have regretted because of drinking": "int",
            "I frequently get into arguments": "int",
            "I find that I cry frequently": "int",
            "I am unable to keep up with my schoolwork": "int",
            "I have thoughts of hurting others": "int",
            "The less I eat, the better I feel about myself": "int",
            "I feel that I have no one who understands me": "int",
            "Seeking Services Due To COVID-19": "bool" }


#############################################################################################
#  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  #
#############################################################################################


def imputation(master_df,
               na_cutoff,
               n_components=10,
               ):
    """
    This function takes the data frame generated using the functions
    in sessioninfo.py and imputes missing values in the patient
    information.

    master_df: master_df after match_patient_info.
    na_cutoff: The proportion of NAs for each column is calcuated.
        Those higher than the cutoff are imputed with the median
        response of those that are not NA; the rest are imputed
        using principle components.
    n_components: the number of principle components used to impute
        for the features below the cutoff.
    """

    na_proportions = master_df.isna().mean(axis=0)
    # If column has NAs, print how many
    print('{0:30}\t{1}'.format('Column', 'Number of NAs'))
    for i, j in zip(master_df.columns, na_proportions):
        if j != 0:
            print('{0:30}\t{1:.1f} %'.format(i, j*100))

    # Seeking Services Due To COVID-19 to zero
    if "Seeking Services Due To COVID-19" in master_df.columns:
        master_df.fillna({"Seeking Services Due To COVID-19": 0}, inplace=True)

    master_df['PatientID'] = master_df['PatientID'].astype(str)
    
    if "TherapistID" in list(master_df.columns):
        master_df['TherapistID'] = master_df['TherapistID'].astype(str)
    else:
        master_df['MainTherapist'] = master_df['MainTherapist'].astype(str)

    col_for_imput = (master_df.select_dtypes(include="number")
                    ).drop(["NumOfAttended", "DateDifference"], axis=1).columns


    median_fill = dict(master_df.loc[:, col_for_imput].dropna().apply(np.median, axis=0))
    temp = master_df.loc[:, col_for_imput].fillna(median_fill)
    temp.fillna(0, inplace=True)
    Matrix = temp.to_numpy()
    print('Calculating SVD')
    U, s, VT = svd(Matrix)

    n = n_components 
    replace = pd.DataFrame(U[:, :n] @ np.diag(s[:n]) @ VT[:n, :],
                        columns=col_for_imput)

    # Create list of columns that will use the SVD imputation
    c_replace = [c for c in col_for_imput if na_proportions[c] < na_cutoff]

    # print(c_replace)

    def to_answer(x, answers):
        '''Rounds x to the closes number in the list set_values'''
        answers = np.array(answers)
        index = np.argmin(np.abs(answers - x))
        return answers[index]

    # Use the replace data frame to fill in NAs
    for c in c_replace:
        a = {'answers': master_df[c].dropna().unique()}
        master_df.loc[master_df[c].isna(), c] = (replace.loc[master_df[c].isna(), c]
                                                ).apply(to_answer, **a)


    master_df.fillna(median_fill, inplace=True)
    # When talking with Dr. Davey Erekson, it was suggested that we aggregate
    # the follow up questions about disabilities and trauma to be a number count of the yes's

    # Disability
    # rename column name that ask about disabilities
    rename = {'Are you registered, with the office for disability services on this campus, as having a documented and diagnosed disability?':
            'Disabilities'}
    master_df.rename(columns=rename, inplace=True)


    disability = [c for c in master_df.columns if 'If you selected, "Yes"' in c]
    disability_values = (
        master_df[disability].apply(lambda x: sum(x.dropna()), axis=1) 
    )  # adds up the number of disabilities

    disability_filter = pd.to_numeric(master_df['Disabilities'], errors='coerce')
    disability_filter = disability_filter.replace({np.nan: 0}) > 0
    disability_values[disability_filter & (disability_values == 0)] += 1  
    # This is included for those who said they had a disability but did not mark the follow up questions
    master_df.loc[:, 'Disabilities'] = disability_values
    # Assigns number of disabilities to Disabilities column

    master_df.drop(disability, axis=1, inplace=True)

    ####### Trauma

    # Childhood
    # column names that ask about childhood trauma
    modified_names = ['ChildhoodTrauma', 'Trauma']
    childhood = [c for c in master_df.columns if 'Childhood' in c]
    # binary y/n
    childhood_values = master_df[childhood].apply(lambda x: 1 if sum(x.dropna()) > 0 else 0, axis=1)
    master_df.insert(loc=master_df.columns.get_loc(
        childhood[0]), column=modified_names[0], value=childhood_values)

    # Trauma
    trauma = [c for c in master_df.columns if (
        'Please select the traumatic event' in c) & ('Childhood' not in c)]
    # Column names that are about trauma but not childhood trauma
    trauma_values = master_df[trauma].apply(lambda x: sum(x.dropna()), axis=1)
    master_df.insert(loc=master_df.columns.get_loc(
        trauma[0]), column=modified_names[1], value=trauma_values)
    master_df.loc[:, 'Trauma'] += master_df[childhood].apply(lambda x: sum(x.dropna()), axis=1).values
    # Trauma is the number of yes's in the trauma columns, including childhood

    master_df.drop(childhood, axis=1, inplace=True)
    master_df.drop(trauma, axis=1, inplace=True)

    # specify dtypes
    for c in master_df.columns:
        master_df[c] = master_df[c].astype(dtypes_[c])

    print('Imputation complete')
    return master_df

