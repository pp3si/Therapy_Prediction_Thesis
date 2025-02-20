import pandas as pd
import numpy as np
import CAPS
from CAPS import ProgressBar
import os.path
import random
from statistics import mode
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

include_apptype = ["Daytime Crisis", "Individual Tx", 
                   'Transfer', 'Individual - in person', 
                   'Intake -  Zoom', 'Quick Care - zoom', 
                   'Quick Care - In person', "Intake", 
                   'Consultation', 'PM Crisis',
                   'Individual - Zoom', 'Intake - in person',
                   'Urgent Concerns', 'Housing Tx',  
                   'Athletic Intake', 'Athletic Individual',
                   'Athletic Transfer', 
                   'Athletics - Daytime Crisis',
                   'Athletics PM Crisis', 
                   'Daytime Crisis/Intake',
                   'Business School Intake - Zoom',
                   'Business School Individual - In person',
                   'Business School Individual - Zoom',
                   'Business School Intake - in person',
                   'Business School Individual',
                   'Business School Intake',
                   '<Business School Individual - Zoom',
                   'Law School Individual',
                   'Law School Intake', 
                   'Business School Quick Care',
                   'Athletics Quick Care', 
                   'Law School Quick Care',
                   'One-Time Consultation']


def small_clean(data):
    # Drop na clients, replace missing therpist ID with -1,
    # and convert ids
    data.replace('#NULL!', np.nan, inplace=True)
    data.dropna(subset=["ClientID"], inplace=True)
    data["ClientID"] = data["ClientID"].astype(float).astype(np.int64).astype(str)
    data.fillna({"TherapistID": -1}, inplace=True)
    data["TherapistID"] = data["TherapistID"].astype(np.int64).astype(str)
    data["AppID"] = data["AppID"].astype(str)
    return data


def load_old_app(data_path):
    old_app = pd.read_csv(os.path.join(*data_path, "Appointments.csv"),
                low_memory=False)
    return small_clean(old_app)


def load_new_app(data_path):
    new_app = pd.read_csv(os.path.join(*data_path, "appointments aug 2018 to april 2023.csv"),
                low_memory=False)
    new_app.columns = ["ClientID", "AppID", "AppType", 
                       "Date", "time", "length", 
                       "TherapistID", "AttendanceDescription", 
                       "AttendanceDescriptionCopy", 
                       "IndividualOrGroup"]

    return small_clean(new_app)


def combine_app(old, new):
    old_c = old.copy()
    old_c["Date"] = pd.to_datetime(old_c["Date"])
    new["Date"] = pd.to_datetime(new["Date"])
    combined = pd.concat([old_c, new])
    combined.index = range(combined.shape[0])
    combined = combined[["ClientID", "AppID", "AppType", 
                   "Date", "TherapistID",
                   "AttendanceDescription"]]
    combined["Date"] = pd.to_datetime(combined["Date"])
    combined.drop_duplicates(subset=["ClientID",
                                     "AppID",
                                     "TherapistID"],
                             inplace=True,
                             keep="last") # We want to 
                             # keep the last (i.e. the newer data)
                             # as some appointments that were just
                             # "Scheduled" in the old are now
                             # "Attended" (example AppID: 904411)
    # Some appointments have multiple clients
    # or therapists, that is why we cannot
    # simply drop_duplicates on AppID
    return combined


def hide_or_no(appoint_data, 
               hide=True,
               data_path_old=[".."],
               data_path_new=["..", "2018-2023"],
               dataset="both"):
    """
    Pass in appointment data and returns
    the data without the hidden clients or
    only the data with the hidden clients
    """
    apts_df = appoint_data.copy()

    # Union of the two lists
    if dataset=="both":
        hide_2 = pd.read_csv(os.path.join(*data_path_new, 'ClientsHiddenAway2018_2023'))
        hide_1 = pd.read_csv(os.path.join(*data_path_old, 'ClientsHiddenAway'))
        hide_ = pd.concat([hide_1, hide_2])
    elif dataset=="old":
        hide_ = pd.read_csv(os.path.join(*data_path_old, 'ClientsHiddenAway'))
    elif dataset=="new":
        hide_ = pd.read_csv(os.path.join(*data_path_new, 'ClientsHiddenAway2018_2023'))
    else:
        raise ValueError("Unrecognized dataset type given to hide_or_no")

    if hide:
        apts_df = apts_df.loc[~apts_df['ClientID'].isin(hide_.iloc[:,0].astype(str)), :]
    else:
        apts_df = apts_df.loc[apts_df['ClientID'].isin(hide_.iloc[:,0].astype(str)), :]
    return apts_df


def make_session(appoint_data,
                 unique_therapist: bool,
                 intake_dif: bool,
                 min_appoint=1,
                 max_diff_between_apps=180,
                 include_apptype=include_apptype
                 ):
    """
    This function takes the appointment data
    and patient information data and returns
    a data frame with the sessions + survey
    information and a data frame with the
    appointments, with the relevant session
    ID number.

    appoint_data: The appointment data.
    unique_therapist: If true, then a student
        who visits two therapists at the same
        time will have two different sessions,
        assuming the other session conditions
        are met, with the possible exception
        of the intake appointment (See 
        intake_dif). 
    intake_dif: Determines whether or not to
        include the intake appointment in a
        session if it was with a different
        therapist. To include such appointments
        pass True. Will pass error if 
        unique_therapist = False.
    min_appoint: The minimum number of
        appointments in a session to be
        considered a session.
    max_diff_between_apps: The max number days 
        between attended appointments before a 
        new session begins.
    include_apptype: The list of the appointment
        types that should be included in the
        cleaning. By default, it a list of all
        the individual appointment types.
    """ 
    if (not unique_therapist) & intake_dif:
        raise ValueError("intake_dif is true but unique_therapist is false. This combination is not allowed.")
    
    dropped = pd.DataFrame(columns=['Why', 'How Many'])

    # Cleaning of Appointments data frame
    apts_df = appoint_data.copy()

    rows = apts_df.shape[0]
    dropped.loc[dropped.shape[0], :] = ('Rows in in passed appointment data', rows)

    # Make Dates a Datetime object
    apts_df['Date'] = pd.to_datetime(apts_df['Date'])

    # Keep appointment types that are of interest: related to individual therapy

    apts_df = apts_df[apts_df["AppType"].isin(include_apptype)]

    # For appointments with more than one client 
    # "Consultation" appears to allow for this
    num_of_clients = apts_df.groupby("AppID")["ClientID"].nunique()
    keep_AppID = list(num_of_clients.loc[num_of_clients == 1].index)
    apts_df = apts_df[apts_df["AppID"].isin(keep_AppID)]

    dropped.loc[dropped.shape[0], :] = ('Remove certain appointment types', apts_df.shape[0] - rows)
    
    # ### Aggregate Sessions

    client_IDs = apts_df['ClientID'].unique().astype(str)


    def closest_day(x, series):
        """
        Returns the smallest number of days
        between x and the pd series of dates.
        Must be nonnegative.
        """
        diff_day = (x - series).apply(lambda x: x.days)
        positive_filter = diff_day >= 0
        if sum(positive_filter) == 0:
            return 0
        out = diff_day.loc[positive_filter].min()
        return out.days


    def elapsed(x):
        """
        Complicated but does what we want.
        elap1: date difference not grouped
            by attendance description
        elap2: date difference grouped by
            by attendance description
        elap3: date difference from the
            closest intake appointment
        """
        attend_ = x["AttendanceDescription"] == "Attended"
        if attend_:
            v_1 = max(x[["elap1", "elap2"]])
        else:
            v_1 = x["elap1"]

        if intake_dif:
            v_2 = x["elap3"]
            return min([v_1, v_2])
        else:
            return v_1


    def start_end_row(Dates):
        x  = Dates.dropna()
        if len(x) >= 1:
            return [min(x), max(x)]
        else:
            return [np.nan, np.nan]


    def start_end(app):
        '''
        Start date and end date function.
        Does not include appointments that were not attended
        '''
        app_ = app.copy()
        mask = app['AttendanceDescription'] != "Attended"
        app_.loc[mask, "Date"] = np.nan
        out = app_.groupby('SessionID')['Date'].apply(start_end_row)
        return out


    n = len(client_IDs)
    
    if unique_therapist:
        master_df_col = ['PatientID', 'TherapistID', 
                        "SessionID", 'StartDate', 
                        'EndDate', 'NumOfAttended',
                        "Crisis"]
    else:
        master_df_col = ['PatientID', 'MainTherapist', 
                        "SessionID", 'StartDate', 
                        'EndDate', 'NumOfAttended',
                        "Crisis"]
    
    master_df = pd.DataFrame([["str", "str", 0, "str", "str", 0, 0]],
                              columns=master_df_col) 
                              # dataframe with entries
                              # to specify dtypes

    master_df = master_df[0:0] # empty

    app_session = pd.DataFrame([[0, "str", "str", "str", "str", "str", 0]],
                                columns=['AppID', "ClientID", 
                                         'TherapistID', 'Date', 
                                         'AppType', 
                                         'AttendanceDescription', 
                                         "SessionID"])[0:0]
    
    app_columns = list(app_session.columns[:-1]) + ['Elapsed','Zoom','Crisis', 'SessionID']
    app_session_rows_list = []
    master_df_rows_list = []

    sess_num = 0
    print('Aggregating Sessions')
    for j, client in enumerate(client_IDs):
        ProgressBar(j, n)

        # Data frame containing the AppIDs TherapistIDs, and Dates for a particular client
        app = apts_df[
                    apts_df['ClientID'] == client
                        ][['AppID', 'ClientID', 'TherapistID', 'Date', 'AppType', 'AttendanceDescription']]

        # Sorts the data frame based on date
        app.sort_values("AttendanceDescription", ascending=False, inplace=True)
        app.sort_values('Date', inplace=True)

        if app.shape[0] < 1:
            # If the data-frame has no appointments, it is not included
            continue
        

        # Create column 'Elapsed' which has the number of days since 
        # the last attended appointment for attended appointments 
        # and days since last appointment of the for non-attended appointments,
        # with an expeption if the first appointment is
        # not attended.
        if app.shape[0] == 1:
            app["Elapsed"] = 0
        else:
            if unique_therapist:
                app["elap1"] = app.groupby("TherapistID")["Date"].diff()
                app["elap2"] = app.groupby(['TherapistID', 'AttendanceDescription'])["Date"].diff()    

                if intake_dif:
                    intake_i = app.loc[app['AppType'].apply(lambda x: "intake" in x.lower()), :].index
                    intake_dates = app.loc[intake_i, 'Date']
                    app["elap3"] = app["Date"].apply(lambda x: closest_day(x, intake_dates))

            else:
                app["elap1"] = app["Date"].diff()
                app["elap2"] = app.groupby(['AttendanceDescription'])["Date"].diff()   

            ####

            for c in ["elap1", "elap2"]:
                app[c] = app[c].apply(lambda x: x.days)    
            app.fillna(0, inplace=True)
                
            app['Elapsed'] = app.apply(elapsed, axis=1)
            
            app.drop([c for c in app.columns if "elap" in c],
                     axis=1,
                     inplace=True)
        ###
        
        # indicator for zoom appointments
        app["Zoom"] = app["AppType"].apply(lambda x: "zoom" in x.lower())
        
        # If the appointment of type crisis, yes(1)/no(0)
        app['Crisis'] = app['AppType'].apply(lambda x: 1 if x.lower()[-6:] == 'crisis' else 0)

        app['SessionID'] = sess_num  # Assigns values and creates a column called SessionID
        
        if unique_therapist & intake_dif:
            catch_intake_split = pd.DataFrame(columns=app.columns, dtypes=app.dtypes)

        # Number sessions
        for therapist in app['TherapistID'].unique():
            sessions = []

            if unique_therapist & intake_dif:
                mask = (app['TherapistID'] == therapist) | (
                    app['AppType'].apply(lambda x: 'intake' in x.lower())
                )
            elif unique_therapist:
                mask = (app['TherapistID'] == therapist)
            else:
                mask = app.index
            
            for i in range(app.loc[mask, :].shape[0] - 1):
                sessions.append(sess_num)
                if app.loc[mask, :].iloc[i + 1]['Elapsed'] > max_diff_between_apps:
                    sess_num += 1
            
            sessions.append(sess_num)

            # In the case that two sessions with two different
            # therapist may both may be close enough to the
            # "intake" appointment to count it in each session
            if unique_therapist & intake_dif:
                # This will catch the intakes that have already
                # been assigned a session
                intake_filter = app.loc[mask, "SessionID"] > 0
                if app.loc[mask, "SessionID"] > 0:
                    catch_intake_split = app.loc[intake_filter, :]

            app.loc[mask, 'SessionID'] = sessions
            sess_num += 1

        # Attach the intake appointments that were overwritten
        if unique_therapist & intake_dif:
            if catch_intake_split.shape[0] > 0:
                app = pd.concat([app, catch_intake_split])

        # Makes a data-frame with columns for the start and end dates
        
        temp = pd.DataFrame(list(start_end(app)), 
                            columns=['StartDate', 'EndDate'])

        temp.insert(loc=0, column='PatientID', value=client)

        # Adds the therapist ID that corresponds to the therapist during the session dates
        # or the main therapist
        val = app.groupby('SessionID')["TherapistID"].apply(mode).values
        if unique_therapist:
            temp.insert(loc=1, column="TherapistID", value=val)
        else:
            temp.insert(loc=1, column="MainTherapist", value=val)

        # Adds the SessionID, so that we can cross reference with app_session/app_oq
        temp["SessionID"] = app.groupby('SessionID')['SessionID'].apply(lambda x: x.iloc[0]).values

        # Adds number of appointments in session    
        temp['NumOfAttended'] = (
            app.groupby('SessionID')['AttendanceDescription'].apply(lambda x: sum(x == "Attended")).values
        )  

        # 1 if the first appointment is crisis, 0 if not
        temp['Crisis'] = (
            app.groupby('SessionID')["Crisis"].apply(lambda x: x.iloc[0]).values
        )

        for i in range(len(app)):
            app_session_rows_list.append(list(app.iloc[i]))
        for i in range(len(temp)):
            master_df_rows_list.append(list(temp.iloc[i]))

        temp_cols = temp.columns

        # app_session = pd.concat([app_session, app])
        # master_df = pd.concat([master_df, temp])

    app_session = pd.DataFrame(app_session_rows_list, columns=app_columns)
    master_df = pd.DataFrame(master_df_rows_list, columns=temp_cols)

    master_df['NumOfAttended'].replace({np.nan: 0}, inplace=True)
    master_df['StartDate'] = pd.to_datetime(master_df['StartDate'])
    master_df['EndDate'] = pd.to_datetime(master_df['EndDate'])
    rows = master_df.shape[0]
    dropped.loc[dropped.shape[0], :] = ('Number of Sessions', rows)
    master_df = master_df.loc[master_df['NumOfAttended'] >= min_appoint, :]
    dropped.loc[dropped.shape[0], :] = (f'Drop Sessions that are fail to have {min_appoint} attended appointment(s)', master_df.shape[0] - rows)
    
    rows = master_df.shape[0]
    # Remove sessions in which a client had 
    # a single appointment with more than one therapist
    # It is rare and proves problematic (infact, appID 900924
    # might involve an recording error), so we remove them
    num_of_therap = app_session.groupby("AppID")["TherapistID"].nunique()
    remove_app_id = list(num_of_therap.loc[num_of_therap > 1].index)
    remove_session = list(app_session.loc[app_session["AppID"].isin(remove_app_id), "SessionID"])
    
    app_session = app_session.loc[~app_session["SessionID"].isin(remove_session), :]
    master_df = master_df.loc[~master_df["SessionID"].isin(remove_session), :]
    
    dropped.loc[dropped.shape[0], :] = ('Drop Sessions with appointments that have multiple therapists in one appointment', master_df.shape[0] - rows)
    master_df.index = range(master_df.shape[0])
    app_session.index = range(app_session.shape[0])

    print(dropped)
    return master_df, app_session


#############################################################################################
#  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  #
#############################################################################################


def column_rename(current, key):
    # call new but could still modify the old
    # since it does not use copy
    new = current
    for j, i in enumerate(new):
        i = i.upper()
        if i in list(key['Column Name']):
            new[j] = str(key[key['Column Name'] == i].Description.values[0])
    return new


def load_keep(keep_path, old: bool):
    with open(os.path.join(*keep_path, "keep.txt"), "r") as f:
        keep = f.read().split("\n")
    
    if old:
        keep.remove('Seeking Services Due To COVID-19')
        return keep
    else:
        return keep


def subset_pat(patient_info):
    '''
    Some surveys are only partly filled out
    This correpsonds to surveys administered
    through out therapy

    We want avoid these since the clients
    did not fill in answers to most of the 
    questions
    '''
    patient_info.replace('<No Response>', np.nan, inplace=True)
    mask_1 = patient_info.iloc[:, -62:].isna().mean(axis=1) < .25 # CCAPS 62
    mask_2 = ~patient_info["Race / Ethnicity"].isna()
    # filled out both parts of the survey
    subset_pat = patient_info.loc[mask_1 & mask_2, :]
    return subset_pat


def load_old_patient_info(data_path, keep_path):
    """
    This function loads, edits, and outputs the patient information surverys from 2012 to 2019.
    The argument keep_path is for locating the keep.txt file.
    """
    pat_df_1 = pd.read_csv(os.path.join(*data_path, "PatientInformation.csv"), low_memory=False)
    pat_df_1['ClientID'] = pat_df_1['ClientID'].astype(float).astype(np.int64).astype(str)
    pat_df_1['notedate'] = pd.to_datetime(pat_df_1['notedate'])
    key = pd.read_csv(os.path.join(*data_path, 'Key.csv'))

    pat_df_1.columns = column_rename(list(pat_df_1.columns), key)

    # Oddly, the non-inactive Gender is all NA values
    pat_df_1['Gender'] = pat_df_1['Gender <inactive>']

    # Drop columns marked as inactive
    drop = [i for i in pat_df_1.columns if 'inactive' in i]
    pat_df_1.drop(drop, axis=1, inplace=True)

    pat_df_1['Which sex was assigned to you at birth?'] = pat_df_1['Gender'] 
    pat_df_1 = pat_df_1.loc[~(pat_df_1['age'] == 0), :]

    pat_df_1['notedate'] = pd.to_datetime(pat_df_1['notedate'])
    pat_df_1['age'] = pd.to_numeric(pat_df_1['age'], errors="coerce")

    def birthday(x):
        if not pd.isna(x['age']) and x['age'] > 5:
            return x['notedate'] - pd.tseries.offsets.DateOffset(days = int(x['age'] * 365.25))
        else:
            return np.nan

    pat_df_1["Date of Birth"] = pat_df_1[["notedate", 'age']].apply(birthday, axis=1)

    # Subset to columns that Dr. Davey Erekson suggested
    keep = load_keep(keep_path, old=True)
    pat_df_1 = pat_df_1[keep]

    # Remove mostly NA rows
    pat_df_1 = subset_pat(pat_df_1)

    return pat_df_1


def load_new_patient_info(data_path, keep_path):
    """
    This function loads, edits, and outputs the patient information surverys from 2018 to 2023.
    The argument keep_path is for locating the keep.txt file.
    """
    ccaps = pd.read_csv(os.path.join(*data_path, "CCAPS aug 2018 april 2023.csv"), low_memory=False)
    sds = pd.read_csv(os.path.join(*data_path, "SDS aug 2018 to april 2023.csv"), low_memory=False)
    ccaps.insert(loc=0, column="ClientID", value=pd.to_numeric(ccaps["client ID"], errors="coerce"))
    sds.insert(loc=0, column="ClientID", value=pd.to_numeric(sds["client ID"], errors="coerce"))
    ccaps.dropna(subset=["ClientID"], inplace=True)
    sds.dropna(subset=["ClientID"], inplace=True)
    ccaps["ClientID"] = ccaps["ClientID"].astype(np.int64).astype(str)
    sds["ClientID"] = sds["ClientID"].astype(np.int64).astype(str)
    ccaps.drop("client ID", axis=1, inplace=True)
    sds.drop("client ID", axis=1, inplace=True)

    key_c = pd.read_excel(os.path.join(*data_path, "CCAPS key.xls"), 
                        header = 5, 
                        names = ["Column Name", "Description"]).iloc[:-2]
    key_s = pd.read_excel(os.path.join(*data_path, "SDS key.xls"), 
                        header = 5, 
                        names = ["Column Name", "Description"]).iloc[:-2]

    ccaps.columns = column_rename(list(ccaps.columns), key_c)
    sds.columns  = column_rename(list(sds.columns), key_s)

    pat_df_2 = pd.merge(sds, ccaps, how="left", 
                    left_on=["notedate", "ClientID"], 
                    right_on = ["notedate", "ClientID"])

    mask_sex_na = pat_df_2.loc[:, 'Which sex was assigned to you at birth?'].isna()
    pat_df_2.loc[mask_sex_na, 'Which sex was assigned to you at birth?'] = (
                pat_df_2.loc[mask_sex_na, "Gender <inactive>"].iloc[:,1]
            )
    # multiple columns are called "Gender <inactive>" 
    # The second one has values where "Which sex..." is na
    
    # Not really inactive but there are two
    pat_df_2["Academic Status"] = pat_df_2["Academic Status <inactive>"].iloc[:,0]

    # Subset to columns that Dr. Davey Erekson suggested
    keep = load_keep(keep_path, old=False)
    pat_df_2 = pat_df_2[keep]


    def dob_nan(dob):
        year = dob[-4:]
        try:
            if int(year) < 2015:
                return dob
            else:
                return ""
        except ValueError:
            return ""
    

    pat_df_2["Date of Birth"] =  pat_df_2["Date of Birth"].apply(dob_nan)
    # some DoBs are on the nearly the same day as the survey date
    # or in the future; fill with an empty string

    pat_df_2 = subset_pat(pat_df_2)

    return pat_df_2


def combine_patient_info(new, old):
    old_col = old.columns
    patient_info = pd.concat([old, new])
    patient_info.drop_duplicates(subset=old_col, inplace=True)
    return patient_info


#############################################################################################
#  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  #
#############################################################################################

scale_times = {'Never':             0, 
               '1 time':            1, 
               '2-3 times':         2, 
               '4-5 times':         3,
               'More than 5 times': 4}

scale_stress = {'Never stressful':     0, 
                'Rarely stressful':    1, 
                'Sometimes stressful': 2,
                'Often stressful':     3, 
                'Always stressful':    4}

scale_freq = {'0':                      0, 
              'Rarely (0-1 nights)':    0, 
              'Sometimes (2-3 nights)': 1, 
              'Often (4+ nights)':      2}

scale_agree = {'Strongly disagree': -2, 
               'Somewhat disagree': -1,
               'Neutral':            0, 
               'Somewhat agree':     1, 
               'Strongly agree':     2}

scale_importance = {'Very unimportant': -2, 
                    'Unimportant':      -1,
                    'Neutral':           0, 
                    'Important':         1, 
                    'Very important':    2}


def match_patient_info(master_df, 
                       patient_data,
                       recent_dict
                       ):
    """
    This function takes in the data frame from make_session and attaches
    a completed intake survey to it if one is available.

    master_df: master_df from make_session.
    patient_data: Pandas data frame of the patient information.
        Should have columns renamed and combined (see load
        functions) before passing.
    recent_dict: The dictionary used for assigning ones or zeros
        to the last time columns. Example,
        {'Within the last 2 weeks': 1, 'Within the last month': 1,
        'Within the last year': 1, 'Within the last 1-5 years': 0,
        'More than 5 years ago': 0, 'Never': 0, '<No Response>': 0}
    """
    dropped = pd.DataFrame(columns=['Why', 'How Many'])
    rows = master_df.shape[0]
    # ### Match patient information to session
    print('Matching patient information to session')
    subset_pat = patient_data.copy()

    # Convert columns to certain types
    master_df['PatientID'] = master_df['PatientID'].astype(str)
    master_df['StartDate'] = pd.to_datetime(master_df['StartDate'])
    master_df['EndDate'] = pd.to_datetime(master_df['EndDate'])
    subset_pat['ClientID'] = subset_pat['ClientID'].astype(str)
    subset_pat['notedate'] = pd.to_datetime(subset_pat['notedate'])

    # Add columns to master_df that are in patient info, except for ClientID
    add_columns = pd.DataFrame(columns=subset_pat.drop('ClientID', axis=1).columns)
    master_df = master_df.join(add_columns)


    stable = ['Date of Birth', 'Which sex was assigned to you at birth?', 
              'Race / Ethnicity', 'International Student',
              "Have you been diagnosed with an autism-spectrum disorder or Asperger's Syndrome?",
              'First Generation', 'Financial Stress Past', 'Religion', 'Religion Importance']

    last_time = [c for c in subset_pat.columns if 'Last time' in c]

    changing = list(set(subset_pat.iloc[:, 3:].columns) - set(stable) - set(last_time))

    convert_dictionary = {'Within the last year': 365, 'More than 5 years ago': 'same',
                        'Within the last 1-5 years': 'same', 'Within the last 2 weeks': 14,
                        'Within the last month': 30, 'Never': 'same'}


    def last_time_convert(value, time_past):
        x = convert_dictionary[value] + time_past
        if x <= 14:
            return 'Within the last 2 weeks'
        if x <= 30:
            return 'Within the last month'
        if x <= 365:
            return 'Within the last year'
        if x <= 1825:
            return 'Within the last 1-5 years'
        return 'More than 5 years ago'

    lost = []

    print(f"Total number of sessions: {master_df.shape[0]}")
    num_outside_default_range = 0
    num_no_survey = 0
    for i in range(master_df.shape[0]):
        ProgressBar(i, master_df.shape[0])
        # Get a session (row) of master_df
        client = master_df.loc[i]
        # Find the patient information for that client
        information = subset_pat[subset_pat['ClientID'] == client['PatientID']].sort_values('notedate')

        if information.shape[0] == 0:
            lost.append(['NoPaper', client['PatientID']])
            num_no_survey += 1
            # If there is not information, then add to lost list so that you may look at these clients
            continue

        # Calculate how difference between session start and survey notedate 
        # and make it the information index
        information.index = (client['StartDate'] - information['notedate']
                            ).apply(lambda x: x.days)  # if notedate is after StartDate, returns a negative value

        information.sort_index()  # low to high

        # Take the nearest survey to assign for patient information of that session in master_df
        try:
            survey = information.loc[
                (information.index >= 0) & (information.index <= 90), :].iloc[0]  # First tries to find a survey 90 days before
            # the session started
        except IndexError:
            num_outside_default_range += 1
            survey = information.iloc[np.argmin(abs(information.index)), :]
            # if try fails, then use closest (absolute) survey


        # Fill missing values using other surveys
        if information.shape[0] > 1:
            for c in stable:
                if np.isnan(survey.isna()[c]):
                    for i in information.drop(survey.index).index:
                        if ~np.isnan(information.isna().loc[i, c]):
                            survey[c] = information.loc[i, c].values
                            break

            # For changing, look only at the surveys that were within
            # 90 days before the session start
            for c in changing:
                if np.isnan(survey.isna()[c]):
                    for i in information.drop(survey.index).index:
                        if (~np.isnan(information.isna().loc[i, c])
                            ) & (information.loc[i].index <= 90
                                ) & (information.loc[i].index >= 0):
                            survey[c] = information.loc[i, c].values
                            break

            for c in last_time:
                if np.isnan(survey.isna()[c]):
                    for i in information.drop(survey.index).index:
                        if (~np.isnan(information.isna().loc[i, c])) & (information.loc[i].index >= 0):
                            value = information.loc[i, c].values
                            time_past = information.loc[i].index
                            fill = value
                            if convert_dictionary[value] != 'same':
                                # Later, answers that are beyond one year will be changed to a 0,
                                # so not it does not need to be changed if it already beyond a year
                                fill = last_time_convert(value, time_past)

                            survey[c] = fill
                            break

        master_df.loc[i, subset_pat.drop('ClientID', axis=1).columns] = survey

    # End of session-survey matching
    print(f"Num with no surveys: {num_no_survey}")
    print(f"Num outside default date range: {num_outside_default_range}")

    rows = master_df.shape[0]
    master_df.dropna(subset='notedate', inplace=True)  # Drop sessions that had no surveys
    dropped.loc[dropped.shape[0], :] = ('No surveys', rows - master_df.shape[0])

    master_df.index = range(master_df.shape[0])

    master_df['notedate'] = pd.to_datetime(master_df['notedate'])
    master_df['StartDate'] = pd.to_datetime(master_df['StartDate'])
    master_df['Date of Birth'] = pd.to_datetime(master_df['Date of Birth'])

    # New column DateDifference that is the difference between the session start date and the notedate
    date_diff = master_df[['StartDate', 'notedate']].apply(
        lambda x: (x['StartDate'] - x['notedate']).days, axis=1)
    master_df.insert(loc=master_df.columns.get_loc('notedate') + 1,
                    column='DateDifference', value=date_diff)

    # Add age
    age = master_df[['StartDate', 'Date of Birth']].apply(
        lambda x: (x['StartDate'] - x['Date of Birth']).days, axis=1) / 365.25
    master_df.insert(loc=master_df.columns.get_loc('Date of Birth') + 1,
                    column='age', value=age)
    
    rows = master_df.shape[0]
    mask_16 = ~(master_df["age"] < 16) # doing >= 16 would drop na rows
    mask_100 = ~(master_df["age"] > 100)
    master_df = master_df.loc[mask_16 & mask_100, :]
    dropped.loc[dropped.shape[0], :] = ('Age less than 16 or over 100', rows - master_df.shape[0])

    # ### To Numeric

    # Answers on a scale are converted to numbers
    scale_list = [scale_times,
                  scale_agree, 
                  scale_freq, 
                  scale_importance,
                  scale_stress]

    # Make replace dictionary
    replace = {}
    for c in master_df.columns:
        unique_v = master_df[c].dropna().unique()
        for scale in scale_list:
            if set(scale.keys()).issuperset(unique_v):
                replace[c] = scale
                break

    master_df.replace(replace, inplace=True)


    last_time = [c for c in master_df.columns if 'Last time' in c]
    for c in last_time:
        master_df[c].replace(recent_dict, inplace=True)

    rename_1 = {}
    for c in last_time:
        master_df[c] = master_df[c].astype(float)
        rename_1[c] = c.replace('Last time', 'Recent')

    master_df.rename(columns=rename_1, inplace=True)

    academic_columns = pd.get_dummies(master_df['Academic Status'], prefix='', prefix_sep='')
    
    acad_list = ["Freshman / First-year", 
                 "Sophomore", "Junior", 
                 "Senior", 
                 'Graduate / professional degree student']

    for c in acad_list:
        master_df.insert(loc=master_df.columns.get_loc('Academic Status'), 
                        column=c, 
                        value=academic_columns[c])
    master_df.drop('Academic Status', axis=1, inplace=True)

    # Convert certain columns to yes(1)/no(0)

    prior = lambda x: 0 if x == 'Never' else np.nan if pd.isna(x) else 1

    for c in ['Prior Counseling', 'Prior Meds']:
        master_df[c] = master_df[c].apply(prior)
    

    rename_2 = {'Which sex was assigned to you at birth?':      'Female',
                'Relationship Status':                     'MarriedMale',
                'Race / Ethnicity':                     'RacialMinority',
                'Sexual Orientation':           'SexOrientationMinority',
                'Religion':                          'ReligiousMinority',
                'In what college is your current major?': 'NursingOrLaw',
                'Housing Other':                              'Homeless'}

    master_df.rename(columns=rename_2, inplace=True)


    female = lambda x: 1 if x == 'Female' else 0 if x == 'Male' else np.nan

    master_df['Female'] = master_df['Female'].apply(female)


    marriedmale = lambda x: 1 if x.iloc[0] == 0 and x.iloc[1] == 'Married' else 0

    master_df['MarriedMale'] = (master_df[['Female', 'MarriedMale']]
                                ).apply(marriedmale, axis=1)


    racialminority = lambda x: 0 if x == 'White' else np.nan if pd.isna(x) else 1

    master_df['RacialMinority'] = (master_df['RacialMinority']
                                   ).apply(racialminority)


    def sexorientminority(x):
        if x == 'Heterosexual / Straight':
            return 0
        elif x == 0:
            return 0
        elif pd.isna(x):
            return np.nan
        else:
            return 0
    

    master_df['SexOrientationMinority'] = (master_df['SexOrientationMinority']
                                           ).apply(sexorientminority)


    def agn_or_ath(x):
        if x == 'Agnostic' or x == 'Atheist':
            return 1
        elif pd.isna(x):
            return np.nan
        else:
            return 0
    

    master_df.insert(loc=master_df.columns.get_loc('ReligiousMinority'),
                    column='AgnosticOrAtheist',
                    value=master_df['ReligiousMinority'].apply(agn_or_ath))


    def religiousminority(x):
        if x == 'Christian' or x == 'Atheist' or x == 'Agnostic':
            return 0
        elif pd.isna(x):
            return np.nan
        else:
            return 0


    master_df['ReligiousMinority'] = master_df['ReligiousMinority'].apply(religiousminority)

    # Dr. Erekson said 'In what college is you major' might be relevent 
    # if the client is a nursing or law student
    nurselaw = lambda x: 1 if (x=='Law' or x=='Nursing') else 0
    master_df['NursingOrLaw'] = master_df['NursingOrLaw'].apply(nurselaw)


    def to_homeless(x):
        x = str(x)
        if 'homeless' in x.lower() or 'none' in x.lower() or x.lower() == 'no':
            return 1
        else:
            return 0


    master_df['Homeless'] = master_df['Homeless'].apply(to_homeless)
    # This is a hard one to convert, since it is not multiple choice
    # After taking a look as some of the answers, looking for homeless and none,
    # somewhere in the string or if the string is simply 'no' seemed like a wise choice

    print(dropped)

    return master_df


####################################################################################################
