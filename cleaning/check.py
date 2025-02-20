import pandas as pd
import numpy as np
import os.path
import inspect
from cleaning import sessioninfo, scores, imputation, filtr


def convert_types(data):
    """
    Convert the types in master_df and app_oq to 
    prepare for check_all. Converts inplace.
    """
    to_str = ["PatientID", "ClientID",
              "TherapistID", "MainTherapist"]
    for c in to_str:
        if c in data.columns:
            data[c] = data[c].astype(str)
    
    to_dt = ["notedate", "Date",
             "StartDate", "EndDate"]
    for c in to_dt:
        if c in data.columns:
            data[c] = pd.to_datetime(data[c], format="mixed")

    return None


def to_string(x):
    """
    Used in dataframe method 'apply' to convert values
    without converting the entire column. That way
    avoids the problem of that applying astype would
    run into if value is np.nan.
    """
    try:
        return str(np.int64(float(x)))
    except ValueError:
        return x


class renamer(): # for columns with the same name
    # https://stackoverflow.com/questions/40774787/
    # renaming-columns-in-a-pandas-dataframe-with-duplicate-column-names
    def __init__(self):
        self.d = dict()

    def __call__(self, x):
        if x not in self.d:
            self.d[x] = 0
            return x
        else:
            self.d[x] += 1
            return "%s_%d" % (x, self.d[x])


def load_raw_data(data_path_old, data_path_new):
    """
    Load and combine data files while modifying them 
    only to make them useable.
    """
    old_app = pd.read_csv(os.path.join(*data_path_old, 
                                       "Appointments.csv"),
                          low_memory=False)
    old_app["Date"] = pd.to_datetime(old_app["Date"], format="%d-%b-%y")

    new_app = pd.read_csv(os.path.join(*data_path_new, 
                                       "appointments aug 2018 to april 2023.csv"),
                          low_memory=False)
    
    new_app.columns = ["ClientID", "AppID", 
                       "AppType", "Date", "time", 
                       "length", "TherapistID", 
                       "AttendanceDescription", 
                       "AttendanceDescriptionCopy", 
                       "IndividualOrGroup"]

    new_app["Date"] = pd.to_datetime(new_app["Date"], format="%m/%d/%Y")
    old_app["AppID"] = old_app["AppID"].astype(int)
    new_app["AppID"] = new_app["AppID"].astype(int)

    id_mask = ~old_app["AppID"].isin(new_app["AppID"])
    # a few appointments (nonattended) are in the old
    # that are not in the new even though dates put it
    # within the scope of the new
    # namely, 900506, 905055, 894967, 904741, 904730, 
    # 904673, 904675, 904676, 900443, 901872, 904262, 
    # 897589, 897590, 898023, 900434, 904356,

    old_app = old_app.loc[id_mask, :]

    app = pd.concat([old_app, new_app])
    
    # patient information

    pat_df_1 = pd.read_csv(os.path.join(*data_path_old, 
                                        "PatientInformation.csv"), 
                           low_memory=False)
    key = pd.read_csv(os.path.join(*data_path_old, 'Key.csv'))

    pat_df_1.columns = sessioninfo.column_rename(list(pat_df_1.columns), key)

    ccaps = pd.read_csv(os.path.join(*data_path_new, 
                                     "CCAPS aug 2018 april 2023.csv"), 
                        low_memory=False)
    sds = pd.read_csv(os.path.join(*data_path_new, 
                                   "SDS aug 2018 to april 2023.csv"), 
                      low_memory=False)

    ccaps.rename(columns={"client ID": "ClientID"}, inplace=True)
    sds.rename(columns={"client ID": "ClientID"}, inplace=True)

    ccaps.dropna(subset=["notedate", "ClientID"], inplace=True)

    key_c = pd.read_excel(os.path.join(*data_path_new, "CCAPS key.xls"), 
                        header = 5, 
                        names = ["Column Name", "Description"]).iloc[:-2]
    key_s = pd.read_excel(os.path.join(*data_path_new, "SDS key.xls"), 
                        header = 5, 
                        names = ["Column Name", "Description"]).iloc[:-2]

    ccaps.columns = sessioninfo.column_rename(list(ccaps.columns), key_c)
    sds.columns  = sessioninfo.column_rename(list(sds.columns), key_s)

    sds.replace({"ClientID": {"#REF!": np.nan}}, inplace=True)
    sds["ClientID"] = sds["ClientID"].astype(float)

    pat_df_2 = pd.merge(sds, ccaps, how="left", 
                        left_on=["notedate", "ClientID"], 
                        right_on=["notedate", "ClientID"])

    
    pat_df_1["notedate"] = pd.to_datetime(pat_df_1["notedate"], format="%m/%d/%Y")
    pat_df_2["notedate"] = pd.to_datetime(pat_df_2["notedate"], format="%m/%d/%Y")

    pat_df_1.rename(columns=renamer(), inplace=True)
    pat_df_2.rename(columns=renamer(), inplace=True)
    pat_info = pd.concat([pat_df_1, pat_df_2], axis=0)

    pat_info.replace({'<No Response>': np.nan}, inplace=True)

    # scores

    scores_old = pd.read_csv(os.path.join(*data_path_old, 
                                          "OQ.csv"), 
                             low_memory=False)
    scores_new = pd.read_csv(os.path.join(*data_path_new, 
                                          "OQ-45 aug 2018 to april 2023.csv"),
                             low_memory=False)
    
    scores_old["AdministrationDate"] = pd.to_datetime(scores_old["AdministrationDate"], 
                                                      format="%m/%d/%Y %H:%M")
    scores_new["AdministrationDate"] = pd.to_datetime(scores_new["AdministrationDate"], 
                                                      format="%m/%d/%Y %H:%M")

    scores_new.rename(columns={'client ID': 'ClientID', 
                               'TotalScore': 'CurrentScore'}, 
                      inplace=True)

    scores_old.replace({"ClientID": {"#VALUE!": np.nan}}, inplace=True)
    scores_old["ClientID"] = scores_old["ClientID"].astype(float)
    scores_new["ClientID"] = scores_new["ClientID"].astype(float)

    scores_df = pd.concat([scores_old, scores_new], axis=0)

    scores_df = scores_df.loc[:, scores.keep]
    scores_df.drop_duplicates(inplace=True)

    # Bad scores
    mask = scores_df["CurrentScore"].apply(lambda x: (x >=0) & (x <= 180))
    scores_df = scores_df.loc[mask, :]

    # Convert ClientID to strings without changing the entire column
    app["ClientID"] = app["ClientID"].apply(to_string)
    pat_info["ClientID"] = pat_info["ClientID"].apply(to_string)
    scores_df["ClientID"] = scores_df["ClientID"].apply(to_string)

    app["TherapistID"] = app["TherapistID"].apply(to_string)

    return app, pat_info, scores_df


#############################################################################################


def check_client(appointments_raw,
                 patient_info_raw,
                 scores_raw,
                 master_df,
                 app_oq,
                 clientID=None):
    '''
    Use to check the cleaning by looking at a
    specific client. If not client ID is passed,
    it samples from master_df.
    '''
    if clientID is None:
        clientID = master_df["PatientID"].sample(1).iloc[0]

    col = ["PatientID", "StartDate", "EndDate", 
           "NumOfAttended", "Crisis", "notedate"]

    data_client = [(master_df, "PatientID"),
                   (master_df.loc[:, col], "PatientID"),
                   (app_oq, "ClientID"),
                   (appointments_raw, "ClientID"),
                   (patient_info_raw, "ClientID"),
                   (scores_raw, "ClientID")]
    
    print(f"Client ID: {clientID}")
    for data, c in data_client:
        print(data.loc[data[c]==clientID, :])
    return None


#############################################################################################
#############################################################################################
##########################################     ##############################################
##########################################     ##############################################
##########################################     ##############################################
##########################################     ##############################################
##################################                     ######################################
#####################################               #########################################
########################################         ############################################
##########################################     ##############################################
############################################ ################################################
#############################################################################################
#############################################################################################


def get_default_args(func):
    # https://stackoverflow.com/questions/
    # 12627118/get-a-function-arguments-default-value
    signature = inspect.signature(func)
    arguments = {k: v.default
                 for k, v in signature.parameters.items()
                 if v.default is not inspect.Parameter.empty}
    return arguments


arg_default = {}
function_list = [sessioninfo.make_session,
                 sessioninfo.match_patient_info,
                 scores.oq_score,
                 imputation.imputation,
                 filtr.filter_date]

for func in function_list:
    temp_dict = get_default_args(func)
    for key in temp_dict.keys():
        arg_default[key] = temp_dict[key]


patient_check_col = ["First Generation",
                     "Confusion about religious beliefs or values",
                     "Gender, ethnic, or racial discrimination",
                     "Perfectionism",
                     "Physical health problems (headaches, GI trouble)",
                     "Sexual concerns",
                     "Sexual orientation or identity"] 
                     # Instead of checking every feature,
                     # we check a subset. These columns
                     # were chosen because they are already
                     # numeric and are na on the same rows


def check_all(appointments_raw,
              patient_info_raw,
              scores_raw,
              master_df,
              app_oq,
              arg_list):
    """
    The function completes the long and arduous task of
    checking that the clean data files are as we indeed
    intend. It checks the following:


    arg_dict: Format by putting each of the function-
        argument dictionaries into a list, with the
        exception of arguments of match_patient_info 
        and include_type argument of make_session. If 
        an argument value is not given, uses default. 
        Order does not matter. 
        Example:
            [{"unique_therapist":      True,
              "intake_dif":           False,
              "min_appoint":              1,
              "max_diff_between_apps":  180},
             {"recent_dict":    recent_dict},
             {"days_before":             30, 
              "days_after":               7},
             {"cutoff_hi":              180, 
              "cutoff_low":               0}]
    """
    # create a single dictionary with all the arguments
    args_names = ["unique_therapist",
                  "intake_dif",
                  "min_appoint",
                  "max_diff_between_apps",
                  "recent_dict",
                  "days_before", 
                  "days_after",
                  "cutoff_hi", 
                  "cutoff_low"]

    arg_dict = {}

    for arg in args_names:
        added = False
        for i in range(len(arg_list)):
            if arg in arg_list[i].keys():
                arg_dict[arg] = arg_list[i][arg]
                added = True
        if not added:
            if arg in arg_default.keys():
                arg_dict[arg] = arg_default[arg]
            else:
                raise ValueError(f"must specify {arg} since it has no default")
    
    # Helpful for later
    app = app_oq.loc[app_oq["AppType"] != "OQ",]
    master_df.index = master_df["SessionID"]
    appointments_raw.index = appointments_raw["AppID"]
    app_g_sess = app.groupby("SessionID")


    ################
    ### ClientID ###
    ################

    print("Checking client ID across master_df and app_oq")

    # One client ID
    num_client = app_g_sess["ClientID"].nunique()
    mask = num_client != 1
    if any(mask):
        print("Some session has more than one client ID \U0001F61E")
        print("Offending session(s):")
        print(list(num_client.loc[mask].index))
        return None

    app_clientids = app_g_sess["ClientID"].apply(lambda x: x.iloc[0])
    mask = master_df["PatientID"] != app_clientids[master_df.index]
    if any(mask):
        print("Client ID does not match across the two data files \U0001F61E")
        print("Offending session(s):")
        print(list(master_df.loc[mask, "SessionID"]))
        return None

    print("- They all match \U0001F44D")

    ###################
    ### TherapistID ###
    ###################

    # Check therapist IDs
    app_therap = app_g_sess["TherapistID"]

    if arg_dict["unique_therapist"]:
        print("Checking therapist ID for uniqueness and constancy across data files")

        # One therapist
        if arg_dict["intake_dif"]:
            intake_mask = app["AppType"].apply(lambda x: "intake" in x.lower())
            app_therap_nointake = app.loc[~intake_mask, :].groupby("SessionID")["TherapistID"]
            therap_num = app_therap_nointake.nunique()
        else:
            therap_num = app_therap.nunique()
        
        mask = therap_num != 1
        if any(mask):
            print("Non-unique therapists \U0001F61E")
            print("Offending session(s):")
            print(list(therap_num.loc[mask].index))
            return None

        # Does master_df match with app_oq
        try:
            therapists = app_therap.apply(lambda x: x.mode()).unstack(level=1)
            # dataframe with the mode therapist(s) 
            mask = therapists.loc[master_df["SessionID"], :].eq(master_df["TherapistID"], axis=0)
            mask = ~mask.apply(any, axis=1) # does MainTherapist not match any of the mode therapists
        except ValueError: # the ValueError would be raise
            # if each session had a single mode
            therapists = app_therap.apply(lambda x: x.mode().iloc[0])
            mask = master_df["TherapistID"] != therapists[master_df["SessionID"]]
       
        if any(mask):
            print("Therapists do not match across master_df and app_oq \U0001F61E")
            print("Offending session(s):")
            print(list(master_df.loc[mask, "SessionID"]))
            return None

    else: # Non-unique therapist
        print("Checking Therapist ID for constancy across data files")
        therapists = app_therap.apply(lambda x: x.mode()).unstack(level=1)
        # dataframe with the mode therapist(s) 
        mask = therapists.loc[master_df["SessionID"], :].eq(master_df["MainTherapist"], axis=0)
        mask = ~mask.apply(any, axis=1) # does MainTherapist not match any of the mode therapists
        if any(mask):
            print("Main therapist does not match across master_df and app_oq \U0001F61E")
            print("Offending session(s):")
            print(list(master_df.loc[mask, "SessionID"]))
            return None

    # Check Therapist ID in appointments
    def check_app_therap(x):
        therap = appointments_raw.loc[x["AppID"], "TherapistID"]
        if type(therap) == pd.Series:
            check = x.loc["TherapistID"] not in list(therap)
        else:
            check = x.loc["TherapistID"] != therap
        # Some appointments have multiple therapists
        return check

    
    mask = app.replace({"TherapistID": {"-1": "#NULL!"}}).apply(check_app_therap, axis=1)
    if any(mask):
        print("Therapists do not match across app_oq and appointments \U0001F61E")
        print("Offending appointment(s):")
        print(list(app.loc[mask, "AppID"]))
        return None

    print("- Looks good \U0001F642")

    #####################
    ### Appointments ####
    #####################
    

    def check_appointments(x):
        """
        This function looks at the appointments in the raw
        appointments data. Rather than checking several 
        times the different aspects of appointments, this 
        function computes all we want at once.
        """
        mask = app["SessionID"] == x.loc["SessionID"]
        appointid = app.loc[mask, "AppID"].values
        subset = appointments_raw.loc[appointid, :]
        attended = subset.loc[subset["AttendanceDescription"] == "Attended", :]

        start = attended["Date"].min()
        end = attended["Date"].max()
        first = subset["Date"].min()
        last = subset["Date"].max()
        diff_between = subset['Date'].diff().apply(lambda x: x.days).fillna(0)

        if start == first:
            # first (and last) is to capture unattended 
            # appointments before the first attended one; 
            # if the first appointment was attended, 
            # we put 'first' far into the future so
            # that when comparing it to last, we do not
            # mark a session for violating the session
            # definition; see also "session criteria"
            # further down.
            first = pd.to_datetime("01/01/2050", format="%d/%m/%Y")

        check_num = x.loc['NumOfAttended'] != attended.shape[0]
        check_start = x.loc["StartDate"] != start 
        check_end = x.loc["EndDate"] != end
        check_within_sess = any(diff_between > arg_dict["max_diff_between_apps"])

        return check_num, check_start, check_end, check_within_sess, first, last


    print("Checking things related to attended appointments")
    print("  namely, number of attended appointments,")
    print("  minimum number of attended appointments, and")
    print("  session criteria related to days between appointments")

    problems = 0

    appoint_check = master_df.apply(check_appointments, axis=1, result_type="expand")
    appoint_check.columns = ["CNumOfAttended", #C for check
                             "CStart", "CEnd",
                             "CWithinSession",
                             "First", "Last"] 

    mask = appoint_check["CNumOfAttended"]
    if any(mask):
        print("Number of attended appointments appears to be incorrect \U0001F61E")
        print("Offending session(s):")
        print(list(master_df.loc[mask, "SessionID"]))
        problems += 1
    

    # Check min appointment
    mask = master_df["NumOfAttended"] < arg_dict["min_appoint"]
    if any(mask):
        print("A session made it into master_df but does not have the required number of attended appointments \U0001F61E")
        print("Offending session(s):")
        print(list(master_df.loc[mask, "SessionID"]))
        problems += 1

    # Check start and end dates
    mask = appoint_check[["CStart", "CEnd"]].apply(lambda x: any(x), axis=1)
    if any(mask):
        print("Start or end date may not be correct \U0001F61E")
        print("Offending session(s):")
        print(list(master_df.loc[mask, "SessionID"]))
        problems += 1

    """
    Session criteria:
    
    This one is kinda complicated.
    The thing we want to make sure of is that
    (1) within a session,
       no two sequential appointments are
       more than the cutoff away
    (2) between sessions, 
      (a) the difference between two sequential
       sessions (with the same therapist if 
       therapist matters) are more than the 
       cutoff and 
      (b) the difference between nonattended 
       appointments for those session are more
       than the cutoff
    
    The following example may help:
    
       AppID  Attendence     Date      SessionID
         1        A       1 Aug 2024       1
         2        A       8 Aug 2024       1
         3        N       16 Aug 2024      1
         4        N       5 May 2025       2
         5        A       7 May 2025       2
    
    A is for attended, N stands in for anything else.
    We check within each session first (done in 
    function above), and check across sessions on 
    attended  appointments (2,5), and between non-
    attended ones (3,4). This example is correct for
    the cutoff 180 days.
   """

    if arg_dict["unique_therapist"]:
        master_df.sort_values(by=["PatientID", "TherapistID", "StartDate"], inplace=True)
        temp = master_df[["PatientID", "TherapistID"]]
        next_mask = (temp == temp.shift(-1).fillna(0)).apply(lambda x: all(x), axis=1)
    else:
        master_df.sort_values(by=["PatientID", "StartDate"], inplace=True)
        next_mask = master_df["PatientID"] == master_df["PatientID"].shift(-1).fillna(0)
    # next_mask to to subset master_df to the part where the next session
    # needs to be checked to see if the appointments meet the cutoff

    master_df_ = master_df.copy()
    master_df_["NextStart"] = pd.to_datetime("01/01/2050", format="%d/%m/%Y")
    master_df_.loc[next_mask, "NextStart"] = master_df_.shift(-1).loc[next_mask, "StartDate"]

    master_df_["CWithinSession"] = appoint_check["CWithinSession"]
    master_df_["First"] = appoint_check["First"]
    master_df_["Last"] = appoint_check["Last"]
    
    master_df_["NextFirst"] = pd.to_datetime("01/01/2050", format="%d/%m/%Y")
    master_df_.loc[next_mask, "NextFirst"] = master_df_.shift(-1).loc[next_mask, "First"]
    
    def meets_dist(x):
        nstart_end = (x["NextStart"] - x["EndDate"]).days
        nfirst_last = (x["NextFirst"] - x["Last"]).days
        check_1 = nstart_end <= arg_dict["max_diff_between_apps"]
        check_2 = nfirst_last <= arg_dict["max_diff_between_apps"]
        check_3 = x.loc["CWithinSession"]        
        return check_1 or check_2 or check_3

    mask = master_df_.apply(meets_dist, axis=1)
    if any(mask):
        print("Not all sessions meet the criteria for distances between appointments \U0001F61E")
        print("Offending session(s):")
        print(list(master_df.loc[mask, "SessionID"]))
        problems += 1

    if problems > 0:
        return
    
    print("- No problems \U0001F44C")
    
    #####################
    ### Patient Info ####
    #####################

    print("Checking patient information")
    for c in patient_check_col:
        patient_info_raw[c] = patient_info_raw[c].astype(float)


    def for_check_patient(subset_i, x):
        not_equal = x.loc[patient_check_col] != subset_i.loc[patient_check_col]
        return any(not_equal) # 1 if any column is not equal
 

    def check_patientinfo(x):
        current_ = x.loc[["PatientID", "notedate"]].values
        mask = patient_info_raw[["ClientID", "notedate"]].values == current_
        mask = np.apply_along_axis(all, 1, mask)
        subset = patient_info_raw.loc[mask, :]
        if subset.shape[0] == 0:
            return True # no patient info
        else:
            check = subset.apply(for_check_patient,
                                 axis=1,
                                 x=x)
            return check.min() == 1 
            # Some clients have multiple
            # surveys taken on the same 
            # day, so we look to see if 
            # all of them don't match

    mask = master_df.apply(check_patientinfo, axis=1)
    if any(mask):
        print("A session had patient information that did not match or exist the raw file \U0001F61E")
        print("Offending session(s):")
        print(list(master_df.loc[mask, "SessionID"]))
        return None
    
    # age
    mask = ~(master_df["age"].apply(lambda x: 16 <= x <= 100))
    if any(mask):
        print("The age of a client does not make sense \U0001F61E")
        print("Offending client(s):")
        print(list(master_df.loc[mask, "PatientID"]))
        return None

    # Date filter
    def check_notedate(x):
        diff = (x.loc["StartDate"] - x.loc["notedate"]).days
        check = (diff > arg_dict["cutoff_hi"]) | (diff < arg_dict["cutoff_low"])
        return None

    
    mask = master_df[["StartDate", "notedate"]].apply(check_notedate, axis=1)
    if any(mask):
        print("The date filter did not work \U0001F61E")
        print("Offending session(s):")
        print(list(master_df.loc[mask, "SessionID"]))
        return None

    print("- Nice \U0000270C")
    ###############
    ### Scores ####
    ###############

    print("Checking that the scores fall within the session period and")
    print("  that the number of scores matches the raw data")


    def oq_check_and_range(data):
        oq_dates = data.dropna(subset="CurrentScore")["Date"]
        app_dates = data.loc[data["AppType"] != "OQ", "Date"]

        before_cutoff = app_dates.min() - pd.Timedelta(days=arg_dict["days_before"])
        after_cutoff = app_dates.max() + pd.Timedelta(days=arg_dict["days_after"])
        check_1 = oq_dates.min() < before_cutoff
        check_2 = oq_dates.max().round(freq="h") > after_cutoff
        # we round because we added some mintues to deal
        # with OQs on the same day

        return pd.Series([(check_1 or check_2), 
                          before_cutoff, 
                          after_cutoff, 
                          len(oq_dates),
                          data["ClientID"].iloc[0]])

    
    oq_check = app_oq.groupby("SessionID").apply(oq_check_and_range)
    oq_check.columns = ["DateCheck", "Start", "End", "OQNum", "ClientID"]

    oq_mask = oq_check['DateCheck']
    if any(oq_mask):
        print("Some OQ score is not within the specificed range \U0001F61E")
        print("Offending session(s):")
        print(list(oq_check.loc[oq_mask, :].index))

    # Number of OQs


    def filter_be(dates, begin, end):
        return begin <= dates <= end


    def check_scorenumber(x):
        mask = scores_raw["ClientID"] == x.loc["ClientID"]
        sub_oq = scores_raw.loc[mask, :]
        mask_2 = sub_oq["AdministrationDate"].apply(filter_be,
                                                    begin=x["Start"],
                                                    end=x["End"])
        client_oq = sub_oq.loc[mask_2, "CurrentScore"].dropna()
        check = len(client_oq) != x.loc["OQNum"]
        if check:
            print(x.loc["ClientID"], len(client_oq), x.loc["OQNum"], x["Start"], x["End"])
        return check


    oq_mask = oq_check.apply(check_scorenumber, axis=1)
    if any(oq_mask):
        print("The number of OQ scores do not match \U0001F61E")
        print("Offending session(s):")
        print(list(oq_check.loc[oq_mask, :].index))
        return None

    print("- Well done \U0001F60E")

    # fin
    print("You have sucessfully run check_all! The cleaning worked \U0001F973")
    return None
