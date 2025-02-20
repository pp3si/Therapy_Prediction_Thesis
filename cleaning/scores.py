import pandas as pd
import numpy as np
import CAPS
from CAPS import ProgressBar
import os.path

keep = ["ClientID", "AdministrationDate",
        "CurrentScore", "Interpersonal Relations", 
        "Social Role", "Symptom Distress"
        ] + [f"Response_{i+1}" for i in range(45)]

response_scale = {"Never":         0,
                  "Rarely":        1,
                  "Sometimes":     2,
                  "Frequently":    3,
                  "Almost Always": 4}


def clean_ClientID(scores):
    # Drop missing observations
    scores.replace('#VALUE!', np.nan, inplace=True)
    scores.dropna(subset=["ClientID"], inplace=True)
    scores["ClientID"] = scores['ClientID'].astype(np.int64)
    return scores


def load_old_scores(data_path):
    scores = pd.read_csv(os.path.join(*data_path, "OQ.csv"), low_memory=False)
    scores = clean_ClientID(scores.loc[:, keep])
    return scores


def load_new_scores(data_path):
    scores = pd.read_csv(os.path.join(*data_path, "OQ-45 aug 2018 to april 2023.csv"),
                low_memory=False)
    scores.rename(columns={'client ID': 'ClientID', 'TotalScore': 'CurrentScore'}, inplace=True)
    scores = clean_ClientID(scores.loc[:, keep])
    return scores


def combine_oq(old, new):
    scores = pd.concat([old, new])
    scores.drop_duplicates(inplace=True)
    return scores


def oq_score(app_session, scores, days_before=30, days_after=7):
    """
    Attaches OQ scores to the appointment data frame.

    app_session: app_session, i.e. the data frame returned by
        make_sessin in sessioninfo.py
    scores: OQ scores data frame. Before passing, exclude columns
        not wanted in the final app_session data frame, such as
        "EmpiricalBlue".
    days_before: The number of days before the first appointment 
        (attended or not) to look for an OQ score belonging
        to a session.
    days_after: The number of days after the last appointment 
        (attended or not) to look for an OQ score belonging
        to a session.
    """
    scores_ = scores.copy()
    scores_.index = range(scores_.shape[0])

    # Drop observations outside the range of the OQ (0-180)
    scores_ = scores_.loc[scores_["CurrentScore"].apply(lambda x: (x >=0) & (x <= 180)), :]

    # Convert OQ times to datetime objects
    scores_["Date"] = pd.to_datetime(scores_["AdministrationDate"])
    scores_.drop("AdministrationDate", axis=1, inplace=True)
    scores_.sort_values(["ClientID", "Date"], inplace=True)

    # Convert Response_i
    replace = {f"Response_{i}": response_scale for i in range(1,46)}
    scores_.replace(replace, inplace=True)

    # Convert types
    scores_['ClientID'] = scores_['ClientID'].astype(np.int64)
    app_session['ClientID'] = app_session['ClientID'].astype(np.int64)
    app_session['Date'] = pd.to_datetime(app_session['Date'])


    # Session data frame: SessionID, Start, End, ClientID
    # Helpful for iterating through clients and sessions
    sess_1 = app_session.groupby("SessionID")["Date"].apply(lambda x: x.iloc[0])
    sess_2 = app_session.groupby("SessionID")["Date"].apply(lambda x: x.iloc[-1])
    sess_3 = app_session.groupby("SessionID")["ClientID"].apply(lambda x: x.iloc[0])
    sess = pd.concat([sess_1, sess_2, sess_3], axis=1)
    sess.columns = ["Start", "End", "ClientID"]
    sess = sess.reset_index()

    # Alter start and end dates
    sess["Start"] -= pd.Timedelta(days=days_before) # to catch OQs before the first app
    sess["End"] += pd.Timedelta(days=days_after)  # to catch OQs after the last app

    # Note: this differs from master_frame by allowing for
    # nonattended appointments to count
 
    # Add SessionID Column
    scores_["SessionID"] = np.nan

    # Mark which session overlap within the same client
    # This will allow for different treatment of those with
    # overlap and those without for some speed benefits
    sess.sort_values(by=["ClientID", "Start"], inplace=True)
    over = (sess["Start"] - sess["End"].shift(1)).apply(lambda x: x.days)
    same_client = sess["ClientID"] == sess["ClientID"].shift(1)
    sess["Overlap"] = (over <= days_before + days_after) & same_client


    def start_end_sess(temp, j):
        return (temp["Start"].iloc[j],
                temp["End"].iloc[j],
                temp["SessionID"].iloc[j])


    def add_function(Client):
        """
        For a client, gather the relevant OQs and return them with session IDs.
        The if-else split allows for some time benefits, since in one,
        we do not have to append data in a loop. In the case of overlaping
        sessions, we want to append data rather than simply modifying, so that 
        the overlaping sessions are not stealing OQs from each other.
        """
        mask_1 = scores_.loc[:, "ClientID"] == Client
        temp = sess.loc[sess["ClientID"] == Client, :]

        if sum(mask_1) == 0: # empty
            return scores_.loc[mask_1, :]
        
        if temp["Overlap"].sum() == 0:
            # Loop through the scores data adding session IDs
            j = 0 # index of the row of the subset of session
            start, end, session_number = start_end_sess(temp, j)
            stop_for = False

            for i in scores_.loc[mask_1, :].index:
                while scores_.loc[i, "Date"] > end:
                    # switch to the next session
                    j += 1
                    if temp.shape[0] == j:
                        stop_for = True
                        break # end for-loop, no more sessions
                    start, end, session_number = start_end_sess(temp, j)

                if stop_for:
                    break

                if start <= scores_.loc[i, "Date"] <= end:
                    scores_.loc[i, "SessionID"] = session_number # assign session ID
                else:
                    continue # Score does not fall within the scope of a session
                

            return scores_.loc[mask_1, :]

        else: # For sessions that overlap
            out = scores_.iloc[:0].copy()
            # for each session, check which OQs scores fall within
            # relevant dates, desigate the session ID then save 
            # those scores to a new data frame
            for j in range(temp.shape[0]):
                Session = temp.iloc[j]
                start = Session["Start"]
                end = Session["End"]
                mask_2 = scores_["Date"].apply(lambda x: start <= x <= end)
                scores_.loc[mask_1 & mask_2, "SessionID"] = Session["SessionID"]
                scores_temp = scores_.loc[mask_1 & mask_2, :]
                out = pd.concat([out, scores_temp])
            return out

    add = scores_.iloc[:0].copy()
    add_cols = add.columns
    add_rows_list = []

    clients = app_session["ClientID"].unique()
    print("organizing scores into sessions")

    for i, client in enumerate(clients):
        ProgressBar(i, len(clients))
        
        temp = add_function(client)
        for i in range(len(temp)):
            add_rows_list.append(list(temp.iloc[i]))
        # add = pd.concat([add, temp])
    
    add = pd.DataFrame(add_rows_list, columns=add_cols)

    # For OQs taken on the same day, add a minute so that 
    # the dates are not identical; this repeats because 
    # some clients took many OQs on the same day (4 is the max)
    while True:
        mask = add[["SessionID", 'Date']] == add[["SessionID", 'Date']].shift(1)
        mask = mask.apply(all, axis=1) #Has a line for each row showing if that row is identical to the previous one
        if sum(mask) == 0:
            break
        add.loc[mask, "Date"] += pd.Timedelta(minutes=1)

    # Add OQ scores to appointment data
    app_oq = app_session.merge(add.dropna(subset="SessionID"),
                               on=["ClientID", "SessionID", "Date"],
                               how="outer")

    # Fill in missing values for the OQ only rows
    app_oq.fillna({"AppType":                    "OQ", 
                   "AttendanceDescription": "Present",
                   "AppID":                        -1,
                   "TherapistID":                  -1}, 
                   inplace=True)

    # Convert dtypes
    app_oq["Date"] = pd.to_datetime(app_oq["Date"])

    to_str = ["AppID", "ClientID", "TherapistID", "SessionID"]
    for c in to_str:
        app_oq[c] = app_oq[c].astype(np.int64).astype(str)
    # Depending on how the data frame is saved,
    # converting to str will not be preserved
    # Putting 0 in place of np.nan guarantees 
    # that pd.read_csv will read them as integers

    app_oq["Zoom"] = app_oq["Zoom"].astype(float)
    # The remainder should be floats
    # though some are really int or bools with missing values
    # as is the case with "Zoom"

    # Put attended after other types; this is important for
    # removing duplicate OQs after this
    app_oq.sort_values("AttendanceDescription", ascending=False, inplace=True)

    # Converting type so that "SessionID" is sorted in a
    # more intelligible way
    app_oq["SessionID"] = app_oq["SessionID"].astype(int)
    app_oq.sort_values(by=["SessionID", "Date"], inplace=True)
    app_oq["SessionID"] = app_oq["SessionID"].astype(str)

    # Sometimes a client had two (or more) appointments on the same
    # date(e.g. an rescheduled to another time); all those appointments
    # could end up getting the same OQ, but we don't want this; 
    # only one should have it, to accurately represent the number of
    # OQs the client has; If there is an "attended" appointment it
    # retains the OQ
    mask = app_oq[["SessionID", "Date"] + keep[2:6]] == app_oq[["SessionID", "Date"] + keep[2:6]].shift(-1)
    # keep[2:6]  OQ score and the subscales; since the individual 
    # reponses can be NA, we look at these to determine if the OQ
    # on a date is identical
    mask = mask.apply(all, axis=1)
    app_oq.loc[mask, keep[2:]] = np.nan

    return app_oq

