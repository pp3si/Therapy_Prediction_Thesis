import os
import pandas as pd
import numpy as np
from cleaning import sessioninfo, imputation, scores, filtr

#The "dataset" keyword option was added by the author.
def get_old_data(data_path_new, data_path_old, keep_path, hide: bool, dataset="both"):
    app_old = sessioninfo.load_old_app(data_path_old)
    patient_old = sessioninfo.load_old_patient_info(data_path_old, keep_path=keep_path)
    scores_old = scores.load_old_scores(data_path_old)

    app_old_hid = sessioninfo.hide_or_no(appoint_data=app_old,
                                     data_path_old=data_path_old,
                                     data_path_new=data_path_new,
                                     hide=hide,
                                     dataset=dataset)
    
    return app_old_hid, patient_old, scores_old


def get_new_data(data_path_new, data_path_old, keep_path, hide: bool):
    app_new = sessioninfo.load_new_app(data_path_new)
    patient_new = sessioninfo.load_new_patient_info(data_path_new, keep_path=keep_path)
    scores_new = scores.load_new_scores(data_path_new)

    app_new_hid = sessioninfo.hide_or_no(appoint_data=app_new,
                                     data_path_old=data_path_old,
                                     data_path_new=data_path_new,
                                     hide=hide)
    
    return app_new_hid, patient_new, scores_new


def get_combined_data(data_path_new, data_path_old, keep_path, hide: bool):
    app_old = sessioninfo.load_old_app(data_path_old)
    app_new = sessioninfo.load_new_app(data_path_new)
    patient_old = sessioninfo.load_old_patient_info(data_path_old, keep_path=keep_path)
    patient_new = sessioninfo.load_new_patient_info(data_path_new, keep_path=keep_path)
    scores_old = scores.load_old_scores(data_path_old)
    scores_new = scores.load_new_scores(data_path_new)

    appointments = sessioninfo.combine_app(old=app_old, new=app_new)
    patient_info = sessioninfo.combine_patient_info(old=patient_old, new=patient_new)
    scores_df = scores.combine_oq(old=scores_old, new=scores_new)

    app_hid = sessioninfo.hide_or_no(appoint_data=appointments,
                                     data_path_old=data_path_old,
                                     data_path_new=data_path_new,
                                     hide=hide)
    
    return app_hid, patient_info, scores_df


def run_all(app_hid, 
            patient_info,
            scores_df,
            session_arg,
            scores_arg,
            match_arg,
            filter_arg):

    master_df_1, app_session = sessioninfo.make_session(appoint_data=app_hid, **session_arg)

    app_oq = scores.oq_score(app_session=app_session, scores=scores_df, **scores_arg)
    
    master_df_2 = sessioninfo.match_patient_info(master_df=master_df_1, 
                                                 patient_data=patient_info,
                                                 **match_arg)

    master_df_3 = imputation.imputation(master_df_2, **imput_arg)

    master_df_4 = filtr.filter_date(master_df_3, **filter_arg)

    return master_df_4, app_oq
