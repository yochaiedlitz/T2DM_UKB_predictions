from Configs.CI_Configs import runs
import random
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, make_scorer, roc_curve, auc, average_precision_score, precision_recall_curve
import sys
from addloglevels import sethandlers
from sklearn.externals import joblib
import pickle
from UKBB_Functions import Filter_CZ  # SAVE_FOLDER, RUN_NAME,
# from Scoreboards.UKBB_Scoreboard_functions import *
from sklearn.metrics import brier_score_loss
import shutil


CHARAC_ID = {"Age at last visit": "21003-3.0", "Sex": "31-0.0", "Ethnic background": "21000-0.0",
             "Type of special diet followed": "20086-0.0"}
ETHNIC_CODE = {-3: "Prefer not to answer", -1: "Do not know", 1: "White", 2: "Mixed", 3: "Asian",
               4: "Black or Black British", 5: "Chinese", 6: "Other ethnic group", 1001: "British", 1002: "Irish",
               1003: "Any other white background", 2001: "White and Black Caribbean",
               2002: "White and Black African", 2003: "White and Asian", 2004: "Any other mixed background",
               3001: "Indian", 3002: "Pakistani", 3003: "Bangladeshi", 3004: "Any other Asian background",
               4001: "Caribbean", 4002: "African", 4003: "Any other Black background"}
SEX_CODE = {"Female": 0, "Male": 1}
DIET_CODE = {"Gluten-free": 8, "Lactose-free": 9, "Low calorie": 10, "Vegetarian": 11, "Vegan": 12, "Other": 13}


def roc_auc_score_proba(y_true, proba):
    return roc_auc_score(y_true, proba)


def standarise_df(df):
    fit_col = df.columns
    x_std_col = [x for x in fit_col if not x.endswith("_na")]
    x_na_col = [x for x in fit_col if x.endswith("_na")]
    x_train_std = df[x_std_col]
    x_train_std = (x_train_std - np.mean(x_train_std, axis=0)) / np.std(x_train_std, axis=0)
    x_train_std_na_col = x_train_std.loc[:, x_train_std.isna().sum() > 0].columns.values
    x_train_std.loc[:, x_train_std.isna().sum() > 0] = df.loc[:, x_train_std_na_col]
    x_train_std[x_na_col] = df[x_na_col]
    return x_train_std


# def Filter_CZ(Features_DF, charac_selected,charac_id):  # CHARAC_SELECTED = {"Age at recruitment": "All", "Sex": "Female", "Ethnic background":
#     # "All","Type of special diet followed": "All"}
#     # CHARAC_ID = {"Age at last visit": 21003, "Sex": 31, "Ethnic background": 21000,
#     # "Type of special diet followed": 20086}
#
#     if charac_selected["Age at last visit"] != "All":
#         Age_max = charac_selected["Age at recruitment"] + 5
#         Age_min = charac_selected["Age at recruitment"] - 5
#         Features_DF = Features_DF.loc[
#                       Features_DF[charac_id["Age at recruitment"]].between(Age_min, Age_max, inclusive=True), :]
#     if "Age_max" in charac_selected:
#         if charac_selected["Age_max"] != "None":
#             Age_max = charac_selected["Age_max"]
#     if "Age_min" in charac_selected:
#         if charac_selected["Age_min"] != "None":
#             Age_min = charac_selected["Age_min"]
#
#     if charac_selected["Sex"] != "All":
#         Features_DF = Features_DF[Features_DF[charac_id["Sex"]] == SEX_CODE[charac_selected["Sex"]]]
#
#     if charac_selected["Ethnic background"] != "All":
#         if charac_selected["Ethnic background"] == 1:
#             Features_DF = Features_DF[(Features_DF[charac_id["Ethnic background"]] == 1) |
#                                       ((Features_DF[charac_id["Ethnic background"]] > 1000) &
#                                        (Features_DF[charac_id["Ethnic background"]] < 2000))]
#         elif charac_selected["Ethnic background"] == 2:
#             Features_DF = Features_DF[(Features_DF[charac_id["Ethnic background"]] == 2) |
#                                       ((Features_DF[charac_id["Ethnic background"]] > 2000) &
#                                        (Features_DF[charac_id["Ethnic background"]] < 3000))]
#         elif charac_selected["Ethnic background"] == 3:
#             Features_DF = Features_DF[(Features_DF[charac_id["Ethnic background"]] == 3) |
#                                       ((Features_DF[charac_id["Ethnic background"]] > 3000) &
#                                        (Features_DF[charac_id["Ethnic background"]] < 4000))]
#         elif charac_selected["Ethnic background"] == 4:
#             Features_DF = Features_DF[(Features_DF[charac_id["Ethnic background"]] == 4) |
#                                       ((Features_DF[charac_id["Ethnic background"]] > 4000) &
#                                        (Features_DF[charac_id["Ethnic background"]] < 5000))]
#         else:
#             Features_DF = Features_DF[(Features_DF[charac_id["Ethnic background"]]
#                                        == charac_selected["Ethnic background"])]
#
#     if charac_selected["Type of special diet followed"] != "All":
#         Features_DF = Features_DF[Features_DF[charac_id["Type of special diet followed"]] ==
#                                   DIET_CODE[charac_selected["Type of special diet followed"]]]
#     # "Maximal_a1c": 38, "Minimal_a1c": 20
#     if "Minimal_a1c" in charac_selected:
#         if charac_selected["Minimal_a1c"] != "All":
#             Features_DF = Features_DF.loc[Features_DF.loc[:, "30750-0.0"] > charac_selected["Minimal_a1c"], :]
#     if "Maximal_a1c" in charac_selected:
#         if charac_selected["Maximal_a1c"] != "All":
#             Features_DF = Features_DF.loc[Features_DF.loc[:, "30750-0.0"] <= charac_selected["Maximal_a1c"], :]
#
#     return Features_DF


def load_lr_data(run_object, Data_file_path, mode="Train", standardise=True, force_update=True,save_data_folder=None,
                 postfix=""):
    feat_file = pd.read_csv(run_object.features_file_path)
    feat_col = feat_file.loc[feat_file["Exclude"] == 0, "Field ID"]
    db_columns = pd.read_csv(Data_file_path, nrows=0).columns.values
    feat_cols_list = list(feat_col.values)
    use_cols = [col for col in db_columns for feat_col in feat_cols_list if
                ((col.startswith(feat_col) and "_na" not in col) or feat_col == col)]
    use_cols = ["eid"] + use_cols
    #30750	Glycated haemoglobin (HbA1c)	Blood biochemistry
    if "30750-0.0" not in use_cols:
        X = pd.read_csv(Data_file_path, usecols=use_cols + ["30750-0.0"],
                        index_col="eid")  # 21003-4.0 is time between visits, 21003-3.0 is age at last visit
        filtered_cols = [x for x in use_cols if not (x.startswith("30750-") or x == "eid")]
    else:
        X = pd.read_csv(Data_file_path, index_col="eid", usecols=use_cols)
        filtered_cols = X.columns.values
    try:
        X = Filter_CZ(X, run_object.charac_selected, run_object.charac_id)
    except AttributeError:
        print("did not filter X")

    X = X.loc[:, filtered_cols]
    if standardise:
        X = standarise_df(X)
    if save_data_folder is None:
        model_paths=run_object.model_paths
    else:
        model_paths = save_data_folder

    X_path = os.path.join(model_paths, "X_LR_"+postfix+".csv")
    y_path = os.path.join(model_paths, "y_LR_"+postfix+".csv")

    train_length = X.shape[0]

    y = pd.read_csv(Data_file_path, usecols=["eid", "2443-3.0"], index_col="eid")
    y = y.loc[X.index, "2443-3.0"]
    Data = X.join(y)
    if mode == "Val":
        Data = Data.sample(n=train_length, replace=True)

    y = Data.loc[:, "2443-3.0"]
    X = Data.drop(axis=1, columns=["2443-3.0"])

    if not os.path.isfile(X_path) or force_update:
        X.to_csv(X_path)
        y.to_csv(y_path)
        with open(os.path.join(model_paths, "use_cols.txt"), "wb") as fp:  # Pickling
            pickle.dump(use_cols, fp)
    return X, y


def compute_lr(run_object, mode, Batch_number, penalizer=[], Choose_N_Fold=3, force_update=True):
    "mode should be either Train or Val"
    AUC_dict = {}
    APS_dict = {}
    score = run_object.score
    for SN in np.arange(Batch_number * run_object.batch_size, (Batch_number + 1) * run_object.batch_size):
        if mode == "Train":
            penalizer = random.randrange(0, 100, 1) / 50.
            train_file_path = run_object.train_file_path
            test_file_path = run_object.val_file_path
            path = run_object.Training_path
        elif mode == "Val" or "Test":
            train_file_path = run_object.train_val_file_path
            test_file_path = run_object.test_file_path
            path = run_object.CI_results_path
        X_train, y_train = load_lr_data(run_object, train_file_path, mode=mode, standardise=run_object.standardise,
                                        force_update=force_update)
        X_test, y_test = load_lr_data(run_object, test_file_path, mode=mode, standardise=run_object.standardise,
                                      force_update=force_update)

        clf = LogisticRegressionCV(cv=Choose_N_Fold, random_state=0, penalty="l2",
                                   scoring=score, class_weight=run_object.class_weight, Cs=[penalizer])
        try:
            clf.fit(X_train, y_train.values.flatten())
            print ("fitting SN:", SN, " was succesful")
            if run_object.save_model:
                pickle.dump(clf, open(os.path.join(path, "LR_model_" + str(int(SN)) + ".sav"), 'wb'))
            y_proba = clf.predict_proba(X_test)
            AUC = roc_auc_score(y_test.values, y_proba[:, 1])
            APS = average_precision_score(y_test.values, y_proba[:, 1])
            AUC_dict[str(SN)] = AUC
            APS_dict[str(SN)] = APS
            # plot_precision_recall(y_test_val, y_pred_val, APS)
            results_df = pd.DataFrame.from_dict(
                {"APS": [APS], "AUC": [AUC], "SN": [SN], "penalizer": [penalizer]})
            results_df = results_df.set_index("SN", drop=True)
            prediction_DF = pd.DataFrame.from_dict({"y_test": y_test.values, "y_pred": y_proba[:, 1]})
            results_df.to_csv(os.path.join(path, "AUC_APS_results_" + str(int(SN)) + ".csv"), index=True)
            prediction_DF.to_csv(os.path.join(path, "y_pred_results_" + str(int(SN)) + ".csv"))
        except:
            print ("fitting SN:", SN, " FAILED")
            AUC_dict[str(SN)] = None
            APS_dict[str(SN)] = None
    return AUC_dict, APS_dict


def optimal_params(run_object, name_start="AUC_APS_results_", ind_col="SN"):
    path = run_object.Training_path
    df_list = [pd.read_csv(os.path.join(path, x), index_col=ind_col) for x in os.listdir(path) if
               x.startswith(name_start)]
    if not df_list:
        return None,None
    df = pd.concat(df_list)
    df.sort_values(by="AUC", inplace=True, ascending=False)
    df.to_csv(run_object.hyper_parameters_summary_table)
    params = df.iloc[0, :]
    return params, df


def LR_CI(run_name, force_update=True,run_object=None,debug = False,calc_new_training=True):  # mode should be explore or None
    sethandlers()  # set handlers for queue_tal jobs
    Qworker = '/home/edlitzy/pnp3/lib/queue_tal/qworker.py'
    if run_object is None:
        run_object = runs(run_name=run_name, force_update=force_update,debug=debug)
    if run_object.debug or debug:
        from queue_tal.qp import fakeqp as qp
        print("Running in debug mode!!!!!!!!!!")
    else:
        from queue_tal.qp import qp

    with qp(jobname="HP" + run_name, q=['himem7.q'], mem_def='4G', trds_def=2, tryrerun=False, max_u=650,
            delay_batch=20, qworker=Qworker) as q:
        os.chdir('/net/mraid08/export/jafar/Microbiome/Analyses/Edlitzy/tempq_files/')
        q.startpermanentrun()
        tkns = []
        # TODO check if model exists
        # Choosing hyper parameter
        num_of_hy_par_batches = int(run_object.hyper_parameter_iter / run_object.batch_size) + 1
        # TODO: Retain the code from remark.
        # Collecting and sorting for optimal hyperparameter

        if not calc_new_training:
            params, _ = optimal_params(run_object)
            if params is None:
                calc_new_training=True
        if calc_new_training:
            for BN in np.arange(num_of_hy_par_batches):
                print ("Building LR training batch number", BN, "for", run_name)
                tkns.append(q.method(compute_lr, [run_object, "Train", BN, False]))
                if BN == num_of_hy_par_batches - 1:
                    q.wait(tkns, assertnoerrors=False)
            params, _ = optimal_params(run_object)
            print ("Training optimal parameters", params)

        penalizer = params["penalizer"]
    with qp(jobname="CI" + run_name, q=['himem7.q'], mem_def='4G', trds_def=1, tryrerun=False, max_u=650,
            delay_batch=10, qworker=Qworker) as q:
        os.chdir('/net/mraid08/export/jafar/Microbiome/Analyses/Edlitzy/tempq_files/')
        q.startpermanentrun()
        if run_object.compute_CI:
            num_of_CI_batches = np.ceil((float(run_object.num_of_bootstraps) / run_object.batch_size))
            waiton = []
            for BN in np.arange(num_of_CI_batches):
                print ("Computing validation number", BN, ", for:", run_name)
                waiton.append(q.method(compute_lr, [run_object, "Val", BN, penalizer]))
            q.wait(waiton, assertnoerrors=False)
            run_object.calc_CI()

#LR_Five_blood_tests_scoreboard / LR_Anthro_scoreboard/LR_Anthro_scoreboard_explore
if __name__=="__main__":
    run_name = "LR_Finrisc"
    sethandlers()
    LR_CI(run_name=run_name)
