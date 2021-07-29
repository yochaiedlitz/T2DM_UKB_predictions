#!/usr/bin/env python
# coding: utf-8

# In[35]:


# Python code to create the above Kaplan Meier curve
import random
import os
from lifelines import KaplanMeierFitter
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from lifelines import CoxPHFitter
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve, auc, \
    brier_score_loss, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, make_scorer
from addloglevels import sethandlers
from CI_Configs import runs
from UKBB_Functions import  Filter_CZ
from sklearn.utils import resample
Qworker = '/home/edlitzy/pnp3/lib/queue_tal/qworker.py'

run_name="SA_GDRS"
USE_FAKE_QUE=False
CALC_CI_ONLY=False

if USE_FAKE_QUE:
    from queue_tal.qp import fakeqp as qp
else:
    from queue_tal.qp import qp

def calc_TTE(row):
    """
    Returns either the time between the first visit to the first appearance of diabetes, or
    if diabetes was not diagnosed, or the time of diagnosis is not given - return the time of last visit"""
    if pd.isnull(row["TTE"]):
        return row["21003-4.0"]
    else:
        return row["TTE"]


def plot_ROC_curve(y_test_val, y_pred_val, AUC):
    fpr, tpr, _ = roc_curve(y_test_val, y_pred_val)
    fig = plt.figure(figsize=(16, 9))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating AUC={0:0.2f}'.format(AUC))
    plt.legend(loc="lower right")
    plt.show()

#     pdf.savefig(fig, dpi=DPI)
#     plt.close(fig)

def plot_precision_recall(y_test_val, y_pred_val, APS):
    precision, recall, _ = precision_recall_curve(y_test_val, y_pred_val)
    fig = plt.figure(figsize=(16, 9))
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(APS))
    plt.show()

    # Plotting ratio graph for precision recall
    rel_prec = precision / precision[0]

    #     fig = plt.figure()
    #     plt.step(recall, rel_prec, color='b', alpha=0.2, where='post')
    #     plt.fill_between(recall, rel_prec, step='post', alpha=0.2, color='b')
    #     plt.xlabel('Recall')
    #     plt.ylabel('Relative Precision')
    #     # plt.ylim([0.0, 1.05 * np.percentile(rel_prec,99.97)])
    #     plt.ylim([0.0, 1.05 * max(rel_prec)])
    #     plt.xlim([0.0, 1.0])
    #     plt.title('2-class Relative-Precision-Recall curve: AP={0:0.2f}'.format(APS))
    #     plt.show()
    #    # Plotting ratio graph for precision recallwith removed maximum value

    fig = plt.figure(figsize=(16, 9))
    plt.step(recall, rel_prec, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, rel_prec, step='post', alpha=0.2,
                     color='b')
    plt.xlabel('Recall')
    plt.ylabel('Relative Precision')
    plt.ylim([0.0, 1.05 * max(np.delete(rel_prec, np.argmax(rel_prec)))])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Relative-Precision-Recall trimmed max: AP={0:0.2f}'.format(APS))
    plt.show()
    # Show graph of True positive Vs.quantiles of predicted probabilities.


def get_rel_score(row):
    """
    A function that is used in apply on Dataframes
    Returns the predicted Survival rate at the visit time
    """
    return row[row.loc["21003-4.0"]]


def get_event_n_duration(path):
    """
    Calculates the time passed from visit to event, or' if no event occurs - to the last known visit
    return durations,event_observed,Diab_age_df.loc[:,['TTE',"2443-3.0"]],Diab_age_df
    """
    data_col = pd.read_csv(path, nrows=0).columns.values
    diab_age_col = [x for x in data_col if x.startswith("2976-")]  # Aged when diabetes first diagnosed
    diab_col = [x for x in data_col if x.startswith("2443-")]  # 1 if duiabetes diagnosed
    Init_age_col = "21003-0.0"
    all_ages_cols = [col for col in data_col if col.startswith("21003-")]
    all_ages_df = pd.read_csv(path, usecols=["eid"] + all_ages_cols, index_col="eid")

    Diab_age_df = pd.read_csv(path, usecols=diab_age_col + ["eid"], index_col="eid")
    Diab_age_df["Min_diab_age"] = Diab_age_df.min(axis=1)
    Diab_age_df = Diab_age_df.join(all_ages_df[Init_age_col])
    Diab_age_df["TTE"] = Diab_age_df["Min_diab_age"] - Diab_age_df[
        "21003-0.0"]  # Calculating time from first visit to diab onset
    neg_diab_age_ind = Diab_age_df.loc[
        Diab_age_df["TTE"] < 0, "TTE"].index  # Getting indexes of events with negative values, to filter them out
    diab_ind = [ind for ind in Diab_age_df.index if ind not in neg_diab_age_ind]
    Diab_age_df = Diab_age_df.loc[diab_ind, :]

    diab = pd.read_csv(path, usecols=["eid", "2443-3.0"], index_col="eid")
    Diab_age_df = Diab_age_df.join(diab)
    Diab_age_df = Diab_age_df.join(all_ages_df["21003-4.0"])  # Time between first and last visit

    Diab_age_df['TTE'] = Diab_age_df.apply(calc_TTE, axis=1)
    durations = Diab_age_df['TTE'].values
    event_observed = Diab_age_df['2443-3.0'].values

    return durations, event_observed, Diab_age_df.loc[:, ['TTE', "2443-3.0", "21003-4.0", "21003-3.0"]], Diab_age_df


def fit_n_plot_cox_train_result(Train_dummy, penalizer=0.2, var_thresh=0.05,plot=False):
    cph = CoxPHFitter(penalizer=penalizer)  ## Instantiate the class to create a cph object
    #     drop_col_diab=Train_dummy.loc[Train_dummy["2443-3.0"]==1,:].var()/Train_dummy.loc[Train_dummy["2443-3.0"]==1,:].median()<var_thresh
    #     drop_col_non_diab=Train_dummy.loc[Train_dummy["2443-3.0"]==1,:].var()/Train_dummy.loc[Train_dummy["2443-3.0"]==1,:].median()<var_thresh
    drop_col_diab = Train_dummy.loc[Train_dummy["2443-3.0"] == 1, :].var() < var_thresh
    drop_col_non_diab = Train_dummy.loc[Train_dummy["2443-3.0"] == 0, :].var() < var_thresh
    # drop_col_non_diab = Train_dummy.loc[Train_dummy["2443-3.0"] == 1, :].var() < var_thresh
    drop_col = drop_col_diab + drop_col_non_diab
    drop_columns = drop_col[drop_col == True].index.values
    #     print("drop_col: ",drop_columns)
    use_cols = drop_col[drop_col == False].index.values
    use_cols_list = list(use_cols)
    if "2443-3.0" not in use_cols_list:
        use_cols_list.append("2443-3.0")
    if "TTE" not in use_cols_list:
        use_cols_list.append("TTE")
    cph.fit(Train_dummy[use_cols_list], duration_col="TTE", event_col='2443-3.0',
            show_progress=True)  ## Fit the data to train the model
    if plot:
        cph.print_summary()  ## HAve a look at the significance of the features
        tr_rows = Train_dummy.iloc[0:10, :-2]
        cph.predict_survival_function(tr_rows).plot(figsize=[16, 9])
        shp = max(5, int(0.2 * len(use_cols_list)))
        f, axs = plt.subplots(1, 1, figsize=(min(15, shp), shp))
        cph.plot(ax=axs)
    if "2443-3.0" in use_cols_list:
        use_cols_list.remove("2443-3.0")
    if "TTE" in use_cols_list:
        use_cols_list.remove("TTE")
    return cph, use_cols_list


def train_cox(run,Train_file_path, penalizer=0.2, var_thresh=0.05,CI=[]):
    features_file_path=run.features_file_path

    feat_file = pd.read_csv(features_file_path)
    feat_col = feat_file.loc[feat_file["Exclude"] == 0, "Field ID"]
    db_columns = pd.read_csv(Train_file_path, nrows=0).columns.values
    feat_cols_list = list(feat_col.values)
    use_cols = [col for col in db_columns for feat_col in feat_cols_list if
                (col.startswith(feat_col) and "_na" not in col)]
    use_cols = ["eid"] + use_cols
    use_cols_plus_HbA1c=use_cols + ["30750-0.0"] # Adding %HbA1c for filtration
    Train_data = pd.read_csv(Train_file_path, usecols=use_cols_plus_HbA1c,
                             index_col="eid")  # 21003-4.0 is time between visits, 21003-3.0 is age at last visit
    _, _, dur_eve_df, _ = get_event_n_duration(Train_file_path)  #
    dur_eve_df = dur_eve_df.drop(["21003-4.0", "21003-3.0"], axis=1)
    Train_data = Train_data.join(dur_eve_df)
    if "21003-4.0" in Train_data.columns:
        Train_data = Train_data.drop(["21003-4.0"], axis=1)  # To avoid colinearity with TTE
    if "21003-3.0" in Train_data.columns:
        Train_data = Train_data.drop(["21003-3.0"], axis=1)  # To avoid colinearity with TTE

    Train_data = Filter_CZ(Train_data, run.charac_selected, run.charac_id)

    Train_data=Train_data.drop(["30750-0.0"],axis=1) #Remove HbA1c

    train_length=Train_data.shape[0]
    if CI=="CI":
        Train_data_boot = Train_data.sample(n=train_length, replace=True)
    else:
        Train_data_boot = Train_data

    Train_dummy = pd.get_dummies(Train_data_boot, drop_first=True)
    cph, use_cols = fit_n_plot_cox_train_result(Train_dummy, penalizer=penalizer, var_thresh=var_thresh,plot=False)
    return cph, use_cols, Train_dummy


def predict_test_results(run,Test_file_path, Results_path, cph, col_names,SN=1,penalizer=[],
                         var_thresh=[],CI=[]):
    """

    :param Test_file_path:
    :param Results_path:
    :param cph:
    :param col_names:
    :param SN:
    :param penalizer:
    :param var_thresh:
    :return: Tot_test_pred, Y_tot, Test_dummy, y_test_val, y_pred_val, AUC, APS
    """
    col_names = list(set(col_names))

    if "eid" not in col_names:
        col_names.append("eid")
    use_cols_plus_HbA1c=col_names+["30750-0.0"] #Adding %HbA1c for filtration

    Test_data = pd.read_csv(Test_file_path, usecols=use_cols_plus_HbA1c,
                            index_col="eid")  # 21003-4.0 is time between visits, 21003-3.0 is age at last visit
    Test_data=Filter_CZ(Test_data, run.charac_selected, run.charac_id)

    Test_data=Test_data.loc[:,col_names]
    _, _, dur_eve_df, _ = get_event_n_duration(Test_file_path)  #
    Test_data_all = Test_data.join(dur_eve_df)  # To avoid colinearity with TTE
    Test_length=Test_data_all.shape[0]
    if CI=="CI":
        Test_data_boot= Test_data_all.sample(n=Test_length,replace=True)
    else:
        Test_data_boot = Test_data_all
    Test_data_clean = Test_data_boot.drop(["21003-4.0", "21003-3.0", "2443-3.0", "TTE"], axis=1)
    Test_dummy = pd.get_dummies(Test_data_clean, drop_first=True)
    #     Test_dummy_rel=Test_dummy.iloc[:,:-2]
    test_predicted = cph.predict_survival_function(Test_dummy)
    dummy_idx = np.arange(0, Test_dummy.shape[0])
    Test_dummy.index=dummy_idx
    Test_data_boot.index=dummy_idx
    test_predicted.columns=dummy_idx
    Tot_test_pred = test_predicted.T.join(Test_data_boot.loc[:, "21003-4.0"])
    Tot_test_pred["21003-4.0"] = Tot_test_pred["21003-4.0"].astype(str)
    col = [str(x) for x in Tot_test_pred.columns.values]
    new_col_dict = dict(zip(Tot_test_pred.columns.values, col))
    Tot_test_pred.rename(columns=new_col_dict, inplace=True)
    Tot_test_pred["pred"] = Tot_test_pred.apply(get_rel_score, axis=1)
    Tot_test_pred.index=np.arange(0,Tot_test_pred.shape[0])
    Test_data_boot.index=np.arange(0,Test_data_boot.shape[0])
    Y_tot = Tot_test_pred.join(Test_data_boot.loc[:,"2443-3.0"]).loc[:,["pred","2443-3.0"]].dropna(axis=1)
    # print("*************~~~~~ Ytot ~~~~~~~~************")
    # print("KeyError: u'the label [2443-3.0] is not in the [columns]'")
    # print (Y_tot)
    # print("*************~~~~~++++++~~~~~~~~************")
    y_test_val = Y_tot.loc[:,"2443-3.0"].values
    y_pred_val = 1 - Y_tot.loc[:,"pred"].values

    AUC = roc_auc_score(y_test_val, y_pred_val)
    # plot_ROC_curve(y_test_val, y_pred_val, AUC)

    APS = average_precision_score(y_test_val, np.array(y_pred_val))
    # plot_precision_recall(y_test_val, y_pred_val, APS)
    results_df = pd.DataFrame.from_dict({"APS": [APS], "AUC": [AUC], "SN": [SN],"penalizer":[penalizer],"var_thresh":[var_thresh]})
    results_df = results_df.set_index("SN", drop=True)
    prediction_DF = pd.DataFrame.from_dict({"y_test_val": y_test_val, "y_pred_val": y_pred_val})
    results_df.to_csv(os.path.join(Results_path, "AUC_APS_results_" + str(int(SN)) + ".csv"),index=True)
    prediction_DF.to_csv(os.path.join(Results_path, "y_pred_results_" + str(int(SN)) + ".csv"))
    # return Tot_test_pred, Y_tot, Test_dummy, y_test_val, y_pred_val, AUC, APS

def Train_Test_Cox(run,Train_file_path,Test_file_path,Results_path,batch,CI=[],penalizer=None,var_thresh=None):
    for ind in np.arange(run.batch_size):
        SN = (batch * run.batch_size + ind)
        if penalizer==None:
            penalizer = random.randrange(0, 100, 1) / 10.
            var_thresh = random.randrange(0, 100, 1) / 100.
        cph, use_cols, Train_dummy = train_cox(run,Train_file_path=Train_file_path,
                                           penalizer=penalizer,
                                           var_thresh=var_thresh,CI=CI)
    #     print (use_cols)
    #     Tot_test_pred, Y_tot, Test_dummy, y_test_val, y_pred_val,AUC, APS =\
    #         predict_test_results(run,val_file_path=val_file_path,Results_path=Results_path,
    #                              cph=cph,col_names=use_cols,SN=SN,penalizer=penalizer,
    #                                            var_thresh=var_thresh,CI=CI)
        predict_test_results(run, Test_file_path=Test_file_path, Results_path=Results_path,
                                     cph=cph,col_names=use_cols,SN=SN,penalizer=penalizer,
                                                   var_thresh=var_thresh,CI=CI)

    # return cph, Train_dummy, Tot_test_pred, Y_tot, Test_dummy, y_test_val, y_pred_val, AUC, APS

def optimal_params(path):
    runs_files_results_names=os.listdir(path)
    runs_files_results_path=[os.path.join(path,x) for x in runs_files_results_names if x.startswith("AUC_APS")]
    runs_results_list=[pd.read_csv(x,index_col="SN") for x in runs_files_results_path]
    runs_Results_df=pd.concat(runs_results_list)
    runs_Results_df.sort_values(by="AUC",inplace=True, ascending=False)
    runs_Results_df.to_csv(os.path.join(path,"runs_results_summary.cav"))
    params=runs_Results_df.loc[0,:]
    return params

def upload_SA_CI_jobs(q):
    waiton = []
    run=runs(run_name)
    #TODO check if model exists
    try:
        params, _ = optimal_params(run)
    except:
        for batch in np.arange(np.ceil(np.float(run.hyper_parameter_iter) / run.batch_size)):
            print ("Sending Batch number: ", batch, "of run: ", run.run_name, "For hyper parameter")
            waiton.append(q.method(Train_Test_Cox, [run, run.train_file_path,
                                                    run.val_file_path, run.Training_path,
                                                    batch]))
        q.wait(waiton, assertnoerrors=False)
        params = optimal_params(run.Training_path)

    print ("Trainin optimal parameters are", params)
    penalizer = params["penalizer"]
    var_thresh = params["var_thresh"]

    for batch in np.arange(np.ceil(np.float(run.num_of_bootstraps)/run.batch_size)):
        print ("Sending Batch number: " ,batch,"of run: ",run.run_name,"For CI calc")
        waiton.append(q.method(Train_Test_Cox, [run, run.train_val_file_path,
                                                run.test_file_path, run.CI_results_path, batch, "CI", penalizer,
                                                var_thresh]))
    q.wait(waiton,assertnoerrors=False)
    run.calc_CI()

def main():
    # CI_Results_list=[]
    # run=runs(run_name)
    sethandlers()
    with qp(jobname="CI_Addings", max_u=800, mem_def="3G",q=['himem7.q'], tryrerun=True,trds_def=1,
            delay_batch=20,qworker=Qworker) as q:
        os.chdir('/net/mraid08/export/jafar/Microbiome/Analyses/Edlitzy/tempq_files/')
        q.startpermanentrun()
        upload_SA_CI_jobs(q)

    # results_path=run.CI_results_summary_table
    # CI_Results_list.append(pd.read_csv(results_path))
    # CI_results_df=pd.concat(CI_Results_list)
    # CI_results_df.to_csv(os.path.join(run.Folder_path,run_name+"_CI.csv"))
    # print("Results df saved to:",os.path.join(run.Folder_path,run_name+"_CI.csv"))
    # print(CI_results_df)

if __name__=="__main__":
    main()
