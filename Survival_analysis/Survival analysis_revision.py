import sys
import random
import os
from lifelines import KaplanMeierFitter,CoxPHFitter
from lifelines.utils import k_fold_cross_validation
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, \
    roc_curve, auc, brier_score_loss, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, make_scorer
from CI_Configs import runs
from UKBB_Functions import  Filter_CZ,to_pickle,from_pickle
from sklearn.utils import resample
from LabData import config_global as config
from LabUtils.addloglevels import sethandlers
from LabQueue.qp import fakeqp
import os
from imports import standarise_df


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
    diab_age_data_path="/net/mraid08/export/jafar/UKBioBank/Data/ukb29741.csv"
    diab_data_col=pd.read_csv(diab_age_data_path, nrows=0).columns.values
    data_col = pd.read_csv(path, nrows=0).columns.values
    diab_age_col = [x for x in diab_data_col if x.startswith("2976-")]  # Aged when diabetes first diagnosed
    # diab_col = [x for x in data_col if x.startswith("2443-")]  # 1 if diabetes diagnosed
    init_age_col = "21003-0.0"
    all_ages_cols = [col for col in data_col if col.startswith("21003-")]
    all_ages_df = pd.read_csv(path, usecols=["eid"] + all_ages_cols, index_col="eid")

    Diab_age_df = pd.read_csv(diab_age_data_path, usecols=diab_age_col + ["eid"], index_col="eid")
    Diab_age_df["Min_diab_age"] = Diab_age_df.min(axis=1)
    Diab_age_df = Diab_age_df.join(all_ages_df[init_age_col],how="right")
    Diab_age_df["TTE"] = Diab_age_df["Min_diab_age"] - Diab_age_df[
        "21003-0.0"]  # Calculating time from first visit to diab onset
    neg_diab_age_ind = Diab_age_df.loc[
        Diab_age_df["TTE"] < 0, "TTE"].index  # Getting indexes of events with negative values, to filter them out
    diab_ind = [ind for ind in Diab_age_df.index if ind not in neg_diab_age_ind]
    Diab_age_df = Diab_age_df.loc[diab_ind, :]

    diab = pd.read_csv(path, usecols=["eid", "2443-3.0"], index_col="eid")
    Diab_age_df = Diab_age_df.join(diab,how="inner")
    Diab_age_df = Diab_age_df.join(all_ages_df["21003-4.0"],how="inner")  # Time between first and last visit
    Diab_age_df['TTE'] = Diab_age_df.apply(calc_TTE, axis=1)
    durations = Diab_age_df['TTE'].values
    event_observed = Diab_age_df['2443-3.0'].values

    # return durations, event_observed, Diab_age_df.loc[:, ['TTE', "2443-3.0", "21003-4.0", "21003-3.0"]], Diab_age_df
    return durations, event_observed, Diab_age_df.loc[:, ['TTE', "2443-3.0"]], Diab_age_df

def fit_n_plot_cox_train_result(
        Train_dummy, penalizer=0.2,l1_ratio=0.5,var_thresh=0,plot=False,freeze_penalizer=False):
    #     drop_col_diab=Train_dummy.loc[Train_dummy["2443-3.0"]==1,:].var()/Train_dummy.loc[Train_dummy["2443-3.0"]==1,:].median()<var_thresh
    #     drop_col_non_diab=Train_dummy.loc[Train_dummy["2443-3.0"]==1,:].var()/Train_dummy.loc[Train_dummy["2443-3.0"]==1,:].median()<var_thresh
    retry=True
    retry_ind=0
    drop_col_diab = Train_dummy.loc[Train_dummy["2443-3.0"] == 1, :].var() <= var_thresh
    drop_col_non_diab = Train_dummy.loc[Train_dummy["2443-3.0"] == 0, :].var() <= var_thresh
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
    while retry:
        try:
            cph = CoxPHFitter(penalizer=penalizer)  ## Instantiate the class to create a cph object
            cph.fit(Train_dummy[use_cols_list], duration_col="TTE", event_col='2443-3.0',
                    show_progress=True)  ## Fit the data to train the model
            success=True
            retry=False
        except:
            if not freeze_penalizer:
                print(("Failed fitting,increased panelizer is: ","{:.2f}".format(penalizer)))
                retry=True
                retry_ind+=1
                if retry_ind==10:
                    print(("Failed last penalizer:", "{:.2f}".format(penalizer)," Exiting"))
                    return None,None,None
                penalizer=penalizer*random.choice([0.01,0.1,10,100])
                l1_ratio=l1_ratio*random.choice([0.01,0.1,10,100])
            else:
                retry=False
                cph=None


    print(("success:",success," last penalizer:","{:.2f}".format(penalizer)))
    if plot and cph is not None:
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
    return cph, use_cols_list, penalizer,l1_ratio


def train_cox(Train_data,penalizer=0.2,l1_ratio=0.5, var_thresh=0.0,CI=[],plot=False,
              freeze_penalizer=False):

    train_length=Train_data.shape[0]
    if CI=="CI":
        Train_data_boot = Train_data.sample(n=train_length, replace=True)
    else:
        Train_data_boot = Train_data
    if "21003-4.0" in Train_data_boot.columns:
        Train_data_boot = Train_data_boot.drop(["21003-4.0"], axis=1)  # To avoid colinearity with TTE
    if "21003-3.0" in Train_data_boot.columns:
        Train_data_boot = Train_data_boot.drop(["21003-3.0"], axis=1)  # To avoid colinearity with TTE

    Train_dummy = pd.get_dummies(Train_data_boot, drop_first=True)
    # Train_dummy, penalizer=0.2,var_thresh=0,plot=False
    cph, use_cols, penalizer,l1_ratio = fit_n_plot_cox_train_result(
        Train_dummy, penalizer=penalizer,l1_ratio=l1_ratio, var_thresh=var_thresh,plot=plot,
        freeze_penalizer=freeze_penalizer)
    return cph, use_cols, Train_dummy,penalizer,l1_ratio


def predict_test_results(Test_data, Results_path, cph,SN=1,penalizer=[],l1_ratio=[],
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

    Test_length=Test_data.shape[0]
    dummy_index=[]
    if CI=="CI":
        Test_data_boot= Test_data.sample(n=Test_length,replace=True)
        dummy_index=np.arange(Test_data_boot.shape[0])
        Test_data_boot.index=dummy_index
    else:
        Test_data_boot = Test_data
    drop_cols=[x for x in ["21003-4.0", "21003-3.0", "2443-3.0", "TTE"] if x in Test_data_boot.columns]
    # drop_cols=[x for x in ["21003-4.0", "21003-3.0", "2443-3.0"] if x in Test_data_boot.columns]
    if len(drop_cols)>0:
        Test_data_clean = Test_data_boot.drop(drop_cols, axis=1)
    Test_dummy = pd.get_dummies(Test_data_clean, drop_first=True)
    #     Test_dummy_rel=Test_dummy.iloc[:,:-2]
    test_predicted = cph.predict_survival_function(Test_dummy)
    # test_predicted =cph.score(Test_dummy)
    dummy_idx = np.arange(0, Test_dummy.shape[0])
    if len(dummy_idx)>0:
        Test_dummy.index=dummy_idx
        Test_data_boot.index=dummy_idx
        test_predicted.columns=dummy_idx
    Tot_test_pred = test_predicted.T.join(Test_data_boot.loc[:,"21003-4.0"])
    Tot_test_pred["21003-4.0"] = Tot_test_pred["21003-4.0"].astype(str)
    col = [str(x) for x in Tot_test_pred.columns.values]
    new_col_dict = dict(list(zip(Tot_test_pred.columns.values, col)))
    Tot_test_pred.rename(columns=new_col_dict, inplace=True)
    Tot_test_pred["pred"] = Tot_test_pred.apply(get_rel_score, axis=1)
    # Tot_test_pred.index=np.arange(0,Tot_test_pred.shape[0])
    # Test_data_boot.index=np.arange(0,Test_data_boot.shape[0])
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
    results_df = pd.DataFrame.from_dict({"APS": [APS], "AUC": [AUC], "SN": [SN],"penalizer":[penalizer],"l1_ratio":l1_ratio,"var_thresh":[var_thresh]})
    results_df = results_df.set_index("SN", drop=True)
    prediction_DF = pd.DataFrame.from_dict({"y_test_val": y_test_val, "y_pred_val": y_pred_val})
    results_df.to_csv(os.path.join(Results_path, "AUC_APS_results_" + str(int(SN)) + ".csv"),index=True)
    prediction_DF.to_csv(os.path.join(Results_path, "y_pred_results_" + str(int(SN)) + ".csv"))
    # return Tot_test_pred, Y_tot, Test_dummy, y_test_val, y_pred_val, AUC, APS

def Train_Test_Cox(run, Train_data, Test_data, Results_path, batch, CI=[],
                   init_penalizer=None,init_l1_ratio=None,plot=False,freeze_penalizer=False):
    for ind in np.arange(run.batch_size):
        SN = (batch * run.batch_size + ind)
        if init_penalizer is None:
            penalizer = random.uniform(0,1)
            l1_ratio=random.uniform(0,1)
        else:
            penalizer = init_penalizer
            l1_ratio=init_l1_ratio
        var_thresh = 0
        cph, use_cols, Train_dummy,panelizer,l1_ratio = train_cox(
            Train_data=Train_data,penalizer=penalizer,l1_ratio=l1_ratio,
                                           var_thresh=var_thresh,CI=[],plot=plot,freeze_penalizer=freeze_penalizer)
    #     print (use_cols)
    #     Tot_test_pred, Y_tot, Test_dummy, y_test_val, y_pred_val,AUC, APS =\
    #         predict_test_results(run,val_file_path=val_file_path,Results_path=Results_path,
    #                              cph=cph,col_names=use_cols,SN=SN,penalizer=penalizer,
    #                                            var_thresh=var_thresh,CI=CI)
        if cph is not None:
            predict_test_results(Test_data=Test_data, Results_path=Results_path,
                                         cph=cph,SN=SN,penalizer=penalizer,l1_ratio=l1_ratio,
                                                       var_thresh=var_thresh,CI=CI)

    # return cph, Train_dummy, Tot_test_pred, Y_tot, Test_dummy, y_test_val, y_pred_val, AUC, APS

def optimal_params(path):
    runs_files_results_names=os.listdir(path)
    runs_files_results_path=[os.path.join(path,x) for x in runs_files_results_names if x.startswith("AUC_APS")]
    runs_results_list=[pd.read_csv(x,index_col="SN") for x in runs_files_results_path]
    runs_Results_df=pd.concat(runs_results_list)
    runs_Results_df.sort_values(by="AUC",inplace=True, ascending=False)
    runs_Results_df.to_csv(os.path.join(path,"runs_results_summary.cav"))
    params=runs_Results_df.iloc[0,:]
    return params

def load_data(run,force_new_data_load=False):
    pickle_data_path=os.path.join(run.model_paths,"data_dict.pkl")
    try:
        comptute_data=False
        data_dict=from_pickle(pickle_data_path)
        print(("Loaded data from:", pickle_data_path))
    except:
        print("Couldnt load data' computing")
        comptute_data=True
    if comptute_data:
        data_path_dict={"train":run.train_file_path,"val":run.val_file_path,"test":run.test_file_path}
        data_dict={"train":pd.DataFrame(),"val":pd.DataFrame(),"test":pd.DataFrame()}
        features_file_path=run.features_file_path
        feat_file = pd.read_csv(features_file_path)
        feat_col = feat_file.loc[feat_file["Exclude"] == 0, "Field ID"]
        cols_file_path =run.test_file_path
        db_columns = pd.read_csv(cols_file_path, nrows=0).columns.values
        feat_cols_list = list(feat_col.values)
        use_cols = [col for col in db_columns for feat_col in feat_cols_list if
                    (col.startswith(feat_col) and "_na" not in col)]
        use_cols = ["eid"] + use_cols
        if "30750-0.0" not in use_cols:
            use_cols_plus_HbA1c=use_cols + ["30750-0.0"] # Adding %HbA1c for filtration
            dropa1c=True
        else:
            use_cols_plus_HbA1c=use_cols
            dropa1c=False
        for key,db_file_path in list(data_path_dict.items()):
            print("loading ", key," data")
            tmp_data = pd.read_csv(db_file_path, usecols=list(set(use_cols_plus_HbA1c+["TTE",'2443-3.0'])),
                                 index_col="eid")  # 21003-4.0 is time between visits, 21003-3.0 is age at last visit
            # _, _, dur_eve_df, _ = get_event_n_duration(db_file_path)  #
            # dur_eve_df = dur_eve_df.drop(["21003-4.0", "21003-3.0"], axis=1)
            # tmp_data = tmp_data.join(dur_eve_df,how="right")
            tmp_data = Filter_CZ(tmp_data, run.charac_selected, run.charac_id)
            if dropa1c:
                tmp_data=tmp_data.drop("30750-0.0",axis=1) #Remove HbA1c

            std_cols = [x for x in tmp_data.columns if x not in ['TTE', '2443-3.0',"21003-4.0"]]
            tmp_data.loc[:, std_cols] = standarise_df(tmp_data.loc[:, std_cols])
            data_dict[key]=tmp_data.dropna()
        to_pickle(data_dict,pickle_data_path)
        print(("Saved data to:", pickle_data_path))
    return data_dict["train"],data_dict["val"],data_dict["test"]

def upload_SA_CI_jobs(q,run,plot):
    waiton = []
    #TODO check if model exists
    Train_data,Val_data,Test_data=load_data(run)
    if len(os.listdir(run.Training_path))>0:
        params= optimal_params(run.Training_path)
    else:
        for batch in np.arange(np.ceil(float(run.hyper_parameter_iter) / run.batch_size)):
            print("Sending Batch number: ", batch, "of run: ", run.run_name, "For hyper parameter")
            params = {"run": run, "Train_data": Train_data, "Test_data": Val_data,
                      "Results_path": run.Training_path, "batch": batch, "CI": [],
                      "init_penalizer": None,"plot":plot,"freeze_penalizer":False}
            waiton.append(
                q.method(Train_Test_Cox,(),params))
        q.wait(waiton, assertnoerrors=False)
        params = optimal_params(run.Training_path)

    print(("Trainin optimal parameters are", params))
    penalizer = params["penalizer"]
    l1_ratio=params["l1_ratio"]
    Train_val_df=pd.concat([Train_data,Val_data])
    for batch in np.arange(np.ceil(np.float(run.num_of_bootstraps)/run.batch_size)):
        print("Sending Batch number: " ,batch,"of run: ",run.run_name,"For CI calc")
        params={"run":run, "Train_data":Train_val_df, "Test_data":Test_data,
                "Results_path":run.CI_results_path, "batch":batch, "CI":"CI",
                "init_penalizer":penalizer,"plot":plot,"init_l1_ratio":l1_ratio,
                "freeze_penalizer":True}
        waiton.append(q.method(Train_Test_Cox,(),params))
    q.wait(waiton,assertnoerrors=False)

def survival_analysis(run_name,use_fake_que=False,debug = False,
                      force_update=False,calc_ci_only=False,plot=False):
    # CI_Results_list=[]
    run=runs(run_name,debug = debug,force_update=force_update)
    if not calc_ci_only:
        if use_fake_que:
            qp = fakeqp
        else:
            qp = config.qp
        sethandlers(file_dir=config.log_dir)
        os.chdir('/net/mraid08/export/genie/LabData/Analyses/Yochai/Jobs')

        with qp(jobname='SA_'+run_name, q=['himem7.q'], _trds_def=2, _mem_def='1G',
                _tryrerun=True, max_r=300) as q:
            os.chdir('/net/mraid08/export/jafar/Microbiome/Analyses/Edlitzy/tempq_files/')
            q.startpermanentrun()
            upload_SA_CI_jobs(q,run=run,plot=plot)

    run.calc_CI()

    # results_path=run.CI_results_summary_table
    # CI_Results_list.append(pd.read_csv(results_path))
    # CI_results_df=pd.concat(CI_Results_list)
    # CI_results_df.to_csv(os.path.join(run.Folder_path,run_name+"_CI.csv"))
    # print("Results df saved to:",os.path.join(run.Folder_path,run_name+"_CI.csv"))
    # print(CI_results_df)

if __name__=="__main__":
    run_name = "SA_Four_BT_no_retic"#SA_GDRS_revision" # "SA_Five_BT"  # "SA_Antro_neto_whr"
    use_fake_que = False
    calc_ci_only = False
    debug = False
    force_update=True
    plot=False
    survival_analysis(run_name,use_fake_que=use_fake_que,calc_ci_only = calc_ci_only,debug = debug,
                      force_update=force_update,plot=plot)


