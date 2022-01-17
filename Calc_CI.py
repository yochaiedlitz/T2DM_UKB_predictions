import pandas as pd
from sklearn.utils import resample
import pickle
import os
import lightgbm as lgb
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from CI_Configs import runs #RunParams
import numpy as np
# model_path="/home/edlitzy/UKBB_Tree_Runs/For_article/Two_hot/"
# model_paths_names=os.listdir(model_path)
# model_paths=[os.path.join(model_path,x) for x in model_paths_names if x.startswith("imp_")]
USE_FAKE_QUE=False
CALC_CI_ONLY=False
from LabData import config_global as config
from LabUtils.addloglevels import sethandlers
from LabQueue.qp import fakeqp
from LabUtils.addloglevels import handlers_were_set
from UKBB_Functions import convert_dict
def Calc_CI_Proba(run, batch_number,sn_list=[]):
    data_dict_list=[]
    if sn_list==[]:
        sn_list=[batch_number*run.batch_size+ind for ind in np.arange(run.batch_size)]
    for run_number in sn_list:
        train_length = run.y_train.shape[0]
        val_length = run.y_val.shape[0]
        train_ind = resample(run.y_train.index, n_samples=train_length)
        val_ind = resample(run.y_val.index, n_samples=val_length)

        X_train = run.X_train.loc[train_ind, :]
        y_train = run.y_train.loc[train_ind]
        X_val = run.X_val.loc[val_ind, :]
        y_val = run.y_val.loc[val_ind]

        lgb_train = lgb.Dataset(X_train.values, label=y_train.values.flatten(), categorical_feature=run.cat_ind,
                                feature_name=run.Rel_Feat_Names, free_raw_data=False)
        run.params_dict=convert_dict(run.params_dict)
        num_trees = int(run.params_dict['num_boost_round'])

        evals_result = {}  # to record eval results for plotting
        gbm = lgb.train(params=run.params_dict, num_boost_round=num_trees, train_set=lgb_train, valid_sets=[lgb_train],
                        valid_names=["Train"], feature_name=run.Rel_Feat_Names, evals_result=evals_result,
                        verbose_eval=run.VERBOSE_EVAL)  #

        y_proba = gbm.predict(X_val, num_iteration=num_trees, raw_score=False)
        #     y_proba_tmp_df = pd.DataFrame(data=y_proba, index=y_val.index)
        #     y_proba_df = pd.concat([y_proba_df, y_proba_tmp_df], axis=0)

        APS = average_precision_score(y_val.values, y_proba)
        AUROC = roc_auc_score(y_val.values, y_proba)
        precision, recall, _ = precision_recall_curve(y_val.values, y_proba)
        fpr, tpr, _ = roc_curve(y_val.values, y_proba)
        data_dict = {"AUROC": AUROC, "APS": APS, "precision": precision, "recall": recall, "fpr": fpr, "tpr": tpr,
                     "y_proba": y_proba,
                     "y_val.values": y_val.values, "y_val_df": y_val}
        # ci_obj.create_dir()
        with open(os.path.join(run.CI_results_path, "CI_Dict_" + str(int(run_number))), 'wb') as fp:
            pickle.dump(data_dict, fp)
        data_dict_list.append(data_dict)
    return data_dict_list


def upload_CI_jobs(q,run):
    waiton = []
    model_path=run.model_paths
    num_of_batches=np.floor(run.num_of_bootstraps/run.batch_size)+1
    print(("calculating:",num_of_batches, " number of bathes, each of size:",run.batch_size))
    if not CALC_CI_ONLY:
        if run.exist_CI_files==[]:
            for bn in np.arange(num_of_batches):
                print(("Model path:", model_path, " Batch_number: ", bn))
                waiton.append(q.method(Calc_CI_Proba,[run,bn]))
            q.waitforresults(waiton)
        else:
            BN_dict={}
            for sn in run.missing_files:
                bn_of_sn=str(np.floor(float(sn)/run.batch_size))
                if bn_of_sn not in list(BN_dict.keys()):
                    BN_dict[bn_of_sn]=[]
                BN_dict[bn_of_sn].append(sn)
            for bn in list(BN_dict.keys()):
                print(("Model path:", model_path, " Batch_number: ", bn))
                waiton.append(q.method(Calc_CI_Proba, [run, bn,BN_dict[bn]]))
            q.waitforresults(waiton)
    run.calc_CI()

def calc_ci(run_name, debug=False):
    run=runs(run_name,debug=debug)
    if run.debug or debug:
        qp=fakeqp
        print("Running in debug mode!!!!!!!!!!")
    else:
        qp=config.qp
    if not handlers_were_set:
        try:
            sethandlers()
        except:
            print("handlers_were_set were already set")
        os.chdir('/net/mraid08/export/jafar/Microbiome/Analyses/Edlitzy/tempq_files/')
    with qp(jobname="calcCI" + run_name, q=['himem7.q'], _mem_def='4G', _trds_def=5,
            _tryrerun=False, max_r=650) as q:
        q.startpermanentrun()
        upload_CI_jobs(q,run)


if __name__=="__main__":
    names=["All"]
    for name in names:
        calc_ci(run_name=name, debug=False)
