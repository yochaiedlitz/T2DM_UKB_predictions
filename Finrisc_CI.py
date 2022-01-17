from imports import *
from CI_Configs import runs

def load_data(run,y_rank_col):
    # feat_file = pd.read_csv(run.features_file_path)
    # feat_col = feat_file.loc[feat_file["Exclude"] == 0, "Field ID"]
    # db_columns = pd.read_csv(run.test_file_path, nrows=0).columns.values
    # feat_cols_list = list(feat_col.values)
    # use_cols = [col for col in db_columns for feat_col in feat_cols_list if
    #                 ((col.startswith(feat_col) and "_na" not in col) or feat_col == col)]
    use_cols = ["eid",y_rank_col]
    X = pd.read_csv(run.test_file_path, index_col="eid", usecols=use_cols)
    y = pd.read_csv(run.test_file_path, usecols=["eid", "2443-3.0"], index_col="eid")
    y = y.loc[X.index, "2443-3.0"]
    data = X.join(y)
    with open(os.path.join(run.model_paths, "use_cols.txt"), "wb") as fp:  # Pickling
        pickle.dump(use_cols, fp)
    return data

def sample_data(data):
    data = data.sample(n=data.shape[0], replace=True)
    y = data.loc[:, "2443-3.0"]
    X = data.drop(axis=1, columns=["2443-3.0"])
    return X, y

def bootstrap(run, data,y_rank_col):
    "mode should be either Train or Val"
    AUC_dict = {}
    APS_dict = {}

    results_list=[]
    for SN in tqdm(np.arange(run.num_of_bootstraps)):
        X,y=sample_data(data)
        y_test=y.values
        try:
            y_proba=X.loc[:, y_rank_col].values
        except:
            y_proba=X.loc[y_rank_col].values
        path = run.CI_results_path
        AUC = roc_auc_score(y_test, y_proba)
        APS = average_precision_score(y_test, y_proba)
        AUC_dict[str(SN)] = AUC
        APS_dict[str(SN)] = APS
        tmp_results_df=pd.DataFrame.from_dict(
            {"APS": [APS], "AUC": [AUC], "SN": [SN]})
        tmp_results_df = tmp_results_df.set_index("SN", drop=True)
        results_list.append(tmp_results_df)
        prediction_DF = pd.DataFrame.from_dict({"y_test": y_test, "y_pred": y_proba})
        prediction_DF.to_csv(os.path.join(path, "y_pred_results_" + str(int(SN)) + ".csv"))
    results_df=pd.concat(results_list)
    results_save_path=os.path.join(path, "AUC_APS_results_" + str(int(SN)) + ".csv")
    results_df.to_csv(results_save_path, index=True)
    print("saved results_df to:",results_save_path)
    return results_df

if __name__=="__main__":
    y_rank_col="Finrisc"
    run_dict={"Finrisc":runs("LR_Finrisc")}
    run=run_dict[y_rank_col]
    data=load_data(run,y_rank_col)
    results_df=bootstrap(run, data,y_rank_col)
    run.calc_CI(results_df)

