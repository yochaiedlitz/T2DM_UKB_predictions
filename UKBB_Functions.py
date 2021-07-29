import os
import sys
import glob
import errno
import random
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import collections  # Used for ordered dictionary
import lightgbm as lgb
import shap
# import Create_Prob

from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, GroupKFold, StratifiedKFold
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve, auc, \
    brier_score_loss, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, make_scorer
from matplotlib.backends.backend_pdf import PdfPages
import time
import datetime
import pickle
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.naive_bayes import GaussianNB
import os.path

# RUN_NAME = "Singles"# Dilluted_Genetics

BASIC_FEAT_PATH = "/home/edlitzy/Biobank/"  # Base Folder where script saves the data to
PRS_PATH = '/net/mraid08/export/genie/Data/Yochai/PRS/PRS_Results/No_UKBB_PRS/No_UKBB_PRS_Normalised.csv'
SNPs_Summary_Path = '/net/mraid08/export/jafar/Yochai/PRS/PRS_Results/Extract_1K_SNPs_UKBB/Final_Results/'
# DATA_PATH = '/net/mraid08/export/jafar/UKBioBank/Data/ukb29741.csv'   # Yochai/DB_UKBB_031218/ukb20971.csv,/net/mraid08/export/jafar/UKBioBank/Data/ukb10194_new.csv
# DATA_PATH = '/net/mraid08/export/jafar/UKBioBank/Data/ukb29741_Diabetes_returned_train_val.csv'
Lite_Folder = "/home/edlitzy/Biobank/Data_Files/"
# SAVE_FOLDER = "/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/" + RUN_NAME + "/"
BASIC_DATA_PATH = "/net/mraid08/export/genie/Data/Yochai/UKBB_Runs/"  # '/net/mraid08/export/jafar/Microbiome/Analyses/Biobank/Comparison/Diabetes/Lifestyle_Diabetes/'
PROBA_FOLDER = "/net/mraid08/export/genie/Data/Yochai/Proba/All_Features_PRS/"

MAX_FEATURES_DISPLAY = 30  # How many features to show in the SHAP Analaysis
DPI = 200  # Resolution of images
SHAP_DPI = 1600
NJOBS = 1  # Used in prediction algorithms to define paralalisem
VERBOSE = -1  # How often the prediction shows progress
VERBOSE_EVAL = 1000  # How Evry
TRAIN = True
TRAIN_SIZE = 0.8
NUM_OF_DEP_PLOT = 40

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
    return roc_auc_score(y_true, proba[:, 1])


def Choose_params(Hyp_params):
    params_dict = {}
    # for key in Hyp_params.keys():
    for key, vals in Hyp_params.iteritems():
        if type(vals) == list:
            params_dict[key] = random.choice(Hyp_params[key])
        else:
            if type(vals) == np.ndarray:
                params_dict[key] = np.random.choice(Hyp_params[key])

            elif type(Hyp_params[key]) == float or type(Hyp_params[key]) == int:
                params_dict[key] = Hyp_params[key]
            else:
                params_dict[key] = Hyp_params[key].rvs()
    return params_dict


def Sort_AUC_APS(job_name, Save_2_folder, final_folder, Target_ID, proba_path, calc_shap, mode, use_proba, pdf_name,
                 Refit_Base_Model_Path):
    path = os.path.join(Save_2_folder,'*_result.csv')
    # y_train, X_train, y_test_df, y_test, X_test, cat_names,Rel_Feat_Names = Load_Saved_Data(final_folder,Lite=False)
    files = glob.glob(path)
    scores = []

    for name in files:  # 'file' is a builtin type, 'name' is a less-ambiguous variable name.
        scores.append(pd.read_csv(name).iloc[-1:, -3:])

    df = pd.concat(scores)
    df.set_index(keys="SN", drop=True, append=False, inplace=True, verify_integrity=False)

    df.to_csv(os.path.join(final_folder,job_name + "_Result.csv"))

    auc_aps = df
    # Meging the parameters file

    path = os.path.join(Save_2_folder, '*parameters.csv')
    print("Sort_AUC_APS concatanation path:", path)
    allFiles = glob.glob(path)
    print("allFiles:", allFiles)
    frame = pd.DataFrame()
    list_ = []

    for file_ in allFiles:
        print(file_)
        df = pd.read_csv(file_, index_col=0)
        df.columns = df.iloc[10]
        list_.append(df)
        if os.path.isfile(file_):
            os.remove(file_)
        else:  # Show an error ##
            print("Error: %s file not found" % file_)
    frame = pd.concat(list_, axis=1)
    frame.index_name = "parameter"
    frame.columns = frame.loc["SN", :]

    frame.to_csv(os.path.join(final_folder ,job_name + "_Parameters_Table.csv"))

    print_to_file(auc_aps, job_name, final_folder, Save_2_folder, frame, Target_ID
                  , proba_path, calc_shap, mode, use_proba, pdf_name, Refit_Base_Model_Path)


def print_to_file(result, job_name, final_folder, Save_2_folder, frame, Target_ID
                  , proba_path, calc_shap, mode, use_proba, pdf_name, Refit_Base_Model_Path):
    print("pdf name in print_to_file is:", pdf_name)  # TODO:

    y_train, X_train, y_test_df, y_test, X_test, cat_names, Rel_Feat_Names = Load_Saved_Data(final_folder, Lite=False)
    metric = frame.loc["metric"].iloc[0]

    if metric == "auc":
        result_sorted = result.sort_values(by="score", ascending=False)
    else:
        result_sorted = result.sort_values(by="score", ascending=True)

    result_sorted.to_csv(os.path.join(final_folder , job_name + "_result_sorted.csv"))
    top_auc_sn = str(int(result_sorted.index.values[0]))
    create_pdf(Save_2_folder, final_folder, job_name, Target_ID, proba_path, calc_shap, mode, use_proba, pdf_name,
               Refit_Base_Model_Path)


def create_pdf(Save_2_folder, final_folder, job_name, Target_ID, proba_path, calc_shap, mode, use_proba, pdf_name,
               Refit_Base_Model_Path):
    # Data = {'df_Features': df_Features, 'DF_Targets': DF_Targets, 'X_display': X_display, 'X_train': X_train,
    #             'y_train': y_train, 'y_test': y_test, 'X_test': X_test, 'cat_names': cat_names,
    #             'Rel_Feat_Names': Rel_Feat_Names}
    with open(os.path.join(final_folder ,job_name + "train_Data"), 'rb') as fp:
        Data = pickle.load(fp)
    y_train = Data['y_train']
    X_train = Data['X_train']
    # y_test_df= Data['y_test_dPdfPagesf']
    y_test = Data['y_test']
    X_test = Data['X_test']
    if use_proba:
        y_val = Data["y_val"].values.flatten()
        X_val = Data["X_val"]

    cat_names = Data["cat_names"]
    Rel_Feat_Names = Data["Rel_Feat_Names"]

    run_name = str(pd.read_csv(filepath_or_buffer=os.path.join(final_folder , job_name + "_result_sorted.csv")).iloc[0, 0])

    # ~~~~~~~~~~~~Loading parameters~~~~~~~~~~~~~~
    parameters = pd.read_csv(os.path.join(final_folder , job_name + "_Parameters_Table.csv"),
                             index_col=0)  # Check that we can read params and build the selected model, train it and make all required drawings
    parameters.index_name = "parameter"
    parameters.columns = parameters.loc["SN", :]
    parameters.drop(index="SN", inplace=True)
    metric = str(parameters.loc["metric", run_name])
    print("metric is: ", metric)
    print("pdf name in create_pdf is:", pdf_name)
    SHAP_PDF_PATH = os.path.join(final_folder , pdf_name + "_" + run_name + ".pdf")
    print("saving pdf at: ", SHAP_PDF_PATH)
    pdf = PdfPages(SHAP_PDF_PATH)
    if isinstance(parameters, (list,)):
        params_dict = parameters[run_name].to_dict()
    else:
        params_dict = parameters.to_dict()[run_name]

    cat_ind = [x for x, name in enumerate(Rel_Feat_Names) if name in cat_names]

    lgb_train = lgb.Dataset(X_train, label=y_train.values.flatten(), categorical_feature=cat_ind,
                            feature_name=Rel_Feat_Names,
                            free_raw_data=False)
    if use_proba:
        lgb_eval = lgb.Dataset(X_val, label=y_val, reference=lgb_train,
                               categorical_feature=cat_ind,
                               free_raw_data=False, feature_name=Rel_Feat_Names)

    # evals_result = np.load(Save_2_folder+run_name + '_evals_result.npy')

    evals_result = {}  # to record eval results for plotting
    start_time = time.time()
    print('Start training non-shap values of final model of: ', job_name, " at:", datetime.datetime.now())
    num_trees = int(params_dict["num_boost_round"])
    if use_proba:
        if Refit_Base_Model_Path == None:
            gbm = lgb.train(init_model=Refit_Base_Model_Path, params=params_dict, num_boost_round=num_trees,
                            train_set=lgb_train, valid_sets=[lgb_train, lgb_eval],
                            valid_names=["Train", "Evaluation"], feature_name=Rel_Feat_Names,
                            evals_result=evals_result, verbose_eval=VERBOSE_EVAL)
        else:
            Base_gbm = lgb.Booster(model_file=Refit_Base_Model_Path)
            gbm = lgb.train(params=params_dict, num_boost_round=Base_gbm.num_trees() + num_trees, train_set=lgb_train,
                            valid_sets=[lgb_train, lgb_eval],
                            valid_names=["Train", "Evaluation"], feature_name=Rel_Feat_Names,
                            evals_result=evals_result, verbose_eval=VERBOSE_EVAL)  #
    else:
        if Refit_Base_Model_Path == None:
            gbm = lgb.train(init_model=Refit_Base_Model_Path, params=params_dict, num_boost_round=num_trees,
                            train_set=lgb_train, valid_sets=[lgb_train],
                            valid_names=["Train"], feature_name=Rel_Feat_Names,
                            evals_result=evals_result, verbose_eval=VERBOSE_EVAL)

        else:
            Base_gbm = lgb.Booster(model_file=Refit_Base_Model_Path)
            gbm = lgb.train(params=params_dict, num_boost_round=Base_gbm.num_trees() + num_trees, train_set=lgb_train,
                            valid_sets=[lgb_train],
                            valid_names=["Train"], feature_name=Rel_Feat_Names,
                            evals_result=evals_result, verbose_eval=VERBOSE_EVAL)  #

    print("Final training of ", job_name, " took %s seconds ---" % (time.time() - start_time))
    gbm.save_model(os.path.join(final_folder , run_name + '_lgbm.txt'))

    with open(os.path.jopin(final_folder , "evals_result"), 'wb') as fp:
        pickle.dump(evals_result, fp)

    ax = lgb.plot_metric(evals_result)
    fig = plt.gcf()
    pdf.savefig(fig, dpi=DPI)
    plt.close()

    # Add here the X_Test and Y_Test results

    y_proba = gbm.predict(X_test, num_iteration=num_trees, raw_score=False)
    # y_proba_df = pd.DataFrame(data=y_proba, index=y_test_df.index, columns=["Y_PROB"])
    y_proba_df = pd.DataFrame(data=y_proba, index=y_test.index, columns=["Y_PROB"])

    y_proba_df.to_csv(os.path.join(final_folder , job_name + "_OnlyPROB.csv"))

    if ((mode == "A") and (use_proba)):
        X_test[job_name + "Y_PROB"] = y_proba
        X_test.to_csv(os.path.join(final_folder , job_name + "_FEATandY_PROB.csv"))
        y_test.to_csv(os.path.join(final_folder , job_name + "_Target.csv"))
        with open(os.path.join(final_folder , job_name + "_Proba"), 'wb') as fp:
            pickle.dump(y_proba, fp)
        with open(os.path.join(final_folder , job_name + "_Feat"), 'wb') as fp:
            pickle.dump(X_test.values, fp)

        X_test.drop(job_name + "Y_PROB", axis=1, inplace=True)

    print('Finished predicting...', job_name, "at time:", datetime.datetime.now())

    y_pred_val = []
    y_test_val = []
    for i, stat in enumerate(np.isnan(y_test.values)):
        if ~stat:
            y_pred_val.append(y_proba[i])
            y_test_val.append(y_test.values.flatten()[i])

    if Target_ID != "21001-0.0":
        APS = average_precision_score(y_test_val, np.array(y_pred_val))
        AUC = roc_auc_score(y_test_val, y_pred_val)

        # Print ROC Curve
        plot_ROC_curve(y_test_val, y_pred_val, AUC, pdf)

        # Print Precision Recall Curve
        plot_precision_recall(y_test_val, y_pred_val, APS, pdf)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ plot_quantile_curve~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        plot_quantiles_curve(y_test_val, y_pred_val, pdf)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ plot_calibration_curve~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        plot_calibration_curve(y_test_val, y_pred_val, pdf, os.path.join(final_folder , job_name), Print_Benefit=True)

    with open(os.path.join(final_folder , "y_pred_val_list"), 'wb') as fp:
        pickle.dump(y_pred_val, fp)
    with open(os.path.join(final_folder ,"y_test_val_list"), 'wb') as fp:
        pickle.dump(y_test_val, fp)

    X_display = X_test.copy()
    X_display.columns = Rel_Feat_Names
    if ~os.path.isfile(os.path.join(final_folder ,job_name + "_X_display.csv")):
        X_display.to_csv(os.path.join(final_folder , job_name + "_X_display.csv"))

    if calc_shap:
        Calc_SHAP(final_folder, pdf, job_name, gbm)

    pdf.close()
    print("finished Create_PDF for:", job_name, " at:", datetime.datetime.now())
    print("Finished", job_name)
    print("saved PDF at:", SHAP_PDF_PATH)


def plot_only(final_folder, calc_shap, job_name, gbm=None):
    run_name = str(pd.read_csv(filepath_or_buffer=os.path.join(final_folder , job_name + "_result_sorted.csv")).iloc[0, 0])

    SHAP_PDF_PATH = os.path.join(final_folder , "Results_" + job_name + "_" + run_name + ".pdf")
    pdf = PdfPages(SHAP_PDF_PATH)
    with open(os.path.join(final_folder , job_name + "_Proba"), 'rb') as fp:
        y_pred_val = pickle.load(fp)
    with open(os.path.join(final_folder , "y_test_val_list"), 'rb') as fp:
        y_test_val = pickle.load(fp)

    APS = average_precision_score(y_test_val, y_pred_val)

    AUC = roc_auc_score(y_test_val, y_pred_val)

    # Print ROC Curve
    plot_ROC_curve(y_test_val, y_pred_val, AUC, pdf)

    # Print Precision Recall Curve
    plot_precision_recall(y_test_val, y_pred_val, APS, pdf)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ plot_quantile_curve~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    plot_quantiles_curve(y_test_val, y_pred_val, pdf)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ plot_calibration_curve~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    plot_calibration_curve(y_test_val, y_pred_val, pdf, os.path.join(final_folder,job_name), Print_Benefit=True)

    if calc_shap:
        Calc_SHAP(final_folder, pdf, job_name, gbm)

    pdf.close()
    print("finished Create_PDF for:", job_name, " at:", datetime.datetime.now())
    print("Finished", job_name)


def Calc_SHAP(final_folder, pdf, job_name, gbm):
    y_train, X_train, y_test_df, y_test, X_test, cat_names, Rel_Feat_Names = Load_Saved_Data(final_folder, Lite=False)
    run_name = str(pd.read_csv(filepath_or_buffer=os.path.join(final_folder , job_name + "_result_sorted.csv")).iloc[0, 0])

    SHAP_PDF_PATH = os.path.join(final_folder , "Results_" + job_name + "_" + run_name + ".pdf")
    # ~~~~~~~~~~~~Loading parameters~~~~~~~~~~~~~~
    parameters = pd.read_csv(os.path.join(final_folder , job_name + "_Parameters_Table.csv"),
                             index_col=0)  # Check that we can read params and build the selected model, train it and make all required drawings
    parameters.index_name = "parameter"
    parameters.columns = parameters.loc["SN", :]
    parameters.drop(index="SN", inplace=True)
    if isinstance(parameters, (list,)):
        params_dict = parameters[run_name].to_dict()
    else:
        params_dict = parameters.to_dict()[run_name]

    cat_ind = [x for x, name in enumerate(Rel_Feat_Names) if name in cat_names]

    evals_result = {}  # to record eval results for plotting
    print("Calculating SHAP Values for:", job_name, " at:", datetime.datetime.now())
    shap_values = gbm.predict(X_test.values, pred_contrib=True, num_threads=10)
    gbm.save_model(os.path.join(final_folder , job_name + "_shap_model.txt"))
    df = pd.DataFrame(shap_values,index=X_test.index,columns=X_test.columns)
    df.to_csv(os.path.join(final_folder ,job_name + "_shap_values.csv"))
    # X_display.to_csv(final_folder+JOB_NAME+"_X_display.csv")

    fig = plt.figure()
    with open(os.path.join(final_folder , "Rel_Feat_Names"), 'rb') as fp:
        Rel_Feat_Names = pickle.load(fp)
    X_display = X_test.copy()
    X_display.columns = Rel_Feat_Names
    if shap_values.shape[0] > 50000:
        shap.summary_plot(shap_values[:50000, :X_display.shape[1]], features=X_display.iloc[:50000, :], show=False,
                          auto_size_plot=True,
                          max_display=30)
    else:
        shap.summary_plot(shap_values[:, :X_display.shape[1]], features=X_display, show=False, auto_size_plot=True,
                          max_display=30)
    plt.savefig(os.path.join(final_folder , "Results_" + job_name + "_" + run_name + 'SHAP.png'), bbox_inches='tight', dpi=SHAP_DPI)
    plt.close("all")
    img = mpimg.imread(os.path.join(final_folder ,"Results_" + job_name + "_" + run_name + 'SHAP.png'))
    imgplot = plt.imshow(img)
    plt.axis('off')
    fig = plt.gcf()
    pdf.savefig(fig, dpi=SHAP_DPI)
    plt.close("all")
    # df.to_csv()

    Ordered_X_idx = list(np.argsort(-np.abs(df.iloc[:, :-1]).sum(0)))
    ranks = [np.abs(df.iloc[:, :-1]).sum(0)[ind] for ind in Ordered_X_idx]
    FEAT_DF = pd.read_csv(os.path.join(final_folder , job_name + "_DF_Features_List.csv"), header=0)
    if "Unnamed: 0" in FEAT_DF.columns.values:
        FEAT_DF.drop(columns="Unnamed: 0")
    FEAT_DF.set_index('Description', drop=True, inplace=True)
    X_Descreption = [X_display.columns.values[ind] for ind in Ordered_X_idx]
    # print X_Descreption
    if FEAT_DF.shape[0] != 0:
        Sorted_Question = FEAT_DF.loc[X_Descreption, :]
        Sorted_Question.drop_duplicates(keep="first", inplace=True)
        # Sorted_Question.loc[:,"Score"] = ranks
        Sorted_Question.to_csv(os.path.join(final_folder, job_name + "_Features_List_Shap_Sorted.csv"))
    print("Printing Dependence plot for:", job_name, " at:", datetime.datetime.now())

    for name in Ordered_X_idx[0:NUM_OF_DEP_PLOT]:
        fig = plt.figure(figsize=(20, 20))
        if shap_values.shape[0] > 50000:
            shap.dependence_plot(name, shap_values[:50000, :X_display.shape[1]], X_display.iloc[:50000, :], show=False)
        else:
            shap.dependence_plot(name, shap_values[:, :X_display.shape[1]], X_display, show=False)

        plt.savefig(os.path.join(final_folder ,"Results_" + job_name + "_" + run_name + 'SHAP.png'), bbox_inches='tight',
                    dpi=SHAP_DPI)
        plt.close("all")
        img = mpimg.imread(os.path.join(final_folder , "Results_" + job_name + "_" + run_name + 'SHAP.png'))
        imgplot = plt.imshow(img)
        plt.axis('off')
        fig = plt.gcf()
        pdf.savefig(fig, dpi=SHAP_DPI)
        plt.close("all")


def plot_ROC_curve(y_test_val, y_pred_val, AUC, pdf):
    fpr, tpr, _ = roc_curve(y_test_val, y_pred_val)
    fig = plt.figure()
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
    pdf.savefig(fig, dpi=DPI)
    plt.close(fig)


def plot_precision_recall(y_test_val, y_pred_val, APS, pdf):
    precision, recall, _ = precision_recall_curve(y_test_val, y_pred_val)
    fig = plt.figure()
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(APS))
    pdf.savefig(fig, dpi=DPI)
    plt.close(fig)

    # Plotting ratio graph for precision recall

    fig = plt.figure()
    rel_prec = precision / precision[0]
    plt.step(recall, rel_prec, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, rel_prec, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Relative Precision')
    # plt.ylim([0.0, 1.05 * np.percentile(rel_prec,99.97)])
    plt.ylim([0.0, 1.05 * max(rel_prec)])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Relative-Precision-Recall curve: AP={0:0.2f}'.format(APS))
    pdf.savefig(fig, dpi=DPI)
    plt.close(fig)

    # Plotting ratio graph for precision recallwith removed maximum value

    fig = plt.figure()
    plt.step(recall, rel_prec, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, rel_prec, step='post', alpha=0.2,
                     color='b')
    plt.xlabel('Recall')
    plt.ylabel('Relative Precision')
    plt.ylim([0.0, 1.05 * max(np.delete(rel_prec, np.argmax(rel_prec)))])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Relative-Precision-Recall trimmed max: AP={0:0.2f}'.format(APS))
    pdf.savefig(fig, dpi=DPI)
    plt.close(fig)

    # Show graph of True positive Vs.quantiles of predicted probabilities.

    fig = plt.figure()
    plt.step(recall, rel_prec, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, rel_prec, step='post', alpha=0.2,
                     color='b')
    plt.xlabel('Recall')
    plt.ylabel('Relative Precision')
    plt.ylim([0.0, 1.05 * max(np.delete(rel_prec, np.argmax(rel_prec)))])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Relative-Precision-Recall trimmed max: AP={0:0.2f}'.format(APS))
    pdf.savefig(fig, dpi=DPI)
    plt.close(fig)


def plot_calibration_curve(y_test, prob_pos, pdf, path, Print_Benefit=True):
    """Plot calibration curve for est w/o and with calibration. """
    nbins_vec = [10, 15, 20, 25, 30, 50, 100]
    Y_df = pd.DataFrame()
    for nbins in nbins_vec:
        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_pos, n_bins=nbins)

        plt.figure(figsize=(10, 10))
        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
        ax2 = plt.subplot2grid((3, 1), (2, 0))

        ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

        ax1.plot(mean_predicted_value, fraction_of_positives, "rs-", label="LGBM Predictor")

        ax2.hist(prob_pos, range=(0, 1), bins=nbins, color="r", label="Original Predictions", histtype="step", lw=2)

        X = np.array(prob_pos)
        y = np.array(y_test)

        ir = IsotonicRegression()
        skf = StratifiedKFold(n_splits=10, random_state=0, shuffle=False)
        y_true_list = []
        y_cal_prob_list = []
        for train_index, test_index in skf.split(X, y):
            # print "test index shape",test_index.shape
            # print "train index shape", train_index.shape

            X_Cal_train, X_Cal_test = X[train_index], X[test_index]
            y_Cal_train, y_Cal_test = y[train_index], y[test_index]

            start_time = time.time()
            ir.fit(X_Cal_train, y_Cal_train)
            p_calibrated = ir.transform(X_Cal_test)
            y_cal_prob_list = np.append(y_cal_prob_list, p_calibrated)
            y_true_list = np.append(y_true_list, y_Cal_test)

        fraction_of_positives, mean_predicted_value = calibration_curve(y_true_list, y_cal_prob_list, n_bins=nbins)

        ax1.plot(mean_predicted_value, fraction_of_positives, "bs-", label="Calibrated_curve")
        ax2.hist(y_cal_prob_list, color="b", bins=nbins, range=(0, 1), label="Calibrated_hist", histtype="step", lw=2)

        ax1.set_ylabel("Fraction of positives")
        max_value = np.max([x for x in fraction_of_positives if str(x) != "nan"])
        # ax1.set_ylim([0, max_value])
        # ax1.set_xlim([0, mean_predicted_value[np.where(fraction_of_positives==max_value)]])

        ax1.legend(loc="lower right")
        ax1.set_title('Calibration plots  (reliability curve) for nbins:' + str(nbins))

        ax2.set_xlabel("Mean predicted value")
        ax2.set_ylabel("Count")
        ax2.legend(loc="upper center", ncol=2)

        plt.tight_layout()
        pdf.savefig(plt.gcf(), dpi=DPI)
        plt.close(plt.gcf())

        Y_df.loc[:, "Orig_prob_nbins_" + str(nbins)] = X
        Y_df.loc[:, "Cal_prob_nbins_" + str(nbins)] = y_cal_prob_list
        Y_df.loc[:, "y_true_nbins_" + str(nbins)] = y_true_list

    Y_df.to_csv(path + "_Cal_prob")

    if Print_Benefit:

        Pt_start = 0
        Pt_stop = 0.5
        Pt_step = 0.001
        True_pos_list = []
        False_pos_list = []
        Net_Benefit = pd.DataFrame(index=np.arange(Pt_start, Pt_stop, Pt_step))
        N = len(y_cal_prob_list)

        true_pos_all = np.sum(y_true_list)
        false_pos_all = len(y_true_list) - true_pos_all

        for pt in np.arange(Pt_start, Pt_stop, Pt_step):
            Pos_cal = Y_df.loc[Y_df.loc[:, "Cal_prob_nbins_25"] >= pt, "y_true_nbins_25"]
            if Pos_cal.count() != 0:
                True_Pos_cal = Pos_cal.sum()
                False_Pos_cal = Pos_cal.shape[0] - True_Pos_cal
            else:
                True_Pos_cal = 0
                False_Pos_cal = 0
            Net_Benefit.loc[pt, "Net benefit calibrated"] = ((True_Pos_cal / N) - (False_Pos_cal / N) * (pt / (1 - pt)))
            Net_Benefit.loc[pt, "Test all Net benefit"] = ((true_pos_all / N) - (false_pos_all / N) * (pt / (1 - pt)))
            Net_Benefit.loc[pt, "Don't test benefit"] = 0

        fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))

        styles1 = ['b-', 'r--', 'g:']
        Net_Benefit.plot(ax=ax1, style=styles1)
        ax1.set_xlabel("Threshold Probability")
        ax1.set_ylabel("Net Benefit")

        ax1.legend(loc="upper right")
        ax1.set_title('Net Benefit plot based on 50 bins')
        #         min_lim=Net_Benefit.loc[Net_Benefit.loc[:,"Net benefit calibrated"]>0].index.values.min()
        ax1.set_ylim(min(-0.005, Net_Benefit.loc[:, ["Net benefit calibrated", "Test all Net benefit"]].min().min()),
                     Net_Benefit.loc[:, ["Net benefit calibrated", "Test all Net benefit"]].max().max())

        plt.tight_layout()
        pdf.savefig(plt.gcf(), dpi=DPI)
        plt.close(plt.gcf())

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))

        styles1 = ['b-', 'r--', 'g:']
        Net_Benefit.plot(ax=ax, style=styles1)
        ax.set_xlabel("Threshold Probability")
        ax.set_ylabel("Net Benefit")

        ax.legend(loc="upper right")
        ax.set_title('Net Benefit plot based on 50 bins')
        #         min_lim=Net_Benefit.loc[Net_Benefit.loc[:,"Net benefit calibrated"]>0].index.values.min()

        ax.set_ylim(min(Net_Benefit.loc[:, ["Net benefit calibrated"]].min().min() - 0.005, -0.005),
                    max(Net_Benefit.loc[:, ["Net benefit calibrated", "Test all Net benefit"]].max().max(), 0))

        plt.tight_layout()

        pdf.savefig(plt.gcf(), dpi=DPI)
        plt.close(plt.gcf())

        Net_Benefit.to_csv(path_or_buf=path + "_Net_Benefit")


def plot_quantiles_curve(y_test_val, y_pred_val, pdf):
    """Plot calibration curve for est w/o and with calibration. """
    quantiles = 100
    df = pd.DataFrame(data={"Y_test": y_test_val, "Y_Pred": y_pred_val})
    df = df.sort_values("Y_Pred", ascending=False)
    Resolution = [5, 10, 25, 50, 100]
    Quants = pd.DataFrame()
    for ind, res in enumerate(Resolution):
        Quants = pd.DataFrame()
        Quants = df.loc[:, "Y_Pred"].quantile(np.arange(1. / res, 1 + 1. / res, 1. / res))
        Rank = pd.DataFrame()
        for ind, quant in enumerate(Quants.values):
            if ind > 0:
                Rank.loc[np.str(ind), "Diagnosed"] = df.loc[((df["Y_Pred"] <= quant) & \
                                                             (df["Y_Pred"] > Quants.values[ind - 1]))].loc[:,
                                                     'Y_test'].sum()
                Rank.loc[np.str(ind), "All"] = df.loc[((df["Y_Pred"] <= quant) & \
                                                       (df["Y_Pred"] > Quants.values[ind - 1]))].loc[:,
                                               'Y_test'].count()
                Rank.loc[np.str(ind), "Ratio"] = Rank.loc[np.str(ind), "Diagnosed"] / Rank.loc[np.str(ind), "All"]
            else:
                Rank.loc[np.str(ind), "Diagnosed"] = df.loc[df["Y_Pred"] <= quant].loc[:, 'Y_test'].sum()
                Rank.loc[np.str(ind), "All"] = df.loc[df["Y_Pred"] <= quant].loc[:, 'Y_test'].count()
                Rank.loc[np.str(ind), "Ratio"] = Rank.loc[np.str(ind), "Diagnosed"] / Rank.loc[np.str(ind), "All"]
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.bar(Rank.index.values, Rank.loc[:, "Ratio"])
        labels = [int((ind + 1) * 100.0 / len(ax.get_xticklabels())) for ind, item in enumerate(ax.get_xticklabels())]
        ax.set_xticklabels(labels)
        ax.set_xlim(0, Rank.index.values.astype(int).max() + 1)
        ax.set_title('Precentage of Ill Vs. Prediction quantile with:' + str(res) + " bins")
        ax.set_xlabel("Prediction quantile")
        ax.set_ylabel("Precentage of Ill in quantile")

        ax2 = ax.twinx()
        Rank.loc[:, "Ratio"] = Rank.loc[:, "Ratio"].fillna(0)
        Rank.loc[:, "Fold"] = Rank.loc[:, "Ratio"].values / Rank.loc[Rank.index[:50], "Ratio"].mean()
        ax2.bar(Rank.index.values, height=Rank.loc[:, "Fold"])
        ax.set_ylabel("Fold prevalence at quantile vs \n mean prevalence below median quantile")

        plt.tight_layout()

        pdf.savefig(plt.gcf(), dpi=DPI)
        plt.close(plt.gcf())


def Order_Targets(DF_Targets, no_symp_code, Sub_Class_ID):
    DF_Targets[DF_Targets.iloc[:, 0] == no_symp_code] = 0
    DF_Targets[DF_Targets.iloc[:, 0] < 0] = np.NaN  # Imputation Replace -1 and -3 values with NAN

    if Sub_Class_ID == "All":
        DF_Targets[DF_Targets.iloc[:, 0] > 0] = 1
    else:
        DF_Targets[DF_Targets.iloc[:, 0] == Sub_Class_ID] = "Target"
        DF_Targets[(DF_Targets.iloc[:, 0] != "Target") & (DF_Targets.iloc[:, 0] != np.NaN)] = 0
        DF_Targets[DF_Targets.iloc[:, 0] == "Target"] = 1

    DF_Targets = DF_Targets.dropna(axis=0)

    return DF_Targets


def Filter_CZ(Features_DF, charac_selected,
              charac_id):  # CHARAC_SELECTED = {"Age at recruitment": "All", "Sex": "Female", "Ethnic background":
    # "All","Type of special diet followed": "All"}
    # CHARAC_ID = {"Age at last visit": 21003, "Sex": 31, "Ethnic background": 21000,
    # "Type of special diet followed": 20086}

    if charac_selected["Age at last visit"] != "All":
        Age_max = charac_selected["Age at recruitment"] + 5
        Age_min = charac_selected["Age at recruitment"] - 5
        Features_DF = Features_DF.loc[
                      Features_DF[charac_id["Age at recruitment"]].between(Age_min, Age_max, inclusive=True), :]
    if "Age_max" in charac_selected:
        if charac_selected["Age_max"] != "None":
            Age_max = charac_selected["Age_max"]
    if "Age_min" in charac_selected:
        if charac_selected["Age_min"] != "None":
            Age_min = charac_selected["Age_min"]

    if charac_selected["Sex"] != "All":
        Features_DF = Features_DF[Features_DF[charac_id["Sex"]] == SEX_CODE[charac_selected["Sex"]]]

    if charac_selected["Ethnic background"] != "All":
        if charac_selected["Ethnic background"] == 1:
            Features_DF = Features_DF[(Features_DF[charac_id["Ethnic background"]] == 1) |
                                      ((Features_DF[charac_id["Ethnic background"]] > 1000) &
                                       (Features_DF[charac_id["Ethnic background"]] < 2000))]
        elif charac_selected["Ethnic background"] == 2:
            Features_DF = Features_DF[(Features_DF[charac_id["Ethnic background"]] == 2) |
                                      ((Features_DF[charac_id["Ethnic background"]] > 2000) &
                                       (Features_DF[charac_id["Ethnic background"]] < 3000))]
        elif charac_selected["Ethnic background"] == 3:
            Features_DF = Features_DF[(Features_DF[charac_id["Ethnic background"]] == 3) |
                                      ((Features_DF[charac_id["Ethnic background"]] > 3000) &
                                       (Features_DF[charac_id["Ethnic background"]] < 4000))]
        elif charac_selected["Ethnic background"] == 4:
            Features_DF = Features_DF[(Features_DF[charac_id["Ethnic background"]] == 4) |
                                      ((Features_DF[charac_id["Ethnic background"]] > 4000) &
                                       (Features_DF[charac_id["Ethnic background"]] < 5000))]
        else:
            Features_DF = Features_DF[(Features_DF[charac_id["Ethnic background"]]
                                       == charac_selected["Ethnic background"])]

    if charac_selected["Type of special diet followed"] != "All":
        Features_DF = Features_DF[Features_DF[charac_id["Type of special diet followed"]] ==
                                  DIET_CODE[charac_selected["Type of special diet followed"]]]
    # "Maximal_a1c": 38, "Minimal_a1c": 20
    if "Minimal_a1c" in charac_selected:
        if charac_selected["Minimal_a1c"] != "All":
            Features_DF = Features_DF.loc[Features_DF.loc[:, "30750-0.0"] > charac_selected["Minimal_a1c"], :]
    if "Maximal_a1c" in charac_selected:
        if charac_selected["Maximal_a1c"] != "All":
            Features_DF = Features_DF.loc[Features_DF.loc[:, "30750-0.0"] <= charac_selected["Maximal_a1c"], :]

    return Features_DF


def Load_Targets(Target_ID, Sub_Class_ID, visits, nrows_return, no_symp_codes, data_path):
    DF_Targets = []
    for ind, name in enumerate(visits):
        print("Load Tergets of visits: ", str(name))
        if ind == 0:
            if isinstance(nrows_return, (int, long)):
                nrows_0 = nrows_return * 5
                DF_Targets.append(pd.read_csv(data_path, usecols=[Target_ID[:-4] + "-" + str(name) + ".0", "eid"],
                                              index_col='eid', nrows=nrows_0))
            else:
                DF_Targets.append(pd.read_csv(data_path, usecols=[Target_ID[:-4] + "-" + str(name) + ".0", "eid"],
                                              index_col='eid', nrows=nrows_return))
        else:
            DF_Targets.append(pd.read_csv(data_path, usecols=[Target_ID[:-4] + "-" + str(name) + ".0", "eid"],
                                          index_col='eid', nrows=nrows_return))
        DF_Targets[ind] = Order_Targets(DF_Targets[ind], no_symp_codes, Sub_Class_ID)
    if Target_ID != 21001 - 0.0:
        DF_Targets[0] = DF_Targets[0][
            DF_Targets[0].iloc[:, 0] == 0]  # Taking only Health persons in round A that flipped in round B

    # To enable good concatatnation of targets
    for ind, name in enumerate(visits):
        DF_Targets[ind].columns = [Target_ID[:-4]]

    if len(DF_Targets) >= 2:
        Targets = pd.concat([DF_Targets[1], DF_Targets[2]], axis=0, join="outer").groupby(
            level=0).max()  # if  participant visits twice, taking the value if he got sick
        Targets = pd.concat([DF_Targets[0], Targets], axis=1, join="inner")

    else:
        Targets = pd.concat([DF_Targets[0], DF_Targets[1]], axis=1, join="inner")

    Targets = Targets.iloc[:, 1]

    if nrows_return != 'None':
        if Targets.shape[0] > nrows_return:
            print("original targets shape:", str(Targets.shape[0]))
            Targets = Targets.iloc[0:nrows_return]
            print("Current targets shape:", str(Targets.shape[0]))

    # Targets.to_csv(fname, header=True, index=True)

    return Targets


# def Load_Healthy_Patients():
#     print ("Define me")

def Load_N_Data(data_path, Target_ID, final_folder, Sub_Class_ID, job_name, feat_path, no_symp_code, mode, use_proba,
                no_symp_dict, debug, use_fake_que, visits, nrows_return, n_rows, charac_selected, charac_id, proba_path,
                use_prs, Use_SNPs, Select_Traits_Gen, prs_cols, prs_path, all_test_as_val=False, Thresh_in_Column=0.8,
                Thresh_in_Row=0.7, split=True, train_set=True,use_explicit_columns=False,
                load_saved_data=False):
    """
    :param Target_ID: The UKBB Target ID with -0.0 postfix for example 2443-0.0 for diabetes
    :param final_folder:  A folder for the Data output
    :param Sub_Class_ID:Should be either "All" or the value that signs the subphenotype in the target
    :param job_name:#The desired Job_name for saving the data
    :param feat_path:#File that tells which feature to use and which to exclude: Diabetes_Features.csv,Diabetes_Features_No_Baseline.csv,Baseline_Features.csv,Diabetes_Features_Lifestyle.csv,Diabetes_Features_No_Baseline.csv, Full_Diabetes_Features # "Diabetes_Features.csv","Diabetes_Features.csv","Diabetes_Features.csv",BMI_Features_Lifestyle.csv
    :param no_symp_code:The symbol for not having the phenotype according to UKBB DataBase
    :param mode: #Whether to Use all participants ("A") and not only the returns ("R")
    :param use_proba:Whether or not to use (in mode R) or Calculate the probabilities of the Returnings
    :param no_symp_dict: A dictionary that maps for each phenotype it no_symptom_sign
    :param debug:ste to True if you dont know what to do (True/False)
    :param use_fake_que:True/False
    :param visits:[0,1,2] Which visits to take into account
    :param nrows_return:How MAny of the returning visits to take into account, set to None if not debugging, set to less than 30000 if want to accelerate priocess
    :param n_rows:
    :param charac_selected:
    :param charac_id:
    :param proba_path:
    :param use_prs:Final PRS score for each phenotype for each person, loaded from a prvious calculated table True/False
    :param Select_Traits_Gen:Adding specific SNPs values (top 1000 SNPs that are significant according to PRS)
    :param prs_cols:Adding PRS -Only final score for each phenotype for each user
    :param prs_path:
    :param all_test_as_val:
    :param Thresh_in_Column:
    :param Thresh_in_Row:
    :param split:
    :return:
    """
    if load_saved_data:
        if train_set == True:
            with open(os.path.join(final_folder, job_name + "train_Data"), 'rb') as fp:
                Data=pickle.load(fp)
        else:
            with open(os.path.join(final_folder, job_name + "test_Data"), 'rb') as fp:
                Data=pickle.load(fp)
    else:
        print("mode is:", mode)
        Target_ID = str(Target_ID)  # Medical conditions
        print("Target ID is: ", Target_ID)

        data_cols = pd.read_csv(data_path, index_col="eid", nrows=0)

        # ~~~~~~~~~~~~~~~Loading Targets~~~~~~~~~~~~~~
        print("mode is:", mode)
        if mode == "R":
            # DF_Targets = Load_Targets(Target_ID, Sub_Class_ID,visits,nrows_return,no_symp_dict[Target_ID],
            #                           data_path=data_path)#no_symp_dict[Target_ID] - alist of codes to be identified as no symptom
            # def Load_Targets(Target_ID, Sub_Class_ID, visits, nrows_return, *no_symp_code):
            DF_Targets = pd.read_csv(data_path, usecols=[Target_ID.split("-")[0] + "-3.0", "eid"], index_col="eid")
        elif mode == "A":
            if use_proba:  # Need to calculate probability file
                Returning_targets = Load_Targets(Target_ID, Sub_Class_ID, visits, nrows_return, no_symp_dict[Target_ID],
                                                 data_path=data_path)
                # DF_Targets = Load_Targets(Target_ID, Sub_Class_ID, visits, nrows_return, no_symp_dict[Target_ID])  # no_symp_dict[Target_ID] - alist of codes to be identified as no symptom

                if isinstance(nrows_return, (int, long)):
                    n_rows = nrows_return * 5
                DF_Targets = pd.read_csv(data_path, usecols=[Target_ID, "eid"], index_col='eid', nrows=n_rows)
                DF_Targets = Order_Targets(DF_Targets, no_symp_dict[Target_ID], Sub_Class_ID)
            else:
                DF_Targets = pd.read_csv(data_path, usecols=[Target_ID, "eid"], index_col='eid', nrows=n_rows)
                DF_Targets = Order_Targets(DF_Targets, no_symp_dict[Target_ID], Sub_Class_ID)
        elif mode == "EHR":

            print("mode EHR")
        else:
            print("mode is neither A nor R, Check capital letters")

        print("Load FEATURES of FIRST visit")

        # ~~~~~~~~Features Extraction~~~~~~~~~~~~~~~~
        FEAT_DF = pd.read_csv(feat_path)  # Read Features files
        FEAT_DF = FEAT_DF[FEAT_DF["Exclude"] != 1]
        Use_Columns = [x for x in FEAT_DF["Field ID"]]
        # Use_Columns.extend(["21003-1.0","21003-2.0"])
        # ~~~~~~Using only columns that exist both in the desired Features and the actual Database~~~~~~~~~~~~
        # filtered_cols = list(set(data_cols.columns.values).intersection(Use_Columns))
        if use_explicit_columns:
            filtered_cols = [str(x) for x in data_cols.columns.values for y in Use_Columns if x==y]
        else:
            filtered_cols = [str(x) for x in data_cols.columns.values for y in Use_Columns if str(x).startswith(str(y))]
        Used_Columns = set([str(y) for y in Use_Columns for x in filtered_cols if str(x).startswith(str(y))])
        Non_Used_Columns = [y for y in Use_Columns if y not in Used_Columns]
        # Non_Used_Columns_short = [x.split(sep, 1)[0] for x in Non_Used_Columns]
        if FEAT_DF.shape[0] != 0:
            FEAT_DF.loc[FEAT_DF["Field ID"].isin(Non_Used_Columns), "Wasn't used"] = True
            FEAT_DF[FEAT_DF["Field ID"].isin(Non_Used_Columns)].to_csv(os.path.join(final_folder ,job_name + "_non_used_col.csv"))
        filtered_cols.append("eid")

        Feat_DF_eid = FEAT_DF.set_index(keys="Field ID", drop=True)
        # Excluding non relevant features
        cat_feat = []
        if FEAT_DF.shape[0] != 0:
            cat_feat = FEAT_DF.loc[
                (FEAT_DF['Datatype'] == 'Categorical') & (
                        FEAT_DF["Exclude"] != 1), "Field ID"].values  # All categorical Features
        # cat_feat = [s for s in cat_feat if s in filtered_cols]  # Leaving only features that exist in our database
        cat_feat = [s for s in filtered_cols for y in cat_feat if
                    str(s).startswith(str(y))]  # Leaving only features that exist in our database

        if (FEAT_DF.shape[0] != 0):
            if not use_explicit_columns:
                cat_names = [Feat_DF_eid.loc[y, "Description"] + "_" + x.split("_")[-1] for x in cat_feat for y
                         in Feat_DF_eid.index.values if y.startswith(x.split("_")[0])]
            else:
                cat_names = [Feat_DF_eid.loc[y, "Description"] for x in cat_feat for y
                         in Feat_DF_eid.index.values if y==x]
        else:
            cat_names = []
        print("Loading all Feaures, n_rows:", str(n_rows))
        if "30750-0.0" not in filtered_cols:
            df_Features = pd.read_csv(data_path, index_col="eid", usecols=filtered_cols + ["30750-0.0"], nrows=n_rows)
            filtered_cols = [x for x in filtered_cols if not (x.startswith("30750-") or x == "eid")]
        else:
            df_Features = pd.read_csv(data_path, index_col="eid", usecols=filtered_cols, nrows=n_rows)
            filtered_cols = df_Features.columns.values

        print('df_Features size before filtering nan', df_Features.shape)

        print("Loaded size of Features is: ", df_Features.shape[0])
        mut_ind = list(set(df_Features.index.values).intersection(DF_Targets.index.values))
        df_Features = df_Features.loc[mut_ind, :]

        if (Thresh_in_Column == "any") or Thresh_in_Column == "all":  # Require that many non-NA values.
            df_Features.dropna(axis=1, how=Thresh_in_Column, inplace=True)
        elif Thresh_in_Column > 0:
            df_Features.dropna(axis=1, thresh=Thresh_in_Column, inplace=True)

        if (Thresh_in_Row == "any") or Thresh_in_Row == "all":  # Require that many non-NA values.
            df_Features.dropna(axis=0, how=Thresh_in_Row, inplace=True)
        elif Thresh_in_Row > 0:
            df_Features.dropna(axis=0, thresh=Thresh_in_Row, inplace=True)

        print("Feature and Targets Mutual size of Features is: ", df_Features.shape[1])

        # df_Features.dropna(subset=[df_Features.columns.values[0]], inplace=True)
        df_Features = Filter_CZ(df_Features, charac_selected, charac_id)

        df_Features = df_Features.loc[:, filtered_cols]

        print("size of Features after filter_CZ: ", df_Features.shape[0])

        if df_Features.shape[0] != 0:
            Rel_Feat = [k for k in df_Features.columns.values]
            Rel_Feat = [str(k) for k in Rel_Feat]
            Features_ids = [str(x) for x in Feat_DF_eid.index.values]
            # cat_names = [Feat_DF_eid.loc[y, "Description"] + "_" + x.split("_")[-1] for x in cat_feat for y
            #              in Feat_DF_eid.index.values if y.startswith(x.split("_")[0])]
            # Rel_Feat_Names = [Feat_DF_eid.loc[x.split("_")[0], "Description"] + x.split("-")[-1] for x in Rel_Feat]
            if not use_explicit_columns:
                Rel_Feat_Names = [Feat_DF_eid.loc[y, "Description"] + "_" + x.split("_")[-1] for x in Rel_Feat
                              for y in Feat_DF_eid.index.values if y.startswith(x.split("_")[0])]
            else:
                Rel_Feat_Names = [Feat_DF_eid.loc[y, "Description"] for x in Rel_Feat for y
                         in Feat_DF_eid.index.values if y==x]

            # Rel_Feat_Names = [x.replace("0.0","") for x in Rel_Feat_Names]
            # Rel_Feat_Names = [x.replace("1.0","") for x in Rel_Feat_Names]
            # Rel_Feat_Names = [x.replace("2.0","") for x in Rel_Feat_Names]
            # Rel_Feat_Names = [x.replace("3.0","") for x in Rel_Feat_Names]
        else:
            Rel_Feat_Names = []
            # Rel_Feat_Names = [Feat_DF_eid.loc[s, "Description"] for s in Rel_Feat if s in Features_ids]
            # if Use_imp_flag:
            #     Rel_Feat_Names=Rel_Feat_Names+[x+" imputation flag" for x in Rel_Feat_Names]
        if use_prs:  # Adding PRS -Only final score for each phenotype for each user
            PRS_Columns_DF = pd.read_csv(filepath_or_buffer=prs_path, index_col="eid", nrows=0)
            exist_prs_col = list(PRS_Columns_DF.columns.values)
            prs_use_col = list(set(exist_prs_col).intersection(prs_cols))
            prs_use_col.append("eid")
            print("prs_use_col")
            PRS_DF = pd.read_csv(filepath_or_buffer=prs_path, index_col="eid", usecols=prs_use_col)
            if df_Features.shape[0] != 0:
                how_merge = "inner"
            else:
                how_merge = "outer"
                print("PRS head():")
                print(PRS_DF.head())
            print("PRS columns: ", PRS_DF.columns.values)
            df_Features = PRS_DF.join(df_Features, how=how_merge, on="eid")
            exist_prs_col = PRS_DF.columns.values
            non_used_prs_names = [x for x in prs_cols if x not in prs_use_col]
            print("non used prs columns ", non_used_prs_names)
            df_Features = df_Features.reset_index().drop_duplicates(subset='eid', keep='first').set_index("eid")
            # df_Features.dropna(axis = "columns",thresh=int(0.5*df_Features.shape[0]), inplace=True)
            prs_list = [item for item in PRS_DF.columns.values]
            print("size of Features after PRS: ", df_Features.shape[0])
            Rel_Feat_Names = prs_list + Rel_Feat_Names

        if Use_SNPs:  # Adding specific SNPs values (top 1000 SNPs that are significant according to PRS)
            Genes_col = []
            if df_Features.shape[0] != 0:
                how_merge = "inner"
            else:
                how_merge = "outer"
            for ind, (trait, Genes_file) in enumerate(Select_Traits_Gen.iteritems()):
                print(trait)
                Gen_DF = pd.read_csv(filepath_or_buffer=Genes_file, index_col="eid")
                temp_Gen_col = Gen_DF.columns.values
                # Excl_Col = Gen_DF.columns.values-list(set(df_Features.columns.values).intersection(Gen_DF.columns.values))
                Excl_Col = [x for x in Gen_DF.columns.values if x not in df_Features.columns.values]
                Excl_Col = Excl_Col[0:max(len(Excl_Col), 20)]
                df_Features = df_Features.join(Gen_DF[Excl_Col], how=how_merge, on="eid")
                Genes_col = Genes_col + Excl_Col
                df_Features = df_Features.reset_index().drop_duplicates(subset='eid', keep='first').set_index("eid")
            Rel_Feat_Names = Rel_Feat_Names + Genes_col
            cat_names = list(cat_names) + Genes_col
            cat_names = [x for x in cat_names if x in Rel_Feat_Names]
            # df_Features.dropna(axis="columns", thresh=int(0.5*df_Features.shape[0]), inplace=True)

            print("size of Features after Genes: ", df_Features.shape[0])
        mut_ind = list(set(df_Features.index.values).intersection(DF_Targets.index.values))
        df_Features = df_Features.loc[mut_ind, :]
        DF_Targets = DF_Targets.loc[mut_ind]
        Rel_Feat_Names = list(Rel_Feat_Names)
        cat_names = list(cat_names)

        # X_display = X_display.loc[df_Features.index, :]
        if ((mode == "A") and use_proba):
            RTI = Returning_targets.index.values  # Returning targets index
            ATI = DF_Targets.index.values  # All Target Index
            # test_index = [item for item in RTI if item in ATI]
            test_index = [item for item in RTI if item in ATI]
            train_index = [item for item in ATI if item not in RTI]

            y_test = Returning_targets.loc[test_index].reset_index().drop_duplicates(subset='eid', keep='first').set_index(
                "eid")
            y_train = DF_Targets.loc[train_index, :].reset_index().drop_duplicates(subset='eid', keep='first').set_index(
                "eid")
            X_test = df_Features.loc[test_index, :].reset_index().drop_duplicates(subset='eid', keep='first').set_index(
                "eid")
            X_train = df_Features.loc[train_index, :].reset_index().drop_duplicates(subset='eid', keep='first').set_index(
                "eid")

            # if all_test_as_val:
            #     X_val=X_test
            #     y_val=y_test
            # else:
            X_val, X_train, y_val, y_train = train_test_split(X_train, y_train, train_size=TRAIN_SIZE, random_state=18)
            with open(os.path.join(final_folder , "y_val"), 'wb') as fp:
                pickle.dump(y_test, fp)
            y_val.to_csv(os.path.join(final_folder , "y_val.csv"))

            with open(os.path.join(final_folder , "X_val"), 'wb') as fp:
                pickle.dump(X_test, fp)
            X_val.to_csv(os.path.join(final_folder , "X_val.csv"))

        else:
            Feat_ind = df_Features.index.values
            Tar_ind = DF_Targets.index.values
            mut_ind = [x for x in Tar_ind if x in Feat_ind]
            df_Features = df_Features.loc[mut_ind, :]
            DF_Targets = DF_Targets.loc[mut_ind]
            X_train, X_test, y_train, y_test = train_test_split(df_Features, DF_Targets.loc[mut_ind], train_size=TRAIN_SIZE)

        X_display = X_test.copy()
        X_display.columns = Rel_Feat_Names

        with open(os.path.join(final_folder , "y_test"), 'wb') as fp:
            pickle.dump(y_test, fp)
        y_test.to_csv(os.path.join(final_folder , "y_test.csv"))

        with open(os.path.join(final_folder , "X_test"), 'wb') as fp:
            pickle.dump(X_test, fp)
        X_test.to_csv(os.path.join(final_folder ,"X_test.csv"))

        with open(os.path.join(final_folder , "y_train"), 'wb') as fp:
            pickle.dump(y_train, fp)
        y_train.to_csv(os.path.join(final_folder , "y_train.csv"))

        with open(os.path.join(final_folder ,"X_train"), 'wb') as fp:
            pickle.dump(X_train, fp)
        X_train.to_csv(os.path.join(final_folder , "X_train.csv"))

        with open(os.path.join(final_folder , "cat_names"), 'wb') as fp:
            pickle.dump(cat_names, fp)
        with open(os.path.join(final_folder ,"Rel_Feat_Names"), 'wb') as fp:
            pickle.dump(Rel_Feat_Names, fp)

        if ~os.path.isfile(os.path.join(final_folder , job_name + "_X_display.csv")):
            X_display.to_csv(os.path.join(final_folder , job_name + "_X_display.csv"))
        if ~os.path.isfile(os.path.join(final_folder , job_name + "_df_Features.csv")):
            df_Features.to_csv(os.path.join(final_folder , job_name + "_df_Features.csv"))
        if ~os.path.isfile(os.path.join(final_folder , job_name + "_DF_Targets.csv")):
            DF_Targets.to_csv(os.path.join(final_folder , job_name + "_DF_Targets.csv"))
        if ~os.path.isfile(os.path.join(final_folder , job_name + "_DF_Features_List.csv")):
            FEAT_DF.to_csv(os.path.join(final_folder , job_name + "_DF_Features_List.csv"))

        if use_proba:
            Data = {'df_Features': df_Features, 'DF_Targets': DF_Targets, 'X_display': X_display, 'X_train': X_train,
                    'y_train': y_train, 'y_test': y_test, 'X_test': X_test, 'cat_names': cat_names, 'y_val': y_val,
                    'X_val': X_val,
                    'Rel_Feat_Names': Rel_Feat_Names}
        else:
            Data = {'df_Features': df_Features, 'DF_Targets': DF_Targets, 'X_display': X_display, 'X_train': X_train,
                    'y_train': y_train, 'y_test': y_test, 'X_test': X_test, 'cat_names': cat_names,
                    'Rel_Feat_Names': Rel_Feat_Names}
        if train_set == True:
            with open(os.path.join(final_folder , job_name + "train_Data"), 'wb') as fp:
                pickle.dump(Data, fp)
        else:
            with open(os.path.join(final_folder , job_name + "test_Data"), 'wb') as fp:
                pickle.dump(Data, fp)
    return Data


def Load_Saved_Data(Data_Folder, Lite, mode="A"):
    if mode != "R":
        y_train = pd.read_csv(Data_Folder + "y_train.csv", index_col='eid')
        if isinstance(y_train, pd.DataFrame) or isinstance(y_train, pd.Series):
            y_train = y_train.values.flatten()
        X_train = pd.read_csv(Data_Folder + "X_train.csv", index_col='eid')

    y_test_df = pd.read_csv(Data_Folder + "y_test.csv", index_col='eid')
    if isinstance(y_test_df, pd.DataFrame) or isinstance(y_test_df, pd.Series):
        y_test = y_test_df.values.flatten()
    X_test = pd.read_csv(Data_Folder + "X_test.csv", index_col='eid')

    with open(Data_Folder + "cat_names", 'rb') as fp:
        cat_names = pickle.load(fp)
    with open(Data_Folder + "Rel_Feat_Names", 'rb') as fp:
        Rel_Feat_Names = pickle.load(fp)
    if mode != "R":
        return y_train, X_train, y_test_df, y_test, X_test, cat_names, Rel_Feat_Names
    else:
        return y_test_df, y_test, X_test, cat_names, Rel_Feat_Names


def Predict(SN, parameters, Save_2_folder, job_name, final_folder, n_fold, use_proba=True, Refit_Base_Model_Path=None):
    with open(os.path.join(final_folder , job_name + "train_Data"), 'rb') as fp:
        Data = pickle.load(fp)
    y_train = Data["y_train"].values.flatten()
    X_train = Data["X_train"]
    y_test = Data['y_test'].values.flatten()
    X_test = Data["X_test"]
    cat_names = Data["cat_names"]
    Rel_Feat_Names = Data["Rel_Feat_Names"]
    if use_proba:
        y_val = Data["y_val"].values.flatten()
        X_val = Data["X_val"]

    start_time = time.time()

    params_df = pd.DataFrame.from_dict(parameters, orient="index")

    run_name = str(SN)
    cat_ind = [x for x, name in enumerate(Rel_Feat_Names) if name in cat_names]
    lgb_train = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_ind,
                            feature_name=Rel_Feat_Names, free_raw_data=False)
    # lgb_test= lgb.Dataset(X_val, label=y_val.values.flatten(), reference=lgb_train, categorical_feature=cat_ind,
    #                            free_raw_data=False, feature_name=Rel_Feat_Names)
    lgb_test = lgb.Dataset(X_val, label=y_val, reference=lgb_train, categorical_feature=cat_ind,
                           free_raw_data=False, feature_name=Rel_Feat_Names)
    evals_result = {}  # to record eval results for plotting

    print("---Start training non-shap values of  model : ", str("SN"),
          ", at Predict, at time: %s ---" % datetime.datetime.now())

    # cv_result = lgb.cv(params=parameters, train_set=lgb_train, feature_name=Rel_Feat_Names,nfold=n_fold,
    #                    stratified=True, shuffle=False)
    start_time = time.time()
    parameters_bu = parameters
    print
    "parameters before train:", parameters
    num_trees = int(parameters['num_boost_round'])
    if Refit_Base_Model_Path == None:
        gbm = lgb.train(params=parameters, num_boost_round=num_trees, train_set=lgb_train,
                        valid_sets=[lgb_train, lgb_test], valid_names=["Train", "Valid"],
                        feature_name=Rel_Feat_Names, evals_result=evals_result, verbose_eval=VERBOSE_EVAL)  #
    else:
        Base_gbm = lgb.Booster(model_file=Refit_Base_Model_Path)
        print
        "Num of trees before refit:", str(Base_gbm.num_trees())
        gbm = lgb.train(init_model=Refit_Base_Model_Path, params=parameters,
                        num_boost_round=Base_gbm.num_trees() + num_trees,
                        train_set=lgb_train, valid_sets=[lgb_train, lgb_test], valid_names=["Train", "Valid"],
                        feature_name=Rel_Feat_Names, evals_result=evals_result, verbose_eval=VERBOSE_EVAL)
        print
        "Num of trees After refit:", str(gbm.num_trees())

    gbm.save_model(os.path.join(final_folder , job_name + "_CV_Model_" + run_name + ".txt"))
    print
    "parameters after training:", parameters
    parameters = parameters_bu

    print
    evals_result
    cv_pd = pd.DataFrame([{"SN": run_name, "score": evals_result['Valid'][parameters['metric']][-1]}])
    cv_pd.set_index("SN", drop=True, inplace=True)
    cv_pd.to_csv(os.path.join(Save_2_folder,run_name + '_cv_result.csv'))

    params_df.loc["SN"] = SN
    print
    "Save params_df in predict to: ", Save_2_folder, run_name, "_parameters.csv"
    params_df.to_csv(os.path.join(Save_2_folder , run_name + "_parameters.csv"))


def Load_Prob_Based_Data(Target_ID, final_folder, Data_Folder, Sub_Class_ID, job_name, feat_path, no_symp_code,
                         nrows_return, disease_proba_dict, use_proba, Lite="False", HowHow="left"):
    print("Feat path is: ", feat_path)
    y_test_df, y_test, X_test, cat_names, Rel_Feat_Names = Load_Saved_Data(Data_Folder, Lite, "R")
    df_Features = X_test

    if use_proba:
        for key, value in disease_proba_dict.iteritems():
            print(key)
            if os.path.isfile(value):
                new_proba = pd.read_csv(value, nrows=X_test.shape[0], index_col="eid")
                df_Features = df_Features.join(new_proba, how=HowHow)
                df_Features.columns.values[-1] = key
                Rel_Feat_Names = Rel_Feat_Names + list(df_Features.columns.values[len(Rel_Feat_Names):])
                print("Probability file for ", key, " loaded from:", value)
            else:
                print("Probability file for ", key, " does not exist at: ", value)
        print(Rel_Feat_Names)

    DF_Targets = y_test_df.loc[df_Features.index.values]
    X_display = df_Features.copy()
    X_display.columns = Rel_Feat_Names
    df_Features = df_Features.loc[DF_Targets.index, :]

    print("Load FEATURES of FIRST visit")

    print("size of Features is: ", df_Features.shape[0])
    ###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    with open(os.path.join(final_folder,"y_test"), 'wb') as fp:
        pickle.dump(y_test, fp)
    with open(os.path.join(final_folder,"X_test"), 'wb') as fp:
        pickle.dump(X_test, fp)
    with open(os.path.join(final_folder,"cat_names"), 'wb') as fp:
        pickle.dump(cat_names, fp)
    with open(os.path.join(final_folder,"Rel_Feat_Names"), 'wb') as fp:
        pickle.dump(Rel_Feat_Names, fp)

    if ~os.path.isfile(os.path.join(final_folder,job_name + "_X_display.csv")):
        X_display.to_csv(os.path.join(final_folder,job_name + "_X_display.csv"))
    if ~os.path.isfile(os.path.join( final_folder, job_name + "_df_Features.csv")):
        df_Features.to_csv(os.path.join(final_folder,job_name + "_df_Features.csv"))
    if ~os.path.isfile(os.path.join(final_folder,job_name + "_DF_Targets.csv")):
        DF_Targets.to_csv(os.path.join(final_folder,job_name + "_DF_Targets.csv"))

    Data = {'df_Features': df_Features, 'DF_Targets': DF_Targets, 'X_display': X_display, 'cat_names': cat_names,
            'Rel_Feat_Names': Rel_Feat_Names}

    return Data


def Predict_prob(BN, parameters, X, y, cat_names, Rel_Feat_Names, Save_2_folder, job_name, n_fold, final_folder,
                 Refit_Return_Model_Path, Refit_Returned,batch_size):
    for ind in np.arange(batch_size):
        SN=int(batch_size*BN+ind)
        run_name = str(SN)
        print("Rel_Feat_Names:", Rel_Feat_Names)
        print("started predict prob of SN:", run_name," in BN:", BN)
        path = os.path.join(Save_2_folder,"SN" + run_name + "_parameters.csv")
        start_time = time.time()
        cv_ind = 0
        Test_ind_list = []
        Train_ind_list = []
        skf = StratifiedKFold(n_splits=n_fold, random_state=0, shuffle=False)
        num_of_trees = parameters['num_boost_round']
        if Refit_Returned:  # model to refit
            if Refit_Return_Model_Path.endswith('.txt'):
                lgbs = [Refit_Return_Model_Path]
                sort_lgbs = lgbs
            else:
                lgbs = [f for f in os.listdir(Refit_Return_Model_Path) if f.endswith('_lgbm.txt')]
                sort_lgbs = [x for i in range(5) for x in lgbs if x[-10] == str(i)]
        for train_index, test_index in skf.split(X, y.values.flatten()):
            Test_ind_list.append(test_index)
            Train_ind_list.append(train_index)
            print("in predict_prob ---time is %s ---" % datetime.datetime.now())

            params_df = pd.DataFrame.from_dict(parameters, orient="index")
            cat_ind = [x for x, name in enumerate(Rel_Feat_Names) if name in cat_names]

            y_train = y.iloc[train_index].values.flatten()
            X_train = X.iloc[train_index, :]
            if isinstance(y_train, pd.DataFrame):
                y_train = y_train.values.flatten()

            lgb_train = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_ind,
                                    feature_name=Rel_Feat_Names, free_raw_data=False)

            evals_result = {}  # to record eval results for plotting
            print('Start training in predict_prob...')
            # train
            if not Refit_Returned:  # Case whithut refit
                cv_result = lgb.cv(params=parameters, num_boost_round=num_of_trees, train_set=lgb_train,
                                   feature_name=Rel_Feat_Names, nfold=n_fold, stratified=True)
            elif len(sort_lgbs) == 1:  # Refitting Transfer learning from first visit
                cv_result = lgb.cv(init_model=sort_lgbs[0], params=parameters, num_boost_round=num_of_trees,
                                   train_set=lgb_train,
                                   feature_name=Rel_Feat_Names, nfold=n_fold, stratified=True)
            else:  # Refitting model calculated before on same data (stacking)
                cv_result = lgb.cv(init_model=Refit_Return_Model_Path + sort_lgbs[cv_ind], params=parameters,
                                   num_boost_round=num_of_trees, train_set=lgb_train,
                                   feature_name=Rel_Feat_Names, nfold=n_fold, stratified=True)
            cv_pd = pd.DataFrame.from_dict(cv_result).iloc[-1:, :]
            cv_pd["CV_ind"] = cv_ind
            cv_pd.columns = ["score", 'stdv', "CV_ind"]
            cv_pd.set_index(keys="CV_ind", drop=True, append=False, inplace=True)
            cv_pd.to_csv(os.path.join(Save_2_folder,run_name + '_cv_ind' + str(cv_ind) + '_cv_result.csv'))
            if cv_ind == 0:
                if os.path.isfile(path):
                    params_df_Total = pd.read_csv(path)
                else:
                    params_df_Total = pd.DataFrame()
                if Refit_Returned:  # Case whithut refit
                    if len(sort_lgbs) == 1:
                        df2 = pd.DataFrame(data=[SN, sort_lgbs[0]], index=["SN", "init_model"])
                    else:  # In case of multiple initial models for cross validation
                        df2 = pd.DataFrame(data=[SN, sort_lgbs[cv_ind]], index=["SN", "init_model"])
                else:
                    df2 = pd.DataFrame(data=[SN], index=["SN"])

                params_df = params_df.append(df2)

                params_df_Total = pd.concat([params_df_Total, params_df], axis=1)

                # params_df_Total.loc["init_model",:].iloc[:,SN]=sort_lgbs[0]
                params_df_Total.to_csv(path)

            cv_ind += 1
            print("Cross validation of cv_ind:", str(cv_ind), " of SN: ", str(SN),
                  "took %s seconds ---" % (time.time() - start_time))

        # params_df_Total.to_csv(path)
        with open(os.path.join(final_folder,"Test_ind_list_" + str(SN)), 'wb') as fp:
            pickle.dump(Test_ind_list, fp)
        print
        "finished saving Test_ind_list of:", str(cv_ind)

        with open(os.path.join(final_folder ,"Train_ind_list" + str(SN)), 'wb') as fp:
            pickle.dump(Train_ind_list, fp)
        print
        "finished saving Train_ind_list of:", str(cv_ind)


def AVG_Prob_AUC_APS_per_SN(job_name, Save_2_folder, final_folder, SN):
    path = os.path.join(Save_2_folder, str(SN) + '_cv_ind*_cv_result.csv')
    files = glob.glob(path)
    scores = []
    print
    "in AVG_Prob_AUC_APS_per_SN files:", files
    for name in files:  # 'file' is a builtin type, 'name' is a less-ambiguous variable name.
        scores.append(pd.read_csv(name).iloc[-1:, -3:])
    df = pd.concat(scores)
    df.set_index(keys="CV_ind", drop=True, append=False, inplace=True, verify_integrity=False)
    # df.drop["cv_ind"]
    avg_score = df.loc[:, "score"].mean()
    fname = os.path.join(final_folder,job_name + "_Score_Table.csv")

    if SN == 0:
        avg_score_df = pd.DataFrame()
    else:
        avg_score_df = pd.read_csv(fname, index_col="SN")

    avg_score_df.index.name = "SN"
    avg_score_df.loc[str(SN), "score"] = avg_score
    avg_score_df.to_csv(fname)
    print
    "Scores Data frame at the end of AVG_Prob_AUC_APS_per_SN:", avg_score_df


def Sort_Prob_AUC_APS(job_name, Save_2_folder, final_folder, Target_ID, calc_shap, use_proba, X, y, X_test,
                      y_test,metric, n_folds,pdf_name):
    print ("pdf name in Sort_Prob_AUC_APS is:", pdf_name)

    fname = os.path.join(final_folder,job_name + "_Score_Table.csv")
    df = pd.read_csv(fname)
    df.set_index(keys="SN", drop=True, append=False, inplace=True, verify_integrity=False)
    auc_aps = df

    # Merging the parameters file
    if os.path.isfile(os.path.join(final_folder ,job_name + "_Parameters_Table.csv")):
        frame = pd.read_csv(os.path.join(final_folder,job_name + "_Parameters_Table.csv"))
        frame.set_index(frame.columns[0], drop=True, inplace=True)
        frame.index.name = "parameter"

    else:
        path = os.path.join(Save_2_folder,"SN*_parameters.csv")
        print("in Sort_Prob_AUC_APS, path:", path)
        print("Sort_AUC_APS concatanation path:", path)
        allFiles = glob.glob(path)
        print("allFiles:", allFiles)
        list_ = []
        for file_ in allFiles:
            print(file_)
            df = pd.read_csv(file_, index_col=0)
            df.columns = df.iloc[10]
            list_.append(df)
            # if os.path.isfile(file_):
            #     os.remove(file_)
            # else:  # Show an error ##
            #     print("Error: %s file not found" % file_)
        frame = pd.concat(list_, axis=1)
        frame.index.name = "parameter"
        # frame.columns = frame.loc["SN", :]
        frame.to_csv(os.path.join(final_folder,job_name + "_Parameters_Table.csv"))

    Sort_Prob(auc_aps, job_name, final_folder, Save_2_folder, frame, Target_ID, calc_shap, X, y,
              X_test, y_test, n_folds, pdf_name)


def Sort_Prob(result, job_name, final_folder, Save_2_folder, frame, Target_ID, calc_shap, X, y,
              X_test, y_test, n_folds, pdf_name):
    print
    "pdf name in Sort_Prob is:", pdf_name

    metric = frame.loc["metric"].iloc[0]
    if metric == "auc":
        result_sorted = result.sort_values(by="score", ascending=False)
    else:
        result_sorted = result.sort_values(by="score", ascending=True)

    result_sorted.to_csv(os.path.join(final_folder ,job_name + "_result_sorted.csv"))
    Calc_Final_Proba(metric, Save_2_folder, final_folder, job_name, Target_ID, calc_shap,
                     X, y, X_test, y_test, n_folds, pdf_name)


def Calc_Final_Proba(metric, Save_2_folder, final_folder, job_name, Target_ID, calc_shap, X, y, X_test, y_test, n_folds,
                     pdf_name
                     , Calc_only_shap=False):
    run_name = str(pd.read_csv(os.path.join(final_folder,job_name + "_result_sorted.csv"),
                               index_col=0).index.values[0])

    print
    "pdf name in Calc_Final_Proba is:", pdf_name

    with open(os.path.join(final_folder,"Rel_Feat_Names"), 'rb') as fp:
        Rel_Feat_Names = pickle.load(fp)
    with open(os.path.join(final_folder,"cat_names"), 'rb') as fp:
        cat_names = pickle.load(fp)

    # SHAP_PDF_PATH = final_folder + pdf_name + "_" + str(run_name) + ".pdf"
    #
    # pdf = PdfPages(SHAP_PDF_PATH)
    # ~~~~~~~~~~~~Loading parameters~~~~~~~~~~~~~~
    parameters = pd.read_csv(os.path.join(final_folder,job_name + "_Parameters_Table.csv"),
                             index_col=0)  # Check that we can read params and build the selected model, train it and make all required drawings
    parameters.index_name = "parameter"
    parameters.columns = parameters.loc["SN", :]
    parameters.drop(index="SN", inplace=True)
    if isinstance(parameters, (list,)):
        params_dict = parameters[run_name].to_dict()
    else:
        params_dict = parameters.to_dict()[run_name]

    try:
        Refit_Return_Model_Path = params_dict["init_model"]  # Getting the path for the model to be refitted
        print("Reffiting based on: ", Refit_Return_Model_Path)

    except:
        Refit_Return_Model_Path = None
        print("Not refitting")
    num_trees = int(params_dict['num_boost_round'])

    cat_ind = [x for x, name in enumerate(Rel_Feat_Names) if name in cat_names]

    y_proba_df = pd.DataFrame()
    shap_values_df = pd.DataFrame()
    cv_ind = 0
    y = pd.DataFrame(data=y, index=X.index)
    print
    'Start training of final model of: ', job_name, " at:", datetime.datetime.now()
    if Refit_Return_Model_Path != None:
        if os.path.isdir(Refit_Return_Model_Path):
            lgbs = [f for f in os.listdir(Refit_Return_Model_Path) if f.endswith('_lgbm.txt')]
            sort_lgbs = [x for i in range(5) for x in lgbs if x[-10] == str(i)]
            print
            "Sorted_lgbs:", sort_lgbs
        elif os.path.isfile(Refit_Return_Model_Path):
            lgbs = [Refit_Return_Model_Path]
            sort_lgbs = [Refit_Return_Model_Path]

    params_bu = params_dict
    X_train = X
    y_train = y
    X_val = X_test
    y_val = y_test
    lgb_train = lgb.Dataset(X_train.values, label=y_train.values.flatten(), categorical_feature=cat_ind,
                            feature_name=Rel_Feat_Names, free_raw_data=False)
    lgb_test = lgb.Dataset(X_val.values, label=y_val.values.flatten(), reference=lgb_train, categorical_feature=cat_ind,
                           free_raw_data=False, feature_name=Rel_Feat_Names)

    # evals_result = np.load(Save_2_folder+run_name + '_evals_result.npy')

    evals_result = {}  # to record eval results for plotting
    print
    'Start training non-shap values of final model of: ', job_name, " at:", datetime.datetime.now(), \
    " Fold Number:", str(cv_ind + 1), "out of:", str(n_folds)
    # train
    start_time = time.time()
    if Refit_Return_Model_Path == None:
        gbm = lgb.train(params=params_dict, num_boost_round=num_trees, train_set=lgb_train, valid_sets=[lgb_train],
                        valid_names=["Train"], feature_name=Rel_Feat_Names, evals_result=evals_result,
                        verbose_eval=VERBOSE_EVAL)  #
    else:
        if os.path.isdir(Refit_Return_Model_Path):
            gbm = lgb.train(init_model=Refit_Return_Model_Path + sort_lgbs[cv_ind], params=params_dict,
                            train_set=lgb_train, feature_name=Rel_Feat_Names, verbose_eval=VERBOSE_EVAL)
        elif os.path.isfile(Refit_Return_Model_Path):
            gbm = lgb.train(init_model=Refit_Return_Model_Path, params=params_dict,
                            train_set=lgb_train, feature_name=Rel_Feat_Names, verbose_eval=VERBOSE_EVAL)
        # gbm = lgb.train(init_model=Refit_Return_Model_Path+sort_lgbs[cv_ind], params=params_dict,
        #                 train_set=lgb_train, valid_sets=[lgb_train],
        #             valid_names=["Train"], feature_name=Rel_Feat_Names,
        #             evals_result=evals_result, verbose_eval=VERBOSE_EVAL)
    gbm.save_model(os.path.join(final_folder,"Results_" + job_name + "_" + run_name + "_lgbm.txt"))
    params_dict = params_bu
    if not Calc_only_shap:
        print('Start predicting non-shap values of final model of: ', job_name, " at:", datetime.datetime.now())
        y_proba = gbm.predict(X_val, num_iteration=num_trees, raw_score=False)
        y_proba_tmp_df = pd.DataFrame(data=y_proba, index=y_val.index)
        y_proba_df = pd.concat([y_proba_df, y_proba_tmp_df], axis=0)
        gbm.save_model(os.path.join(final_folder,job_name + run_name + '_lgbm.txt'))

    if calc_shap:
        print("Calculating SHAP Values for:", job_name, " at:", datetime.datetime.now())
        X_val_shap = X_val.sample(np.min([np.max([1000,int(0.2*X_val.shape[0])]),X_val.shape[0]]))
        shap_values = gbm.predict(X_val_shap.values, num_iteration=gbm.best_iteration, pred_contrib=True, num_threads=10)
        shap_values_df = pd.DataFrame(data=shap_values,index=X_val_shap.index,columns=list(X_val_shap.columns.values)+["BiaShap"])
    # Add here the X_Test and Y_Test results

    print
    "Final training of ", job_name, " took %s seconds ---" % (time.time() - start_time)
    if ~Calc_only_shap:
        with open(os.path.join(final_folder,job_name + "_evals_result"), 'wb') as fp:
            pickle.dump(evals_result, fp)
        with open(os.path.join(final_folder,job_name + "y_proba"), 'wb') as fp:
            pickle.dump(y_proba, fp)
        y_proba_df.to_csv(os.path.join(final_folder ,job_name + "_OnlyPROB.csv"))

    if calc_shap:
        shap_values_df.to_csv(os.path.join(final_folder,job_name + "_shap_values.csv"), index=True)

    print('Finished predicting...', job_name, "at time:", datetime.datetime.now())
    if ~Calc_only_shap:
        create_pdf_prob(metric, Save_2_folder, final_folder, job_name, Target_ID, calc_shap, n_folds, y_test, X_test,
                        pdf_name)


def create_pdf_prob(metric, Save_2_folder, final_folder, job_name, Target_ID, calc_shap, n_folds, y_test, X_test,
                    pdf_name):
    print
    "pdf name in create_pdf_prob is:", pdf_name,
    run_name = str(
        pd.read_csv(filepath_or_buffer=os.path.join(final_folder,job_name + "_result_sorted.csv"), index_col=0).iloc[0, 0])
    SHAP_PDF_PATH = os.path.join(final_folder ,pdf_name + "_" + str(run_name) + ".pdf")
    pdf = PdfPages(SHAP_PDF_PATH)
    y_pred_val = []
    y_test_val = []
    evals_result_list = []
    shap_values_list = []

    y_proba_df = pd.read_csv(os.path.join(final_folder ,job_name + "_OnlyPROB.csv"), index_col=0)
    if y_proba_df.shape[1] > 1:
        y_proba_df = y_proba_df.drop(y_proba_df.columns[0], axis=1)

    y_proba = y_proba_df.values.flatten()
    y_test = y_test.values.flatten()
    for i, stat in enumerate(np.isnan(y_test)):
        if ~stat:
            y_pred_val.append(y_proba[i])
            y_test_val.append(y_test[i])

    APS = average_precision_score(y_test_val, y_pred_val)
    AUC = roc_auc_score(y_test_val, y_pred_val)

    # Print ROC Curve
    plot_ROC_curve(y_test_val, y_pred_val, AUC, pdf)

    # Print Precision Recall Curve
    plot_precision_recall(y_test_val, y_pred_val, APS, pdf)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ plot_quantile_curve~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    plot_quantiles_curve(y_test_val, y_pred_val, pdf)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ plot_calibration_curve~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    plot_calibration_curve(y_test_val, y_pred_val, pdf, os.path.join(final_folder ,job_name), Print_Benefit=True)

    with open(os.path.join(final_folder ,"y_pred_val_list"), 'wb') as fp:
        pickle.dump(y_pred_val, fp)
    with open(os.path.join(final_folder ,"y_test_val_list"), 'wb') as fp:
        pickle.dump(y_test_val, fp)

    with open(os.path.join(final_folder,"Rel_Feat_Names"), 'rb') as fp:
        Rel_Feat_Names = pickle.load(fp)
    X_display = X_test.copy()
    X_display.columns = Rel_Feat_Names
    if ~os.path.isfile(os.path.join(final_folder,job_name + "_X_display.csv")):
        X_display.to_csv(os.path.join(final_folder , job_name + "_X_display.csv"))

    if calc_shap:
        if os.path.isfile(os.path.join(final_folder ,job_name + "_shap_values.csv")):
            shap_values_df = pd.read_csv(os.path.join(final_folder , job_name + "_shap_values.csv"),
                                         index_col="eid")
            df = shap_values_df.values
        else:
            print
            "Calculating SHAP Values for:", job_name, " at:", datetime.datetime.now()
            Calc_Final_Proba(metric, Save_2_folder, final_folder, job_name, Target_ID, calc_shap, X_test,
                             y_test
                             , n_folds, pdf_name, Calc_only_shap=True)
        # print "plotting SHAP Values for:", job_name, " at:", datetime.datetime.now()
        # fig = plt.figure(figsize=(40, 30))
        df = pd.read_csv(os.path.join(final_folder, job_name + "_shap_values.csv"),index_col="eid")
        shap_values = df.values
        X_display=X_display.loc[df.index,:]
        # shap.summary_plot(shap_values[:, :X_display.shape[1]], features=X_display, show=False, auto_size_plot=True,
        #                   max_display=30)
        shap.summary_plot(shap_values[:, :X_display.shape[1]], features=X_display, show=False,
                          auto_size_plot=True,
                          max_display=30)
        # if X_display.shape[0]>50000:
        #     shap.summary_plot(shap_values[:50000,:], features=X_display.iloc[:50000,:], show=False, auto_size_plot=False, max_display=30)
        # else:
        # shap.summary_plot(shap_values[:, :X_display.shape[1]], features=X_display, show=False,
        #                   auto_size_plot=True,
        #                   max_display=30)
        ax = plt.gca()
        ax.tick_params(axis="y", labelsize="20")
        plt.savefig(os.path.join(final_folder, "Results_" + job_name + "_" + run_name + 'SHAP.png'),
                    bbox_inches='tight', dpi=DPI)
        plt.close("all")
        img = mpimg.imread(os.path.join(final_folder ,"Results_" + job_name + "_" + run_name + 'SHAP.png'))
        imgplot = plt.imshow(img)
        plt.axis('off')
        fig = plt.gcf()
        pdf.savefig(fig, dpi=DPI)
        plt.close("all")

        # df.to_csv()

        Ordered_X_idx = list(np.argsort(-np.abs(df.iloc[:, :-1]).sum(0)))
        ranks = [np.abs(df.iloc[:, :-1]).sum(0)[ind] for ind in Ordered_X_idx]
        # FEAT_DF = pd.read_csv(final_folder + job_name + "_DF_Features_List.csv", header=0)
        # if "Unnamed: 0" in FEAT_DF.columns.values:
        #     FEAT_DF.drop(columns="Unnamed: 0")
        # FEAT_DF.set_index('Description', drop=True, inplace=True)
        X_Descreption = [X_display.columns.values[ind] for ind in Ordered_X_idx]
        # print X_Descreption
        Sorted_Question = pd.DataFrame(data=X_Descreption)
        # Sorted_Question = FEAT_DF.loc[X_Descreption, :]
        # Sorted_Question.drop_duplicates(keep="first", inplace=True)
        Sorted_Question["Score"] = ranks
        Sorted_Question.to_csv(os.path.join(final_folder , job_name + "_Features_List_Shap_Sorted.csv"))
        print
        "Printing Dependence plot for:", job_name, " at:", datetime.datetime.now()

        for name in Ordered_X_idx[0:NUM_OF_DEP_PLOT]:
            fig = plt.figure(figsize=(20, 20))
            if 'A' in df.columns:
                shap.dependence_plot(name, shap_values[:, :X_display.shape[1]], X_display, show=False,
                                     interaction_index=X_display.columns.get_loc("Diabetes Probabilities"))
            else:
                shap.dependence_plot(name, shap_values[:, :X_display.shape[1]], X_display, show=False)

            plt.savefig(os.path.join(final_folder , "Results_" + job_name + "_" + run_name + 'SHAP.png'), bbox_inches='tight',
                        dpi=DPI)
            plt.close("all")
            img = mpimg.imread(os.path.join(final_folder , "Results_" + job_name + "_" + run_name + 'SHAP.png'))
            imgplot = plt.imshow(img)
            plt.axis('off')
            fig = plt.gcf()
            pdf.savefig(fig, dpi=DPI)
            plt.close("all")

    pdf.close()
    print
    "finished Create_PDF for:", job_name, " at:", datetime.datetime.now()
    print
    "Finished:", job_name
    print
    "saved PDF at:", SHAP_PDF_PATH
