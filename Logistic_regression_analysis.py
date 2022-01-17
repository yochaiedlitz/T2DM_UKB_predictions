import pandas as pd
import numpy as np
import pickle
import os
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, make_scorer,roc_curve, auc,average_precision_score,precision_recall_curve
import matplotlib.pyplot as plt
import matplotlib
import shap
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg
import sys
from addloglevels import sethandlers
import glob
from UKBB_Func_Final import SAVE_FOLDER, RUN_NAME
plt.ion()  # Enables closing of plt figures on the run
sethandlers()  # set handlers for queue_tal jobs
Qworker = '/home/edlitzy/pnp3/lib/queue_tal/qworker.py'
BASIC_FOLDER_NAME=SAVE_FOLDER
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)
LR_folder_name=os.path.join(BASIC_FOLDER_NAME, "LR_comparison") #Folder to save LR comparison results
all_folders=os.listdir(BASIC_FOLDER_NAME)
relevant_folder_names=[x for x in all_folders if not(x.endswith("pdfs") or x.endswith("shap_folder") or x.endswith("imputed"))]

SHAP_FOLDER_PATH=os.path.join(BASIC_FOLDER_NAME,LR_folder_name,"shap_folder")
PDF_FOLDER_PATH=os.path.join(BASIC_FOLDER_NAME,LR_folder_name,"pdfs")
if not os.path.exists(LR_folder_name):
    os.makedirs(LR_folder_name)

if not os.path.exists(SHAP_FOLDER_PATH):
    os.makedirs(SHAP_FOLDER_PATH)

if not os.path.exists(PDF_FOLDER_PATH):
    os.makedirs(PDF_FOLDER_PATH)
Use_Fake_Que=False
if Use_Fake_Que:
    from queue_tal.qp import fakeqp as qp
else:
    from queue_tal.qp import qp
#     # Job_name_Array = ["Q_Vasc", "Q_Diab", "Q_Heart", "Q_Stroke", "Q_Angina", "Q_B_Pressure"]
#     # if ALL_FEATURES:
#     #     FEAT_PATH = ["Diabetes_Features.csv", "Diabetes_Features.csv", "Diabetes_Features.csv", "Diabetes_Features.csv",
#     #                  "Diabetes_Features.csv", "Diabetes_Features.csv"]  # Full Diabetes features
#     # else:
#     #     FEAT_PATH = ["Top_Vasc_Features.csv", "Top_Diabetes_Features.csv", "Top_Vasc_Features.csv", "Top_Vasc_Features.csv",
#     #                  "Top_Vasc_Features.csv",
#     #                  "Top_Vasc_Features.csv"]  # Full Diabetes features
#     # Job_ID = ["6150-0.0", "2443-0.0", "6150-0.0", "6150-0.0", "6150-0.0", "6150-0.0"]
#     # File_Name_Array = ["Vascular_Healthy_Comb.csv", "Diabetes_Healthy_Comb.csv", "Heart_att_Healthy_Comb.csv",
#     #                    "Stroke_Healthy_Comb.csv", "Angina_Healthy_Comb.csv", "Blood_P_Healthy_Comb.csv"]
#     # # No_symp_array = [0, -7,  ]
#     # Sub_Class_array = ["All", "All", 1, 3, 2, 4, ]

# "2443":"Diabetes diagnosed by doctor",
#     1	    Yes
#     0	    No
#     -1	Do not know
#     -3	Prefer not to answer

# "6150":Vascular/heart problems diagnosed by doctor"
#     1 Heart attack
#     2 Angina
#     3 Stroke
#     4 High blood pressure
#     -7 None of the above
#     -3 Prefer not to answer
# "6152":"Blood clot, DVT, bronchitis, emphysema, asthma, rhinitis, eczema, allergy diagnosed by doctor",
#     5 Blood clot in the leg (DVT)
#     7 Blood clot in the lung
#     6 Emphysema/chronic bronchitis
#     8 Asthma
#     9 Hayfever, allergic rhinitis or eczema
#     -7 None of the above
#     -3 Prefer not to answer
# "2453":"Cancer diagnosed by doctor"
#     1 Yes - you will be asked about this later by an interviewer
#     0 No
#     -1 Do not know
#     -3 Prefer not to answer
# "2463":"Fractured/broken bones in last 5 years"
#     1 Yes
#     0 No
#     -1 Do not know
#     -3 Prefer not to answer

# "4041": Gestational Diabetes
#      1	Yes
#      0	No
#     -2	Not applicable
#     -1	Do not know
#     -3	Prefer not to answer

sys.path
# explicitly require this experimental feature
# from sklearn.experimental import enable_iterative_imputer  # noqa
# now you can import normally from sklearn.impute

def roc_auc_score_proba(y_true, proba):
    return roc_auc_score(y_true, proba)

auc_score = make_scorer(roc_auc_score_proba, needs_proba=True)


def standarise_df(df):
# if standarize:
    fit_col=df.columns
    x_std_col=[x for x in fit_col if not x.endswith("_na")]
    x_na_col=[x for x in fit_col if x.endswith("_na")]
    x_train_std=df[x_std_col]
#     print( np.std(x_train_std,axis=0))
    x_train_std=(x_train_std-np.mean(x_train_std,axis=0))/ np.std(x_train_std,axis=0)
#     print x_train_std.loc[:,x_train_std.isna().sum()>0]
    x_train_std_na_col=x_train_std.loc[:,x_train_std.isna().sum()>0].columns.values
    x_train_std.loc[:,x_train_std.isna().sum()>0]=df.loc[:,x_train_std_na_col]
#     print x_train_std.loc[:,x_train_std_na_col]
    x_train_std[x_na_col]=df[x_na_col]
    return x_train_std


def compute_lr(job_name, Basic_folder_name="/home/edlitzy/UKBB_Tree_Runs/For_article/", penalty="l2",
               Prob_HYP_PAR_ITER=200, Choose_N_Fold=3, impute_val_dict={},
               strategy='most_frequent', score=auc_score, standarize=True, impute=False):
    Hyp_Param_Dict_LR_cs = Prob_HYP_PAR_ITER
    final_folder = os.path.join(Basic_folder_name, job_name, "Diabetes_Results")
    train_data_path = os.path.join(final_folder, "Diabetestrain_Data")
    test_data_path = os.path.join(final_folder, "Diabetestest_Data")

    with open(train_data_path, 'rb') as fp:
        train_Data = pickle.load(fp)
    with open(test_data_path, 'rb') as fp:
        test_Data = pickle.load(fp)

    y_train = train_Data["DF_Targets"]
    X_train = train_Data["df_Features"]
    y_test = test_Data["DF_Targets"]
    X_test = test_Data["df_Features"]
    cat_names = test_Data["cat_names"]
    Rel_Feat_Names = test_Data["Rel_Feat_Names"]

    X_train.dropna(how="any", axis=1, inplace=True)
    X_test.dropna(how="any", axis=1, inplace=True)

    X_train.dropna(how="any", axis=0, inplace=True)
    X_test.dropna(how="any", axis=0, inplace=True)

    X_train_fit = X_train
    X_test_fit = X_test

    if standarize:
        X_train_fit = standarise_df(X_train_fit)
        X_test_fit = standarise_df(X_test_fit)

    clf = LogisticRegressionCV(cv=Choose_N_Fold, random_state=None, penalty=penalty,
                               scoring=score, class_weight="balanced")
    clf.fit(X_train_fit, y_train.values.flatten())

    y_proba = clf.predict_proba(X_test_fit)
    return X_train_fit, y_train, X_test_fit, y_test, X_train, X_test, y_proba, clf, Rel_Feat_Names, cat_names
    # imputation of median instead on nan


def pdf_save(pdf,current_figure=[],DPI=200,plot=False):
    if pdf:
        print("saving pdf at plot_roc to: ",pdf)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0,rect=[0, 0.03, 1, 0.95])
        pdf.savefig(current_figure, dpi=DPI)
        plt.close(current_figure)
    if plot:
        plt.show()


def finalise_roc(ax,lw=2,font_size=10):
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=font_size)
    ax.set_ylabel('True Positive Rate', fontsize=font_size)
    ax.tick_params(axis="x", labelsize=font_size - 4)
    ax.tick_params(axis="y", labelsize=font_size - 4)
    ax.legend(loc="best")
    # ax.set_title('Receiver operating curve', fontsize=font_size)
    return ax

def finalise_PR(ax,lw=2,font_size=10):
        ax.set_xlabel('Recall', fontsize=font_size)
        ax.set_ylabel('Precision', fontsize=font_size)
        ax.tick_params(axis="x", labelsize=font_size-4)
        ax.tick_params(axis="y", labelsize=font_size-4)
        ax.legend(loc='best')
        ax.set_ylim([0, 1.01])
        ax.set_xlim([0, 1.])
        # ax.set_title('Precision recall curve', fontsize=font_size )
        return ax

def plot_roc(y, y_prob, pdf=[], legend="ROC", ax=[],color="b"):
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)
    lw = 2
    ax.plot(fpr, tpr, lw=lw, color=color,label=legend + " curve (area = %0.2f)" % roc_auc)
    return ax, roc_auc


def plot_aps(y,y_prob,legend="ROC",ax=[],lw = 2,color="b"):
    precision, recall, _ = precision_recall_curve(y, y_prob)
    aps = average_precision_score(y,y_prob)

    ax.step(recall, precision, where='post',
            label=legend + ' APS={0:0.2f}'.format(aps) + ', Prevalence= {0:0.3f}'.format(precision[0]), lw=lw,
            color=color)
    # axi = ax.twinx()
    # # set limits for shared axis
    # axi.set_ylim(ax.get_ylim())
    # # set ticks for shared axis
    # relative_ticks = []
    # label_format = '%.1f'
    # for tick in ax.get_yticks():
    #     tick = tick / precision[0]
    #     relative_ticks.append(label_format % (tick, ))
    # axi.set_yticklabels(relative_ticks,fontsize=font_size)
    # axi.set_ylabel('Precision fold', fontsize=font_size+4)
    ax.axhline(y=precision[0], color='r', linestyle='--')
    return ax,aps


def Linear_shap(clf,X_train_fit,Rel_Feat_Names,X_test_fit,x_lim=(-10, 10),max_features=10,pdf=[],shap_img_path=[]):
    explainer = shap.LinearExplainer(clf, X_train_fit.values, feature_dependence="independent")
    shap_values = explainer.shap_values(X_test_fit.values).astype(np.double)
    # X_test_array = X_test_fit.values.tolist() # we need to pass a dense version for the plotting functions
    # plt.xlim(x_lim)
    shap.summary_plot(shap_values, X_test_fit.values, feature_names=Rel_Feat_Names,sort=True,max_display=max_features,
                      show=False)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0, rect=[0, 0.03, 1, 0.95])
    plt.savefig(shap_img_path, bbox_inches='tight', dpi=800)
    plt.close("all")
    img = mpimg.imread(shap_img_path)
    plt.imshow(img)
    plt.axis("off")
    fig = plt.gcf()
    pdf_save(pdf=pdf,current_figure=fig)
    plt.close("all")



def linear_features_importance(Rel_Feat_Names,clf,title="Feature importance using Lasso Model",figsize=(12,9),
                               num_feat=10,pdf=[]):
    imp_coef = pd.DataFrame({"Covariates names":Rel_Feat_Names,"Covariate coefficient":clf.coef_.flatten()})
    imp_coef_sorted=imp_coef.sort_values(by="Covariate coefficient",ascending=True)
    imp_coef_sorted.set_index("Covariates names",inplace=True)
    top_feat=list(np.arange(0,num_feat))
    bot_feat=list(np.arange(-num_feat,0))
    tot_feat=top_feat+bot_feat
    top_20_coeff=imp_coef_sorted.iloc[tot_feat,:]
    matplotlib.rcParams['figure.figsize'] = figsize
    top_20_coeff.plot.barh(figsize=figsize)
    #.figure(figsize=figsize)
    ax=plt.gca()
    ax.set_yticklabels(top_20_coeff.index.values,fontsize=12)
    ax.set_xlabel("Covariates coefficients",fontsize=12)
    ax.set_title(title,fontsize=14)
    # plt.ylabel("Covariates names",fontsize=18)
    pdf_save(pdf=pdf,current_figure=plt.gcf())


def plot_quantiles_curve(y_pred_val, y_test_val, test_label="Quantiles", bins=100, low_quantile=0.8, top_quantile=1,
                         figsize=(16, 9), ax=None, pop1=None, pop1_legend=None, font_size=96, color="plasma", alpha=1,
                         legend="Precentile", plot_now=False, pdf=[]):
    vals_df = pd.DataFrame(data={"Y_test": y_test_val, "Y_Pred": y_pred_val})
    res = 1. / bins
    quants_bins = [int(x * 100) / 100. for x in np.arange(low_quantile, top_quantile + res / 2, res)]
    vals_df = vals_df.sort_values("Y_Pred", ascending=False)
    Quants = vals_df.loc[:, "Y_Pred"].quantile(quants_bins)
    Rank = pd.DataFrame()
    for ind, quant in enumerate(Quants.values[:-1]):
        #         print(quant)
        Rank.loc[np.str(ind), "Diagnosed"] = vals_df.loc[((vals_df["Y_Pred"] <= Quants.values[ind + 1]) & \
                                                          (vals_df["Y_Pred"] > quant))].loc[:,
                                             'Y_test'].sum()
        Rank.loc[np.str(ind), "All"] = vals_df.loc[((vals_df["Y_Pred"] > quant) & \
                                                    (vals_df["Y_Pred"] <= Quants.values[ind + 1]))].loc[:,
                                       'Y_test'].count()
        Rank.loc[np.str(ind), "Ratio"] = Rank.loc[np.str(ind), "Diagnosed"] / Rank.loc[np.str(ind), "All"]

    #     fig, ax = plt.subplots(1, 1, figsize=figsize)
    try:
        my_colors = sns.color_palette(color, Rank.shape[0])
    except:
        my_colors = color
    width = 0.8

    x = [item - width / 2 for item in np.arange(len(Rank.index.values))]
    labels = [str(int(100 * item)) for item in np.arange(low_quantile + res, top_quantile + res / 2, res)]
    pop = ax.bar(left=x, height=Rank.loc[:, "Ratio"], width=width, align='center', color=my_colors, tick_label=labels,
                 alpha=alpha)
    ax.set_xlabel("Prediction quantile", fontsize=font_size)
    ax.set_ylabel("Prevalence in quantile", fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size - 6, rotation=70)
    ax.tick_params(axis='both', which='minor', labelsize=font_size - 8, rotation=70)
    if pop1:
        plt.legend([pop, pop1], [legend, pop1_legend], fontsize=font_size)
    return ax, pop

def sort_csv(csvs_path):
    print(csvs_path)
    print((glob.glob(csvs_path + "/*.csv")))
    all_filenames = [i for i in glob.glob(csvs_path + "/*.csv")]  # combine all files in the list
    combined_csv = pd.concat([pd.read_csv(f, index_col=0) for f in all_filenames])
    combined_csv=combined_csv.sort_values(by="LR ROC AUC", ascending=False)
    combined_csv.to_csv(os.path.join(csvs_path,"ranks_csv.csv"))
    # [os.remove(file) for file in all_filenames]
    return combined_csv

def summary_logistics_plots(Basic_folder_name, job_name, pdf_path, strategy='most_frequent',
                            score=auc_score, Prob_HYP_PAR_ITER=100, N_Fold=3):
    # if the variable pdf_path=[], then the functions will not save to PDF baut rather will plot.show()
    data_name = job_name
    shap_img_path=os.path.join(SHAP_FOLDER_PATH,job_name+".png")
    pdf=PdfPages(pdf_path)
    results_df = pd.DataFrame(index=[job_name], columns=["LR ROC AUC", "LR P-R APS", "GBDT ROC AUC", "GBDT P-R APS"])

    X_train_fit, y_train, X_test_fit, y_test, X_train, X_test, y_proba, clf, Rel_Feat_Names, cat_names = compute_lr(
        job_name=job_name, Basic_folder_name=Basic_folder_name, penalty="l2",
        Prob_HYP_PAR_ITER=Prob_HYP_PAR_ITER,
        Choose_N_Fold=N_Fold, impute_val_dict={}, strategy=strategy, score=score)

    with open(os.path.join(Basic_folder_name, job_name, "Diabetes_Results/y_pred_val_list"), 'rb') as fp:
        y_pred_val = pickle.load(fp)
    with open(os.path.join(Basic_folder_name, job_name, "Diabetes_Results/y_test_val_list"), 'rb') as fp:
        y_test_val = pickle.load(fp)

    fig,ax=plt.subplots(2,1)

    ax[0], results_df.loc[job_name, "LR ROC AUC"] = plot_roc(y_test, y_proba[:, 1],legend="LR AUC ",
                                                             ax=ax[0],color="g")
    ax[0], results_df.loc[job_name, "GBDT ROC AUC"] = plot_roc(y_test_val, y_pred_val, pdf=pdf,
                                                               legend="GBDT AUC ", ax=ax[0],color="r")

    ax[0]=finalise_roc(ax[0], lw=2,font_size=10)

    ax[1], results_df.loc[job_name, "LR P-R APS"] = plot_aps(y_test, y_proba[:, 1], legend="LR APS ",
                                                             ax=ax[1],color="r")

    ax[1], results_df.loc[job_name, "GBDT P-R APS"] = plot_aps(y_test_val, y_pred_val,
                                                               legend="GBDT APS ", ax=ax[1],color="g")

    ax[1] = finalise_PR(ax[1],lw=2,font_size=10)

    fig.suptitle("ROC and P-R plots of "+job_name)

    pdf_save(pdf, fig)
    results_df.to_csv(os.path.join(Basic_folder_name, LR_folder_name, job_name + ".csv"))

    Linear_shap(clf=clf, X_train_fit=X_train_fit, Rel_Feat_Names=Rel_Feat_Names, X_test_fit=X_test_fit,
                 x_lim=(-10, 10), pdf=pdf,shap_img_path=shap_img_path)
    linear_features_importance(Rel_Feat_Names, clf, title=data_name + " features importance using Lasso Model",
                               num_feat=5, pdf=pdf)

    fig,ax=plt.subplots(1,1)

    ax, pop = plot_quantiles_curve(y_proba[:, 1], y_test.values.flatten(), test_label="data_name",
                                   bins=100, low_quantile=0.5, top_quantile=1, figsize=(16, 9), font_size=12,
                                   color="yellow", alpha=0.8, legend="LR",plot_now=False,ax=ax)
    ax,pop = plot_quantiles_curve(y_pred_val, y_test_val, test_label="data_name", bins=100, low_quantile=0.5, top_quantile=1,
                         figsize=(16, 9), ax=ax, font_size=12, color="blue", alpha=0.3, legend="GBDT", pop1=pop,
                         pop1_legend="LR", plot_now=True, pdf=pdf)

    ax.set_title('Quantiles comparison', fontsize=12)
    pdf_save(pdf=pdf, current_figure=fig)
    pdf.close()
    print(("PDF save to: ",pdf_path))

def main():
    with qp(jobname="LogReg", q=['himem7.q'], mem_def='4G', trds_def=1, tryrerun=True,max_u=650, delay_batch=30) as q:
        os.chdir('/net/mraid08/export/jafar/Microbiome/Analyses/Edlitzy/tempq_files/')
        q.startpermanentrun()
        tkns = []
        pdf=[]
        for ind,job_name in enumerate(relevant_folder_names):
            pdf_path=os.path.join(PDF_FOLDER_PATH,job_name+".pdf")
            param=(BASIC_FOLDER_NAME, job_name,pdf_path)
            print("job_name:", job_name)
            tkns.append(q.method(summary_logistics_plots, param))
            if ind == (len(relevant_folder_names) - 1):
                print ("Waiting for create_PDF to finish")
                q.waitforresults(tkns)
    results_df=sort_csv(os.path.join(BASIC_FOLDER_NAME, LR_folder_name))
    print(results_df)

if __name__=="__main__":
    sethandlers()
    main()
