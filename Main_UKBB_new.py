load_saved_data=False
from S1_Age_n_Sex import *
from UKBB_Functions import *
# from Logistic_regression_analysis import *
import os
import sys
import matplotlib.pyplot as plt
import datetime
from Calc_CI import calc_ci as Main_CI
from CI_Configs import runs
Calc_LR = False
CalcOnlyLR = False
Calc_CI=True
from sklearn.linear_model import LogisticRegressionCV
from LabData import config_global as config
from LabUtils.addloglevels import sethandlers
from LabQueue.qp import fakeqp

"""Script used to calculate probabilities either for Returning participants based on Non-Returning participants model
if mode=="ALL" and USE_PROBA==True (i.e. the returning participants will be test set), otherwise, if mode==Retrurning, will only calculate the Returning participants 
results based on themselves (i.e. part of it will be train and part will be test sets., if mode=="ALL" and 
USE_PROBA==False, will ignore the returning participants """
plt.ion()  # Enables closing of plt figures on the run
NAME=BASIC_PROB_BASED_JOB_NAME[0]
print("NAME:",NAME)
check_run=runs(NAME,only_check_model=True) #Checkss if a model exists
SAVE_FOLDER=check_run.model_paths
TRAIN_PATH=check_run.train_val_file_path
TEST_PATH=check_run.test_file_path

print("Mode:",check_run.mode)
print("TRAIN_PATH:",TRAIN_PATH)
print("TEST_PATH:",TEST_PATH)

try:
    Batch_size=check_run.batch_size
except:
    Batch_size=20.0
# Job_ID = ["2443-0.0"]  #
#   # Data_Job_Names = {"6150-0.0": "Vascular", "2443-0.0": "Diabetes", "2453-0.0": "Cancer", "4041-0.0": "Gestational diabetes","21001-0.0":'BMI'}
#
# FEAT_PATH = ["Diabetes_Features.csv"]  #Diabetes_Features_No_Baseline.csv,Baseline_Features.csv,Diabetes_Features_Lifestyle.csv,Diabetes_Features_No_Baseline.csv, Full_Diabetes_Features # "Diabetes_Features.csv","Diabetes_Features.csv","Diabetes_Features.csv",BMI_Features_Lifestyle.csv
#
#
# BASIC_JOB_NAME = "Diabetes_40_45_Female"
# CHARAC_SELECTED = {"Age at recruitment": 42.5, "Sex": "Female", "Ethnic background": "All",
#                    "Type of special diet followed": "All"}
#
# PROBA_FOLDER = "/net/mraid08/export/jafar/Microbiome/Analyses/Biobank/Proba/Proba_All_Returned/"
#
# DISEASE_PROBA_DICT = {"Diabetes Probabilities": PROBA_FOLDER+"Diabetes_OnlyPROB.csv",
#                       "CVD Probabilities": PROBA_FOLDER+"Vascular_OnlyPROB.csv",
#                       "Cancer Probabilities": PROBA_FOLDER+"Cancer_OnlyPROB.csv"}
#
# BASIC_PROB_BASED_JOB_NAME = "RE_"+BASIC_JOB_NAME + "All_Feat_No_prob"
#
# Lite = False #Used for debug
# USE_FAKE_QUE = True
# DEBUG = False
# MODE = "A"  # "A" all participants in 1st visit, "Returning" returning visits
# USE_PROBA = True  # Whether or not to either calculate probability if working on all participants or to use probabilities
# #  calculated if working with returning participants
# NFOLD = 10
# HYP_PAR_ITER = 15
# MEM = '20G'
# N_THREADS = 2
# P_THREADS = 2
#
# Calc_Base_Prob = False
# NROWS = None  # 1-500000 or None
# CALC_SHAP = True  # Whether or not to calculate the SHAP values for the basic probabilities
# Finalize_Only = False
#
#
# Calc_Prob_Based_Prob = True
# RE_USE_PROBA = False
# HowHow = "left" #"inner" - take only participants who has probabilities for other disease as well, "left" - take all
# NROWS_RETURN = None  # How many returning participants to load
# CALC_P_SHAP = True  # Whether or not to calculate the SHAP values for the Preob based predictions
# Finalize_Prob_Based_Only = False
#
#
# # ALL_FEATURES = True  # Use all features all selected
# # # RUN_NAME=None #"5"
# # # PROB_RUN_NAME=None #"5"
#
#  #Only for Finalize_Only, SN for thye chosen model
#
# VISITS = [0,1,2]
# NUM_OF_DEP_PLOT = 2
# EARLY_STOPPING_ROUNDS = 100
#
#
# # CHARAC_SELECTED = {"Age at recruitment": "All", "Sex": "All", "Ethnic background": "All",
# #                    "Type of special diet followed": "All"}
# CHARAC_ID = {"Age at recruitment": "21022-0.0", "Sex": "31-0.0", "Ethnic background": "21000-0.0",
#              "Type of special diet followed": "20086-0.0"}
# ETHNIC_CODE = {-3: "Prefer not to answer", -1: "Do not know", 1: "White", 2: "Mixed", 3: "Asian",
#                4: "Black or Black British", 5: "Chinese", 6: "Other ethnic group", 1001: "British", 1002: "Irish",
#                1003: "Any other white background", 2001: "White and Black Caribbean",
#                2002: "White and Black African", 2003: "White and Asian", 2004: "Any other mixed background",
#                3001: "Indian", 3002: "Pakistani", 3003: "Bangladeshi", 3004: "Any other Asian background",
#                4001: "Caribbean", 4002: "African", 4003: "Any other Black background"}
# SEX_CODE = {"Female": 0, "Male": 1}
# DIET_CODE = {"Gluten-free": 8, "Lactose-free": 9, "Low calorie": 10, "Vegetarian": 11, "Vegan": 12, "Other": 13}
#
#
# Job_name_dict = {"6150-0.0": "Vascular", "2443-0.0": "Diabetes", "2453-0.0": "Cancer", "4041-0.0": "Gestational diabetes","21001-0.0":'BMI'}  #,"Diabetes", "Cancer",	"Gestational diabetes","Vascular"
#
# # if ALL_FEATURES:
# #     FEAT_PATH = ["Diabetes_Features.csv"]  # Full_Diabetes_Features # "Diabetes_Features.csv","Diabetes_Features.csv","Diabetes_Features.csv"
# # else:
# #     FEAT_PATH = ["Top_Diabetes_Features.csv", "Top_Diabetes_Features.csv","Top_Diabetes_Features.csv"]  # Full Diabetes features
#
# # File_Name_Array = ["Vascular_Healthy_Comb.csv","Diabetes_Healthy_Comb.csv"]
# No_symp_dict = {"6150-0.0": -7, "2443-0.0": 0, '2453-0.0': 0, '21001-0.0': "nan"}
# Sub_Class_array = ["All"]  # "All",, "All"
try:
    use_explicit_columns=USE_EXPLICIT_COLUMNS #Used in FINRISC in order to take only the exact deatures, and not columns starting with the features names
except:
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("!!!!!!!! NOT USING EXPLICIT COLUMN NAMES !!!!!!!")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    use_explicit_columns = False

if USE_FAKE_QUE:
    qp = fakeqp
    print("Running in debug mode!!!!!!!!!!")
    JOB_NAME = BASIC_JOB_NAME
    PROBA_PATH = [PROBA_FOLDER+ x + "/" for x in BASIC_JOB_NAME ] # Folder to store data of probabilities of returened participants according to learning from all other participants
    for x in PROBA_PATH:
        if not os.path.exists(x):
            os.makedirs(x)
elif DEBUG:
    qp = config.qp
    JOB_NAME = ["DBG_" + x for x in  BASIC_JOB_NAME]
    PROBA_PATH = [PROBA_FOLDER+ x + "/" for x in BASIC_JOB_NAME ] # Folder to store data of probabilities of returened participants according to learning from all other participants
    for x in PROBA_PATH:
        if not os.path.exists(x):
            os.makedirs(x)
else:
    qp = config.qp
    #from queue_tal.qp import fakeqp as qp
    JOB_NAME = BASIC_JOB_NAME
    PROBA_PATH = [PROBA_FOLDER+ x + "/" for x in BASIC_JOB_NAME ] # Folder to store data of probabilities of returened participants according to learning from all other participants
    for x in PROBA_PATH:
        if not os.path.exists(x):
            os.makedirs(x)
    # Job_name_Array = ["Q_Vasc", "Q_Diab", "Q_Heart", "Q_Stroke", "Q_Angina", "Q_B_Pressure"]
    # if ALL_FEATURES:
    #     FEAT_PATH = ["Diabetes_Features.csv", "Diabetes_Features.csv", "Diabetes_Features.csv", "Diabetes_Features.csv",
    #                  "Diabetes_Features.csv", "Diabetes_Features.csv"]  # Full Diabetes features
    # else:
    #     FEAT_PATH = ["Top_Vasc_Features.csv", "Top_Diabetes_Features.csv", "Top_Vasc_Features.csv", "Top_Vasc_Features.csv",
    #                  "Top_Vasc_Features.csv",
    #                  "Top_Vasc_Features.csv"]  # Full Diabetes features
    # Job_ID = ["6150-0.0", "2443-0.0", "6150-0.0", "6150-0.0", "6150-0.0", "6150-0.0"]
    # File_Name_Array = ["Vascular_Healthy_Comb.csv", "Diabetes_Healthy_Comb.csv", "Heart_att_Healthy_Comb.csv",
    #                    "Stroke_Healthy_Comb.csv", "Angina_Healthy_Comb.csv", "Blood_P_Healthy_Comb.csv"]
    # # No_symp_array = [0, -7,  ]
    # Sub_Class_array = ["All", "All", 1, 3, 2, 4, ]

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

def upload_jobs(q):
    """Calculate """
    mode_A="A"
    for ind, Target_ID in enumerate(Job_ID):
        Hyp_Param_Dict = Hyp_Param_Dict_A
        no_symp_code = No_symp_dict[Target_ID]
        Sub_Class_ID = Sub_Class_array[ind] #Should be either "All" or the value that signs the subphenotype in the target
        job_name = Job_name_dict[Target_ID]
        #file_name = File_Name_Array[ind]
        feat_path = BASIC_FEAT_PATH+FEAT_PATH[ind]

        print(("Started working on", JOB_NAME[ind]))
        Save_2_folder = + JOB_NAME[ind]+"_" + job_name + "/"
        final_folder = Save_2_folder + job_name + "_Results/"

        if not os.path.exists(Save_2_folder):
            os.makedirs(Save_2_folder)

        if not os.path.exists(final_folder):
            os.makedirs(final_folder)

        waiton = []
        TempPDFsQ = []
        SN = 0  # Serial Number for the file names

        # Data = Load_N_Data(file_name, no_symp_code, Target_ID, final_folder, Sub_Class_ID, job_name)
        if not Finalize_Only:
            Load_N_Data(TRAIN_PATH, Target_ID, final_folder, Sub_Class_ID, job_name, feat_path, no_symp_code, mode_A
                        ,USE_PROBA, No_symp_dict, DEBUG, USE_FAKE_QUE, VISITS, NROWS_RETURN, NROWS,
                        check_run.charac_selected, CHARAC_ID, PROBA_PATH[ind], USE_PRS,Use_SNPs, Select_Traits_Gen,
                        PRS_COLS,PRS_PATH, ALL_TEST_AS_VAL, Thresh_in_Column, Thresh_in_Row, Split,
                        use_explicit_columns=use_explicit_columns,load_saved_data=load_saved_data)

            for SN in np.arange(Basic_HYP_PAR_ITER):
                Params_dict = Choose_params(Hyp_Param_Dict)

                print(("Job Name:", job_name, " SN: ", SN))
                waiton.append(q.method(Predict,(),
                                       {"SN":SN, "Save_2_folder":Save_2_folder, "job_name":job_name,
                                        "final_folder":final_folder,"Choose_N_Fold":Choose_N_Fold,
                                        "USE_PROBA":USE_PROBA,"Refit_Model":Refit_Model,
                                        "Hyp_Param_Dict":Hyp_Param_Dict}))
                if SN == (len(Basic_HYP_PAR_ITER)-1):
                    q.waitforresults(waiton)
        if SORT:
            print(("Finalizing: ", job_name))
            TempPDFsQ.append(q.method(Sort_AUC_APS, tuple(
                [job_name]+[Save_2_folder]+[final_folder]+[Target_ID]+[PROBA_PATH[ind]]+[CALC_SHAP]+[mode_A]+[USE_PROBA] +
                [BASIC_JOB_NAME[ind]]+[Refit_Model])))
            if ind == (len(Job_ID)-1):
                print ("Waiting for TempPDFsQ to finish")
                q.waitforresults(TempPDFsQ)
        elif Finalize_Only:

            TempPDFsQ.append(q.method(create_pdf, tuple(
                [Save_2_folder]+[final_folder]
                +[job_name]+[Target_ID]+[PROBA_PATH[ind]]+[CALC_SHAP]+[mode_A]+[USE_PROBA]+[BASIC_JOB_NAME[ind]]+[Refit_Model])))

            if ind == (len(Job_ID)-1):
                print ("Waiting for create_PDF to finish")
                q.waitforresults(TempPDFsQ)


def upload_prob_jobs(q):
    print(("Feat path is: ", RET_FEAT_PATH))
    R_MODE="R"
    Hyp_Param_Dict = Hyp_Param_Dict_R

    # Hyp_Param_Dict['feval'] = [APS]  #Mean Precision Recall

    # Hyp_Param_Dict['eval_at'] = [[1, 10, 100]]

    for ind, Target_ID in enumerate(Job_ID):
        no_symp_code = No_symp_dict[Target_ID]
        Sub_Class_ID = Sub_Class_array[ind]
        job_name = Job_name_dict[Target_ID]
        #file_name = File_Name_Array[ind]
        feat_path = BASIC_FEAT_PATH+RET_FEAT_PATH[ind]
        print(("Started working on", job_name))
        Save_2_folder = check_run.model_paths+"/"
        final_folder = os.path.join(Save_2_folder, job_name + "_Results/")
        Data_Folder = BASIC_DATA_PATH + JOB_NAME[ind]+"_" + job_name + "/" + job_name + "_Results/" #BASIC_DATA_PATH is where original y_train stored
        if REFIT_SERIAL_MODELS: #Checking wether to refit a model folder just made in previous step, or use a pedefined folder
            pre_trained_model_folder = check_run.model_paths + JOB_NAME[ind] + "_" + job_name+"/" + job_name+"_Results/"
            model=pre_trained_model_folder
        else:
            pre_trained_model_folder = Refit_Return_Model_Path
            model=pre_trained_model_folder
        if not os.path.exists(Save_2_folder):
            os.makedirs(Save_2_folder)
        if not os.path.exists(final_folder):
            os.makedirs(final_folder)
        waiton = []
        TempPDFsQ = []
        SN = 0  # Serial Number for the file names

        # Data = Load_N_Data(file_name, no_symp_code, Target_ID, final_folder, Sub_Class_ID, job_name)
        if not RE_USE_PROBA:
            print("Loading new data")
            Data = Load_N_Data(TRAIN_PATH, Target_ID, final_folder, Sub_Class_ID, job_name, feat_path, no_symp_code, R_MODE,
                           RE_USE_PROBA, No_symp_dict, DEBUG, USE_FAKE_QUE, VISITS, NROWS_RETURN, NROWS,
                           check_run.charac_selected, CHARAC_ID, PROBA_PATH, USE_PRS,Use_SNPs,Select_Traits_Gen, PRS_COLS, PRS_PATH,
                           ALL_TEST_AS_VAL,Thresh_in_Column, Thresh_in_Row,Split,
                               use_explicit_columns=use_explicit_columns,load_saved_data=load_saved_data)

        elif os.path.exists(final_folder) and RE_USE_PROBA:
            Data = Load_Prob_Based_Data(Target_ID, final_folder, Data_Folder, Sub_Class_ID, job_name, feat_path,
                                        no_symp_code, NROWS_RETURN, DISEASE_PROBA_DICT, RE_USE_PROBA, Lite, HowHow)
        elif (not os.path.exists(final_folder)) and RE_USE_PROBA:
            print((final_folder, " required for probability data"))
            sys.exit()

        X = Data["df_Features"]
        y = Data["DF_Targets"]

        if Calc_Transfer_Learning: #If Calculating new model based on model trained on first visit
            Models = [os.path.join(os.path.dirname(pre_trained_model_folder),f)
                      for f in os.listdir(pre_trained_model_folder) if
                      (f.endswith('.txt') and ("CV_Model" in f))]
        BN_range=np.ceil(float(Prob_HYP_PAR_ITER) / Batch_size)
        for BN in np.arange(BN_range):
            print(("Started Job Name:", job_name, " BN: ", BN))
            if Calc_Transfer_Learning:
                model=random.choice(Models) #Each iteration choosing new model for testing
            waiton.append(q.method(Predict_prob,(),
                                   {"BN":BN, "X":X, "y":y, "cat_names":Data['cat_names'],
                                    "Rel_Feat_Names":Data['Rel_Feat_Names'],
                                    "Save_2_folder":Save_2_folder,
                                    "n_fold":Choose_N_Fold, "final_folder":final_folder,
                                    "Refit_Return_Model_Path":Refit_Returned,
                                    "batch_size":Batch_size,"Hyp_Param_Dict":Hyp_Param_Dict,
                                    "Refit_Returned":Refit_Returned}))
            if BN == (BN_range - 1):
                print ("Waiting for create_PDF to finish")
                q.waitforresults(waiton)
        # q.waitforresults(waiton)



        # Sort_Prob_AUC_APS(job_name, Save_2_folder, final_folder, Target_ID, CALC_P_SHAP, USE_PROBA, X, y,
        #                   Hyp_Param_Dict['metric'], NFOLD,BASIC_PROB_BASED_JOB_NAME)


def Finalize_prob_jobs(q):
    print(("Feat path is: ", RET_FEAT_PATH))
    R_MODE="R"
    Hyp_Param_Dict = Hyp_Param_Dict_R
    for ind, Target_ID in enumerate(Job_ID):
        no_symp_code = No_symp_dict[Target_ID]
        Sub_Class_ID = Sub_Class_array[ind]
        job_name = Job_name_dict[Target_ID]
        #file_name = File_Name_Array[ind]
        feat_path = BASIC_FEAT_PATH+RET_FEAT_PATH[ind]
        print(("Finalizing", job_name))

        Save_2_folder = check_run.model_paths + "/"
        final_folder = os.path.join(Save_2_folder, job_name + "_Results/")
        Data_Folder = BASIC_DATA_PATH + JOB_NAME[ind]+"_" + job_name + "/" + job_name + "_Results/" #BASIC_DATA_PATH is where original y_train stored

        if REFIT_SERIAL_MODELS: #Checking wether to refit a model folder just made in previous step, or use a pedefined folder
            pre_trained_model_folder = check_run.model_paths + JOB_NAME[ind] + "_" + job_name+"/" + job_name+"_Results/"
            model=pre_trained_model_folder
        else:
            pre_trained_model_folder = Refit_Return_Model_Path
            model=pre_trained_model_folder
        if not os.path.exists(Save_2_folder):
            os.makedirs(Save_2_folder)
        if not os.path.exists(final_folder):
            os.makedirs(final_folder)
        TempPDFsQ = []
        print("Loading data")
        Data = Load_N_Data(TRAIN_PATH, Target_ID, final_folder, Sub_Class_ID, job_name, feat_path, no_symp_code, R_MODE,
                       RE_USE_PROBA, No_symp_dict, DEBUG, USE_FAKE_QUE, VISITS, NROWS_RETURN, NROWS,
                       check_run.charac_selected, CHARAC_ID, PROBA_PATH, USE_PRS,Use_SNPs,Select_Traits_Gen, PRS_COLS, PRS_PATH,
                       ALL_TEST_AS_VAL,Thresh_in_Column, Thresh_in_Row,Split,
                           use_explicit_columns=use_explicit_columns,load_saved_data=load_saved_data)
        Test_Data = Load_N_Data(TEST_PATH, Target_ID, final_folder, Sub_Class_ID, job_name, feat_path, no_symp_code, R_MODE,
                       RE_USE_PROBA, No_symp_dict, DEBUG, USE_FAKE_QUE, VISITS, NROWS_RETURN, NROWS,
                       check_run.charac_selected, CHARAC_ID, PROBA_PATH, USE_PRS,Use_SNPs,Select_Traits_Gen, PRS_COLS, PRS_PATH,
                       ALL_TEST_AS_VAL,Thresh_in_Column, Thresh_in_Row,Split,train_set=False,
                                use_explicit_columns=use_explicit_columns,load_saved_data=load_saved_data)

        X = Data["df_Features"]
        y = Data["DF_Targets"]

        X_test=Test_Data["df_Features"]
        y_test=Test_Data["DF_Targets"]

        print ("Sorting results")
        if not os.path.isfile(final_folder + job_name + "_Score_Table.csv"):
            for SN in np.arange(Prob_HYP_PAR_ITER):
                AVG_Prob_AUC_APS_per_SN(job_name, Save_2_folder, final_folder, SN)
        TempPDFsQ.append(q.method(Sort_Prob_AUC_APS, tuple([job_name]+[Save_2_folder] + [final_folder]+[Target_ID] +
                         [CALC_P_SHAP] + [USE_PROBA] + [X] + [y] +[X_test]+[y_test]+ [Hyp_Param_Dict['metric']] +
                          [NFOLD]+[BASIC_PROB_BASED_JOB_NAME[ind]])))
        if ind == (len(Job_ID) - 1):
            print ("Waiting for create_PDF to finish")
            q.waitforresults(TempPDFsQ)

        # if Finalize_Prob_Based_Only:
        #     Test_ind_list=[]
        #     skf = StratifiedKFold(n_splits=NFOLD, random_state=None, shuffle=False)
        #     for train_index, test_index in skf.split(X, y.values.flatten()):
        #         Test_ind_list.append(test_index)
        #
        #     Flat_Test_ind_list=np.concatenate(Test_ind_list)
        #
        #     TempPDFsQ.append(q.method(create_pdf_prob, tuple(
        #         [Hyp_Param_Dict['metric']] + [Save_2_folder] + [final_folder]
        #         + [job_name] + [Target_ID] + [CALC_P_SHAP] +
        #         [NFOLD] + [y.iloc[Flat_Test_ind_list]] + [X.iloc[Flat_Test_ind_list, :]] +
        #         [BASIC_PROB_BASED_JOB_NAME[ind]])))
        #
        #
        #     if ind == (len(Job_ID) - 1):
        #         print ("Waiting for create_PDF to finish")
        #         q.waitforresults(TempPDFsQ)

        if Calc_CI:
            Main_CI(BASIC_PROB_BASED_JOB_NAME[ind])

def main():
    # if CALC_PROB:
    sethandlers()
    os.chdir('/net/mraid08/export/jafar/Microbiome/Analyses/Edlitzy/tempq_files/')
    if not CalcOnlyLR:
        if Calc_Base_Prob:
            print(("Working on:" , BASIC_JOB_NAME," at:", datetime.datetime.now()))
            with qp(jobname="GBT_"+check_run.run_name, q=['himem7.q'], _mem_def=MEM, _trds_def=NJOBS*N_THREADS,
                    _tryrerun=True, max_r=650) as q:
                q.startpermanentrun()
                upload_jobs(q)

        if Calc_Prob_Based_Prob:
            if not Finalize_Prob_Based_Only:
                print(("Working on" , BASIC_PROB_BASED_JOB_NAME, " at:", datetime.datetime.now()))
                with qp(jobname="prob_GBT" + check_run.run_name, q=['himem7.q'], _mem_def=MEM,
                        _trds_def=NJOBS * N_THREADS,
                        _tryrerun=True, max_r=650) as q:
                    os.chdir('/net/mraid08/export/jafar/Microbiome/Analyses/Edlitzy/tempq_files/')
                    q.startpermanentrun()
                    upload_prob_jobs(q)
        print(("Finalizing on", BASIC_PROB_BASED_JOB_NAME, " at:", datetime.datetime.now()))
        with qp(jobname="Fin_GBT" + check_run.run_name, q=['himem7.q'], _mem_def=MEM, _trds_def=NJOBS * N_THREADS,
                _tryrerun=True, max_r=650) as q:
            os.chdir('/net/mraid08/export/jafar/Microbiome/Analyses/Edlitzy/tempq_files/')
            q.startpermanentrun()
            Finalize_prob_jobs(q)
    # else:
    #     print("In order to calculate GBDT probabilities - change the CalcOnlyLR variable at the top of Main_UKBB_Final script")
    # if Calc_LR:
    #     all_folders = os.listdir(BASIC_FOLDER_NAME)
    #     all_folders=[]
    #     if not os.path.exists(check_run.model_paths):
    #         os.makedirs(check_run.model_paths)
    #
    #     relevant_folder_names = [x for x in all_folders if
    #                              not (x.endswith("pdfs") or x.endswith("shap_folder") or x.endswith("imputed") or x.endswith("LR_comparison"))]
    #     with qp(jobname="LogReg", q=['himem7.q'], mem_def='5G', trds_def=1, tryrerun=True,max_u=650, delay_batch=20) as q:
    #         os.chdir('/net/mraid08/export/jafar/Microbiome/Analyses/Edlitzy/tempq_files/')
    #         q.startpermanentrun()
    #         tkns = []
    #         pdf=[]
    #         for ind,job_name in enumerate(relevant_folder_names):
    #             pdf_path=os.path.join(PDF_FOLDER_PATH,job_name+".pdf")
    #             param=(BASIC_FOLDER_NAME, job_name,pdf_path)
    #             print ("job_name:", job_name)
    #             tkns.append(q.method(summary_logistics_plots, param))
    #             if ind == (len(relevant_folder_names) - 1):
    #                 print ("Waiting for create_PDF to finish")
    #                 q.waitforresults(tkns)
    #     results_df=sort_csv(LR_folder_name)
    #     print(results_df)
    # if REFIT_SERIAL_MODELS:
    #     print "Working on" , BASIC_TL_JOB_NAME, " at:", datetime.datetime.now()
    #     with qp(jobname=BASIC_TL_JOB_NAME, max_u=400 / (NJOBS * P_THREADS), mem_def=MEM, trds_def=NJOBS * P_THREADS,
    #             q=['himem7.q'],
    #             tryrerun=True) as q:
    #         os.chdir('/net/mraid08/export/jafar/Microbiome/Analyses/Edlitzy/tempq_files/')
    #         q.startpermanentrun()
    #         upload_TL_jobs(q)

if __name__=="__main__":
    main()
