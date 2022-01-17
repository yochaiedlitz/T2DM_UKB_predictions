BASIC_JOB_NAME = "dbg_BMI_Prediction_All"
BASIC_PROB_BASED_JOB_NAME = "Re_"+BASIC_JOB_NAME + "No_prob"

Job_ID = ["21001-0.0"]  #
  # Data_Job_Names = {"6150-0.0": "Vascular", "2443-0.0": "Diabetes", "2453-0.0": "Cancer", "4041-0.0": "Gestational diabetes","21001-0.0":'BMI'}
FEAT_PATH = ["BMI_Features.csv"]  #Diabetes_Features_Lifestyle.csv,Diabetes_Features_No_Baseline.csv, Full_Diabetes_Features # "Diabetes_Features.csv","Diabetes_Features.csv","Diabetes_Features.csv","BMI_Features_Lifestyle.csv"

PROBA_FOLDER = "/net/mraid08/export/jafar/Microbiome/Analyses/Biobank/Proba/Proba_All_Returned/"

DISEASE_PROBA_DICT = {"Diabetes Probabilities": PROBA_FOLDER+"Diabetes_OnlyPROB.csv",
                      "CVD Probabilities": PROBA_FOLDER+"Vascular_OnlyPROB.csv",
                      "Cancer Probabilities": PROBA_FOLDER+"Cancer_OnlyPROB.csv"}


Lite=False #Used for debug
USE_FAKE_QUE = True
DEBUG = False
MODE = "R"  # "A" all participants in 1st visit, "Returning" returning visits
USE_PROBA = True  # Whether or not to either calculate probability if working on all participants or to use probabilities
#  calculated if working with returning participants
NFOLD = 5
HYP_PAR_ITER = 4
MEM = '3G'
N_THREADS = 5
P_THREADS = 5

Calc_Base_Prob = False
NROWS = None  # 1-500000 or None
CALC_SHAP = True  # Whether or not to calculate the SHAP values for the basic probabilities
Finalize_Only = False

Calc_Prob_Based_Prob = True
RE_USE_PROBA = False
HowHow = "left" #"inner" - take only participants who has probabilities for other disease as well, "left" - take all
NROWS_RETURN = 5000  # How many returning participants to load
CALC_P_SHAP = True  # Whether or not to calculate the SHAP values for the Preob based predictions
Finalize_Prob_Based_Only = False


# ALL_FEATURES = True  # Use all features all selected
# # RUN_NAME=None #"5"
# # PROB_RUN_NAME=None #"5"

 #Only for Finalize_Only, SN for thye chosen model

VISITS = [0,1,2]
NUM_OF_DEP_PLOT = 2
EARLY_STOPPING_ROUNDS = 100

CHARAC_SELECTED = {"Age at recruitment": "All", "Sex": "All", "Ethnic background": "All",
                   "Type of special diet followed": "All"}
# CHARAC_SELECTED = {"Age at recruitment": "All", "Sex": "All", "Ethnic background": "All",
#                    "Type of special diet followed": "All"}

Job_name_dict = {"6150-0.0": "Vascular", "2443-0.0": "Diabetes", "2453-0.0": "Cancer", "4041-0.0":
    "Gestational diabetes","21001-0.0": 'BMI'}  #,"Diabetes", "Cancer",	"Gestational diabetes","Vascular"

# if ALL_FEATURES:
#     FEAT_PATH = ["Diabetes_Features.csv"]  # Full_Diabetes_Features # "Diabetes_Features.csv","Diabetes_Features.csv","Diabetes_Features.csv"
# else:
#     FEAT_PATH = ["Top_Diabetes_Features.csv", "Top_Diabetes_Features.csv","Top_Diabetes_Features.csv"]  # Full Diabetes features

# File_Name_Array = ["Vascular_Healthy_Comb.csv","Diabetes_Healthy_Comb.csv"]
No_symp_dict = {"6150-0.0": -7, "2443-0.0": 0, '2453-0.0': 0, '21001-0.0': "nan"}
Sub_Class_array = ["All"]  # "All",, "All"
