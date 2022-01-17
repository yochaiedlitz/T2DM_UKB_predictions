import collections  # Used for ordered dictionary
Hyp_Param_Dict_A = collections.OrderedDict()
Hyp_Param_Dict_R = collections.OrderedDict()
ALL_TEST_AS_VAL = True

BASIC_JOB_NAME = "C_BF_ATAV_auc"
BASIC_PROB_BASED_JOB_NAME = "RE_"+BASIC_JOB_NAME+"No_Prob"

Job_ID = ["2453-0.0"]  #
  # Data_Job_Names = {"6150-0.0": "Vascular", "2443-0.0": "Diabetes", "2453-0.0": "Cancer", "4041-0.0": "Gestational diabetes","21001-0.0":'BMI'}

FEAT_PATH = ["Full_Baseline_Features.csv"]  #Diabetes_Features_No_Baseline.csv,Baseline_Features.csv,Diabetes_Features_Lifestyle.csv,Diabetes_Features_No_Baseline.csv, Full_Diabetes_Features # "Diabetes_Features.csv","Diabetes_Features.csv","Diabetes_Features.csv",BMI_Features_Lifestyle.csv


CHARAC_SELECTED = {"Age at recruitment": "All", "Sex": "All", "Ethnic background": "All",
                   "Type of special diet followed": "All"}

PROBA_FOLDER = "/net/mraid08/export/jafar/Microbiome/Analyses/Biobank/Proba/Proba_All_Returned/"

DISEASE_PROBA_DICT = {"Diabetes Probabilities": PROBA_FOLDER+"Diabetes_OnlyPROB.csv",
                      "CVD Probabilities": PROBA_FOLDER+"Vascular_OnlyPROB.csv",
                      "Cancer Probabilities": PROBA_FOLDER+"Cancer_OnlyPROB.csv"}


Lite = False #Used for debug
USE_FAKE_QUE = False
DEBUG = False
# MODE = "A"  # "A" all participants in 1st visit, "R" returning visits,"AR"
USE_PROBA = True  # Whether or not to either calculate probability if working on all participants or to use probabilities
#  calculated if working with returning participants
NFOLD = 10
HYP_PAR_ITER = 20
MEM = '5G'
N_THREADS = 10
P_THREADS = 5

Calc_Base_Prob = True
NROWS = None  # 1-500000 or None
CALC_SHAP = True  # Whether or not to calculate the SHAP values for the basic probabilities
Finalize_Only = False


Calc_Prob_Based_Prob = True
RE_USE_PROBA = False
HowHow = "left" #"inner" - take only participants who has probabilities for other disease as well, "left" - take all
NROWS_RETURN = None  # How many returning participants to load
CALC_P_SHAP = True  # Whether or not to calculate the SHAP values for the Preob based predictions
Finalize_Prob_Based_Only = False

VISITS = [0,1,2]
NUM_OF_DEP_PLOT = 10
EARLY_STOPPING_ROUNDS = 100

# CHARAC_SELECTED = {"Age at recruitment": "All", "Sex": "All", "Ethnic background": "All",
#                    "Type of special diet followed": "All"}
CHARAC_ID = {"Age at recruitment": "21022-0.0", "Sex": "31-0.0", "Ethnic background": "21000-0.0",
             "Type of special diet followed": "20086-0.0"}
ETHNIC_CODE = {-3: "Prefer not to answer", -1: "Do not know", 1: "White", 2: "Mixed", 3: "Asian",
               4: "Black or Black British", 5: "Chinese", 6: "Other ethnic group", 1001: "British", 1002: "Irish",
               1003: "Any other white background", 2001: "White and Black Caribbean",
               2002: "White and Black African", 2003: "White and Asian", 2004: "Any other mixed background",
               3001: "Indian", 3002: "Pakistani", 3003: "Bangladeshi", 3004: "Any other Asian background",
               4001: "Caribbean", 4002: "African", 4003: "Any other Black background"}
SEX_CODE = {"Female": 0, "Male": 1}
DIET_CODE = {"Gluten-free": 8, "Lactose-free": 9, "Low calorie": 10, "Vegetarian": 11, "Vegan": 12, "Other": 13}

Job_name_dict = {"6150-0.0": "Vascular", "2443-0.0": "Diabetes", "2453-0.0": "Cancer", "4041-0.0": "Gestational diabetes","21001-0.0":'BMI'}  #,"Diabetes", "Cancer",	"Gestational diabetes","Vascular"

No_symp_dict = {"6150-0.0": -7, "2443-0.0": 0, '2453-0.0': 0, '21001-0.0': "nan"}
Sub_Class_array = ["All"]  # "All",, "All"

Hyp_Param_Dict_A["colsample_bytree"] = [0.25, 0.5, 0.7, 1]
Hyp_Param_Dict_A['is_unbalance'] = [True]
Hyp_Param_Dict_A['objective'] = ['binary']
Hyp_Param_Dict_A['boosting_type'] = ['gbdt']
Hyp_Param_Dict_A['metric'] = ["auc"]  #MAP, aliases: mean_average_precision,kldiv, Kullback-Leibler divergence, aliases: kullback_leibler
Hyp_Param_Dict_A['num_boost_round'] = [1000, 2000, 4000, 8000]
Hyp_Param_Dict_A['verbose'] = [-1]
Hyp_Param_Dict_A['learning_rate'] = [0.005, 0.01, 0.05,0.1]
Hyp_Param_Dict_A["min_child_samples"] = [50, 100, 250]
Hyp_Param_Dict_A["subsample"] = [0.5, 0.7, 0.9, 1]
Hyp_Param_Dict_A["colsample_bytree"] = [0.25, 0.5, 0.7, 1]
Hyp_Param_Dict_A["boost_from_average"] = [True]
Hyp_Param_Dict_A['num_threads'] = [N_THREADS]
Hyp_Param_Dict_A['lambda_l1'] = [0, 0.9, 0.99]
Hyp_Param_Dict_A['lambda_l2'] = [0, 0.9, 0.99]

Hyp_Param_Dict_R["colsample_bytree"] = [0.25, 0.5, 0.7, 1]
Hyp_Param_Dict_R['is_unbalance'] = [True]
Hyp_Param_Dict_R['objective'] = ['binary']
Hyp_Param_Dict_R['boosting_type'] = ['gbdt']
Hyp_Param_Dict_R['metric'] = ["auc"]  #MAP, aliases: mean_average_precision,kldiv, Kullback-Leibler divergence, aliases: kullback_leibler
Hyp_Param_Dict_R['num_boost_round'] = [1000,2000,4000,8000]
Hyp_Param_Dict_R['verbose'] = [-1]
Hyp_Param_Dict_R['learning_rate'] = [0.005, 0.01, 0.05]
Hyp_Param_Dict_R["min_child_samples"] = [10, 25]
Hyp_Param_Dict_R["subsample"] = [0.5, 0.7, 0.9, 1]
Hyp_Param_Dict_R["colsample_bytree"] = [0.25, 0.5, 0.7, 1]
Hyp_Param_Dict_R["boost_from_average"] = [True]
Hyp_Param_Dict_R['num_threads'] = [N_THREADS]
Hyp_Param_Dict_R['lambda_l1'] = [0, 0.9, 0.99]
Hyp_Param_Dict_R['lambda_l2'] = [0, 0.9, 0.99]