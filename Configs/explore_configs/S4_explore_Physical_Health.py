import collections  # Used for ordered dictionary
from PRS import PRS_sumstats
from UKBB_Functions import PROBA_FOLDER
import sys
Top_Gen_Dict = PRS_sumstats.Get_Top_Gen_Dict()
Hyp_Param_Dict_A = collections.OrderedDict()
Hyp_Param_Dict_R = collections.OrderedDict()
# TRAIN_PATH = '/net/mraid08/export/jafar/UKBioBank/Data/ukb29741_Diabetes_returned_extended_train.csv'
# TEST_PATH = '/net/mraid08/export/jafar/UKBioBank/Data/ukb29741_Diabetes_returned_extended_test.csv'
TRAIN_PATH=Imputed_TRAIN_TEST_PATH = '/net/mraid08/export/jafar/UKBioBank/Data/ukb29741_Diabetes_returned_extended_Imputed_train.csv'
TEST_PATH = '/net/mraid08/export/jafar/UKBioBank/Data/ukb29741_Diabetes_returned_extended_Imputed_test.csv'
# ['Diabetes_all','Age_and_Sex','Anthropometry','Blood_Tests','BP_and_HR',
#  'Diet','Early_Life_Factors','Family_and_Ethnicity','Lifestyle_and_physical_activity','Medication',
#  'Mental_health','Non_Diabetes_Diagnosis','Physical_health','Socio_demographics','HbA1c']
ALL_TEST_AS_VAL = True
BASIC_JOB_NAME = ['Physical_health']#['Mental_health','Non_Diabetes_Diagnosis','Physical_health','Socio_demographics','HbA1c']
BASIC_PROB_BASED_JOB_NAME = ["Val_" + x for x in BASIC_JOB_NAME]
Sub_Class_array = ["All"]  # "All",, "All"
Job_ID = ["2443-0.0"]
RET_FEAT_file_names = BASIC_JOB_NAME

feat_list_folder="Diabetes_Features_lists/For_article/" #Folder where the features lists located
FEAT_file_names = [
    "Diabetes_Features_0705"]  # Diabetes_Features.csv,Diabetes_Features_No_Baseline.csv,Baseline_Features.csv,Diabetes_Features_Lifestyle.csv,Diabetes_Features_No_Baseline.csv, Full_Diabetes_Features # "Diabetes_Features.csv","Diabetes_Features.csv","Diabetes_Features.csv",BMI_Features_Lifestyle.csv
# Features File name without ending
# Features File name without ending

FEAT_PATH = [feat_list_folder + x + ".csv" for x in FEAT_file_names]
RET_FEAT_PATH = [feat_list_folder + x + ".csv" for x in RET_FEAT_file_names]

#
# Data_Job_Names = {"6150-0.0": "Vascular", "2443-0.0": "Diabetes", "2453-0.0": "Cancer", "4041-0.0": "Gestational diabetes","21001-0.0":'BMI'}


CHARAC_SELECTED = {"Age at last visit": "All", "Sex": "All", "Ethnic background": "All",
                   "Type of special diet followed": "All"}

DISEASE_PROBA_DICT = {"Diabetes Probabilities": PROBA_FOLDER + "Diabetes_OnlyPROB.csv",
                      "CVD Probabilities": PROBA_FOLDER + "Vascular_OnlyPROB.csv",
                      "Cancer Probabilities": PROBA_FOLDER + "Cancer_OnlyPROB.csv"}

# PRS_COLS -Adding PRS -Only final score for each phenotype for each user
PRS_COLS = ['PRS_MAGIC_HbA1C', 'PRS_cigs_per_day', 'PRS_MAGIC_Scott_FG', 'PRS_ln_HOMA-IR', 'PRS_MAGIC_Scott_FI',
            'PRS_height', 'PRS_Manning_FI', 'PRS_Leptin_BMI', 'PRS_cardio', 'PRS_triglycerides',
            'PRS_Manning_FG', 'PRS_anorexia', 'PRS_Magic_2hrGlucose', 'PRS_Non_Diabetic_glucose2', 'PRS_ever_smoked',
            'PRS_age_smoke', 'PRS_MAGIC_fastingProinsulin', 'PRS_Leptin_Unadjusted_BMI',
            'PRS_MAGIC_Scott_FI_adjBMI', 'PRS_MAGIC_Scott_2hGlu', 'PRS_glucose_iris', 'PRS_ln_FastingInsulin',
            'PRS_bmi', 'PRS_overweight', 'PRS_hba1c', 'PRS_alzheimer', 'PRS_whr', 'PRS_ln_HOMA-B',
            'PRS_ldl', 'PRS_obesity_class2', 'PRS_obesity_class1', 'PRS_diabetes_BMI_Unadjusted',
            'PRS_Manning_BMI_ADJ_FG', 'PRS_waist', 'PRS_ashtma', 'PRS_HBA1C_ISI', 'PRS_HbA1c_MANTRA',
            'PRS_diabetes_BMI_Adjusted', 'PRS_Heart_Rate', 'PRS_Manning_BMI_ADJ_FI', 'PRS_cholesterol', 'PRS_hdl',
            'PRS_FastingGlucose', 'PRS_hips']

# Select_Top_Traits_Gen_arr_names = ['HbA1c_MANTRA','t2d_mega_meta',"MAGIC_Scott_FG","triglycerides",'Magic_2hrGlucose','Manning_Fasting_Insulin'] #Keep empty if None
Select_Top_Traits_Gen_arr_names = ['HbA1c_MANTRA', 't2d_mega_meta', "MAGIC_Scott_FG", 'Magic_2hrGlucose',
                                   'bmi', 'anorexia', 'cardio', 'hips', 'waist', "overweight", 'obesity_class1',
                                   'obesity_class2',
                                   "ever_smoked", "hdl", "ldl", 'triglycerides', 'cholesterol',
                                   'diabetes_BMI_Unadjusted',
                                   'diabetes_BMI_Adjusted', 'FastingGlucose', 'ln_HOMA-B', 'ln_HOMA-IR',
                                   'ln_FastingInsulin',
                                   'Leptin_BMI', 'Leptin_Unadjusted_BMI', 'Heart_Rate', 'MAGIC_fastingProinsulin',
                                   'MAGIC_Scott_FI_adjBMI', 'MAGIC_Scott_FI', 'MAGIC_HbA1C', 'Manning_FG',
                                   'Manning_BMI_ADJ_FG',
                                   'Manning_Fasting_Insulin', 'Manning_BMI_ADJ_FI', 'HBA1C_ISI']  #
USE_FAKE_QUE = False
NROWS = None  # 1-500000 or None
NROWS_RETURN = None  # How many returning participants to load
Split = True #Wheter or not to split data to train and test, should be false only for final testing
Use_imp_flag=True
Logistic_regression=False #"Should be LR for Linear regression or LGBM for treees"

DEBUG = False
USE_PROBA = True  # Whether or not to either calculate probability if working on all participants or to use probabilities
#  calculated if working with returning participants
USE_PRS = False  # wether to use PRS reults
Use_SNPs = False

NFOLD = 5
Choose_N_Fold = 3  # How many CV to make for the initial Cross validation when choosing the hyperparameters

Basic_HYP_PAR_ITER = 20
Prob_HYP_PAR_ITER = 100
MEM = '30G'
N_THREADS = 10
P_THREADS = 2

Calc_Base_Prob = False
CALC_SHAP = True  # Whether or not to calculate the SHAP values for the basic probabilities
SORT = True  # Used mostly for debugging to activate the SORT_AUC_APS function
# Refit_model - path to model to be refitted in the first visit
Refit_Model = None  # '/net/mraid08/export/jafar/UKBioBank/Yochai/UKBB_Runs/Refit/Refit_BL2AF_Diabetes/Diabetes_Results/Diabetes_shap_model.txt'#None##Name of the model to be refitted or None
# /net/mraid08/export/jafar/Yochai/UKBB_Runs/AF_To_refit2_Diabetes/Diabetes_Results
Finalize_Only = False

Calc_Prob_Based_Prob = True
RE_USE_PROBA = False
Calc_Transfer_Learning = False  # Used when we would like torefit several base models and not a specific model
REFIT_SERIAL_MODELS = False  # #Checking wether to refit a model folder just made in previous step, or use a pedefined folder
# Refit_Return_Model_Path - path to model to be refitted in the first visit
Refit_Return_Model_Path = None  # '/net/mraid08/export/jafar/Yochai/UKBB_Runs/mock_refit/Diabetes_Results/'#'/net/mraid08/export/jafar/UKBioBank/Yochai/UKBB_Runs/Refit/Refit_BL2AF_Diabetes/Diabetes_Results/'#None#
HowHow = "left"  # "inner" - take only participants who has probabilities for other disease as well, "left" - take all
CALC_P_SHAP = True  # Whether or not to calculate the SHAP values for the Preob based predictions
SORT_Prob = True
Finalize_Prob_Based_Only = False

if REFIT_SERIAL_MODELS or Refit_Return_Model_Path:
    Refit_Returned = True
else:
    Refit_Returned = False

VISITS = [0, 1, 2]  # [0,1,2]
NUM_OF_DEP_PLOT = 10

Lite = False  # Used for debug

Thresh_in_Column = 0.7
Thresh_in_Row = 0.7

# CHARAC_SELECTED = {"Age at last visit": "All", "Sex": "All", "Ethnic background": "All",
#                    "Type of special diet followed": "All"}
CHARAC_ID = {"Age at last visit": "21022-0.0", "Sex": "31-0.0", "Ethnic background": "21000-0.0",
             "Type of special diet followed": "20086-0.0"}
ETHNIC_CODE = {-3: "Prefer not to answer", -1: "Do not know", 1: "White", 2: "Mixed", 3: "Asian",
               4: "Black or Black British", 5: "Chinese", 6: "Other ethnic group", 1001: "British", 1002: "Irish",
               1003: "Any other white background", 2001: "White and Black Caribbean",
               2002: "White and Black African", 2003: "White and Asian", 2004: "Any other mixed background",
               3001: "Indian", 3002: "Pakistani", 3003: "Bangladeshi", 3004: "Any other Asian background",
               4001: "Caribbean", 4002: "African", 4003: "Any other Black background"}
SEX_CODE = {"Female": 0, "Male": 1}
DIET_CODE = {"Gluten-free": 8, "Lactose-free": 9, "Low calorie": 10, "Vegetarian": 11, "Vegan": 12, "Other": 13}

Job_name_dict = {"6150-0.0": "Vascular", "2443-0.0": "Diabetes", "2453-0.0": "Cancer",
                 "4041-0.0": "Gestational diabetes",
                 "21001-0.0": 'BMI'}  # ,"Diabetes", "Cancer",	"Gestational diabetes","Vascular"

No_symp_dict = {"6150-0.0": -7, "2443-0.0": 0, '2453-0.0': 0, '21001-0.0': "nan"}

# Hyp_Param_Dict_A['max_depth']=[2,4,8,16]
Hyp_Param_Dict_A['num_leaves'] = [4, 8, 16, 32, 64, 128, 256]
Hyp_Param_Dict_A['is_unbalance'] = [True]
Hyp_Param_Dict_A['objective'] = ['binary']
Hyp_Param_Dict_A['boosting_type'] = ['gbdt']  # ,'rf','dart','goss'
Hyp_Param_Dict_A['metric'] = ["auc"]  # MAP, aliases: mean_average_precision,kldiv, Kullback-Leibler divergence, aliases: kullback_leibler
Hyp_Param_Dict_A['num_boost_round'] = [10, 50, 100, 250, 500, 1000]  # ,1000, 2000, 4000, 8000
Hyp_Param_Dict_A['learning_rate'] = [0.005, 0.01, 0.05, 0.1]
Hyp_Param_Dict_A["min_child_samples"] = [10, 25, 50, 250, 500]
Hyp_Param_Dict_A["subsample"] = [0.1, 0.25, 0.5, 0.7, 0.9, 1]
Hyp_Param_Dict_A["colsample_bytree"] = [0.03, 0.1, 0.25, 0.5, 0.7, 1]
Hyp_Param_Dict_A["boost_from_average"] = [True]
Hyp_Param_Dict_A['num_threads'] = [N_THREADS]
Hyp_Param_Dict_A['lambda_l1'] = [0, 0.5, 0.9, 0.99, 0.999]
Hyp_Param_Dict_A['lambda_l2'] = [0, 0.5, 0.9, 0.99, 0.999]
Hyp_Param_Dict_A['bagging_freq'] = [0, 1, 5]
Hyp_Param_Dict_A['bagging_fraction'] = [0.25, 0.5, 0.75, 1]

# Hyp_Param_Dict_R['max_depth']=[2,4,8,16]
Hyp_Param_Dict_A['num_leaves'] = [2, 4, 8, 16, 32, 64, 128]
Hyp_Param_Dict_R['is_unbalance'] = [True]
Hyp_Param_Dict_R['objective'] = ['binary']
Hyp_Param_Dict_R['boosting_type'] = ['gbdt']
Hyp_Param_Dict_R['metric'] = [
    "auc"]  # MAP, aliases: mean_average_precision,kldiv, Kullback-Leibler divergence, aliases: kullback_leibler
Hyp_Param_Dict_R['num_boost_round'] = [50, 100, 250, 500, 1000]  # ,,1000, 2000, 4000, 8000
Hyp_Param_Dict_R['verbose'] = [-1]
Hyp_Param_Dict_R['learning_rate'] = [0.005, 0.01, 0.05]
Hyp_Param_Dict_R["min_child_samples"] = [5, 10, 25, 50]
Hyp_Param_Dict_R["subsample"] = [0.5, 0.7, 0.9, 1]
Hyp_Param_Dict_R["colsample_bytree"] = [0.01, 0.05, 0.1, 0.25, 0.5, 0.7, 1]
Hyp_Param_Dict_R["boost_from_average"] = [True]
Hyp_Param_Dict_R['num_threads'] = [P_THREADS]
Hyp_Param_Dict_R['lambda_l1'] = [0, 0.25, 0.5, 0.9, 0.99, 0.999]
Hyp_Param_Dict_R['lambda_l2'] = [0, 0.25, 0.5, 0.9, 0.99, 0.999]
Hyp_Param_Dict_A['bagging_freq'] = [0, 1, 5]
Hyp_Param_Dict_A['bagging_fraction'] = [0.5, 0.75, 1]

Select_Traits_Gen = {}
for name in Select_Top_Traits_Gen_arr_names:
    Select_Traits_Gen[name] = Top_Gen_Dict[name]

if (len(BASIC_JOB_NAME) != len(Sub_Class_array) or (len(BASIC_JOB_NAME) != len(Sub_Class_array)) or
        (len(BASIC_JOB_NAME) != len(Job_ID))):
    sys.exit("BASIC_JOB_NAME,Sub_Class_array and Job_ID should be same size")