BASIC_JOB_NAME = "D_55_60_F_AF"
BASIC_PROB_BASED_JOB_NAME = "RE_"+BASIC_JOB_NAME

Job_ID = ["2443-0.0"]  #
  # Data_Job_Names = {"6150-0.0": "Vascular", "2443-0.0": "Diabetes", "2453-0.0": "Cancer", "4041-0.0": "Gestational diabetes","21001-0.0":'BMI'}

FEAT_PATH = ["Diabetes_Features.csv"]  #Diabetes_Features_No_Baseline.csv,Baseline_Features.csv,Diabetes_Features_Lifestyle.csv,Diabetes_Features_No_Baseline.csv, Full_Diabetes_Features # "Diabetes_Features.csv","Diabetes_Features.csv","Diabetes_Features.csv",BMI_Features_Lifestyle.csv


CHARAC_SELECTED = {"Age at recruitment": 57.5, "Sex": "Female", "Ethnic background": "All",
                   "Type of special diet followed": "All"}

PROBA_FOLDER = "/net/mraid08/export/jafar/Microbiome/Analyses/Biobank/Proba/Proba_All_Returned/"

DISEASE_PROBA_DICT = {"Diabetes Probabilities": PROBA_FOLDER+"Diabetes_OnlyPROB.csv",
                      "CVD Probabilities": PROBA_FOLDER+"Vascular_OnlyPROB.csv",
                      "Cancer Probabilities": PROBA_FOLDER+"Cancer_OnlyPROB.csv"}


Lite = False #Used for debug
USE_FAKE_QUE = False
DEBUG = False
MODE = "A"  # "A" all participants in 1st visit, "Returning" returning visits
USE_PROBA = True  # Whether or not to either calculate probability if working on all participants or to use probabilities
#  calculated if working with returning participants
NFOLD = 10
HYP_PAR_ITER = 15
MEM = '20G'
N_THREADS = 2
P_THREADS = 2

Calc_Base_Prob = False
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
NUM_OF_DEP_PLOT = 2
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