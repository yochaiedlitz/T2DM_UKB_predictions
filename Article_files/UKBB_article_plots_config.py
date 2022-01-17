from imports import *

plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
mpl.rcParams['axes.facecolor'] = 'w'  # Or any suitable colour...

UKBB_labels_dict = {"31-0.0": "Sex", "31-0.0_1.0": "Male", "31-0.0_0.0": "Female", '48-0.0': "Waist circumference",
                    "49-0.0": "Hip circumference", "50-0.0": "Height", '189-0.0': "Townsend deprivation index",
                    '904-0.0': "Physical activity >10 minutes (Days)", '924-0.0_3.0': "Brisk walking pace",
                    '4080-0.0': "Systolic blood pressure", '4948-3.0': "Waist-hips ratio",
                    '6150-0.0_-7.0': "No Vascular/heart problems diagnosed",
                    '6153-0.0_1.0': "Taking Cholesterol lowering medication",
                    '6153-0.0_-1.0': 'Do not know if taking specified medication*',
                    '6153-0.0_-7.0': "Not taking specified medication*", '21001-0.0': "BMI", '21002-0.0': 'Weight',
                    '21003-0.0': "Current Age", '21003-3.0': "Age at repeated visit",
                    '21003-4.0': "Time between visits", '20022-0.0': "Birth weight",
                    '20107-0.0_-13.0': "Father's illness - preffer not to answer",
                    "20107-0.0_-11.0": "Father's illness - Do not know", '20107-0.0_3.0': "Lung cancer of father",
                    '20107-0.0_9.0': "Diabetes of father", "20107-0.0_6.0": "Chronic bronchitis/emphysema of father",
                    '20107-0.0_12.0': "Severe depression of father",
                    '20110-0.0_-13.0': "Mother's illness - preffer not to answer",
                    "20110-0.0_-17.0": "Mother's illness - not listed", "20110-0.0_1.0": "Heart disease of mother",
                    '20110-0.0_3.0': "Lung cancer of mother", "20110-0.0_5.0": "Breast cancer of mother",
                    '20110-0.0_8.0': "High blood pressure of mother", '20110-0.0_9.0': "Diabetes of mother",
                    '20110-0.0_12.0': "Severe depression of mother", '20111-0.0_-17.0': 'Siblings illness-not listed',
                    '20111-0.0_-13.0': "Siblings illness-Prefrer to not answer",
                    '20111-0.0-4.0': "Bowel cancer of siblings", '20111-0.0_9.0': "Diabetes of siblings",
                    '20459-0.0_2.0': "Very happy with own health", '20459-0.0_1.0': "Extremely happy with own health",
                    '30000-0.0': "White blood cell (leukocyte) count",
                    '30010-0.0': "Red blood cell (erythrocyte) count", '30030-0.0': "Haematocrit percentage",
                    '30050-0.0': "Mean corpuscular haemoglobin", '30120-0.0': 'Lymphocyte count',
                    '30140-0.0': "Neutrophill count", '30180-0.0': "Lymphocyte percentage",
                    '30200-0.0': "Neutrophill percentage", '30300-0.0': "High light scatter reticulocyte count",
                    '30620-0.0': "Alanine aminotransferase", '30640-0.0': "Apolipoprotein B",
                    '30650-0.0': "Aspartate aminotransferase", "30750-0.0": "Baseline HbA1c%", "30620-0.0": "ALT",
                    "30010-0.0": "Erythrocyte count", "30830-0.0": "SHBG", "30850-0.0": "Testosterone",
                    "30690-0.0": "Cholesterol", "30760-0.0": "HDL Cholesterol"}

final_dict = {
    "eid": "eid",
    "31-0.0": "Sex",

    "21003-0.0": "Age at first visit",
    "21003-3.0": "Age at last visit",

    "2443-0.0": "% Diabetic at first visit",
    "2443-3.0": "% Diabetic at last visit",

    "30750-0.0": "%Hba1c at first visit",
    "30750-3.0": "%Hba1c at last return",

    "21001-0.0": "BMI at first visit",
    "21001-3.0": "BMI at last visit",

    "21002-0.0": "Weight at first visit",
    "21002-3.0": "Weight at last visit",

    "50-0.0": "Height at first visit",
    "50-3.0": "Height at last visit",

    "48-0.0": "Waist circumfurance at first visit",
    "48-3.0": "Waist circumfurance at last visit",

    "49-0.0": "Hips circumferance at first visit",
    "49-3.0": "Hips circumferance at last visit",
    "21003-4.0":"Time between visits"

}
all_data_dict = {
    "eid": "eid",
    "31-0.0": "Sex",

    "21003-0.0": "Age at first visit",
    "21003-1.0": "Age at first visit",
    "21003-2.0": "Age at first visit",

    "2443-0.0": "% Diabetic",
    "2443-1.0": "% Diabetic",
    "2443-2.0": "% Diabetic",

    "30750-0.0": "Hba1c",
    "30750-1.0": "Hba1c",

    "21001-0.0": "BMI",
    "21001-1.0": "BMI",
    "21001-2.0": "BMI",

    "21002-0.0": "Weight",
    "21002-1.0": "Weight",
    "21002-2.0": "Weight",

    "50-0.0": "Height",
    "50-1.0": "Height",
    "50-2.0": "Height",

    "48-0.0": "Waist circumfurance",
    "48-1.0": "Waist circumfurance",
    "48-2.0": "Waist circumfurance",

    "49-0.0": "Hips circumferance",
    "49-1.0": "Hips circumferance",
    "49-2.0": "Hips circumferance"}

encodings_list = ["48", "49", "50", "30750", "21001", "21002", "2443", "21003"]

column_names_list=['%Hba1c at last visit', 'Weight at first visit',
       'Hips circumferance at last visit',
       'Waist circumfurance at last visit', 'Weight at last visit',
       'Height at first visit',
       'Age at last visit', 'Waist circumfurance at first visit',
       'Hips circumferance at first visit', 'Height at last visit',
       'BMI at first visit', '%Hba1c at first visit',"%Hba1c at last return",
       'Age at first visit', 'BMI at last visit','Time between visits']

final_encoding_list = list(final_dict.keys())
final_encoding_list.remove("eid")
final_encoding_list_0 = [x for x in final_encoding_list if x.endswith("0.0")]



