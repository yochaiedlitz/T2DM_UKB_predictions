import pandas as pd
import numpy as np
import os
from . import UKBB_Func
import pickle
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from matplotlib import rcParams as rc
import seaborn as sns
import pylustrator
pylustrator.start()
from . import UKBB_results_summ_funcs as usf
directories_path="/home/edlitzy/UKBB_Tree_Runs/For_article/Singles/" #For adding categories
results_directory_name="Diabetes_Results"
pred_list_name="y_pred_val_list"
test_list_name="y_test_val_list"
all_directories=os.listdir(directories_path)
predictions_files=[os.path.join(directories_path,x,results_directory_name,pred_list_name) for x in all_directories]
y_files=[os.path.join(directories_path,x,results_directory_name,test_list_name) for x in all_directories]
predictions_files
sorted_label_name=["Physical health","Genetics","Non Diabetes Diagnosis","Age and Sex","Diet","Family and Ethnicity",
                  "Anthropometry","Hemoglobin-a1c" ,"Medication","Blood Tests","BP and HR",
                   "Mental Health","Socio demographics","Lifestyle and physical activity","Early Life Factors"]
labels=sorted_label_name
singles_df=pd.DataFrame({"y_files":y_files, "predictions_files":predictions_files,"labels":labels})
singles_df=singles_df.set_index("labels")
singles_df=singles_df.loc[sorted_label_name,:]
for ind, label in enumerate(labels):
    with open(singles_df.predictions_files.values[ind], 'rb') as fp:
        y_pred_val=pickle.load(fp)
    with open(singles_df.y_files.values[ind], 'rb') as fp:
        y_test_val=pickle.load(fp)
    singles_df.loc[label,"AUC"] = roc_auc_score(y_test_val, y_pred_val)
    singles_df.loc[label,"APS"] = average_precision_score(y_test_val, y_pred_val)
singles_df=singles_df.sort_values(by="AUC")

singles_precision_list,singles_recall_list = usf.calc_precision_recall_list(singles_df,sort=False)
singles_fpr_list,singles_tpr_list = usf.calc_roc_list(singles_df,sort=False)
usf.plot_summary_curves(df=singles_df,precision_list=singles_precision_list,recall_list=singles_recall_list,
                    fpr_list=singles_fpr_list,tpr_list=singles_tpr_list,
                    figsize=(30,30),dpi=900,color_palette='plasma',
                    file_name="summary_curves_Ssingles_FigSize30x30_fs_24_lgs_18_nw3_plasma",
                    legend_len=3,font_size=28,legend_size=18,sort=False)


# directories_path="/home/edlitzy/UKBB_Tree_Runs/For_article/Basic_Groups_and_singles" #For seldom categories
directories_path="/home/edlitzy/UKBB_Tree_Runs/For_article/Adding/" #For adding categories
results_directory_name="Diabetes_Results"
pred_list_name="y_pred_val_list"
test_list_name="y_test_val_list"
all_directories=os.listdir(directories_path)
predictions_files=[os.path.join(directories_path,x,results_directory_name,pred_list_name) for x in all_directories]
y_files=[os.path.join(directories_path,x,results_directory_name,test_list_name) for x in all_directories]

sorted_label_name=["Age and Sex","Non Diabetes Diagnosis","Lifestyle and physical activity","Family and Ethnicity",
                  "Diet","Medication","Socio demographics","Mental Health","Blood Tests","Genetics","Anthropometry",
                   "Hemoglobin-a1c","Early Life Factors","Physical health","BP and HR"]
Adding_label_order=['Age and Sex','Hemoglobin-a1c','Blood Tests','Lifestyle and physical activity','BP and HR',
                     'Medication','Physical health','Mental Health', 'Anthropometry','Non Diabetes Diagnosis',
                     'Family and Ethnicity','Socio demographics','Genetics','Early Life Factors','Diet']
labels=sorted_label_name
adding_df=pd.DataFrame({"y_files":y_files, "predictions_files":predictions_files,"labels":sorted_label_name})
adding_df=adding_df.set_index("labels")


# In[44]:


for ind, label in enumerate(labels):
    with open(adding_df.predictions_files.values[ind], 'rb') as fp:
        y_pred_val=pickle.load(fp)
    with open(adding_df.y_files.values[ind], 'rb') as fp:
        y_test_val=pickle.load(fp)
    adding_df.loc[label,"AUC"] = roc_auc_score(y_test_val, y_pred_val)
    adding_df.loc[label,"APS"] = average_precision_score(y_test_val, y_pred_val)
adding_df=adding_df.loc[Adding_label_order,:]


# In[37]:


adding_precision_list,adding_recall_list=usf.calc_precision_recall_list(adding_df,sort=False)
adding_fpr_list,adding_tpr_list=usf.calc_roc_list(adding_df,sort=False)
usf.plot_summary_curves(df=adding_df,precision_list=adding_precision_list,recall_list=adding_recall_list,
                    fpr_list=adding_fpr_list,tpr_list=adding_tpr_list,
                    figsize=(30,30),dpi=900,color_palette='plasma',
                    file_name="summary_curves_Ssingles_FigSize30x30_fs_24_lgs_18_nw3_plasma",
                    legend_len=3,font_size=28,legend_size=18,sort=False)


# ## Run Singles quantiles

# In[300]:


usf.plot_quantiles_curve(test_label="Blood Tests", bins=100,low_quantile=0.8,top_quantile=1,figsize=(20,10))
