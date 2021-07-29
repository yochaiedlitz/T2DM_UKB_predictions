from UKBB_article_plots_config import *
ALL_PATH='/net/mraid08/export/jafar/UKBioBank/Data/ukb29741.csv'
# ONLY_TRAIN_PATH = '/net/mraid08/export/jafar/UKBioBank/Data/ukb29741_Diabetes_returned_extended_Imputed_train'
# ONLY_VAL_PATH='/net/mraid08/export/jafar/UKBioBank/Data/ukb29741_Diabetes_returned_extended_Imputed_test.csv'
tables_path="/home/edlitzy/UKBB_Tree_Runs/For_article/Tables/Statistics"
returned_extended_path="/net/mraid08/export/jafar/UKBioBank/Data/ukb29741_Diabetes_returned_extended_train.csv"
returned_extended=pd.read_csv(returned_extended_path,index_col="eid")
final_encoding_list=final_dict.keys()
final_encoding_list.remove("eid")
final_encoding_list_0=[x for x in final_encoding_list if x.endswith("0.0")]

final_dict["21003-4.0"]="Time between visits"

# ## Describe all Data


All_data_raw=pd.read_csv(ALL_PATH,usecols=all_data_dict.keys(),index_col="eid")
All_data=All_data_raw.copy()
All_data.loc[:,["30750-0.0","30750-1.0"]]=All_data.loc[:,["30750-0.0","30750-1.0"]].apply(convert_hba1c_mmol_mol_2_percentage,axis=1)

df_all_describe=All_data[final_encoding_list_0].describe()
df_all_describe.rename(final_dict,inplace=True,axis=1)

df_all_describe.to_csv(os.path.join(tables_path,"All_data_Stats.csv"),index=True)
descrebtion_names_dict={}
descrebtion_names_dict["All"]=df_all_describe.copy()

TRAIN_Val_PATH = '/net/mraid08/export/jafar/UKBioBank/Data/ukb29741_Diabetes_returned_extended_Imputed_train_test.csv'
TRAIN_PATH = '/net/mraid08/export/jafar/UKBioBank/Data/ukb29741_Diabetes_returned_extended_Imputed_train.csv'
VAL_PATH='/net/mraid08/export/jafar/UKBioBank/Data/ukb29741_Diabetes_returned_extended_Imputed_test.csv'
TEST_PATH = '/net/mraid08/export/jafar/UKBioBank/Data/ukb29741_Diabetes_returned_extended_Imputed_val.csv'

test_index=pd.read_csv(TEST_PATH,usecols=["eid"],index_col="eid")
test_data=All_data.loc[test_index.index]
test_data=extract_latest_values(test_data,encodings_list=encodings_list)
test_data_index=test_data.loc[:,"2443-3.0"].dropna().index
test_data_clean=test_data.loc[test_data_index,:]
test_data_final=test_data_clean.loc[:,final_encoding_list]
test_data_final["21003-4.0"]=test_data_final["21003-3.0"]-test_data_final["21003-0.0"]
# test_data_final.loc[:,["30750-0.0","30750-3.0"]]=test_data_final.loc[:,["30750-0.0","30750-3.0"]].apply(convert_hba1c_mmol_mol_2_percentage,axis=1)
test_data_final.rename(final_dict,inplace=True,axis=1)
df_test_describe=test_data_final.describe()
descrebtion_names_dict["test"]=df_test_describe.copy()
df_test_describe.to_csv(os.path.join(tables_path,"Test_data_Stats.csv"),index=True)
TRAIN_Val_index=pd.read_csv(TRAIN_Val_PATH,usecols=["eid"],index_col="eid")
TRAIN_Val_data=All_data.loc[TRAIN_Val_index.index]
TRAIN_Val_data=extract_latest_values(TRAIN_Val_data,encodings_list=encodings_list)
TRAIN_Val_data_index=TRAIN_Val_data.loc[:,"2443-3.0"].dropna().index
TRAIN_Val_data_clean=TRAIN_Val_data.loc[TRAIN_Val_data_index,:]
TRAIN_Val_data_final=TRAIN_Val_data_clean.loc[:,final_encoding_list]
TRAIN_Val_data_final["21003-4.0"]=TRAIN_Val_data_final["21003-3.0"]-TRAIN_Val_data_final["21003-0.0"]
TRAIN_Val_data_final.rename(final_dict,inplace=True,axis=1)
df_TRAIN_Val_describe=TRAIN_Val_data_final.describe()
df_TRAIN_Val_describe.to_csv(os.path.join(tables_path,"TRAIN_Val_data_Stats.csv"),index=True)
df_TRAIN_Val_describe
descrebtion_names_dict["train_val"]=df_TRAIN_Val_describe.copy()
TRAIN_index=pd.read_csv(TRAIN_PATH,usecols=["eid"],index_col="eid")
TRAIN_data=All_data.loc[TRAIN_index.index]
TRAIN_data=extract_latest_values(TRAIN_data,encodings_list=encodings_list)
TRAIN_data_index=TRAIN_data.loc[:,"2443-3.0"].dropna().index
TRAIN_data_clean=TRAIN_data.loc[TRAIN_data_index,:]
TRAIN_data_final=TRAIN_data_clean.loc[:,final_encoding_list]
TRAIN_data_final["21003-4.0"]=TRAIN_data_final["21003-3.0"]-TRAIN_data_final["21003-0.0"]
# TRAIN_data_final.loc[:,["30750-0.0","30750-3.0"]]=TRAIN_data_final.loc[:,["30750-0.0","30750-3.0"]].apply(convert_hba1c_mmol_mol_2_percentage,axis=1)
TRAIN_data_final.rename(final_dict,inplace=True,axis=1)
df_TRAIN_describe=TRAIN_data_final.describe()
df_TRAIN_describe.to_csv(os.path.join(tables_path,"TRAIN_data_Stats.csv"),index=True)
df_TRAIN_describe
descrebtion_names_dict["train"]=df_TRAIN_describe
val_index=pd.read_csv(VAL_PATH,usecols=["eid"],index_col="eid")
val_data=All_data.loc[val_index.index]
val_data=extract_latest_values(val_data,encodings_list=encodings_list)
val_data_index=val_data.loc[:,"2443-3.0"].dropna().index
val_data_clean=val_data.loc[val_data_index,:]
val_data_final=val_data_clean.loc[:,final_encoding_list]
val_data_final["21003-4.0"]=val_data_final["21003-3.0"]-val_data_final["21003-0.0"]
# val_data_final.loc[:,["30750-0.0","30750-3.0"]]=val_data_final.loc[:,["30750-0.0","30750-3.0"]].apply(convert_hba1c_mmol_mol_2_percentage,axis=1)
val_data_final.rename(final_dict,inplace=True,axis=1)
df_val_describe=val_data_final.describe()
df_val_describe.to_csv(os.path.join(tables_path,"val_data_Stats.csv"),index=True)
descrebtion_names_dict["val"]=df_val_describe.copy()
all_returned_final=pd.concat([TRAIN_Val_data_final,test_data_final])
df_returned_describe=all_returned_final.describe()
df_returned_describe.to_csv(os.path.join(tables_path,"Returned_data_Stats.csv"),index=True)
descrebtion_names_dict["returned"]=df_returned_describe.copy()
stat_summary_table=pd.DataFrame(index=descrebtion_names_dict.keys())
descrebtion_names_dict["returned"].columns.values
for df_i in descrebtion_names_dict.keys():
    stat_summary_table.loc[df_i,"Number of participants"]=str(int(
        descrebtion_names_dict[df_i].loc["count","Sex"]))
    stat_summary_table.loc[df_i,"%Males"]=(
        int(1000*descrebtion_names_dict[df_i].loc["mean","Sex"]))/10.
    stat_summary_table.loc[df_i,"% Diabetic at first visit"]=np.round(100*descrebtion_names_dict[df_i].loc[
        "mean","% Diabetic at first visit"],decimals=1)
    if stat_summary_table.loc[df_i,"% Diabetic at first visit"]==0:
        stat_summary_table.loc[df_i,"% Diabetic at first visit"]="0"
    try:
        stat_summary_table.loc[df_i,"% Diabetic at last visit"]=np.round(100*descrebtion_names_dict[df_i].loc[
        "mean","% Diabetic at last visit"],decimals=2)
    except:
        stat_summary_table.loc[df_i,"% Diabetic at last visit"]="-"
    for param in column_names_list:
        try:
            stat_summary_table.loc[df_i,param]=str(
                np.round(descrebtion_names_dict[df_i].loc["mean",param],
                         decimals=1))+u"\u00B1"+str(
                np.round(descrebtion_names_dict[df_i].loc["std",param],
                         decimals=1))
        except:
            stat_summary_table.loc[df_i,param]="-"
stat_summary_table_T=stat_summary_table.T
stat_summary_table_T.to_excel(os.path.join(tables_path,"Stats_Summary.xls"))

columns=pd.read_csv(ALL_PATH,nrows=0).columns.values
rel_ids=["20003-0","6177-0","6153-0"]
metformin=1140884600
Insulin=3
metformin_columns=[x for x in columns for y in rel_ids if x.startswith("20003-0")]
Insulin_columns=[x for x in columns for y in rel_ids if x.startswith("6153-0") or x.startswith("6177-0")]
rel_columns=metformin_columns+Insulin_columns


All_disease_data=pd.read_csv(ALL_PATH,usecols=rel_columns+["eid"],index_col="eid")

df_healthy_train_val,train_test_ill,old_train_val_df=return_clean_data(df_to_filter_path=TRAIN_PATH,All_disease_data_df=All_disease_data)

df_healthy_test,test_ill,old_test_df=return_clean_data(df_to_filter_path=TEST_PATH,All_disease_data_df=All_disease_data)

df_healthy_only_val,only_val_ill,old_val_df=return_clean_data(df_to_filter_path=VAL_PATH,All_disease_data_df=All_disease_data)

df_healthy_only_train,only_train_ill,old_train_df=return_clean_data(df_to_filter_path=TRAIN_PATH,All_disease_data_df=All_disease_data)

Train_disease_data=All_disease_data.loc[TRAIN_index.index,:]
Train_disease_data["Metformin"]=Train_disease_data.apply(func=find_code,code=metformin,axis=1)
Train_disease_data["Insulin"]=Train_disease_data.loc[:,Insulin_columns].apply(func=find_code,code=Insulin,axis=1)
Train_disease_data["met_or_ill"]=Train_disease_data["Insulin"]+Train_disease_data["Metformin"]
Train_healthy_index=Train_disease_data.loc[Train_disease_data.loc[:,"met_or_ill"]==0,:].index
Train_healthy=TRAIN_data_final.loc[Train_healthy_index,:]


# In[826]:


Test_disease_data=All_disease_data.loc[test_data_index,:]
Test_disease_data["Metformin"]=Test_disease_data.apply(func=find_code,code=metformin,axis=1)
Test_disease_data["Insulin"]=Test_disease_data.loc[:,Insulin_columns].apply(func=find_code,code=Insulin,axis=1)
Test_disease_data["met_or_ill"]=Test_disease_data["Insulin"]+Test_disease_data["Metformin"]
Test_healthy_index=Test_disease_data.loc[Test_disease_data.loc[:,"met_or_ill"]==0,:].index
Test_healthy=test_data_final.loc[Test_healthy_index,:]


# # Check for participants with HbA1c>6.5%

# In[174]:


TRAIN_PATH = '/net/mraid08/export/jafar/UKBioBank/Data/ukb29741_Diabetes_returned_extended_Imputed_train_test.csv'
TEST_PATH = '/net/mraid08/export/jafar/UKBioBank/Data/ukb29741_Diabetes_returned_extended_Imputed_val.csv'
ALL_PATH='/net/mraid08/export/jafar/UKBioBank/Data/ukb29741.csv'
ONLY_TRAIN_PATH = '/net/mraid08/export/jafar/UKBioBank/Data/ukb29741_Diabetes_returned_extended_Imputed_train.csv'
ONLY_VAL_PATH='/net/mraid08/export/jafar/UKBioBank/Data/ukb29741_Diabetes_returned_extended_Imputed_test.csv'
BASIC_PATH="/net/mraid08/export/jafar/UKBioBank/Data/"



ax_train,df_train,df_train_clean=filter_hba1c(path="/net/mraid08/export/jafar/UKBioBank/Data/ukb29741_Diabetes_returned_No_medicine_train_test.csv")


df_train.to_csv("/net/mraid08/export/jafar/UKBioBank/Data/ukb29741_Diabetes_returned_No_medicine_train_test.csv",index=True)

df_train_clean.to_csv(TRAIN_PATH,index=True)

ax_test,df_test,df_test_clean=filter_hba1c("/net/mraid08/export/jafar/UKBioBank/Data/ukb29741_Diabetes_returned_No_medicine_val.csv")

df_test.to_csv("/net/mraid08/export/jafar/UKBioBank/Data/ukb29741_Diabetes_returned_No_medicine_val.csv",index=True)

df_test_clean.to_csv(TEST_PATH,index=True)

ax_only_train,df_only_train,df_only_train_clean=filter_hba1c("/net/mraid08/export/jafar/UKBioBank/Data/ukb29741_Diabetes_returned_No_medicine_train.csv")

df_only_train.to_csv("/net/mraid08/export/jafar/UKBioBank/Data/ukb29741_Diabetes_returned_No_medicine_train.csv",index=True)

df_only_train_clean.to_csv(ONLY_TRAIN_PATH,index=True)

ax_only_val,df_only_val,df_only_val_clean=filter_hba1c("/net/mraid08/export/jafar/UKBioBank/Data/ukb29741_Diabetes_returned_No_medicine_test.csv")

col0_name=df_only_val.columns[0]

df_only_val.to_csv("/net/mraid08/export/jafar/UKBioBank/Data/ukb29741_Diabetes_returned_No_medicine_test.csv",index=True)

(df_only_val["HbA1c%"]>6.5).sum()

df_only_val_clean.to_csv(ONLY_VAL_PATH,index=True)

diab=[x for x in df_train_clean.columns if x.startswith("2443")]

df_only_val_clean[df_only_val_clean.loc[:,'2443-0.0_0.0']==1].shape

df_only_val_clean.shape

df_only_val_clean.loc[:,'2443-1.0_0.0']


A1c_low_Test_data_path="/home/edlitzy/UKBB_Tree_Runs/For_article/A1c_strat/Strat_L20_H39_Antro_neto_whr_Diabetes/Diabetes_Results/Diabetestest_Data"
A1c_low_Train_data_path="/home/edlitzy/UKBB_Tree_Runs/For_article/A1c_strat/Strat_L20_H39_Antro_neto_whr_Diabetes/Diabetes_Results/Diabetestrain_Data"

with open(A1c_low_Test_data_path, 'rb') as fp:
        A1c_low_Test_data= pickle.load(fp)

with open(A1c_low_Train_data_path, 'rb') as fp:
        A1c_low_Train_data = pickle.load(fp)


A1c_high_Test_data_path="/home/edlitzy/UKBB_Tree_Runs/For_article/A1c_strat/Strat_L39_Antro_neto_whr_Diabetes/Diabetes_Results/Diabetestest_Data"
A1c_high_Train_data_path="/home/edlitzy/UKBB_Tree_Runs/For_article/A1c_strat/Strat_L39_Antro_neto_whr_Diabetes/Diabetes_Results/Diabetestrain_Data"

