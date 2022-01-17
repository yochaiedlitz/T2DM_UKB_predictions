import pandas as pd
import pickle
import os
import numpy as np
import sys
import UKBB_Func_Final as UKBBFF
from sklearn.metrics import roc_auc_score, make_scorer
class runs:
    def __init__(self,run_name,only_check_model=False):
        self.run_name=run_name
        self.hyper_parameter_iter = 200
        self.num_of_bootstraps = 1000
        self.batch_size=20
        self.charac_selected = {"Age at last visit": "All", "Sex": "All", "Ethnic background": "All",
                                "Type of special diet followed": "All"}
        self.choose_model()
        self.compute_CI = True
        if not only_check_model:
            self.charac_id= {"Age at last visit": "21022-0.0", "Sex": "31-0.0", "Ethnic background": "21000-0.0",
                 "Type of special diet followed": "20086-0.0"}
            self.new_features_file_path_for_testing = None
            self.model_paths = None

            if self.model_type=="SA" or self.model_type=="LR":
                self.Val_file_path = None
                try: #Not all models have self, mode, hence the try expression
                    #Use self.mode=="Exploring" when you want to explore on the training data without looking at validation data
                    if self.mode=="Exploring" or self.mode=="exploring" or self.mode=="explore": #Exploring so not using real val data
                        self.Train_file_path ="/net/mraid08/export/jafar/UKBioBank/Data/ukb29741_Diabetes_returned_extended_Imputed_train.csv"
                        self.Test_file_path ="/net/mraid08/export/jafar/UKBioBank/Data/ukb29741_Diabetes_returned_extended_Imputed_test.csv"
                        self.Train_Test_file_path ="/net/mraid08/export/jafar/UKBioBank/Data/ukb29741_Diabetes_returned_extended_Imputed_train.csv"
                        self.Val_file_path = "/net/mraid08/export/jafar/UKBioBank/Data/ukb29741_Diabetes_returned_extended_Imputed_test.csv"
                    else:
                        self.Train_file_path = "/net/mraid08/export/jafar/UKBioBank/Data/ukb29741_Diabetes_returned_extended_Imputed_train.csv"
                        self.Test_file_path = "/net/mraid08/export/jafar/UKBioBank/Data/ukb29741_Diabetes_returned_extended_Imputed_test.csv"
                        self.Train_Test_file_path = "/net/mraid08/export/jafar/UKBioBank/Data/ukb29741_Diabetes_returned_extended_Imputed_train_test.csv"
                        self.Val_file_path = "/net/mraid08/export/jafar/UKBioBank/Data/ukb29741_Diabetes_returned_extended_Imputed_val.csv"
                except:
                    print("Using true validation data")
                    self.Train_file_path = "/net/mraid08/export/jafar/UKBioBank/Data/ukb29741_Diabetes_returned_extended_Imputed_train.csv"
                    self.Test_file_path = "/net/mraid08/export/jafar/UKBioBank/Data/ukb29741_Diabetes_returned_extended_Imputed_test.csv"
                    self.Train_Test_file_path = "/net/mraid08/export/jafar/UKBioBank/Data/ukb29741_Diabetes_returned_extended_Imputed_train_test.csv"
                    if self.Val_file_path ==None:
                        print("Using general Val file path")
                        self.Val_file_path = "/net/mraid08/export/jafar/UKBioBank/Data/ukb29741_Diabetes_returned_extended_Imputed_val.csv"
                self.model_paths=os.path.join(self.Folder_path,self.run_name)
                self.CI_results_path = os.path.join(self.model_paths, "CI")

            if self.model_type=="gbdt" or self.model_type=="GBDT":
                self.Set_GBDT()
                if self.model_paths==None:
                    self.model_paths = os.path.join(self.Folder_path, self.run_name)
                self.CI_results_path = os.path.join(self.model_paths+"_Diabetes/Diabetes_Results/", "CI")

            self.Training_path = os.path.join(self.model_paths, "training_Results")
            self.CI_results_summary_table = os.path.join(self.CI_results_path, self.run_name + "_CI.csv")
            self.hyper_parameters_summary_table = os.path.join(self.CI_results_path,
                                                               self.run_name + "_hyper_parameters_summary.csv")
            self.save_model = True
            self.save_model_filename = os.path.join(self.Folder_path, self.run_name, "LR_Model.sav")
            if self.hyper_parameter_iter == []:
                self.hyper_parameter_iter == 200
            if self.num_of_bootstraps == []:
                self.num_of_bootstraps=1000
            self.model_paths = os.path.join(self.Folder_path, self.run_name)
            self.create_dir()

    def choose_model(self):
# "" "Physical_health" "Mental_health" "Medication" "Lifestyle_and_physical_activity" "HbA1c"
# 'Family_and_Ethnicity' 'Early_Life_Factors' 'Diet' 'BT_No_A1c_No_Gluc' 'BT_No_A1c' 'BP_and_HR'
#"Blood_Tests" "Anthropometry" "All_No_gen"
        if self.run_name == "LR_Socio_demographics"\
                or self.run_name == "LR_Age_and_Sex"\
                or self.run_name == "LR_Physical_health"\
                or self.run_name == "LR_Mental_health"\
                or self.run_name == "LR_Medication"\
                or self.run_name == "LR_Lifestyle_and_physical_activity"\
                or self.run_name == "LR_HbA1c"\
                or self.run_name == "LR_Family_and_Ethnicity"\
                or self.run_name == "LR_Early_Life_Factors"\
                or self.run_name == "LR_Diet"\
                or self.run_name == "LR_BT_No_A1c_No_Gluc"\
                or self.run_name == "LR_BT_No_A1c"\
                or self.run_name == "LR_BP_and_HR"\
                or self.run_name == "LR_Blood_Tests"\
                or self.run_name == "LR_Anthropometry"\
                or self.run_name == "LR_All_No_gen":
            self.Folder_path = "/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Singles_LR/"
            if  self.run_name == "LR_All_No_gen":
                self.features_file_path = "/home/edlitzy/Biobank/Diabetes_Features_lists/For_article/All.csv"
            else:
                self.features_file_path = "/home/edlitzy/Biobank/Diabetes_Features_lists/For_article/"+self.run_name[3:]+".csv"
            self.model_type = "LR"
            self.batch_size = 50

        elif self.run_name == "LR_Antro_whr_family"\
                or self.run_name == "LR_Antro_neto_whr"\
                or self.run_name == "LR_Finrisc"\
                or self.run_name=="LR_A11_minimal":
            self.Folder_path = "/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Baseline_compare/"
            self.features_file_path = "/home/edlitzy/Biobank/Diabetes_Features_lists/For_article/" + self.run_name[3:]
            self.model_type = "LR"
            self.batch_size = 50

        elif self.run_name == "LR_Anthropometry_NO_whr":
            self.Folder_path = "/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Baseline_compare/"
            self.features_file_path = \
                "/home/edlitzy/Biobank/Diabetes_Features_lists/For_article/Anthropometry.csv"
            self.model_type = "LR"
            self.batch_size=50

        elif self.run_name == "LR_All_No_A1c_No_Gluc":
            self.Folder_path = "/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Addings/"
            self.features_file_path = "/home/edlitzy/Biobank/Diabetes_Features_lists/For_article/All_No_A1c_No_Gluc.csv"
            self.model_type = "LR"
            self.compute_CI = True
            self.batch_size=20

        elif self.run_name == "LR_BT_No_A1c":
            self.Folder_path = "/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Addings/"
            self.features_file_path = "/home/edlitzy/Biobank/Diabetes_Features_lists/For_article/BT_No_A1c.csv"
            self.model_type = "LR"
            self.batch_size = 50

        elif self.run_name == "LR_Strat_L39_Antro_neto_whr":
            self.Folder_path = "/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/A1c_strat/"
            self.features_file_path = "/home/edlitzy/Biobank/Diabetes_Features_lists/For_article/Antro_neto_whr.csv"
            self.model_type = "LR"
            self.batch_size=20
            self.charac_selected = {"Age at last visit": "All", "Sex": "All", "Ethnic background": "All",
                                    "Type of special diet followed": "All", "Minimal_a1c": 39}

        elif self.run_name == "LR_Strat_L20_H39_Antro_neto_whr":
            self.Folder_path = "/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/A1c_strat/"
            self.features_file_path = "/home/edlitzy/Biobank/Diabetes_Features_lists/For_article/Antro_neto_whr.csv"
            self.model_type = "LR"
            self.batch_size=20
            self.charac_selected = {"Age at last visit": "All", "Sex": "All", "Ethnic background": "All",
                                    "Type of special diet followed": "All", "Minimal_a1c": 20, "Maximal_a1c": 39}

        elif self.run_name == "LR_Strat_L39_Finrisc":
            self.Folder_path = "/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/A1c_strat/"
            self.features_file_path = "/home/edlitzy/Biobank/Diabetes_Features_lists/For_article/Finrisc.csv"
            self.model_type = "LR"
            self.charac_selected = {"Age at last visit": "All", "Sex": "All", "Ethnic background": "All",
                                    "Type of special diet followed": "All", "Minimal_a1c": 39}
            self.batch_size=50

        elif self.run_name == "LR_Strat_L20_H39_Finrisc":
            self.Folder_path = "/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/A1c_strat/"
            self.features_file_path = "/home/edlitzy/Biobank/Diabetes_Features_lists/For_article/Finrisc.csv"
            self.model_type = "LR"
            self.charac_selected = {"Age at last visit": "All", "Sex": "All", "Ethnic background": "All",
                                    "Type of special diet followed": "All", "Minimal_a1c": 20, "Maximal_a1c": 39}
            self.batch_size=50

        elif self.run_name == "LR_A1c_Strat_low_All_No_A1c_No_Gluc":
            self.Folder_path = "/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/A1c_strat/"
            self.features_file_path = "/home/edlitzy/Biobank/Diabetes_Features_lists/For_article/All_No_A1c_No_Gluc.csv"
            self.model_type = "LR"
            self.charac_selected={"Age at last visit": "All", "Sex": "All", "Ethnic background": "All",
                   "Type of special diet followed": "All","Minimal_a1c":20,"Maximal_a1c":39}

        elif self.run_name == "LR_A1c_Strat_high_All_No_A1c_No_Gluc":
            self.Folder_path = "/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/A1c_strat/"
            self.features_file_path = "/home/edlitzy/Biobank/Diabetes_Features_lists/For_article/All_No_A1c_No_Gluc.csv"
            self.model_type = "LR"
            self.charac_selected = {"Age at last visit": "All", "Sex": "All", "Ethnic background": "All",
                                    "Type of special diet followed": "All", "Minimal_a1c": 39}
            self.batch_size=50


        elif self.run_name=="GDRS_SA":
            self.features_file_path = "/home/edlitzy/Biobank/Diabetes_Features_lists/For_article/GDRS.csv"
            self.model_type = "SA"
            self.Folder_path = "/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Baseline_compare/"
            self.compute_CI=True

        elif self.run_name=="Strat_L39_GDRS_SA":
            self.features_file_path = "/home/edlitzy/Biobank/Diabetes_Features_lists/For_article/GDRS.csv"
            self.model_type="SA"
            self.Folder_path = "/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/A1c_strat/"
            self.compute_CI=True
            self.charac_selected={"Age at last visit": "All", "Sex": "All", "Ethnic background": "All",
                   "Type of special diet followed": "All","Minimal_a1c":39}

        elif self.run_name=="Strat_L20_H39_GDRS_SA":
            self.features_file_path = "/home/edlitzy/Biobank/Diabetes_Features_lists/For_article/GDRS.csv"
            self.model_type="SA"
            self.Folder_path = "/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/A1c_strat/"
            self.compute_CI=True
            self.charac_selected={"Age at last visit": "All", "Sex": "All", "Ethnic background": "All",
                   "Type of special diet followed": "All","Maximal_a1c":39,"Minimal_a1c":20}

        elif self.run_name == "Strat_L39_Antro_neto_whr"\
                or self.run_name ==  "Strat_L39_A11_minimal"\
                or self.run_name ==  "Strat_L39_All":
            self.Folder_path = "/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/A1c_strat/"
            self.model_type = "gbdt"
            self.charac_selected = {"Age at last visit": "All", "Sex": "All", "Ethnic background": "All",
                                    "Type of special diet followed": "All", "Minimal_a1c": 39}
            self.batch_size = 5

        elif self.run_name == "Strat_L20_H39_Antro_neto_whr"\
             or self.run_name == "Strat_L20_H39_All":
            self.Folder_path = "/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/A1c_strat/"
            self.model_type = "gbdt"
            self.charac_selected = {"Age at last visit": "All", "Sex": "All", "Ethnic background": "All",
                                    "Type of special diet followed": "All", "Minimal_a1c": 20, "Maximal_a1c": 39}
            self.batch_size = 10

        elif self.run_name == "Anthro_based_min"\
                or self.run_name == "Antro_whr_family":
            self.Folder_path = "/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Baseline_compare/"
            self.model_type = "gbdt"

        elif self.run_name == "Age_and_Sex" \
                or self.run_name == "Only_genetics" \
                or self.run_name == "Genetics_Age_and_Sex"\
                or self.run_name == "BP_and_HR" \
                or self.run_name == "Socio_demographics" \
                or self.run_name == "Family_and_Ethnicity" \
                or self.run_name == "Physical_health" \
                or self.run_name == "Mental_health" \
                or self.run_name == "Medication" \
                or self.run_name == "Lifestyle_and_physical_activity" \
                or self.run_name == "HbA1c" \
                or self.run_name == "Family_and_Ethnicity" \
                or self.run_name == "Early_Life_Factors" \
                or self.run_name == "BT_No_A1c_No_Gluc" \
                or self.run_name == "BT_No_A1c" \
                or self.run_name == "Blood_Tests" \
                or self.run_name == "Anthropometry"\
                or self.run_name == "Diet":
            self.Folder_path = "/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Singles/"
            self.model_type = "gbdt"
            self.batch_size=5

        elif self.run_name == "A1_BT__Anthro"\
                or self.run_name == "A2_Anthro__Physical_Health"\
                or self.run_name == "A3_Physical_Health__Lifestyle"\
                or self.run_name == "A4_Lifestyle__BP_n_HR"\
                or self.run_name == "A5_BP_n_HR__ND_Diagnosis"\
                or self.run_name == "A6_ND_Diagnosis__Mental"\
                or self.run_name == "A7_Mental__Medication"\
                or self.run_name == "A8_Medication__Diet"\
                or self.run_name == "A9_Diet__Family"\
                or self.run_name == "A10_Family__ELF"\
                or self.run_name == "A11_ELF__Socio" \
                or self.run_name == "A12_Socio_Genetics"\
                or self.run_name == "All":
            self.Folder_path = "/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Addings/"
            self.model_type = "gbdt"

        else:
            sys.exit("Model:",self.run_name, " was not found in CI_configs")

    def Set_GBDT(self,job_name="Diabetes"):
        self.job_name = job_name
        self.paths =os.path.join(self.Folder_path,self.run_name + "_"+self.job_name)
        self.final_folder =os.path.join(self.paths, "Diabetes_Results")
        self.CI_results_path = os.path.join(self.final_folder, "CI")

        self.VERBOSE_EVAL = 1000
        self.best_run = str(pd.read_csv(os.path.join(self.final_folder,self.job_name +
                                                     "_result_sorted.csv"), index_col=0).index.values[0])

        with open(os.path.join(self.final_folder,"Rel_Feat_Names"), 'rb') as fp:
            self.Rel_Feat_Names = pickle.load(fp)
        with open(os.path.join(self.final_folder,"cat_names"), 'rb') as fp:
            self.cat_names = pickle.load(fp)

        self.parameters = pd.read_csv(os.path.join(self.final_folder,self.job_name + "_Parameters_Table.csv"),
                                      index_col=0)  # Check that we can read params and build the selected model, train it and make all required drawings
        self.parameters = self.parameters.loc[['SN', 'boost_from_average', 'boosting_type', 'colsample_bytree',
                                               'is_unbalance', 'lambda_l1', 'lambda_l2', 'learning_rate', 'metric',
                                               'min_child_samples', 'num_boost_round', 'num_threads', 'objective',
                                               'subsample', 'verbose'], :]
        self.parameters.columns = self.parameters.loc["SN", :]
        self.parameters.drop(index="SN", inplace=True)
        self.params_dict = self.parameters.loc[:, self.best_run].to_dict()

        self.cat_ind = [x for x, name in enumerate(self.Rel_Feat_Names) if name in self.cat_names]
        self.params_bu = self.params_dict
        self.CI_load_data()

    def CI_load_data(self):
        data_path = self.final_folder

        with open(os.path.join(data_path, "Diabetestrain_Data"), 'rb') as fp:
            Train_Data = pickle.load(fp)
        with open(os.path.join(data_path, "Diabetestest_Data"), 'rb') as fp:
            Test_Data = pickle.load(fp)
        self.X_train = Train_Data["df_Features"]
        self.y_train = Train_Data["DF_Targets"]
        self.X_val = Test_Data["df_Features"]
        self.y_val = Test_Data["DF_Targets"]
        if self.new_features_file_path_for_testing!=None:
            self.choose_new_GBDT_test_data()
        # return self.X_train, self.y_train, self.X_val, self.y_val
    def choose_new_GBDT_test_data(self):
        new_features_path=self.new_features_file_path_for_testing
        all_data=pd.read_csv(self.Val_file_path,index_col="eid",usecols=["eid","30750-0.0"])
        try:
            use_index_df=all_data.loc[all_data["30750-0.0"]>self.minimal_a1c]
        except:
            print ("self.minimal_a1c is not defined")
        try:
            use_index_df=all_data.loc[all_data["30750-0.0"]>self.minimal_a1c]\
                .loc[all_data["30750-0.0"]<self.maximal_a1c].index
        except:
            print ("self.maximal_a1c is not defined")

        use_index=use_index_df.index
        self.FEAT_DF = pd.read_csv(new_features_path)  # Read Features files
        self.FEAT_DF = self.FEAT_DF[self.FEAT_DF["Exclude"] != 1]
        Use_Columns = [x for x in self.FEAT_DF["Field ID"]]
        self.X_val = self.X_val.loc[use_index,Use_Columns]
        self.y_val = self.y_val .loc[use_index,Use_Columns]
    def create_dir(self):
        if not os.path.exists(self.Folder_path):
            os.makedirs(self.Folder_path)
        if not os.path.exists(self.Training_path):
            os.makedirs(self.Training_path)
        if not os.path.exists(self.CI_results_path):
            os.makedirs(self.CI_results_path)

    def calc_CI(self):
        if self.model_type=="SA" or self.model_type=="LR":
            results_frame_list = [pd.read_csv(os.path.join(self.CI_results_path,f)) for
                              f in os.listdir(self.CI_results_path) if
                              (os.path.isfile(os.path.join(self.CI_results_path, f)) and
                               (f.startswith("AUC_APS_results_")) or f.startswith("AUC_APS_results_")) ]
            results_df=pd.concat(results_frame_list)
            aucroc_list=list(results_df.loc[:,"AUC"].values)
            aps_list=list(results_df.loc[:,"APS"].values)
            self.AUROC_lower,self.AUROC_upper=self.calc_CI_percentile(aucroc_list)
            self.APS_lower,self.APS_upper=self.calc_CI_percentile(aps_list)
            self.APS_median=np.median(np.array(aps_list))
            self.APS_mean = np.mean(np.array(aps_list))
            self.AUROC_median = np.median(np.array(aucroc_list))
            self.AUROC_mean = np.mean(np.array(aucroc_list))
            CI_Results_DF = pd.DataFrame.from_dict({"AUROC_mean": [self.AUROC_mean], "AUROC_median": [self.AUROC_median],
                                                    "AUROC_upper": [self.AUROC_upper], "AUROC_lower": [self.AUROC_lower],
                                                    "APS_mean": [self.APS_mean], "APS_median": [self.APS_median],
                                                    "APS_upper": [self.APS_upper], "APS_lower": [self.APS_lower]})
            CI_Results_DF.index = [self.run_name]
            CI_Results_DF.to_csv(self.CI_results_summary_table)
            print(("Results of",self.run_name,"saved to: ",self.CI_results_summary_table))
            print(("Results are: ",CI_Results_DF))

        elif self.model_type=="gbdt":
            aucroc_list = []
            aps_list = []
            onlyfiles = [f for f in os.listdir(self.CI_results_path) if
                         (os.path.isfile(os.path.join(self.CI_results_path, f)) and f.startswith("CI_Dict"))]
            for f in onlyfiles:
                with open(os.path.join(self.CI_results_path, f), 'rb') as fp:
                    self.data_dict = pickle.load(fp)
                aucroc_list.append(self.data_dict["AUROC"])
                aps_list.append(self.data_dict["APS"])
            self.AUROC_lower, self.AUROC_upper = self.calc_CI_percentile(aucroc_list)
            self.APS_lower, self.APS_upper = self.calc_CI_percentile(aps_list)
            self.APS_median = np.median(np.array(aps_list))
            self.APS_mean = np.mean(np.array(aps_list))
            self.AUROC_median = np.median(np.array(aucroc_list))
            self.AUROC_mean = np.mean(np.array(aucroc_list))
            CI_Results_DF = pd.DataFrame.from_dict(
                {"AUROC_mean": [self.AUROC_mean], "AUROC_median": [self.AUROC_median],
                 "AUROC_upper": [self.AUROC_upper], "AUROC_lower": [self.AUROC_lower],
                 "APS_mean": [self.APS_mean], "APS_median": [self.APS_median],
                 "APS_upper": [self.APS_upper], "APS_lower": [self.APS_lower]})
            CI_Results_DF.index = [self.run_name]
            results_path=os.path.join(self.CI_results_path, "CI_results.csv")
            CI_Results_DF.to_csv(results_path)

            print(("CI_Results_DF saved to:",results_path))
            print(("CI_Results_DF are:",CI_Results_DF))

    def calc_CI_percentile(self,metric_list,alpha = 0.95):
        p = ((1.0 - alpha) / 2.0) * 100
        lower = max(0.0, np.percentile(metric_list, p))
        p = (alpha + ((1.0 - alpha) / 2.0)) * 100
        upper = min(1.0, np.percentile(metric_list, p))
        print(('%.1f confidence interval %.1f%% and %.1f%%' % (alpha * 100, lower * 100, upper * 100)))
        return lower, upper



# class RunParams:
#     #For GBDT runs
#     def __init__(self,model_path_name=[],BASIC_PROB_BASED_JOB_NAME=[],num_of_bootstraps=1000):
#         print("Fetching params for predicitng", model_path_name," CI")
#         self.path= model_path_name+"_Diabetes"
#         self.CI_results_path = os.path.join(model_path_name, "Diabetes_Results", "CI")
#         self.job_name = "Diabetes"
#         self.RUN_NAME = self.path.split("/")[-2]
#         print("RUN_NAME:", self.RUN_NAME)
#         self.num_of_bootstraps=num_of_bootstraps
#         self.VERBOSE_EVAL = 1000
#         if BASIC_PROB_BASED_JOB_NAME==[]:
#             self.BASIC_PROB_BASED_JOB_NAME = "_".join((self.path.split("/")[-1]).split("_")[:-1])
#         else:
#             self.BASIC_PROB_BASED_JOB_NAME=BASIC_PROB_BASED_JOB_NAME
#         print("BASIC_PROB_BASED_JOB_NAME:", self.BASIC_PROB_BASED_JOB_NAME)
#         self.SAVE_FOLDER = "/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Minimal/" + self.RUN_NAME + "/"
#         self.Save_2_folder = self.SAVE_FOLDER + self.BASIC_PROB_BASED_JOB_NAME + "_" + self.job_name + "/"
#         self.final_folder = self.Save_2_folder + self.job_name + "_Results/"
#         self.run_name = str(pd.read_csv(self.final_folder + self.job_name + "_result_sorted.csv", index_col=0).index.values[0])
#
#         with open(self.final_folder + "Rel_Feat_Names", 'rb') as fp:
#             self.Rel_Feat_Names = pickle.load(fp)
#         with open(self.final_folder + "cat_names", 'rb') as fp:
#             self.cat_names = pickle.load(fp)
#
#         self.parameters = pd.read_csv(self.final_folder + self.job_name + "_Parameters_Table.csv",
#                                  index_col=0)  # Check that we can read params and build the selected model, train it and make all required drawings
#         self.parameters =self.parameters.loc[['SN','boost_from_average', 'boosting_type', 'colsample_bytree',
#                                               'is_unbalance', 'lambda_l1', 'lambda_l2', 'learning_rate','metric',
#                                               'min_child_samples', u'num_boost_round', u'num_threads','objective',
#                                               'subsample', 'verbose'],:]
#         self.parameters.columns = self.parameters.loc["SN", :]
#         self.parameters.drop(index="SN", inplace=True)
#         self.params_dict = self.parameters.loc[:,self.run_name].to_dict()
#
#         self.cat_ind = [x for x, name in enumerate(self.Rel_Feat_Names) if name in self.cat_names]
#         self.params_bu = self.params_dict
#         self.create_dir()
#         self.CI_load_data()
#
#
#     def CI_load_data(self):
#         """path should be equal to path"""
#         data_path = os.path.join(self.path, "Diabetes_Results")
#
#         with open(os.path.join(data_path, "Diabetestrain_Data"), 'rb') as fp:
#             Train_Data = pickle.load(fp)
#         with open(os.path.join(data_path, "Diabetestest_Data"), 'rb') as fp:
#             Test_Data = pickle.load(fp)
#         self.X_train = Train_Data["df_Features"]
#         self.y_train = Train_Data["DF_Targets"]
#         self.X_val = Test_Data["df_Features"]
#         self.y_val = Test_Data["DF_Targets"]
#         return self.X_train, self.y_train, self.X_val, self.y_val
#
#     def calc_CI(self):
#         aucroc_list=[]
#         aps_list=[]
#         onlyfiles = [f for f in os.listdir(self.CI_results_path) if (os.path.isfile(os.path.join(self.CI_results_path, f)) and  f.startswith("CI_Dict"))]
#         for f in onlyfiles:
#             with open(os.path.join(self.CI_results_path, f), 'rb') as fp:
#                 self.data_dict=pickle.load(fp)
#             aucroc_list.append(self.data_dict["AUROC"])
#             aps_list.append(self.data_dict["APS"])
#         self.AUROC_lower,self.AUROC_upper=self.calc_CI_percentile(aucroc_list)
#         self.APS_lower,self.APS_upper=self.calc_CI_percentile(aps_list)
#         self.APS_median=np.median(np.array(aps_list))
#         self.APS_mean = np.mean(np.array(aps_list))
#         self.AUROC_median = np.median(np.array(aucroc_list))
#         self.AUROC_mean = np.mean(np.array(aucroc_list))
#         CI_Results_DF = pd.DataFrame.from_dict({"AUROC_mean": [self.AUROC_mean], "AUROC_median": [self.AUROC_median],
#                                                 "AUROC_upper": [self.AUROC_upper], "AUROC_lower": [self.AUROC_lower],
#                                                 "APS_mean": [self.APS_mean], "APS_median": [self.APS_median],
#                                                 "APS_upper": [self.APS_upper], "APS_lower": [self.APS_lower]})
#         CI_Results_DF.index = [self.BASIC_PROB_BASED_JOB_NAME]
#         CI_Results_DF.to_csv(os.path.join(self.CI_results_path,"CI_results.csv"))
#
#     def calc_CI_percentile(self,metric_list,alpha = 0.95):
#         p = ((1.0 - alpha) / 2.0) * 100
#         lower = max(0.0, np.percentile(metric_list, p))
#         p = (alpha + ((1.0 - alpha) / 2.0)) * 100
#         upper = min(1.0, np.percentile(metric_list, p))
#         print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha * 100, lower * 100, upper * 100))
#         return lower, upper
#
#     def create_dir(self):
#         if not os.path.exists(self.CI_results_path):
#             os.makedirs(self.CI_results_path)
