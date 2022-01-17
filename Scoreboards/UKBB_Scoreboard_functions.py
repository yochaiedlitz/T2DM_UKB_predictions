# import matplotlib.pyplot as plt
from LR_CI import *
from Article_files.UKBB_article_plots_functions import *
from sklearn.metrics import average_precision_score as aps
from sklearn.metrics import roc_auc_score as auroc
import re
import os
from Configs.CI_Configs import runs

class ScoreBoard:
    def __init__(self, scoreboard_type,
                 build_new_database=False,
                 train_val_test_type="Updated",
                 base_path="/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/",
                 save_to_scoreboards_basic_path="/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/Scoreboards/",
                 pdo=10,
                 max_or=1000,
                 save_database=False,
                 use_woe_values_for_lr=True,
                 save_ci_reults_dict=True, calculate_scoreboard=True,debug=False,
                 recover=False,
                 min_num_of_ills=1,
                 force_update_LR=True,
                 force_update_ancestor_LR=False,
                 leg_dict=None):  # Anthro,Five_blood
        """
                 mode should be either explore or None, if mode=="explore" it will not work with the validation data,
                 unless otherwise was defined in the Scoreboard_configs file
                 train_val_test_type - Old to use files that were used before the update with ScoreBoard or Updated to use the files that were updated
            """
        self.mode=None
        self.debug=debug
        if recover:
            self.scoreboard_was_calculated = False
        self.scoreboard=None
        self.min_num_of_ills=min_num_of_ills
        self.use_woe_values_for_lr = use_woe_values_for_lr
        self.CI_Results_DF = None
        self.save_ci_reults_dict = save_ci_reults_dict
        self.bins_df = None
        self.final_scores = None
        self.ci_auroc_scores = []
        self.ci_aps_scores = []
        self.y_pred = None
        self.train_bins_df = None
        self.train_bins_df_dict = None
        self.train_bins_labels_dict = None

        self.test_bins_df = None
        self.test_bins_df_dict = None
        self.test_bins_labels_dict = None

        self.val_bins_df = None
        self.val_bins_df_dict = None
        self.val_bins_labels_dict = None

        self.train_val_bins_df = None
        self.train_val_bins_df_dict = None
        self.train_val_bins_labels_dict = None
        self.leg_dict=leg_dict

        self.woe_res_dict = None
        self.scoreboard_type = scoreboard_type
        self.build_new_database = build_new_database
        self.save_database = save_database
        self.train_val_test_type = train_val_test_type
        self.base_path = base_path
        self.save_to_scoreboards_basic_path = save_to_scoreboards_basic_path
        self.set_woe_params(pdo, max_or)
        self.choose_model()
        self.make_dir_if_not_exist(self.save_to_scoreboards_basic_path)

        self.run_object = runs(self.run_name, mode=self.mode,debug=self.debug,force_update=force_update_LR)
        self.ancestor_object = runs(self.ancestor_object_name, mode=self.mode,debug=self.debug,
                                    force_update=force_update_ancestor_LR)
        if not "scoreboard" in self.run_object.features_file_path:
            print("self.run_object.features_file_path did not include 'scoreboard' in it - adding scoreboard to name")
            self.run_object.features_file_path=self.run_object.features_file_path.replace(".csv","scoreboard.csv")
        self.new_features_file_path = self.run_object.features_file_path  # Location to save the new features file
        self.ancestor_object_features_file_path = self.ancestor_object.features_file_path  # Location of the original LR ,odel path
        self.save_to_folder = os.path.join(self.save_to_scoreboards_basic_path, self.run_name)
        self.summary_folder =os.path.join(self.save_to_folder,"summary_files")
        self.woe_csv_path = os.path.join(self.summary_folder,"woe_csv.csv")
        self.woe_dict_path = os.path.join(self.save_to_folder,"woe_dict.pkl")
        self.features_importance_fig_path = os.path.join(
            self.base_path, "figures", "Scoreboard", self.run_name +
                                                     "_features_importance.png")
        self.tables_save_path = os.path.join(self.base_path,"Tables")
        self.ci_summary_table_name= self.run_name + "_features_importance_summary.csv"
        self.features_importance_tables_save_path=os.path.join(self.tables_save_path,self.run_name+
                                                        "_importance.csv")

        self.make_dir_if_not_exist(self.save_to_folder)
        self.make_dir_if_not_exist(self.summary_folder)
        self.features_file_df, self.features_dict = read_features_file_df(
            ancestor_object_features_file_path=self.ancestor_object_features_file_path)

        self.set_datasets_paths()
        if calculate_scoreboard:
            self.calculate_scoreboard()

    def calculate_scoreboard(self):
        if self.build_new_database:
            self.load_datasets()
            self.bin_dataframes()
            if self.save_database:
                self.add_binned_data_to_saved_dataset()
        else:
            self.load_saved_data()
        # TODO change the names in the woe_dict indexes to be similar to the lr models features
        self.woe_dict = build_and_save_woe_dict(self.train_bins_df_dict, self.save_to_folder,
                                                bins_labels_dict=self.train_bins_labels_dict,
                                                features_file_df=self.features_file_df)  # train_bins_df_dict
        self.build_IV_df()

        if self.use_woe_values_for_lr:
            self.train_bins_df = self.set_woe_values_to_df(self.train_bins_df)
            self.test_bins_df = self.set_woe_values_to_df(self.test_bins_df)
            self.val_bins_df = self.set_woe_values_to_df(self.val_bins_df)
            self.train_val_bins_df = self.set_woe_values_to_df(self.train_val_bins_df)

        self.feat_df = create_feature_file(
            new_features_file_path=self.new_features_file_path,
            new_features_names=self.train_bins_df.columns, Type=self.scoreboard_type)

        if self.save_database:
            self.add_binned_data_to_saved_dataset()
        self.scoreboard_was_calculated = True

    def set_woe_values_to_df(self, df):
        # df=self.test_bins_df.columns
        columns = df.columns
        categories = list(set([re.split(' <| >| :', x)[0] for x in columns]))
        categories_woe = [x + "_woe" for x in categories]
        df_woe_values = pd.DataFrame(data=0, index=df.index, columns=categories_woe)
        for category in categories:
            for col in columns:
                if category in col:
                    df_woe_values[category + "_woe"] += df[col] * self.woe_dict[category].loc[col, "WOE"]
        return df_woe_values

    def choose_model(self):  # mode=explore or None, when "explore", working on the validation data

        if self.scoreboard_type == "Five_blood_explore":
            self.run_name = "LR_Five_blood_tests_scoreboard_explore"
            self.ancestor_object_name = "LR_Five_Blood_Tests_explore"
            self.lr_coefficient_path = os.path.join(self.base_path, "Tables/five_blood_fetures.csv")
            self.woe_res_dict = {'Age at last visit': 6, 'Years between visits': 6,
                                 'Gamma glutamyltransferase': 10, 'Glycated haemoglobin (HbA1c)': 17,
                                 'HDL cholesterol': 9, 'Triglycerides': 10,
                                 'Reticulocyte count': 10}  # Number of bins for each features
            self.mode = "explore"

        elif self.scoreboard_type == "Five_blood_debug":
            self.run_name = "LR_Five_blood_tests_scoreboard_debug"
            self.ancestor_object_name = "LR_Five_Blood_Tests_explore"
            self.lr_coefficient_path = os.path.join(self.base_path, "Tables/five_blood_fetures.csv")
            self.woe_res_dict = {'Age at last visit': 6, 'Years between visits': 6,
                                 'Gamma glutamyltransferase': 10, 'Glycated haemoglobin (HbA1c)': 17,
                                 'HDL cholesterol': 9, 'Triglycerides': 10,
                                 'Reticulocyte count': 10} # Number of bins for each features
            self.mode = "explore"
            self.debug = True

        elif self.scoreboard_type == "Anthro_explore"\
                or self.scoreboard_type == "LR_Anthro_scoreboard_explore":
            self.run_name = "LR_Anthro_scoreboard_explore"
            self.ancestor_object_name = "LR_Antro_neto_whr_explore"
            self.lr_coefficient_path = os.path.join(self.base_path, "Tables/Anthropometrics_features.csv")
            self.woe_res_dict = {"Years between visits": 6,
                                 "Age at last visit": 6,
                                 "Weight": 10,
                                 "Standing height": 5,
                                 "Body mass index (BMI)": 10,
                                 "Waist-hips ratio": 10, "Hip circumference": 10,
                                 "Waist circumference": 10,
                                 "Sex": 2}  # Number of bins for each features

            self.mode = "explore"

        elif self.scoreboard_type == "Anthro_debug":
            self.run_name = "LR_Anthro_scoreboard_debug"
            self.ancestor_object_name = "LR_Antro_neto_whr_explore"
            self.lr_coefficient_path = os.path.join(self.base_path, "Tables/Anthropometrics_features.csv")
            self.woe_res_dict = {"Years between visits": 6,
                                 "Age at last visit": 6,
                                 "Weight": 10,
                                 "Standing height": 5,
                                 "Body mass index (BMI)": 10,
                                 "Waist-hips ratio": 10, "Hip circumference": 10,
                                 "Waist circumference": 10,
                                 "Sex": 2}  # Number of bins for each features

            self.mode = "explore"
            self.debug = True


        elif self.scoreboard_type == "Five_blood":
            self.run_name = "LR_Five_blood_tests_scoreboard"
            self.ancestor_object_name = "LR_Five_Blood_Tests"
            self.lr_coefficient_path = os.path.join(self.base_path, "Tables/five_blood_fetures.csv")
            self.woe_res_dict = {'Age at last visit': 1,'Years between visits': 6,
                                 'Gamma glutamyltransferase': 8, 'Glycated haemoglobin (HbA1c)': 8,
                                 'HDL cholesterol': 7, 'Triglycerides': 6,
                                 'Reticulocyte count': 7}  # Number of bins for each features
            self.mode = "None"

        elif self.scoreboard_type == "Anthro":
            self.run_name = "LR_Anthro_scoreboard"
            self.ancestor_object_name = "LR_Antro_neto_whr"
            self.lr_coefficient_path = os.path.join(self.base_path, "Tables/Anthropometrics_features.csv")
            self.woe_res_dict = {"Years between visits": 6,
                                 "Age at last visit": 4,
                                 "Weight": 2,
                                 "Standing height": 3,
                                 "Body mass index (BMI)": 7,
                                 "Waist-hips ratio": 7, "Hip circumference": 4,
                                 "Waist circumference": 6,
                                 "Sex": 2}  # Number of bins for each features
            self.mode = "None"
        # Antro_neto_whr_scoreboard
        elif self.scoreboard_type == "LR_Strat_H39_Antro_neto_whr_scoreboard":
            self.run_name = "LR_Strat_H39_Antro_neto_whr_scoreboard"
            self.ancestor_object_name = "LR_Strat_H39_Antro_neto_whr"
            self.lr_coefficient_path = os.path.join(self.base_path, "Tables/Anthropometrics_features.csv")
            self.woe_res_dict = {"Years between visits": 3,
                                 "Age at last visit": 1,
                                 "Weight": 2,
                                 "Standing height": 3,
                                 "Body mass index (BMI)": 7,
                                 "Waist-hips ratio": 5, "Hip circumference": 3,
                                 "Waist circumference": 4,
                                 "Sex": 2}  # Number of bins for each features
            self.mode = "None"

        elif self.scoreboard_type == "LR_Strat_L20_H39_Antro_neto_whr_scoreboard":
            self.run_name = "LR_Strat_L20_H39_Antro_neto_whr_scoreboard"
            self.ancestor_object_name = "LR_Strat_L20_H39_Antro_neto_whr"
            self.lr_coefficient_path = os.path.join(self.base_path, "Tables/Anthropometrics_features.csv")
            self.woe_res_dict = {"Years between visits": 3,
                                 "Age at last visit": 1,
                                 "Weight": 2,
                                 "Standing height": 3,
                                 "Body mass index (BMI)": 7,
                                 "Waist-hips ratio": 5, "Hip circumference": 3,
                                 "Waist circumference": 4,
                                 "Sex": 2}  # Number of bins for each features
            self.mode = "None"

        elif self.scoreboard_type == "LR_Strat_L39_Antro_neto_whr_scoreboard":
            self.run_name = "LR_Strat_L39_Antro_neto_whr_scoreboard"
            self.ancestor_object_name = "LR_Strat_L39_Antro_neto_whr"
            self.lr_coefficient_path = os.path.join(self.base_path, "Tables/Anthropometrics_features.csv")
            self.woe_res_dict = {"Years between visits": 6,
                                 "Age at last visit": 4,
                                 "Weight": 2,
                                 "Standing height": 3,
                                 "Body mass index (BMI)": 7,
                                 "Waist-hips ratio": 7, "Hip circumference": 4,
                                 "Waist circumference": 6,
                                 "Sex": 2}  # Number of bins for each features
            self.mode = "None"

        elif self.scoreboard_type == "No_reticulocytes_scoreboard":
            self.run_name = "LR_No_reticulocytes_scoreboard"
            self.ancestor_object_name = "LR_No_reticulocytes"
            self.lr_coefficient_path = os.path.join(self.base_path, "Tables/Anthropometrics_features.csv")
            self.woe_res_dict = self.woe_res_dict = {'Age at last visit': 4, 'Years between visits': 6,
                                 'Gamma glutamyltransferase': 8, 'Glycated haemoglobin (HbA1c)': 10,
                                 'HDL cholesterol': 7, 'Triglycerides': 6}  # # Number of bins for each features
            self.mode = "None"

        elif self.scoreboard_type == "LR_explore_No_reticulocytes_scoreboard":
            self.run_name = "LR_explore_No_reticulocytes_scoreboard"
            self.ancestor_object_name = "LR_explore_No_reticulocytes"
            self.lr_coefficient_path = os.path.join(self.base_path, "Tables/Anthropometrics_features.csv")
            self.woe_res_dict = self.woe_res_dict = {'Age at last visit': 4, 'Years between visits': 6,
                                 'Gamma glutamyltransferase': 8, 'Glycated haemoglobin (HbA1c)': 10,
                                 'HDL cholesterol': 7, 'Triglycerides': 6}  # # Number of bins for each features
            self.model = "explore"

        elif self.scoreboard_type == "LR_Strat_H39_Four_BT_scoreboard_orig":
            self.run_name = "LR_Strat_H39_Four_BT_scoreboard_orig"
            self.ancestor_object_name = "LR_No_reticulocytes"
            self.lr_coefficient_path = os.path.join(self.base_path, "Tables/Anthropometrics_features.csv")
            self.woe_res_dict = self.woe_res_dict = {'Age at last visit': 4, 'Years between visits': 6,
                                 'Gamma glutamyltransferase': 8, 'Glycated haemoglobin (HbA1c)': 10,
                                 'HDL cholesterol': 7, 'Triglycerides': 6}  # # Number of bins for each features
            self.mode = "None"

        elif self.scoreboard_type == "LR_Strat_H39_Four_BT_scoreboard_custom":
            self.run_name = "LR_Strat_H39_Four_BT_scoreboard_custom"
            self.ancestor_object_name = "LR_Strat_H39_Four_Blood_Tests"
            self.lr_coefficient_path = os.path.join(self.base_path, "Tables/Anthropometrics_features.csv")
            self.woe_res_dict = self.woe_res_dict = {'Age at last visit': 1, 'Years between visits': 3,
                                 'Gamma glutamyltransferase': 7, 'Glycated haemoglobin (HbA1c)': 6,
                                 'HDL cholesterol': 5, 'Triglycerides': 6}  # # Number of bins for each features
            self.mode = "None"

        elif self.scoreboard_type == "LR_Strat_L20_H39_Four_Blood_Tests_scoreboard":
            self.run_name = "LR_Strat_L20_H39_Four_Blood_Tests_scoreboard"
            self.ancestor_object_name = "LR_Strat_L20_H39_Four_Blood_Tests"
            self.lr_coefficient_path = os.path.join(self.base_path, "Tables/Anthropometrics_features.csv")
            self.woe_res_dict = self.woe_res_dict = {'Age at last visit': 1, 'Years between visits': 3,
                                 'Gamma glutamyltransferase': 7, 'Glycated haemoglobin (HbA1c)': 6,
                                 'HDL cholesterol': 5, 'Triglycerides': 6}  # # Number of bins for each features
            self.mode = "None"

        elif self.scoreboard_type == "LR_Strat_L39_Four_Blood_Tests_scoreboard_custom":
            self.run_name = "LR_Strat_L39_Four_Blood_Tests_scoreboard_custom"
            self.ancestor_object_name = "LR_Strat_L39_Four_Blood_Tests"
            self.lr_coefficient_path = os.path.join(self.base_path, "Tables/Anthropometrics_features.csv")
            self.woe_res_dict = self.woe_res_dict = {'Age at last visit': 4, 'Years between visits': 6,
                                 'Gamma glutamyltransferase': 8, 'Glycated haemoglobin (HbA1c)': 10,
                                 'HDL cholesterol': 7, 'Triglycerides': 6}  # # Number of bins for each features
            self.mode = "None"

        elif self.scoreboard_type == "LR_Strat_L39_Four_BT_SB_revision_explore":
            self.run_name = "LR_Strat_L39_Four_BT_SB_revision"
            self.ancestor_object_name = "LR_No_reticulocytes"
            self.lr_coefficient_path = os.path.join(self.base_path, "Tables/Anthropometrics_features.csv")
            self.woe_res_dict = self.woe_res_dict = {'Age at last visit': 4, 'Years between visits': 6,
                                                     'Gamma glutamyltransferase': 8, 'Glycated haemoglobin (HbA1c)': 10,
                                                     'HDL cholesterol': 7,
                                                     'Triglycerides': 6}  # # Number of bins for each features
            self.mode = "explore"

        elif self.scoreboard_type == "LR_Strat_L39_Antro_SB_revision":
            self.run_name = "LR_Strat_L39_Antro_SB_revision"
            self.ancestor_object_name = "LR_Strat_L39_Antro_neto_whr"
            self.lr_coefficient_path = os.path.join(self.base_path, "Tables/Anthropometrics_features.csv")
            self.woe_res_dict =  {"Years between visits": 6,
                                 "Age at last visit": 6,
                                 "Weight": 10,
                                 "Standing height": 5,
                                 "Body mass index (BMI)": 10,
                                 "Waist-hips ratio": 10, "Hip circumference": 10,
                                 "Waist circumference": 10,
                                 "Sex": 2}   # # Number of bins for each features
            self.mode = "None"
        else:
            sys.exit("Did not find the correcte scoreboard name in Scoreboard.choose_model() "
                     "in UKBB_Scoreboard_functions")

    def set_woe_params(self, pdo, max_or):
        self.pdo = pdo
        self.max_or = max_or
        self.Factor = self.pdo / np.log(2)
        self.Offset = 100 - self.Factor * np.log(self.max_or)
        print('Factor:', round(self.Factor, 2), 'Offset:', round(self.Offset, 2))

    def set_datasets_paths(self):
        if self.train_val_test_type == "Old":
            # self.train_file_path = "/net/mraid08/export/jafar/UKBioBank/Data/ukb29741_Diabetes_returned_extended_Imputed_train.csv"
            # self.val_file_path = "/net/mraid08/export/jafar/UKBioBank/Data/ukb29741_Diabetes_returned_extended_Imputed_test.csv"
            # self.train_val_file_path = "/net/mraid08/export/jafar/UKBioBank/Data/ukb29741_Diabetes_returned_extended_Imputed_train_test.csv"
            # self.test_file_path = "/net/mraid08/export/jafar/UKBioBank/Data/ukb29741_Diabetes_returned_extended_Imputed_val.csv"
            self.train_file_path ="/net/mraid08/export/jafar/UKBioBank/Data/ukb29741_a1c_below_65_train.csv"
            self.val_file_path = "/net/mraid08/export/jafar/UKBioBank/Data/ukb29741_a1c_below_65_val.csv"
            self.train_val_file_path = "/net/mraid08/export/jafar/UKBioBank/Data/ukb29741_a1c_below_65_train_val.csv"
            self.test_file_path = "/net/mraid08/export/jafar/UKBioBank/Data/ukb29741_a1c_below_65_test.csv"
        elif self.train_val_test_type == "Updated" \
                or self.train_val_test_type == "Update" \
                or self.train_val_test_type == "updated" \
                or self.train_val_test_type == "update":
            self.train_file_path = self.run_object.train_file_path
            self.val_file_path = self.run_object.val_file_path
            self.train_val_file_path = self.run_object.train_val_file_path
            self.test_file_path = self.run_object.test_file_path
        else:
            self.train_file_path = None
            self.val_file_path = None
            self.test_file_path = None
            print("please set train_val_test_type to either 'Old' or 'Updated'")

        # ### Path for saving the updated data base
        self.New_Train_file_path = self.run_object.train_file_path
        self.New_Val_file_path = self.run_object.val_file_path
        self.New_Train_Val_file_path = self.run_object.train_val_file_path
        self.New_Test_file_path = self.run_object.test_file_path

    def load_datasets(self):
        self.X_train, self.y_train = load_lr_data(self.ancestor_object, self.train_file_path, mode="train",
                                                  save_data_folder=self.run_object.model_paths,
                                                  standardise=False,postfix="train")
        self.X_val, self.y_val = load_lr_data(self.ancestor_object, self.val_file_path, mode="train", standardise=False,
                                              save_data_folder=self.run_object.model_paths,postfix="val")
        self.X_test, self.y_test = load_lr_data(self.ancestor_object, self.test_file_path, mode="train",
                                                standardise=False,save_data_folder=self.run_object.model_paths,
                                                postfix="test")
        # ## Rename datasets columns
        self.X_train.rename(columns=self.features_dict, inplace=True)
        self.X_test.rename(columns=self.features_dict, inplace=True)
        self.X_val.rename(columns=self.features_dict, inplace=True)

        self.df_train = pd.concat([self.X_train, self.y_train], axis=1)
        self.df_val = pd.concat([self.X_val, self.y_val], axis=1)
        self.df_test = pd.concat([self.X_test, self.y_test], axis=1)

    def bin_dataframes(self):
        """
        Convert the dataframe to a binned dataframe for the ScoreBoard
        :return:
        """
        self.train_bins_df, self.train_bins_df_dict, self.train_bins_labels_dict,self.woe_res_dict = bin_df(
            self.X_train, self.y_train, self.features_file_df, max_bins=5, res_dict=self.woe_res_dict,
            min_num_of_ills=self.min_num_of_ills)
        # I could place woe_values in train_bins_df_dict instead of using train_bins_df
        self.train_bins_df.to_csv(os.path.join(self.save_to_folder, "train_bins_df.csv"))
        self.pickle_dict(variable=self.train_bins_df_dict, filename="train_bins_df_dict.pkl", task="dump")
        self.pickle_dict(variable=self.train_bins_labels_dict, filename="train_bins_labels_dict.pkl", task="dump")

        self.val_bins_df, self.val_bins_df_dict, self.val_bins_labels_dict,self.woe_res_dict = \
            bin_df(self.X_val, self.y_val, self.features_file_df, max_bins=5, res_dict=self.woe_res_dict,
                   bins_labels_dict=self.train_bins_labels_dict,min_num_of_ills=self.min_num_of_ills)
        self.val_bins_df.to_csv(os.path.join(self.save_to_folder, "val_bins_df.csv"))

        self.pickle_dict(variable=self.val_bins_df_dict, filename="val_bins_df_dict.pkl", task="dump")
        self.pickle_dict(variable=self.val_bins_labels_dict, filename="val_bins_labels_dict.pkl", task="dump")

        self.train_val_bins_df = pd.concat([self.train_bins_df, self.val_bins_df], axis=0)
        self.train_val_bins_df.to_csv(os.path.join(self.save_to_folder, "train_val_bins_df.csv"))
        self.pickle_dict(variable=self.train_val_bins_df_dict, filename="train_val_bins_df_dict.pkl", task="dump")
        self.pickle_dict(variable=self.train_val_bins_labels_dict, filename="train_val_bins_labels_dict.pkl",
                         task="dump")

        self.test_bins_df, self.test_bins_df_dict, self.test_bins_labels_dict,self.woe_res_dict = \
            bin_df(self.X_test, self.y_test, self.features_file_df, max_bins=5, res_dict=self.woe_res_dict,
                   bins_labels_dict=self.train_bins_labels_dict,min_num_of_ills=self.min_num_of_ills)
        self.test_bins_df.to_csv(os.path.join(self.save_to_folder, "test_bins_df.csv"))

        self.pickle_dict(variable=self.test_bins_df_dict, filename="test_bins_df_dict.pkl", task="dump")
        self.pickle_dict(variable=self.test_bins_labels_dict, filename="test_bins_labels_dict.pkl", task="dump")

    def add_binned_data_to_saved_dataset(self):
        concat_train_val_test_dataframes(self.train_file_path, self.train_bins_df,
                                                             self.New_Train_file_path)
        concat_train_val_test_dataframes(self.val_file_path, self.val_bins_df,
                                                           self.New_Val_file_path)
        concat_train_val_test_dataframes(self.train_val_file_path, self.train_val_bins_df,
                                                                 self.New_Train_Val_file_path)
        concat_train_val_test_dataframes(self.test_file_path, self.test_bins_df,
                                                            self.New_Test_file_path)

    def pickle_dict(self, variable, filename, task="dump"):
        """

        :param variable:
        :param filename:
        :param task:
        :return:
        """
        if task == "dump" or task == "save":
            pickle_to = os.path.join(self.save_to_folder, filename)
            with open(pickle_to, 'wb') as handle:
                pickle.dump(variable, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)
        elif task == "load":
            pickle_from = os.path.join(self.save_to_folder, filename)
            with open(pickle_from, 'rb') as handle:
                variable = pickle.load(handle)
            return variable
        else:
            sys.exit("task should be either 'load' or 'dump'")

    def load_saved_data(self):
        self.New_Train_file_path = "/net/mraid08/export/jafar/UKBioBank/Data/ukb29741_scoretable_train.csv"
        self.New_Val_file_path = "/net/mraid08/export/jafar/UKBioBank/Data/ukb29741_scoretable_val.csv"
        self.New_Train_Val_file_path = "/net/mraid08/export/jafar/UKBioBank/Data/ukb29741_scoretable_train_val.csv"
        self.New_Test_file_path = "/net/mraid08/export/jafar/UKBioBank/Data/ukb29741_Diabetes_scoretable_test.csv"

        self.X_train, self.y_train = load_lr_data(self.run_object, self.New_Train_file_path, mode="train",
                                                  standardise=False,postfix="train")
        self.train_bins_df = pd.read_csv(os.path.join(self.save_to_folder, "train_bins_df.csv"), index_col="eid")

        self.X_val, self.y_val = load_lr_data(self.run_object, self.New_Val_file_path, mode="train", standardise=False,
                                              postfix="val")
        self.val_bins_df = pd.read_csv(os.path.join(self.save_to_folder, "val_bins_df.csv"), index_col="eid")

        self.X_test, self.y_test = load_lr_data(self.run_object, self.New_Test_file_path, mode="train",
                                                standardise=False,postfix="test")
        self.test_bins_df = pd.read_csv(os.path.join(self.save_to_folder, "test_bins_df.csv"), index_col="eid")

        self.X_train_val, self.y_train_val = load_lr_data(self.run_object, self.New_Test_file_path, mode="train",
                                                          standardise=False,postfix="train_val")
        self.train_val_bins_df = pd.read_csv(os.path.join(self.save_to_folder, "train_val_bins_df.csv"),
                                             index_col="eid")

        self.test_bins_df_dict = self.pickle_dict(variable=self.test_bins_df_dict, filename="test_bins_df_dict.pkl",
                                                  task="load")
        self.test_bins_labels_dict = self.pickle_dict(variable=self.test_bins_labels_dict,
                                                      filename="test_bins_labels_dict.pkl")
        self.train_bins_df_dict = self.pickle_dict(variable=self.train_bins_df_dict, filename="train_bins_df_dict.pkl",
                                                   task="load")
        self.train_bins_labels_dict = self.pickle_dict(variable=self.train_bins_labels_dict,
                                                       filename="train_bins_labels_dict.pkl", task="load")
        self.val_bins_df_dict = self.pickle_dict(variable=self.val_bins_df_dict, filename="val_bins_df_dict.pkl",
                                                 task="load")
        self.val_bins_labels_dict = self.pickle_dict(variable=self.val_bins_labels_dict,
                                                     filename="val_bins_labels_dict.pkl", task="load")
        self.train_val_bins_df_dict = self.pickle_dict(variable=self.train_val_bins_df_dict,
                                                       filename="train_val_bins_df_dict.pkl", task="load")
        self.train_val_bins_labels_dict = self.pickle_dict(variable=self.train_val_bins_labels_dict,
                                                           filename="train_val_bins_labels_dict.pkl", task="load")

    def build_IV_df(self):
        self.IV_df = pd.DataFrame(index=self.woe_dict.keys(), columns=["IV"])
        for key in self.woe_dict.keys():
            tmp_df = self.woe_dict[key]
            self.IV_df.loc[key, "IV"] = sum((tmp_df.healthy_distr - tmp_df.sick_distr) * tmp_df.WOE)
        IV_df_save_name = "IV_df.csv"
        self.IV_df.to_csv(os.path.join(self.save_to_folder, IV_df_save_name), index=True)
        print("IV_df:", self.IV_df)

    @staticmethod
    def make_dir_if_not_exist(path_to_create):
        if not os.path.exists(path_to_create):
            os.makedirs(path_to_create)

    def load_scores_csv(self):
        self.scoring = pd.read_csv(self.woe_csv_path)
        self.scoring.set_index(self.scoring.columns[0], inplace=True, drop=True)
        self.scoring.index.rename("Category", inplace=True)
        return self.scoring

    def calc_scores(self,mode=None,return_results=False,save=True):
        assert mode is None or mode=="explore", "mode should be either None or explore"
        original_mode = self.mode
        if mode is not None:
            self.mode=mode
            prefix="external_mode_"+mode+"_"
        else:
            prefix=""

        scoring = self.load_scores_csv()
        if self.mode == "explore":
            self.bins_df = pd.read_csv(os.path.join(self.save_to_folder, "val_bins_df.csv"), index_col="eid")
        else:
            self.bins_df = pd.read_csv(os.path.join(self.save_to_folder, "test_bins_df.csv"), index_col="eid")

        self.final_scores = self.bins_df.dot(scoring.loc[self.bins_df.columns, "Score"])
        if save:
            self.final_scores.to_csv(os.path.join(self.save_to_folder, prefix+"final_scores.csv"), index=True)
            self.final_scores.to_csv(os.path.join(self.summary_folder, prefix+"final_scores.csv"), index=True)
            self.bins_df.to_csv(os.path.join(self.save_to_folder, "bins_df.csv"), index=True)
            self.y_val.to_csv(os.path.join(self.save_to_folder, "y_val.csv"), index=True)
        self.mode = original_mode
        if return_results:
            return self.final_scores
    def calculate_scoreboards_ci(self,force_recalculate_ci=False):
        if self.final_scores is None or force_recalculate_ci:
            print("Calculating scores")
            self.calc_scores()
        if self.CI_Results_DF is None or force_recalculate_ci:
            print("Calculating ci")
            self.calc_ci_from_results()

        print("CI_Results_DF:", self.CI_Results_DF)
        print("Saving ScoreBoard to ", self.save_to_folder)
        with open(os.path.join(self.save_to_folder, "scoreboard_object_" + self.scoreboard_type+".pkl"), 'wb') as fp:
            pickle.dump(self, fp)
    print("finished CI calculation")

    def calc_ci_from_results(self):
        """
        Bootstrapping from results anc calculating the CI
        :return:
        """
        if self.mode == "explore":
            y_eval=self.y_val
        else:
            y_eval = self.y_test
        y_eval=y_eval.dropna()
        results_frame_list = []
        data_dict_list = []
        path = self.run_object.CI_results_path
        final_scores=self.final_scores.dropna()
        if len(final_scores)==0:
            sys.exit("final_scores is nan")
        y_eval=y_eval.loc[final_scores.index]
        for SN in np.arange(1, self.run_object.num_of_bootstraps + 1, 1):
            y_val_ci = y_eval.sample(n=y_eval.shape[0], replace=True)
            y_proba_ci = self.final_scores.loc[y_val_ci.index]
            APS = aps(y_val_ci, y_proba_ci)
            AUC = auroc(y_val_ci, y_proba_ci)
            precision, recall, _ = precision_recall_curve(y_val_ci, y_proba_ci)
            fpr, tpr, _ = roc_curve(y_val_ci, y_proba_ci)
            data_dict = {"AUROC": AUC, "APS": APS, "precision": precision, "recall": recall, "fpr": fpr, "tpr": tpr,
                         "y_proba": y_proba_ci,
                         "y_val.values": y_val_ci.values, "y_val_df": y_val_ci}
            data_dict_list.append(data_dict)
            results_df = pd.DataFrame.from_dict(
                {"APS": [APS], "AUC": [AUC], "SN": [SN]})
            results_df = results_df.set_index("SN", drop=True)
            # results_df.to_csv(os.path.join(path, "AUC_APS_results_" + str(int(SN)) + ".csv"), index=True)
            results_frame_list.append(results_df)
        ci_results_df = pd.concat(results_frame_list)
        self.CI_Results_DF = self.run_object.calc_CI(ci_results_df)
        if self.save_ci_reults_dict:
            for sn, data_dict in enumerate(data_dict_list):
                with open(os.path.join(path, "CI_Dict_" + str(int(sn))), 'wb') as fp:
                    pickle.dump(data_dict, fp)

    def calculate_scoreboards_bins_scores(self,
                                          build_new_feature_importance=True,
                                          leg_dict=None,
                                          figsize=(30, 24), font_size=36):
        # #Calculate the Feature importance for the anthropometrics
        # file_name==run_name ... ex. 'LR_Five_blood_tests_scoreboard'
        self.leg_dict = merge_two_dicts(self.leg_dict, leg_dict)
        # used_labels_df_path="/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/Scoreboards/WOE_computation/"+file_name+"/used_labels_df.csv"
        labels_dict, UKBIOBANK_labels_df = upload_ukbb_dict()
        file_names_list = [self.run_name]
        Folder_path = [self.save_to_folder]
        lr_features_coeff_pkl_path = os.path.join(self.save_to_folder, "coeffs_table_obj.pkl")
        fbt_hue_colors_dict = {"LR_Five_blood_tests_scoreboard": "red", "LR_Anthro_scoreboard": "green"}
        if build_new_feature_importance:
            try:
                os.rename(lr_features_coeff_pkl_path, lr_features_coeff_pkl_path[:-4] + "_old.pkl")
            except:
                print("!!!fbt_pickle does not exist!!! : \n", lr_features_coeff_pkl_path)
        try:
            with open(lr_features_coeff_pkl_path, "rb") as fp:  # Pickling
                lr_features_coeff_obj = pickle.load(fp)
        except:
            lr_features_coeff_obj = LR_Feature_importance(
                folder_path=Folder_path,
                positive_colors="red",
                negative_colors="pink",
                linespacing=1,
                plot=True,
                font_scale=5,
                leg_labels=leg_dict[self.run_name],
                leg_title=None,
                leg_pos=[0.9, 0.9],
                hue_type="binary",
                figsize=figsize,
                font_size=font_size,
                n_boots=10,
                space=0.2,
                fig_path=self.features_importance_fig_path,
                labels_dict=labels_dict,
                table_save_path=self.features_importance_tables_save_path,
                hue_colors_dict=fbt_hue_colors_dict,
                file_names_list=file_names_list,
                show_values_on_bar=True,
                remove_legend=False,
                ci_summary_table_name=self.ci_summary_table_name,
                build_new=build_new_feature_importance
            )
        print("coeffs_table_obj.long_name_order: ", lr_features_coeff_obj.long_name_order)
        print("coeffs_table_obj.mean_coefficients_table_df : ", lr_features_coeff_obj.mean_coefficients_table_df)
        categories = [re.split(' <| >| :', x)[0] for x in lr_features_coeff_obj.mean_coefficients_table_df.index]
        lr_features_coeff_obj.mean_coefficients_table_df["category"] = categories
        mean_coefficients_table_df = lr_features_coeff_obj.mean_coefficients_table_df

        print ("mean_coefficients_table_df:", mean_coefficients_table_df)
        with open(self.woe_dict_path, "rb") as fp:
            woe_dict = pickle.load(fp)

        n = len(mean_coefficients_table_df.loc[:, "category"].unique()) - 1  # Number of independent variablles in the model
        bias = mean_coefficients_table_df.loc["Bias", "Covariate coefficient"]

        pdo = 10
        max_or = 50
        Factor = pdo / np.log(2)
        Offset = 100 - Factor * np.log(max_or)

        print('Factor:', round(Factor, 2), 'Offset:', round(Offset, 2))
        woe_list = []
        for category in mean_coefficients_table_df.index:
            print("category:", category)
            category_features = []
            feat_coeff = mean_coefficients_table_df.loc[category, "Covariate coefficient"]
            print("feat_coeff:", feat_coeff)
            if category != "Bias":
                for ind, feature in enumerate(woe_dict[category].index):
                    category_features.append(feature)
                    woe_i = woe_dict[category].loc[feature, "WOE"]
                    woe_dict[category].loc[feature, "Score"] = int(
                        (woe_i * feat_coeff + bias / n) * Factor + Offset / n)
                    if ind == 0:
                        min_feat = woe_dict[category].loc[feature, "Score"]
                    else:
                        if woe_dict[category].loc[feature, "Score"] < min_feat:
                            min_feat = woe_dict[category].loc[feature, "Score"]
                woe_dict[category].loc[category_features, "Score"] = woe_dict[category].loc[
                                                                         category_features, "Score"] - min_feat
                woe_list.append(woe_dict[category])
        woe_df = pd.concat(woe_list)
        self.scoreboard=woe_df
        woe_df.to_csv(self.woe_csv_path)
        print ("woe_df:", woe_df)

def recover_scoreboard(
        scoreboard_type,save_to_scoreboards_basic_path,base_path):
    scoreboard = ScoreBoard(scoreboard_type=scoreboard_type,
                            build_new_database=False,
                            save_database=False,
                            train_val_test_type="Updated",
                            base_path=base_path,
                            save_to_scoreboards_basic_path=save_to_scoreboards_basic_path,
                            pdo=10,
                            max_or=50, save_ci_reults_dict=True, calculate_scoreboard=False,recover=True,
                            force_update_LR=False)

    print("Loading ScoreBoard", scoreboard_type, " from ", scoreboard.save_to_folder)
    with open(os.path.join(scoreboard.save_to_folder, "scoreboard_object_" + scoreboard_type+".pkl"), 'rb') as fp:
        scoreboard = pickle.load(fp)
    return scoreboard

def calculate_scoreboards_ci(scoreboard_type="Anthro_explore",
                             calc_ci=True,
                             save_scoreboard=True,
                             load_scoreboard=True,
                             base_path="/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/",
                             save_to_scoreboards_basic_path="/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/Scoreboards/",
                             ):

    scoreboard = ScoreBoard(scoreboard_type=scoreboard_type,
                            build_new_database=False,
                            save_database=False,
                            train_val_test_type="Updated",
                            base_path=base_path,
                            save_to_scoreboards_basic_path=save_to_scoreboards_basic_path,
                            pdo=10,
                            max_or=50, save_ci_reults_dict=True, calculate_scoreboard=False)

    if load_scoreboard:
        try:
            print("Loading ScoreBoard", scoreboard_type," from ", scoreboard.save_to_folder)
            with open(os.path.join(scoreboard.save_to_folder, "scoreboard_object_" + scoreboard_type+".pkl"), 'rb') as fp :
                scoreboard = pickle.load(fp)
        except:
            print("Failed to load ScoreBoard, calculating ScoreBoard", scoreboard_type," from ", scoreboard.save_to_folder)
            scoreboard.calculate_scoreboard()
            if save_scoreboard:
                with open(os.path.join(scoreboard.save_to_folder, "scoreboard_object_" + scoreboard_type+".pkl"), 'wb') as fp:
                    pickle.dump(scoreboard, fp)
    else:
        print("Calculating ScoreBoard ", scoreboard_type)
        scoreboard.calculate_scoreboard()  # Anthro, Five_blood
        if save_scoreboard:
            print("Saving ScoreBoard to ", scoreboard.save_to_folder)
            with open(os.path.join(scoreboard.save_to_folder, "scoreboard_object_" + scoreboard_type+".pkl"), 'wb') as fp:
                pickle.dump(scoreboard, fp)

    if calc_ci:
        if scoreboard.final_scores is None:
            print("Calculating scores")
            final_scores = scoreboard.calc_scores()
        if scoreboard.CI_Results_DF is None:
            print("Calculating ci")
            CI_Results_DF=scoreboard.calc_ci_from_results()
        else:
            CI_Results_DF=scoreboard.CI_Results_DF
        print ("CI_Results_DF:",CI_Results_DF)
    if save_scoreboard:
        print("Saving ScoreBoard to ",scoreboard.save_to_folder)
        with open(os.path.join(scoreboard.save_to_folder, "scoreboard_object_" + scoreboard_type)+".pkl", 'wb') as fp:
            pickle.dump(scoreboard, fp)
    print("finished")
    return scoreboard



# ## read_features_file_df
def read_features_file_df(ancestor_object_features_file_path):
    features_file_df = pd.read_csv(ancestor_object_features_file_path,
                                   usecols=["Field ID", "Description", "Exclude", "Datatype"])
    features_file_df = features_file_df.loc[features_file_df.loc[:, "Exclude"] == 0, :]
    features_file_df = features_file_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    features_dict = features_file_df[["Field ID", "Description"]].set_index('Field ID').to_dict()['Description']
    features_dict["31-0.0_0.0"] = "Sex"
    return features_file_df, features_dict

# ## create_feature_file

def create_feature_file(
        new_features_file_path,
        new_features_names,
        Type,
        sample_features_path="/home/edlitzy/Biobank/Diabetes_Features_lists/For_article/Antro_neto_whr.csv"):
    """
    new_features_file_path='/home/edlitzy/Biobank/Diabetes_Features_lists/For_article/Antro_scoreboard.csv'
    new_features_names=train_val_bins_df.columns
    """
    feat_file = pd.read_csv(sample_features_path)
    tmp_feat_df = pd.DataFrame(columns=feat_file.columns, index=new_features_names)
    feat_file.loc[:, "Exclude"] = 1
    tmp_feat_df.loc[:, "Field ID"] = new_features_names
    tmp_feat_df.loc[:, "Description"] = new_features_names
    tmp_feat_df.loc[:, "Datatype"] = "Categorical"
    tmp_feat_df.loc[:, "Exclude"] = 0
    tmp_feat_df.loc[:, "Type"] = Type + " ScoreBoard"
    tmp_feat_df.set_index("Field ID", inplace=True, drop=True)
    feat_file.set_index("Field ID", drop=True, inplace=True)
    new_feat_df = pd.concat([feat_file, tmp_feat_df], axis=0)
    new_feat_df.to_csv(new_features_file_path, index=True)
    return new_feat_df

def remove_cols(df, col_name_starts_with="2443"):
    remove_col = [x for x in df.columns if x.startswith("2443")]
    if remove_col != []:
        df.drop(remove_col, axis=1)
    return df

def bin_df(X_df, y, features_file_df, max_bins=5, res_dict=None, bins_labels_dict="Train",min_num_of_ills=1):
    X_df = remove_cols(X_df, col_name_starts_with="2443")
    features_file_df = features_file_df.reset_index().set_index("Description", drop=True)
    if bins_labels_dict == "Train":
        bins_df, bins_df_dict, bins_labels_dict,res_dict = bin_df_train(X_df=X_df, y=y,
                                                                        features_file_df=features_file_df,
                                                                        res_dict=res_dict,min_num_of_ills=min_num_of_ills)
    else:
        bins_df, bins_df_dict, bins_labels_dict = bin_df_val_test_df(df=X_df, y=y, features_file_df=features_file_df,
                                                                     bins_labels_dict=bins_labels_dict)
    bins_df = replace_df_dummy_labels(bins_df, bins_labels_dict,features_file_df=features_file_df)
    bins_df = remove_cols(bins_df, col_name_starts_with="2443")
    print("Final woe_res_dict:",res_dict)
    return bins_df, bins_df_dict, bins_labels_dict,res_dict

def set_category_labels(col):
    """
    Manually setting the labels for categorical features
    :param col:
    :return:
    """
    if col=="Sex":
        lables=["Female","Male"]
    else:
        sys.exit(col, " labels are not defined in set_category labels")
    return lables

def ensure_ills_in_bin(X_df,y,col,nbins,min_num_of_ills=1):
    qcut = pd.qcut(X_df.loc[:, col], q=nbins, retbins=True, duplicates='drop')
    qc_df=pd.DataFrame(data={"bins":qcut[1],"y_sum":None})
    bottom_boundary=qc_df["bins"].iloc[:-1].values.flatten()
    top_boundary=qc_df["bins"].iloc[1:].values.flatten()
    for ind, (bottom_bin, top_bin) in enumerate(zip(bottom_boundary,top_boundary)):
        qc_df.loc[qc_df.index[ind], "y_min_ills_not_satisfied"] = y.loc[
            (X_df.loc[:, col] >= bottom_bin) & (X_df.loc[:, col] < top_bin)].sum()<min_num_of_ills
    if qc_df.loc[:, "y_min_ills_not_satisfied"].sum():
        nbins,qcut=ensure_ills_in_bin(X_df,y, col, nbins=nbins-1, min_num_of_ills=min_num_of_ills)
        return nbins,qcut
    else:
        return nbins,qcut

def bin_continous_categories(bins_res, len_unique, X_df, col,y,min_num_of_ills=1):
    #% recursively checks if the qcut provides unique values, if not, reduces number of bins
    bins_res = np.min([bins_res, len_unique + 1])
    bins_res,qcut=ensure_ills_in_bin(X_df,y, col, bins_res, min_num_of_ills=min_num_of_ills)
    labels = round_bins_limits(qcut[1])

    if len(set(labels))==bins_res+1: #Check that there are no two identical bin's limits
        print("col:",col,", bins resolution:",bins_res, ", bins:",bins_res)
        return labels,bins_res
    else:
        bins_res-=1
        return bin_continous_categories(bins_res=bins_res, len_unique=len_unique,X_df=X_df, col=col,y=y,min_num_of_ills=min_num_of_ills)


def bin_df_train(X_df, y, features_file_df, res_dict=None,min_num_of_ills=1):
    """
    Binning of the inputs using quantiles if not categorical, otherwise using the unique values.
    df_dummies is a dataframe with a one hot encoder for all the bins, bins_labels_dict is a a dictionary for the
    labels limits
    """
    df_dummy_list = []
    bins_labels_dict = {}
    bins_df_dict = {}
    for col in X_df.columns:
        print("col is ", col)
        col = col.strip()
        if col == "2443-3.0":
            continue
        unique_labels = list(set(X_df.loc[:, col].values))
        len_unique = len(unique_labels)
        if features_file_df.loc[col, "Datatype"] == "Categorical":
            df_bin = X_df.loc[:, col]
            df_dummies = pd.get_dummies(df_bin, prefix=col, prefix_sep=':')
            bins_labels_dict[col] = set_category_labels(col)
        else:
            try:
                bins_res = res_dict[col]
            except:
                print("could not set ",col," bins resolution")
                sys.exit("could not set"+col+"bins resolution")
            labels, bins_res= bin_continous_categories(bins_res, len_unique, X_df,col,y=y,min_num_of_ills=min_num_of_ills)
            res_dict[col]=bins_res
            df_bin = X_df.loc[:, col].apply(find_bin, bin_limits=labels)
            bins_labels_dict[col] = labels
            df_dummies = pd.get_dummies(df_bin, prefix=col, prefix_sep=':')

        bins_df_dict[col] = pd.concat([df_bin, y], axis=1)
        df_dummy_list.append(df_dummies)
    #     df_dummy_list.append(y)
    df_dummies = pd.concat(df_dummy_list, axis=1)
    return df_dummies, bins_df_dict, bins_labels_dict,res_dict

def bin_df_val_test_df(df, y, features_file_df, bins_labels_dict):
    """
    Binning of the inputs using quantiles if not categorical, otherwise using the unique values.
    df_dummies is a dataframe with a one hot encoder for all the bins, bins_labels_dict is a a dictionary for the
    labels limits
    """
    bins_df_dict = {}
    df_dummy_list = []
    for col in df.columns:
        col = col.strip()
        if col == "2443-3.0":
            continue
        if features_file_df.loc[col, "Datatype"] == "Categorical":
            df_bin = df.loc[:, col]
            # unique_labels = bins_labels_dict[col]
            # if len(unique_labels) == 2:
            #     df_dummies = df_bin
            # else:
            df_dummies = pd.get_dummies(df_bin, prefix=col, prefix_sep=':')
        else:
            bin_limits = bins_labels_dict[col]
            df_bin = df.loc[:, col].apply(find_bin, bin_limits=bin_limits)
            df_dummies = pd.get_dummies(df_bin, prefix=col, prefix_sep=':')
        df_dummy_list.append(df_dummies)
        bins_df_dict[col] = pd.concat([df_bin, y])
    #     df_dummy_list.append(y)
    df_dummies = pd.concat(df_dummy_list, axis=1)
    return df_dummies, bins_df_dict, bins_labels_dict

def find_bin(row, bin_limits):
    for ind, val in enumerate(bin_limits):
        if ind < (len(bin_limits) - 1) and (row <= bin_limits[ind + 1]):
            return ind + 1
        elif ind == (len(bin_limits) - 1) and (row >= bin_limits[ind]):
            return ind
    sys.exit("No bin was found for val:" + str(val))

def round_bins_limits(bins_list):
    new_bins_list = []
    for element in bins_list:
        if element != 0:
            order = int(np.floor(np.log10(element)))
            if order < 1:
                element = np.round(element, np.abs(order - 1))
            elif element < 2:
                element = np.round(element, 2)
            else:
                element = np.round(element,1)
        if (element).is_integer():
            element = int(element)
        new_bins_list.append(element)
    print("new_bins_list:", new_bins_list)
    return new_bins_list

def replace_woe_df_labels(name,current_woe_indexes,bins_labels_list,features_file_df):
    num_of_categories = len(current_woe_indexes)
    indexes_map_dict={}
    print(name)
    features_file_df=features_file_df.reset_index(drop=True).set_index("Description")
    if features_file_df.loc[name, "Datatype"] == "Categorical":
        indexes_map_dict={current_woe_indexes[ind]: name+" : " + category for ind, category in enumerate(bins_labels_list)}
    else:
        for num in current_woe_indexes:
            num=int(num)
            if num==1:
                indexes_map_dict[num]= name + " <= " + str(bins_labels_list[1])
            elif num<num_of_categories:
                indexes_map_dict[num] = name + " > " + str(bins_labels_list[num - 1]) + " and <= " + str(bins_labels_list[num])
            elif num==num_of_categories:
                indexes_map_dict[num] = name + " > " + str(bins_labels_list[num - 1])
            else:
                print("num in current_woe_indexes is:", num," check if not exceeding the num_of_categories:",num_of_categories)
                sys.exit("in replace_woe_df_labels()")
    return indexes_map_dict

def replace_woe_dict_dummy_labels(name,woe_df, bins_labels_dict,features_file_df):
    if name.startswith("2443"):
        sys.exit("2443 should not be in a feature (replace_df_dummy_labels()) of bins_df")
    print("in name:", name)

    current_woe_indexes=woe_df.index.values
    bins_labels_list=bins_labels_dict[name]
    indexes_map_dict=replace_woe_df_labels(name,current_woe_indexes,bins_labels_list,features_file_df)
    print(indexes_map_dict)
    woe_df.rename(index=indexes_map_dict,inplace=True)
    return woe_df

def replace_df_dummy_labels(bins_df, bins_labels_dict,features_file_df):
    """
    Replacing the dummy column labels of the dummy dataframe
    """
    new_col_names = []
    for col in bins_df.columns:
        print("in col:", col)
        if ":" in col:
            col_parts = col.split(":")
            name = col_parts[0]
            assert type(
                name) == str, "Oh no! col.split(':')[0] of" + col + "is not a string type infreplace_df_dummy_labels"
            if name.startswith("2443"):
                sys, exit("2443 should not be in a feature (replace_df_dummy_labels()) of bins_df")
            num_of_categories = len(bins_labels_dict[name])
            category = int(float(col_parts[1]))
            if features_file_df.loc[name, "Datatype"] == "Categorical":
                cat_name = name + " : "+ bins_labels_dict[name][category]
            else:
                if category == 1:
                    cat_name = name + " <= " + str(bins_labels_dict[name][1])
                else:
                    if num_of_categories > category + 1:
                        cat_name = name + " > " + str(bins_labels_dict[name][category - 1]) + " and <= " + str(
                            bins_labels_dict[name][category])
                    else:
                        cat_name = name + " > " + str(bins_labels_dict[name][category - 1])

        else:
            print("col:", col)
            sys.exit(': is not in col')
        print("cat_name:",cat_name)
        assert type(cat_name)==str, "Oh no! type(cat_name)!=str"
        new_col_names.append(cat_name)
    bins_df.columns = new_col_names
    return bins_df

def calc_woe(df, col_name,labels_dict):
    columns = ['counts', 'sick', 'healthy', 'total_distr', 'sick_distr', 'healthy_distr', 'WOE', 'WOE%']
    ind = list(set(df.iloc[:, 0].values))#TODO replace indices with names of features
    ind.sort()
    woe_df = pd.DataFrame(index=ind, columns=columns)
    woe_df["counts"] = df.groupby(col_name).count().loc[ind, :]
    woe_df["sick"] = df.groupby([col_name]).sum().loc[ind, :]
    woe_df["healthy"] = woe_df["counts"] - woe_df["sick"]
    woe_df["total_distr"] = woe_df["counts"] / woe_df["counts"].sum()
    woe_df["sick_distr"] = woe_df["sick"] / woe_df["sick"].sum()
    woe_df["healthy_distr"] = woe_df["healthy"] / woe_df["healthy"].sum()
    woe_df['WOE'] = np.log(woe_df.sick_distr/ woe_df.healthy_distr )
    woe_df['WOE%'] = woe_df.WOE * 100
    woe_df.rename(index=labels_dict)
    return woe_df

def concat_train_val_test_dataframes(Orig_df_file_path, scoretable_df, save_to_path):
    """
    Adding the scoretables paramaters to the original database
    :param Orig_df_file_path:
    :param scoretable_df:
    :param save_to_path:
    :param save:
    :return:
    """
    Orig_df = pd.read_csv(Orig_df_file_path, index_col="eid")
    if "Unnamed: 0" in Orig_df.columns:
        Orig_df.drop("Unnamed: 0", axis=1, inplace=True)
    if "2443-3.0" in scoretable_df.columns:
        scoretable_df.drop("2443-3.0", axis=1, inplace=True)
    scoretable_col = scoretable_df.columns
    new_columns = [x for x in scoretable_col if x not in Orig_df.columns]

    # exist_columns=[x for x in scoretable_col if x  in Orig_df.columns]

    # Orig_df.loc[:,exist_columns]=scoretable_df.loc[Orig_df.index,exist_columns]
    if new_columns !=[]:
        new_df = Orig_df.join(scoretable_df.loc[:, new_columns],how="left")
    else:
        new_df=Orig_df.copy()

    mut_cols=[x for x in scoretable_col if x in Orig_df.columns]
    if mut_cols != []:
        mut_index=[x for x in scoretable_df.index if x in new_df.index]
        new_df.loc[mut_index,mut_cols]=scoretable_df.loc[mut_index,mut_cols]

    new_df.to_csv(save_to_path, index=True)
    return new_df

def plot_woe_graphs(woe_dict,save_to_folder):
    for key in woe_dict.keys():
        fig, ax = plt.subplots(1, 1, figsize=(9, 6))
        woe_dict[key].loc[:, "WOE"].plot(ax=ax)
        ax.set_title("WOE plot of " + key, fontsize=16)
        ax.set_xlabel("Bin number", fontsize=16)
        ax.set_ylabel("WOE score", fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=16)
        filename = key + "_WOE.jpeg"
        save_to=os.path.join(save_to_folder, filename)
        plt.savefig(save_to)
        print("save graph to ",save_to)

def build_and_save_woe_dict(bins_df_dict, save_to_folder,bins_labels_dict,features_file_df):#train_bins_df_dict
    woe_dict = {}
    for col in bins_df_dict.keys():
        print(col)
        df = bins_df_dict[col]
        woe_df=calc_woe(df, col,bins_labels_dict)
        woe_dict[col]= replace_woe_dict_dummy_labels(col,woe_df,bins_labels_dict,features_file_df)
        save_name =os.path.join(save_to_folder, col + "_woe_dict_as_csv.csv")
        print(woe_dict[col])
        woe_dict[col].to_csv(save_name, index=False)
        print("woe[",col,"] saved csv to ",save_name)
    pickle_to=os.path.join(save_to_folder, "woe_dict.pkl")
    with open(pickle_to, 'wb') as handle:
        pickle.dump(woe_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("pickled doe dict to:",pickle_to)
    return woe_dict


def calculate_scoreboards_ci(scoreboard_type="Anthro_explore",
                             calc_ci=True,
                             save_scoreboard=True,
                             load_scoreboard=True,
                             scoreboard=None):

    if scoreboard is None:
        scoreboard = ScoreBoard(scoreboard_type=scoreboard_type,
                                build_new_database=False,
                                save_database=False,
                                train_val_test_type="Updated",
                                base_path="/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/",
                                save_to_scoreboards_basic_path="/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/Scoreboards/",
                                pdo=10,
                                max_or=50, save_ci_reults_dict=True, calculate_scoreboard=False)
        if load_scoreboard:
            try:
                print("Loading ScoreBoard", scoreboard_type," from ", scoreboard.save_to_folder)
                with open(os.path.join(scoreboard.save_to_folder, "scoreboard_object_" + scoreboard_type+".pkl"), 'rb') as fp :
                    scoreboard = pickle.load(fp)
            except:
                print("Failed to load ScoreBoard, calculating ScoreBoard", scoreboard_type," from ", scoreboard.save_to_folder)
                scoreboard.calculate_scoreboard()
                if save_scoreboard:
                    with open(os.path.join(scoreboard.save_to_folder, "scoreboard_object_" + scoreboard_type+".pkl"), 'wb') as fp:
                        pickle.dump(scoreboard, fp)
        else:
            print("Calculating ScoreBoard ", scoreboard_type)
            scoreboard.calculate_scoreboard()  # Anthro, Five_blood
            if save_scoreboard:
                print("Saving ScoreBoard to ", scoreboard.save_to_folder)
                with open(os.path.join(scoreboard.save_to_folder, "scoreboard_object_" + scoreboard_type+".pkl"), 'wb') as fp:
                    pickle.dump(scoreboard, fp)

    if calc_ci:
        if scoreboard.final_scores is None:
            print("Calculating scores")
            final_scores = scoreboard.calc_scores()
        if scoreboard.CI_Results_DF is None:
            print("Calculating ci")
            CI_Results_DF=scoreboard.calc_ci_from_results()
        else:
            CI_Results_DF=scoreboard.CI_Results_DF
        print ("CI_Results_DF:",CI_Results_DF)
    if save_scoreboard:
        print("Saving ScoreBoard to ",scoreboard.save_to_folder)
        with open(os.path.join(scoreboard.save_to_folder, "scoreboard_object_" + scoreboard_type+".pkl"), 'wb') as fp:
            pickle.dump(scoreboard, fp)
    print("finished")
    return scoreboard

def merge_two_dicts(x, y):
    if y is not None:
        z = x.copy()   # start with x's keys and values
        z.update(y)    # modifies z with y's keys and values & returns None
        return z
    else:
        return x


def main():
    print("In CI_Configs")
if __name__=="__main__":
    main()