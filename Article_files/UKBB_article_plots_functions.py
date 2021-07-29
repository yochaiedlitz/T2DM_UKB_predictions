from UKBB_article_plots_config import *
from imports import *


def update_dictionary(current_dict, new_codes_csv_file, save=True):
    """New_codes_csv should be with columns 'Codes' without -0.0 and 'Labels'  """
    labels_df = pd.read_csv(new_codes_csv_file, index_col="Code")
    for code in labels_df.index.values:
        #         print (code)
        current_dict[code] = labels_df.loc[code, "Label"]
    new_labels_df = pd.DataFrame.from_dict(current_dict, orient='index')
    new_labels_df.index.rename("Code", inplace=True)
    new_labels_df.columns = ["Label"]
    if save:
        new_labels_df.to_csv(new_codes_csv_file)
    return current_dict, new_labels_df


def upload_ukbb_dict(path="/home/edlitzy/UKBB_Tree_Runs/For_article/UKBIOBANK_Labels.csv",
                     dropped_df_path="/home/edlitzy/UKBB_Tree_Runs/For_article/dropped_UKBIOBANK_Labels.csv"):
    UKBIOBANK_dict = {}
    UKBIOBANK_labels_df = pd.read_csv(path, index_col="Code")
    dropped_df = UKBIOBANK_labels_df[UKBIOBANK_labels_df.index.duplicated(keep='first')]
    UKBIOBANK_labels_df = UKBIOBANK_labels_df[~UKBIOBANK_labels_df.index.duplicated(keep='first')]
    dropped_df.to_csv(dropped_df_path, index=True)
    print("dropped the following lines due to duplicated indexes:")
    print(dropped_df)
    print("A dataframe with the dropped columns was saved to: ", dropped_df_path)
    for key in UKBIOBANK_labels_df.index.values:
        UKBIOBANK_dict[key] = UKBIOBANK_labels_df.loc[key, "Label"]
    return UKBIOBANK_dict, UKBIOBANK_labels_df


def replace_chars(value, deletechars='\/:*?"<>|'):
    """
    Removes non-legal chars
    :param value:
    :param deletechars:
    :return:
    """
    for c in deletechars:
        value = value.replace(c, '')
    value.replace(" ", "_")
    return value;


def get_cmap(n, name='inferno'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def sort_scores(y_files, predictions_files, labels, test="AUC"):
    y_scores_df = pd.DataFrame({
        "y_files": y_files,
        "predictions_files": predictions_files,
        "labels": labels
    })
    y_scores_df = y_scores_df.set_index("labels")
    for ind, label in enumerate(labels):
        with open(predictions_files[ind], 'rb') as fp:
            y_pred_val = pickle.load(fp)
        with open(y_files[ind], 'rb') as fp:
            y_test_val = pickle.load(fp)
        y_scores_df.loc[label, "AUC"] = roc_auc_score(y_test_val, y_pred_val)
        y_scores_df.loc[label, "APS"] = average_precision_score(
            y_test_val, y_pred_val)
    return y_scores_df


def wrap_labels(labels, num_of_chars):
    if type(labels) == list or type(labels) == np.ndarray:
        labels = ['\n'.join(wrap(l, num_of_chars, break_long_words=False, break_on_hyphens=True)) for l in labels]
    else:
        labels = '\n'.join(wrap(labels, num_of_chars))
    return labels


def calc_precision_recall_list(df, sort=True):
    # How can be "simple" or "relative"
    precision_list = []
    recall_list = []
    if sort:
        df = df.sort_values('APS')
    y_files = df.y_files.values
    predictions_files = df.predictions_files.values
    labels = df.index.values
    APS = df.APS.values
    n = len(labels)
    if (all(len(x) == n for x in [y_files, predictions_files])):
        for ind, label in enumerate(labels):
            with open(df.predictions_files.values[ind], 'rb') as fp:
                y_pred_val = pickle.load(fp)
            with open(df.y_files.values[ind], 'rb') as fp:
                y_test_val = pickle.load(fp)
            precision, recall, _ = precision_recall_curve(
                y_test_val, y_pred_val)
            precision_list.append(precision)
            recall_list.append(recall)
    return precision_list, recall_list


# ## Calc_roc_list

# In[10]:


def calc_roc_list(df, sort=True):
    # How can be "simple" or "relative"
    fpr_list = []
    tpr_list = []
    if sort:
        df = df.sort_values('AUC')
    y_files = df.y_files.values
    predictions_files = df.predictions_files.values
    labels = df.index.values
    AUC = df.AUC.values
    n = len(labels)
    if (all(len(x) == n for x in [y_files, predictions_files])):
        for ind, label in enumerate(labels):
            with open(df.predictions_files.values[ind], 'rb') as fp:
                y_pred_val = pickle.load(fp)
            with open(df.y_files.values[ind], 'rb') as fp:
                y_test_val = pickle.load(fp)
            fpr, tpr, _ = roc_curve(y_test_val, y_pred_val)
            fpr_list.append(fpr)
            tpr_list.append(tpr)
    return fpr_list, tpr_list


def calc_prevalence(df):
    try:
        file_name = df.y_files.values[0]
    except:
        file_name = df.y_files
    #     print (df)
    with open(file_name, 'rb') as fp:
        y_test_val = pickle.load(fp)
    prevalence = np.sum(y_test_val) / len(y_test_val)
    return prevalence


class create_folder_table:
    def __init__(self, directories_path, CI_Results_path=[],
                 conversion_files_df=[],
                 Save_to_Table_path=[], Labels=None,
                 update_labels_dict=[],
                 save_to_file=True,
                 save_prefix=[]):
        if type(Save_to_Table_path) == list:
            Save_to_Table_path = directories_path
        if type(save_prefix) == list:
            save_prefix = os.path.dirname(directories_path)
        self.directories_path = directories_path
        self.name = self.get_folder_name()
        folder_directories = [x for x in os.listdir(self.directories_path) if
                              (x != "shap_folder" and x != "LR_comparison" and
                               os.path.isdir(os.path.join(self.directories_path, x)))]
        File_names = [x.replace('_Diabetes', '') for x in folder_directories]

        if type(conversion_files_df) == list:
            try:
                conversion_files_df = pd.read_csv(root_path=CI_Results_path, index_col="File name")
            except:
                conversion_files_df = CI_Summary_table_object(root_path=CI_Results_path, save_to_file=save_to_file,
                                                              save_prefix=save_prefix).get()
        elif conversion_files_df.index.name != 'File name':
            conversion_files_df = conversion_files_df.reset_index().set_index('File name', drop=True)

        conversion_files_df = conversion_files_df.loc[File_names, :]
        #         print("above 'conversion_files_df.set_index('Label',drop=False,inplace=True)', conversion_files_df.head(): ",conversion_files_df.head())

        conversion_files_df = conversion_files_df.reset_index()
        conversion_files_df['Label'] = conversion_files_df.apply(
            lambda row: row['File name'] if row['Label'] == None else row['Label'], axis=1)
        conversion_files_df = conversion_files_df.set_index('Label', drop=True)

        Articles_table_df = conversion_files_df.loc[:, [u'APS_lower',
                                                        u'APS_mean', u'APS_upper', u'AUROC_lower', u'AUROC_mean',
                                                        u'AUROC_upper']]
        Articles_table_df["APS"] = Articles_table_df.apply(lambda row: self.return_aps_plus_ci(row), axis=1)
        Articles_table_df["AUROC"] = Articles_table_df.apply(lambda row: self.return_auroc_plus_ci(row), axis=1)
        if Labels == None:
            self.Folder_table_df = Articles_table_df.loc[:, ["APS", "AUROC", "APS_mean", "AUROC_mean"]]
        else:
            self.Folder_table_df = Articles_table_df.loc[Labels, ["APS", "AUROC", "APS_mean", "AUROC_mean"]]
        self.Folder_table_df.sort_values(by="AUROC", inplace=True)
        self.Folder_table_df = self.Folder_table_df.reset_index().dropna().set_index('Label')
        self.Labels = self.Folder_table_df.index.values
        self.name = self.get_folder_name()
        self.name = self.replace_chars()
        self.save_path = os.path.join(Save_to_Table_path, self.name + "_Summary_Table.csv")
        if type(update_labels_dict) != list:
            self.update_labels(update_labels_dict)
        else:
            self.Folder_table_df.to_csv(self.save_path)

    def get_folder_name(self):
        if self.directories_path[-1] == "/":
            return os.path.basename(os.path.dirname(self.directories_path))
        else:
            return os.path.basename(self.directories_path)

    def return_aps_plus_ci(self, row):
        aps = str(row["APS_mean"]) + " (" + str(row["APS_lower"]) + "-" + str(row["APS_upper"]) + ")"
        return aps

    def return_auroc_plus_ci(self, row):
        auroc = str(row["AUROC_mean"]) + " (" + str(row["AUROC_lower"]) + "-" + str(row["AUROC_upper"]) + ")"
        return auroc

    def replace_chars(self, deletechars='\/:*?"<>|'):
        for c in deletechars:
            self.name = self.name.replace(c, '')
        self.name.replace(" ", "_")
        return self.name;

    def update_labels(self, replace_dict, save=True):
        tmp_df = self.Folder_table_df.reset_index()
        for ind, tmp_label in enumerate(tmp_df.loc[:, "Label"].values):
            for key in replace_dict.keys():
                if key in tmp_label:
                    tmp_label = tmp_label.replace(key, replace_dict[key])
            tmp_df.loc[ind, "Label"] = tmp_label
        tmp_df = tmp_df.set_index(keys="Label", drop=True)
        self.Folder_table_df = tmp_df
        self.to_csv()
        return self.Folder_table_df

    def get_columns(self):
        return self.Folder_table_df.columns

    def get_index(self):
        return self.Folder_table_df.index

    def set_val(self, ind, col, val):
        self.Folder_table_df.loc[ind, col] = val

    def get_val(self, ind, col, val):
        return self.Folder_table_df.loc[ind, col]

    def get(self, ):
        return self.Folder_table_df

    def to_csv(self, path=None):
        if path == None:
            path = self.save_path
        self.Folder_table_df.to_csv(path, index=True)
        self.save_path = path
        print("Saved to: ", self.save_path)

    def quant_update_folder_table(self, folder_path,
                                  ci_path="/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/Imputed_screened_CI_Summary.csv"):
        #   labels_dict=create_labels_dict()
        CI_df = pd.read_csv(ci_path, index_col="Label")
        base_name = os.path.basename(folder_path)
        folder_table_path = os.path.join(folder_path, base_name + "_Summary_Table.csv")
        Folder_table = pd.read_csv(folder_table_path, index_col="Label")
        Folder_table.loc[:, "Quantiles fold"] = CI_df.loc[Folder_table.index, "quant_CI"]
        Folder_table.to_csv(folder_table_path, index=True)
        return Folder_table


def correct_csv_endings(dir_list, wrong_ending=".0.csv", correct_ending=".csv"):
    for tmp_dir in dir_list:
        for filename in os.listdir(tmp_dir):
            if filename.endswith(wrong_ending):
                src = os.path.join(tmp_dir, filename)
                dst = src.replace(wrong_ending, correct_ending)
                #                 print dst
                os.rename(src, dst)


class CI_Summary_table_object:
    def __init__(self, root_path, save_to_file=False, update=True,
                 save_prefix="Article"):
        print("Root path:", root_path)
        try:
            save_path = os.path.join(root_path, save_prefix + "_CI_Summary.csv")
        except:
            save_path = root_path
        self.save_path = save_path
        self.color_dict = {}
        self.UKBIOBANK_dict = {}
        self.colors_csv = pd.DataFrame()
        self.UKBIOBANK_labels_df = pd.DataFrame()
        self.colors_csv_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/colors.csv"
        self.ci_path_dict = {}

        if update:
            try:
                old_CI_table = pd.read_csv(self.save_path, index_col="File name")
            except:
                old_CI_table = pd.DataFrame(
                    columns=["File name", 'Label', 'Type', 'APS_lower', 'APS_mean', 'APS_median',
                             'APS_upper', 'AUROC_lower', 'AUROC_mean', 'AUROC_median',
                             'AUROC_upper', 'Path'])
                old_CI_table.set_index("File name", inplace=True)
        file_names = old_CI_table.index.values
        added_file_list = []
        ci_csv_list = []
        for path, subdirs, files in os.walk(root_path):
            for name in files:
                if ((name.endswith("CI.csv") or name == "CI_results.csv")
                        and not name.startswith("CI_Dict")
                        and not name.startswith("Articles_CI_Summary")
                        and not (name in file_names)):
                    tmp_file_path = os.path.join(path, name)
                    #             all_file_list.append(tmp_file_path)
                    tmp_csv = pd.read_csv(tmp_file_path, index_col=None)
                    split_path = tmp_file_path.split("/")
                    if "CI_results.csv" in split_path:
                        tmp_run_type = split_path[-5]
                        tmp_folder_name = split_path[-4][:-9]
                    else:
                        tmp_run_type = split_path[-4]
                        tmp_folder_name = split_path[-3]
                    tmp_csv["Type"] = tmp_run_type
                    tmp_csv.iloc[0, 0] = tmp_folder_name
                    tmp_file_name = tmp_csv.iloc[0, 0]
                    tmp_csv = tmp_csv.set_index(tmp_csv.columns[0], drop=True)
                    tmp_csv.index.name = "File name"
                    if "Label" not in tmp_csv.columns:
                        tmp_csv["Label"] = tmp_file_name
                    tmp_csv["Path"] = tmp_file_path
                    tmp_list_value_no_Label = list(tmp_csv.columns.values)
                    tmp_list_value_no_Label.remove("Label")
                    tmp_csv = tmp_csv.loc[:, ["Label"] + (tmp_list_value_no_Label)]
                    for col in tmp_csv.columns:
                        if type(tmp_csv[col].values[0]) == np.float64:
                            tmp_csv[col] = str("{0:.2f}".format(tmp_csv[col].values[0]))
                    ci_csv_list.append(tmp_csv)
                    added_file_list.append(name)
        uploaded_new_CI_table = pd.concat(ci_csv_list)
        uploaded_new_CI_table.index.name = "File name"

        new_CI_Summary_table = pd.concat([old_CI_table, uploaded_new_CI_table])

        file_name = list(new_CI_Summary_table.index.values)
        duplicates = set([x for x in file_name if file_name.count(x) > 1])
        if len(duplicates) > 0:
            print ("!!!! PAY ATTENTION dropped the following duplicates !!!!!!", duplicates)
            #             print("new_CI_Summary_table:",new_CI_Summary_table)
            new_CI_Summary_table = new_CI_Summary_table.reset_index().drop_duplicates(
                subset=["File name"], keep='first').set_index("File name", drop=True)
        self.CI_Summary_table = new_CI_Summary_table
        self.Old_CI_Summary_table = old_CI_table

        self.columns = self.CI_Summary_table.columns.values.tolist()
        self.columns.remove("Label")
        self.columns.remove("Type")
        self.columns = ["Label", "Type"] + self.columns
        self.Old_CI_Summary_table = self.Old_CI_Summary_table.loc[:, self.columns]
        self.CI_Summary_table = self.CI_Summary_table.loc[:, self.columns]
        self.index = self.CI_Summary_table.index.values.tolist()
        self.check_nan_labels()
        if save_to_file:
            self.Old_CI_Summary_table.to_csv(os.path.join(root_path, "Articles_CI_Summary_previous_version.csv")
                                             , index=True)
            #             print ("saving:", self.conversion_files_obj.head(),save_path)
            self.CI_Summary_table.to_csv(save_path, index=True)
        self.load_color_dict()
        self.upload_ukbb_dict()
        self.calc_ci_path_dict()

    def update(self, ind_name, col_name, val, save=True):
        self.CI_Summary_table.loc[ind_name, col_name] = val
        if save:
            self.to_csv(save_path=self.save_path)

    def to_csv(self, save_path):
        self.CI_Summary_table.to_csv(save_path, index=True)

    def get(self):
        return self.CI_Summary_table

    def check_nan_labels(self):
        nan_labels_list = self.CI_Summary_table.loc[:, "Label"].isna()
        self.nan_labels = self.CI_Summary_table.loc[nan_labels_list, :]
        print ("The folowing files has no labels:", self.nan_labels)
        replace_nan_labels = self.CI_Summary_table.loc[nan_labels_list, :].index
        replace_nan_labels = [x.replace("_", " ") for x in replace_nan_labels]
        replace_nan_labels = [x.replace("LR ", "") + " LR" for x in replace_nan_labels if "LR " in x]
        self.CI_Summary_table.loc[nan_labels_list, "Label"] = replace_nan_labels

    def load_color_dict(self):
        self.colors_csv = pd.read_csv(self.colors_csv_path, index_col="File name")
        colors_csv = self.colors_csv
        #         print (self.colors_csv)
        for key in colors_csv.loc[:, "Label"]:
            #             print(key)
            #             print("colors_csv.loc[colors_csv.loc[:,'Label']==key,'color'].values:",colors_csv.loc[colors_csv.loc[:,"Label"]==key,"color"].values)
            self.color_dict[key] = colors_csv.loc[colors_csv.loc[:, "Label"] == key, "color"].values[0]
        return self.color_dict, self.colors_csv

    def get_color_dict(self):
        return self.color_dict

    def save_color_dict(self):
        np.save("/home/edlitzy/UKBB_Tree_Runs/For_article/color_dict.npy", self.color_dict)
        self.colors_csv.to_csv(self.colors_csv_path, index=True)

    def update_color_dict(self, label, color,filename):
        self.old_color_dict = self.color_dict
        print("Old version of dolor_dictionary is stored at old_color_dict")
        self.color_dict[label] = color
        self.colors_csv.reset_index().set_index("File name",inplace=True)
        self.colors_csv.loc[filename,["color","Label"]]=[color,label]
        self.save_color_dict()

    def upload_ukbb_dict(self, path="/home/edlitzy/UKBB_Tree_Runs/For_article/UKBIOBANK_Labels.csv"):
        self.UKBIOBANK_labels_df = pd.read_csv(path, index_col="Code")
        for key in self.UKBIOBANK_labels_df.index.values:
            self.UKBIOBANK_dict[key] = self.UKBIOBANK_labels_df.loc[key, "Label"]

    def get_UKBB_dict(self):
        return self.UKBIOBANK_dict

    def get_UKBIOBANK_labels_df(self):
        return self.UKBIOBANK_labels_df

    def calc_ci_path_dict(self):
        df = self.CI_Summary_table
        for ind, key in enumerate(df.index.values):
            full_path = df.Path.values[ind]
            dirname = (os.path.dirname(full_path))
            dirname = dirname + "/"
            label = df.Label.values[ind]
            self.ci_path_dict[label] = dirname

    def get_ci_path_dict(self):
        return self.ci_path_dict

    def get_specific_ci_path_dict(self, dirs_list):
        new_dict = {}
        for dir_name in dirs_list:
            if dir_name.endswith("_Diabetes"):
                dir_name = dir_name.replace("_Diabetes", "")
            new_dict[dir_name] = self.ci_path_dict[dir_name]
        return new_dict


def calc_roc_aps_lists(directories_path,
                       dirs_names=None,
                       conversion_files_df=[],
                       directories_to_skip=[],
                       CI_Results_path=[],
                       Save_to_Table_path=[]):
    """
    sorted_label_name - list of the label names as sorted when uploading files from directory
    returns:df, precision_list, recall_list, fpr_list, tpr_list
    """
    print("In calc roc aps lists")
    if type(Save_to_Table_path) == list:
        Save_to_Table_path = directories_path
    if type(conversion_files_df) == list:
        conversion_files_object = CI_Summary_table_object(root_path=CI_Results_path)
        print("\nconversion_files_object=", conversion_files_object)
        conversion_files_df = conversion_files_object.get()
        print("conversion_files_df:\n", conversion_files_df)
    if dirs_names == None:
        all_directories = [x for x in os.listdir(directories_path) if
                           x != "shap_folder" and x != "LR_comparison" and
                           os.path.isdir(os.path.join(directories_path, x)) and ("Quantiles" not in x)]
    else:
        all_directories = dirs_names
    print("\ndirectories_path=", directories_path,
          "\nSave_to_Table_path=", Save_to_Table_path,
          "all_directories=", all_directories)
    predictions_files = []
    y_files = []
    for dirname in all_directories:
        print ("dirname:", dirname)
        if (dirname not in directories_to_skip) and ("Quantiles_" not in dirname):
            if (dirname.startswith("LR_") or dirname.endswith("_LR") or dirname.startswith("SA_")
                    or dirname.endswith("_SA")):
                if (dirname.startswith("LR_") or dirname.endswith("_LR")):  # For logistic regression files
                    col_test = "y_test"
                    col_pred = "y_pred"
                elif (dirname.startswith("SA_") or dirname.endswith("_SA")):
                    col_test = "y_test_val"
                    col_pred = "y_pred_val"
                results_directory_name = "CI"
                dirname_path = os.path.join(directories_path, dirname, results_directory_name)
                y_pred_val = []
                y_test_val = []
                y_df_path = os.path.join(dirname_path, "y_pred_results_0.csv")
                y_df = pd.read_csv(y_df_path, index_col=0)
                for i, stat in enumerate(np.isnan(y_df.loc[:, col_test].values)):
                    if ~stat:
                        y_pred_val.append(y_df.loc[i, col_pred])
                        y_test_val.append(y_df.loc[i, col_test])
                with open(os.path.join(dirname_path, "y_pred_val_list"), 'wb') as fp:
                    pickle.dump(y_pred_val, fp)
                with open(os.path.join(dirname_path, "y_test_val_list"), 'wb') as fp:
                    pickle.dump(y_test_val, fp)
                predictions_files.append(os.path.join(dirname_path, "y_pred_val_list"))
                y_files.append(os.path.join(dirname_path, "y_test_val_list"))
            else:
                results_directory_name = "Diabetes_Results"
                pred_list_name = "y_pred_val_list"
                test_list_name = "y_test_val_list"
                predictions_files.append(os.path.join(directories_path, dirname, results_directory_name,
                                                      pred_list_name))
                y_files.append(os.path.join(directories_path, dirname, results_directory_name, test_list_name))
        else:
            print ("skipped dirname:", dirname)
    sorted_label_name = []
    sorted_predictions_files = []
    sorted_y_files = []

    #     print("in calc_roc_aps_lists,conversion_files_df.head(): ",conversion_files_df.head())
    for ind, x in enumerate(all_directories):
        if x.endswith("_Diabetes"):
            x = x[:-9]
        if x in conversion_files_df.index:  # Add this
            sorted_label_name.append(conversion_files_df.loc[x, "Label"])
            sorted_predictions_files.append(predictions_files[ind])
            sorted_y_files.append(y_files[ind])
        else:
            print (x, " was not found in the conversion file")
    labels = sorted_label_name
    y_files = sorted_y_files
    predictions_files = sorted_predictions_files

    df = pd.DataFrame({
        "y_files": y_files,
        "predictions_files": predictions_files,
        "labels": labels})
    df = df.set_index("labels")

    df = df.loc[sorted_label_name, :]
    conversion_files_df = conversion_files_df.reset_index().set_index("Label", drop=True)

    for ind, label in enumerate(labels):
        #         print("****ind,label:",ind," ",label)
        #         print "****df.loc[label,:]: ",df.loc[label,:]
        #         print "****conversion_files_dCI_Results_pathf.loc[label,:]: ", conversion_files_df.loc[label,:]
        df.loc[label, "AUC"] = conversion_files_df.loc[label, "AUROC_mean"]
        df.loc[label, "APS"] = conversion_files_df.loc[label, "APS_mean"]
    df = df.sort_values(by="AUC")
    precision_list, recall_list = calc_precision_recall_list(df, sort=False)
    fpr_list, tpr_list = calc_roc_list(df, sort=False)
    if type(CI_Results_path) == list:
        CI_table = create_folder_table(directories_path=directories_path, conversion_files_df=conversion_files_df,
                                       Save_to_Table_path=Save_to_Table_path).get()
    else:
        CI_table = create_folder_table(directories_path=directories_path, CI_Results_path=CI_Results_path,
                                       Save_to_Table_path=Save_to_Table_path).get()
    df["APS_95CI"] = CI_table.loc[:, "APS"]
    df["AUC_95CI"] = CI_table.loc[:, "AUROC"]
    return df, precision_list, recall_list, fpr_list, tpr_list


def Load_CI_dictionaries(Folder, metric="AUROC"):
    metric_list = []
    F1_files = os.listdir(Folder)
    F1_files = [x for x in F1_files if x.startswith("CI_Dict")]
    for CI_dict in F1_files:
        path = os.path.join(Folder, CI_dict)
        with open(path, 'rb') as fp:
            data_dict = pickle.load(fp)
        metric_list.append(data_dict[metric])
        metric_array = np.array(metric_list)
    #     print(Folder,"statistics: mean=",metric_array.mean(),"stdv=",metric_array.std())
    return metric_array, metric_array.mean(), metric_array.std()


def Calc_P_value(Folder_1, Folder_2, metric="AUROC"):
    metric_array1, metric_mean1, metric_std1 = Load_CI_dictionaries(Folder_1, metric)
    metric_array2, metric_mean2, metric_std2 = Load_CI_dictionaries(Folder_2, metric)
    ttest, pval = mannwhitneyu(metric_array1, metric_array2)
    print("p-value:", pval)
    print("ttest:", ttest)
    if pval < 0.05:
        print("we reject null hypothesis")
    else:
        print("we accept null hypothesis")
    return ttest, pval, metric_array1, metric_array2


def plot_ax(img_path, ax, txt=None, pos=(-0.04, 1), aspect=None, label_size=16):
    """
    aspect : ['auto' | 'equal' | scalar], optional, default: None
    If 'auto', changes the image aspect ratio to match that of the
    axes.

    If 'equal', and `extent` is None, changes the axes aspect ratio to
    match that of the image. If `extent` is not `None`, the axes
    aspect ratio is changed to match that of the extent.

    If None, default to rc ``image.aspect`` value.

    """
    ax.tick_params(axis='both', which='major', labelsize=label_size)
    ax.axis('off')
    if txt != None:
        ax.text(pos[0], pos[1], txt, transform=ax.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')
    img = mpimg.imread(img_path)
    #     ax.imshow(img,aspect="auto")
    ax.imshow(img, aspect=aspect)


def plot_ROC_curves(fpr_list, tpr_list, df, ax=None, legend_length=3, font_size=16, sort=True,
                    legend_font_size=80, figsize=(30, 24), dpi=50, ROC_path=[], plot=True, ticks_font_weight=36,
                    color_dict={}, x_txt=0.65, y_txt=0.48, txt_font_size=48, alpha=0.8,
                    leg_pos=(0.5, 0.7), short_labels=False, txt=None, frameon=False):
    if ax:
        ax = ax
        fig = plt.gcf()
    else:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    if sort:
        df = df.sort_values('AUC')
    y_files = df.y_files.values
    predictions_files = df.predictions_files.values
    labels = df.index.values
    new_labels = wrap_labels(labels, num_of_chars=legend_length)
    AUC = df.AUC.values
    n = len(labels)
    lw = 6
    if all(len(x) == n for x in [y_files, predictions_files]):
        #         cmap = sns.color_palette(color, df.shape[0])
        for ind, label in enumerate(new_labels):
            fpr = fpr_list[ind]
            tpr = tpr_list[ind]
            if short_labels:
                #                 leg_label=wrap_labels(label + ' ({0:0.2f})'.format(AUC[ind]),legend_length)
                #                 leg_label=leg_label.replace("GBDT","BDT")
                #                 ax.plot(fpr,tpr,c=color_dict[labels[ind]],lw=lw,label=leg_label)
                ax.plot(fpr, tpr, c=color_dict[labels[ind]], lw=lw, label='{0:0.2f}'.format(AUC[ind]))

            else:
                leg_label = wrap_labels(label + '({0:0.2f})'.format(AUC[ind]), legend_length)
                leg_label = leg_label.replace("GBDT", "BDT")
                ax.plot(fpr, tpr, c=color_dict[labels[ind]], lw=lw, label=leg_label)

        ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=font_size)
        ax.set_ylabel('True Positive Rate', fontsize=font_size)
        handles, labels = ax.get_legend_handles_labels()
        leg = ax.legend(reversed(handles), reversed(labels), title='Line', loc=leg_pos,
                        fontsize=legend_font_size, fancybox=False, labelspacing=0.2, frameon=frameon)
        leg.set_title("AUROC", prop={'size': legend_font_size})
        ax.tick_params(axis='both', which='major', labelsize=ticks_font_weight)
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        ax.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"], fontsize=ticks_font_weight)
        ax.set_xticks(np.arange(0, 1.1, 0.2))
        ax.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"], fontsize=ticks_font_weight)
    else:
        print("all input lists should be same length, y_test_val_list:",
              len(y_files), ", y_pred_val_list:", len(predictions_files),
              ", labels:", len(labels))
    plt.tight_layout()
    if txt != None:
        ax.text(x_txt, y_txt, txt, fontsize=txt_font_size,
                horizontalalignment='left', verticalalignment='center', transform=ax.transAxes,
                )
        # bbox=dict(fc="white",pad=15,alpha=alpha)
    if ROC_path != []:
        plt.savefig(ROC_path, dpi=dpi, frameon=False)
    if plot:
        plt.show()
    return ax


def plot_roc_ci(df, ci_dict_path, color_dict,
                num_of_files=100,
                lw=2,
                alpha=0.1,
                num_of_chars=30,
                leg_pos=(0.2, 0.8),
                figsize=(12, 9),
                fontsize=18,
                legend_font_size=18,
                leg_text="AUROC(Area Under ROC)",
                leg_title_pos=(-10, 0),
                ax1=None,
                xlabels=None,
                plot=True,
                DPI=100,
                roc_ci_path=None,
                short_labels=True,
                text=None,
                x_txt=0.65,
                y_txt=0.57,
                facecolor="white",
                ):
    legend_list = []
    if ax1 == None:
        fig1, ax1 = plt.subplots(1, 1, figsize=figsize)
    data_dict = {}
    prevalence = calc_prevalence(df)

    for label in df.index.values:
        rel_ci_files = []
        #         print label
        data_dict[label] = {}
        color = color_dict[label]
        print(label, ":", color)
        all_files = os.listdir(ci_dict_path[label])
        rel_ci_files = [x for x in all_files if x.startswith("CI_Dict")]
        if len(rel_ci_files) > 0:
            if num_of_files != None:
                rel_ci_files = rel_ci_files[:num_of_files]
            for ci_file in rel_ci_files:
                with open(os.path.join(ci_dict_path[label], ci_file), 'rb') as fp:
                    data_dict[label][ci_file] = pickle.load(fp)
        else:
            rel_ci_files = [x for x in all_files if x.startswith("y_pred_results")]
            for ci_file in rel_ci_files:
                data = {}
                csv_file = pd.read_csv(os.path.join(ci_dict_path[label], ci_file), index_col=0)
                try:
                    y_pred = csv_file.loc[:, "y_pred"].values
                    y_test = csv_file.loc[:, "y_test"].values
                except:
                    y_pred = csv_file.loc[:, "y_pred_val"].values
                    y_test = csv_file.loc[:, "y_test_val"].values
                data["fpr"], data["tpr"], _ = roc_curve(y_test, y_pred)
                data_dict[label][ci_file] = data
        for ci_file in data_dict[label].keys():
            fpr = data_dict[label][ci_file]["fpr"]
            tpr = data_dict[label][ci_file]["tpr"]

            ax1.plot(fpr, tpr, alpha=alpha, color=color, lw=lw)
        if short_labels:
            AUROC = df.loc[label, "AUC_95CI"]
            legend_list.append(mpatches.Patch(color=color_dict[label],
                                              label=AUROC))
        else:
            new_label = wrap_labels(label, num_of_chars)
            legend_list.append(mpatches.Patch(color=color_dict[label], label=new_label, lw=lw))

    # ax1_leg_list= legend_list+mpatches.Patch(color=color_dict[label], label=new_label)
    dash_zorder = max([_.zorder for _ in ax1.get_children()]) + 1

    ax1.plot([0, 1], [0, 1], color='navy', lw=6, linestyle='--', zorder=dash_zorder)
    ax1.set_ylim(0, 1)
    ax1.set_xlim(0, 1)
    leg = ax1.legend(handles=legend_list, fontsize=fontsize, loc=leg_pos)

    handles, labels = ax1.get_legend_handles_labels()
    #     leg=ax1.legend(reversed(handles), reversed(labels), loc=leg_pos,
    #                       fontsize=legend_font_size,fancybox=False,labelspacing=0.15,frameon=False)
    if leg_text != None:
        leg.set_title(leg_text, prop={'size': legend_font_size})
        leg.get_title().set_position(leg_title_pos)  # -10 is a guess
    leg._legend_box.align = "left"
    frame = leg.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor("white")

    #     ax1.spines['right'].set_visible(False)
    #     ax1.spines['top'].set_visible(False)

    ax1.set_ylabel("TPR", fontsize=fontsize)
    ax1.set_xlabel("FPR", fontsize=fontsize)

    if xlabels != None:
        ax1.set_xticks([float(x) for x in xlabels])
        ax1.set_xticklabels(xlabels, fontsize=fontsize)
        ax1.set_yticks([float(x) for x in xlabels])
        ax1.set_yticklabels(xlabels, fontsize=fontsize)

    ax1.tick_params(labelsize=fontsize)
    ax1.set_facecolor(facecolor)
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    ax1.spines['bottom'].set_visible(True)
    ax1.spines['left'].set_visible(True)
    ax1.spines['bottom'].set_color('black')
    ax1.spines['left'].set_color('black')

    plt.tight_layout()
    if text != None:
        ax1.text(x_txt, y_txt, text, fontsize=fontsize,
                 horizontalalignment='left', verticalalignment='center', transform=ax1.transAxes)
    if roc_ci_path != None:
        plt.savefig(roc_ci_path, dpi=DPI, frameon=False)
    if plot:
        plt.show()
    return ax1


def plot_APS_curves(precision_list, recall_list, df, ax=None, legend_length=20,
                    font_size=16, sort=True, legend_font_size=80, APS_path=[],
                    dpi=50, figsize=(30, 24), plot=True, ticks_font_size=36,
                    color_dict={}, x_txt=0.6, y_txt=0.55, txt_font_size=56, alpha=0.8,
                    leg_pos=(0.6, 0.7), leg_text="Average\nprecision\nscore",
                    Text_text=None, short_labels=True, leg_title_pos=(-10, 0),
                    show_APS_reult=True, fold_ticks=None, xy_lines=None, lines_color="Navy", xlabels=None):
    # How can be "simple" or "relative"
    if ax:
        ax = ax
        fig = plt.gcf()
    else:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    if sort:
        df = df.sort_values('APS', ascending=True)
    else:
        df = df
    y_files = df.y_files.values
    predictions_files = df.predictions_files.values
    labels = df.index.values
    #     print df
    APS = df.APS.values
    n = len(labels)
    lw = 6
    max_fold = 1
    prevalence = calc_prevalence(df)
    if (all(len(x) == n for x in [y_files, predictions_files])):
        #         cmap = sns.color_palette(color, df.shape[0])
        for ind, label in enumerate(labels):
            precision = precision_list[ind]
            recall = recall_list[ind]
            if short_labels:
                ax.step(recall, precision, c=color_dict[labels[ind]], where='post', label='{0:0.2f}'.format(APS[ind]),
                        lw=lw)
            else:
                if show_APS_reult:
                    leg_label = wrap_labels(
                        label + ' ({0:0.2f})'.format(APS[ind]),
                        num_of_chars=legend_length)
                else:
                    leg_label = wrap_labels(label, num_of_chars=legend_length)

                ax.step(recall, precision, c=color_dict[labels[ind]], where='post', label=leg_label, lw=lw)
            #             print precision
            current_fold = np.max(precision) / prevalence
            #             print current_fold
            if current_fold > max_fold:
                max_fold = current_fold
        #                 print max_fold
        ax.set_ylim([0.0, 1.05 * max(precision)])
        axi = ax.twinx()
        # set limits for shared axis
        axi.set_ylim(0, 1.05 * max_fold)
        # set ticks for shared axis
        if fold_ticks:
            rel_ticks_pos = fold_ticks
            relative_ticks_labels = [str("{0:.2f}").format(x) for x in rel_ticks_pos]
            relative_ticks_labels[0] = "0"
            axi.set_yticks(rel_ticks_pos)
            axi.set_yticklabels(relative_ticks_labels)
        axi.set_ylabel('Precision fold', fontsize=font_size)
        axi.tick_params(axis='both', which='major', labelsize=ticks_font_size)
        ax.set_xlim([0.0, 1.0])
        ax.set_xlabel('Recall', fontsize=font_size)
        ax.set_ylabel('Precision', fontsize=font_size)
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        ax.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"], fontsize=ticks_font_size)
        if xlabels == None:
            ax.set_xticks(np.arange(0, 1.1, 0.2))
            ax.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"], fontsize=ticks_font_size)
        else:
            ax.set_xticks([float(x) for x in xlabels])
            ax.set_xticklabels(xlabels, fontsize=ticks_font_size)

        ax.axhline(y=prevalence, color='black', linestyle='--', lw=5,
                   label="Prevalence:{0:0.1f}%".format(100 * prevalence))

        handles, labels = ax.get_legend_handles_labels()
        leg = ax.legend(reversed(handles), reversed(labels), loc=leg_pos,
                        fontsize=legend_font_size, fancybox=False, labelspacing=0.15, frameon=False)
        leg.set_title(leg_text, prop={'size': legend_font_size})
        leg._legend_box.align = "left"
        leg.get_title().set_position(leg_title_pos)  # -10 is a guess

        frame = leg.get_frame()
        frame.set_facecolor("white")
        frame.set_edgecolor("white")

    else:
        print("all input lists should be same length, y_test_val_list:",
              len(y_files), ", y_pred_val_list:", len(predictions_files),
              ", labels:", len(labels))
        print("please make sure that how==simple or how==relative")

    if xy_lines != None:
        try:
            color = color_dict[lines_color]
        except:
            color = lines_color
        ax.vlines(x=xy_lines[0], ymin=0, ymax=xy_lines[1], colors=color, linestyles="dashed", lw=lw)
        ax.hlines(y=xy_lines[1], xmin=0, xmax=xy_lines[0], colors=color, linestyles="dashed", lw=lw)
    plt.tight_layout()
    if Text_text != None:
        ax.text(x_txt, y_txt, Text_text, fontsize=txt_font_size,
                horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
    #                     bbox=dict(fc="white",pad=15,alpha=alpha)
    if APS_path != []:
        plt.savefig(APS_path, dpi=dpi, frameon=False)
    if plot:
        plt.show()
    return ax


def plot_aps_ci(df,
                ci_dict_path,
                color_dict,
                num_of_files=100,
                lw=2,
                alpha=0.1,
                num_of_chars=50,
                leg_pos=(0.2, 0.8),
                figsize=(12, 9),
                fontsize=18,
                leg_text="Average precision score",
                leg_title_pos=(-10, 0),
                ax1=None,
                xy_lines=None,
                text=None,
                lines_color="red",
                xlabels=None,
                ylabels=None,
                plot=True,
                DPI=100,
                pr_ci_path=None,
                x_txt=None,
                y_txt=None,
                short_labels=True,
                legend_dict={},
                facecolor="white"):
    plt.rc('font', size=fontsize)
    legend_list = []
    if ax1 == None:
        fig1, ax1 = plt.subplots(1, 1, figsize=figsize)
    data_dict = {}
    prevalence = calc_prevalence(df)
    if ylabels == None:
        ylabels = xlabels
    for label in df.index.values:
        rel_ci_files = []
        #         print label
        data_dict[label] = {}
        color = color_dict[label]
        all_files = os.listdir(ci_dict_path[label])
        rel_ci_files = [x for x in all_files if x.startswith("CI_Dict")]
        if len(rel_ci_files) > 0:
            if num_of_files != None:
                rel_ci_files = rel_ci_files[:num_of_files]
            for ci_file in rel_ci_files:
                with open(os.path.join(ci_dict_path[label], ci_file), 'rb') as fp:
                    data_dict[label][ci_file] = pickle.load(fp)
        else:
            rel_ci_files = [x for x in all_files if x.startswith("y_pred_results")]
            for ci_file in rel_ci_files:
                data = {}
                csv_file = pd.read_csv(os.path.join(ci_dict_path[label], ci_file), index_col=0)
                try:
                    y_pred = csv_file.loc[:, "y_pred"].values
                    y_test = csv_file.loc[:, "y_test"].values
                except:
                    y_pred = csv_file.loc[:, "y_pred_val"].values
                    y_test = csv_file.loc[:, "y_test_val"].values
                data["precision"], data["recall"], _ = precision_recall_curve(y_test, y_pred)
                data_dict[label][ci_file] = data
        for ci_file in data_dict[label].keys():
            recall = data_dict[label][ci_file]["recall"]
            precision = data_dict[label][ci_file]["precision"]

            ax1.step(recall, precision, alpha=alpha, color=color, lw=lw)
        if short_labels:
            APS = df.loc[label, "APS_95CI"]
            legend_list.append(mpatches.Patch(color=color_dict[label], label=APS))
        else:
            if label in legend_dict.keys():
                new_label = wrap_labels(legend_dict[label], num_of_chars)
            else:
                new_label = wrap_labels(label, num_of_chars)
            legend_list.append(mpatches.Patch(color=color_dict[label], label=new_label, lw=lw))

    # ax1_leg_list= legend_list+mpatches.Patch(color=color_dict[label], label=new_label)
    dash_zorder = max([_.zorder for _ in ax1.get_children()]) + 1
    if xy_lines != None and type(xy_lines) == list:
        for ind, lines in enumerate(xy_lines):
            try:
                color = color_dict[lines_color[ind]]
            except:
                color = lines_color[ind]
            ax1.vlines(x=lines[0], ymin=0, ymax=lines[1], colors=color, linestyles="dashed",
                       lw=6, zorder=dash_zorder)
            ax1.hlines(y=lines[1], xmin=0, xmax=lines[0], colors=color, linestyles="dashed",
                       lw=6, zorder=dash_zorder)
    else:
        print("~~~~~~ xy_lines must be list of tuples ~~~~~~~~~")
    legend_list.append(mpatches.Patch(color='black', linestyle='--',
                                      label="Prevalence:{0:0.1f}%".format(100 * prevalence),
                                      fill=False, lw=6))
    #     print legend_list
    ax1.axhline(y=prevalence, color='black', linestyle='--', lw=6,
                label="Prevalence:{0:0.1f}%".format(100 * prevalence), zorder=dash_zorder)
    leg = ax1.legend(handles=legend_list, fontsize=fontsize, loc=leg_pos)
    ax1.set_ylim(0, 1)
    ax1.set_xlim(0, 1)
    handles, labels = ax1.get_legend_handles_labels()
    #     leg=ax1.legend(reversed(handles), reversed(labels), loc=leg_pos,
    #                       fontsize=legend_font_size,fancybox=False,labelspacing=0.15,frameon=False)
    leg.set_title(leg_text, prop={'size': fontsize})
    leg._legend_box.align = "left"
    leg.get_title().set_position(leg_title_pos)  # -10 is a guess
    frame = leg.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor("white")

    ax1.set_ylabel("Precision", fontsize=fontsize)
    ax1.set_xlabel("Recall", fontsize=fontsize)
    if xlabels != None:
        ax1.set_xticks([float(x) for x in xlabels])
        ax1.set_xticklabels(xlabels, fontsize=fontsize)
        ax1.set_yticks([float(x) for x in ylabels])
        ax1.set_yticklabels(ylabels, fontsize=fontsize)
    else:
        print("~~~~~~ xy_lines must be list of tuples ~~~~~~~~~")

    ax1.set_facecolor(facecolor)
    ax1.tick_params(labelsize=fontsize)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_color('black')
    ax1.spines['left'].set_color('black')
    if text != None:
        ax1.text(x=x_txt, y=y_txt, s=text, fontsize=fontsize,
                 horizontalalignment='left', verticalalignment='center', transform=ax1.transAxes)
    plt.tight_layout()
    if pr_ci_path != None:
        plt.savefig(pr_ci_path, dpi=DPI, frameon=False)
    if plot:
        plt.show()
    return ax1


def plot_aps_summary(df, ax=None, font_size=16, color='plasma', sort=True):
    if ax:
        ax = ax
    else:
        ax = plt.gca()
    my_colors = sns.color_palette(color, df.shape[0])
    color_list = my_colors
    if sort:
        aps_df = df.sort_values(["APS"]).loc[:, "APS"]
    else:
        aps_df = df.loc[:, "APS"]
    labels = aps_df.index.values
    new_labels = wrap_labels(labels, num_of_chars=3)
    ax.bar(new_labels, height=aps_df.values, align='center', color=color_list)

    ax.xaxis.label.set_visible(False)
    ax.set_ylabel("Average precision score", fontsize=font_size)
    #     ax.set_title('Average precision recall summary of adding features',fontsize=font_size+2)
    ax.tick_params(labelrotation=90, labelsize=font_size - 4)
    return ax


def plot_roc_auc_summary(df, ax=None, font_size=16, sort=True,
                         roc_auc_sum_path=[], dpi=50, figsize=(30, 24), plot=True, ticks_font_size=72, color_dict={},
                         num_of_chars=4, linespacing=1, x_txt=0.02, y_txt=0.9, Plot_legend=True):
    if ax:
        ax = ax
        fig = plt.gcf()
    else:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    if sort:
        values = df.sort_values(["AUC"]).loc[:, "AUC"].values
    else:
        values = df.loc[:, "AUC"].values

    if sort:
        auc_df = df.sort_values(["AUC"]).loc[:, "AUC"]
    else:
        auc_df = df.loc[:, "AUC"]
    labels = auc_df.index.values
    new_labels = wrap_labels(labels, num_of_chars)
    x = np.arange(len(new_labels))  # the label locations
    width = 0.8
    clrs = [color_dict[r] for r in labels]
    ax.bar(left=x - width / 2, height=auc_df.values, width=width, align='center', color=clrs,
           tick_label=new_labels)
    ax.set_xticklabels(new_labels, linespacing=linespacing, ha='center')
    ax.xaxis.label.set_visible(False)
    ax.set_ylabel(
        "AUROC", fontsize=font_size)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1])
    ax.set_yticklabels(["0.5", "0.6", "0.7", "0.8", "0.9", "1"], fontsize=ticks_font_size)
    ax.set_ylim([0.5, 1.1 * max(values)])
    ax.set_xlim([-1, len(new_labels) - 1])

    ax.tick_params(axis="x", labelrotation=90, labelsize=ticks_font_size)
    if Plot_legend:
        txt = "LR     - Logistic regression\nSA     - Survival analysis\nGBDT- Gradient boosting decision trees"
        ax.text(x_txt, y_txt, txt, horizontalalignment='left',
                verticalalignment='center', transform=ax.transAxes, fontsize=font_size - 10,
                bbox=dict(fc="none", pad=15))

    plt.tight_layout()
    if roc_auc_sum_path != []:
        plt.savefig(roc_auc_sum_path, dpi=dpi, frameon=False)
    if plot:
        plt.show()
    return ax


def plot_summary_curves(df, precision_list, recall_list, fpr_list, tpr_list, figsize=(30, 24), dpi=1200,
                        color_palette="Set1", file_name="temp", legend_len=4, font_size=24, legend_size=18,
                        sort=True):
    file_name = os.path.join("/home/edlitzy/UKBB_Tree_Runs/For_article/plots/",
                             file_name)
    rc.update({'font.size': font_size})
    plt.rc('legend', fontsize=legend_size)
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    ax1 = plot_roc_auc_summary(df, ax1, font_size, color_palette, sort=sort)
    ax1.text(-0.04, 1, 'A', transform=ax1.transAxes, fontsize=font_size, fontweight='bold', va='top', ha='right')

    ax2 = plot_aps_summary(df, ax2, font_size, color_palette, sort=sort)
    ax2.text(-0.04, 1, 'B', transform=ax2.transAxes, fontsize=font_size, fontweight='bold', va='top', ha='right')
    plt.tight_layout()

    ax3 = plot_APS_curves(precision_list, recall_list, df, ax3, color_palette, legend_length=legend_len,
                          font_size=font_size,
                          sort=sort, legend_font_size=legend_size)
    ax3.text(-0.04, 1, 'C', transform=ax3.transAxes, fontsize=font_size, fontweight='bold', va='top', ha='right')

    ax4 = plot_ROC_curves(fpr_list, tpr_list, df, ax4, color_palette, legend_length=legend_len, font_size=font_size,
                          sort=sort, legend_font_size=legend_size)
    ax4.text(-0.04, 1, 'D', transform=ax4.transAxes, fontsize=font_size, fontweight='bold', va='top', ha='right')

    plt.tight_layout()
    plt.savefig(file_name, dpi=dpi, frameon=False)
    plt.show()


def plot_quantiles_curve(
        singles_df, test_label="Blood Tests", bins=10, low_quantile=0.8, top_quantile=1,
        figsize=(30, 24), ax=None, font_size=96, color="plasma", plot=False, thresh_index=1,
        ticks_font_weight=36, print_title=False, inset=False, rel_fold=None, quantile_path=[], dpi=50,
        position=[], num_of_chars=20, prev_fold=False, in_inset=False, y_label_text=None):
    if ax:
        ax = ax
        fig = plt.gcf()
    else:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    with open(singles_df.loc[test_label, "predictions_files"], 'rb') as fp:
        y_pred_val = pickle.load(fp)
    with open(singles_df.loc[test_label, "y_files"], 'rb') as fp:
        y_test_val = pickle.load(fp)

    vals_df = pd.DataFrame(data={"Y_test": y_test_val, "Y_Pred": y_pred_val})
    res = 1. / bins
    quants_bins = [int(x * 100) / 100. for x in np.arange(low_quantile, top_quantile + res / 2, res)]
    vals_df = vals_df.sort_values("Y_Pred", ascending=False)
    Quants = vals_df.loc[:, "Y_Pred"].quantile(quants_bins)
    Rank = pd.DataFrame()
    for ind, quant in enumerate(Quants.values[:-1]):
        Rank.loc[np.str(ind), "Diagnosed"] = vals_df.loc[(
                (vals_df["Y_Pred"] <= Quants.values[ind + 1]) & (vals_df["Y_Pred"] > quant))].loc[:,
                                             'Y_test'].sum()
        Rank.loc[np.str(ind), "All"] = vals_df.loc[(
                (vals_df["Y_Pred"] > quant) & (vals_df["Y_Pred"] <= Quants.values[ind + 1]))].loc[:,
                                       'Y_test'].count()
        Rank.loc[np.str(ind), "Ratio"] = Rank.loc[np.str(ind), "Diagnosed"] / Rank.loc[np.str(ind), "All"]

    #     fig, ax = plt.subplots(1, 1, figsize=figsize)
    my_colors = sns.color_palette(color, Rank.shape[0])
    width = 0.8
    x = [item - width / 2 for item in np.arange(len(Rank.index.values))]
    labels = [str(int(100 * item)) for item in np.arange(low_quantile + res, top_quantile + res / 2, res)]
    labels = wrap_labels(labels, num_of_chars)
    ax.bar(left=x, height=Rank.loc[:, "Ratio"], width=width, align='center', color=my_colors,
           tick_label=labels)

    ax.tick_params(axis='both', which='major', labelsize=ticks_font_weight)
    ax.tick_params(axis='x', labelrotation=0, labelsize=ticks_font_weight)

    ax2 = ax.twinx()
    Rank.loc[:, "Ratio"] = Rank.loc[:, "Ratio"].fillna(0)
    if rel_fold == None:
        min_ratio_first_decile = np.max([Rank.loc["0", "Diagnosed"], 1]) / Rank.loc["0", "All"]
    elif rel_fold == "prev":
        prev = Rank.loc[:, "Diagnosed"].sum() / Rank.loc[:, "All"].sum()
        min_ratio_first_decile = np.max([Rank.loc["0", "Diagnosed"], 1]) / Rank.loc["0", "All"]
    else:
        min_ratio_first_decile = rel_fold

    Rank.loc[:, "Fold"] = Rank.loc[:, "Ratio"].values / min_ratio_first_decile
    ax2.bar(left=x, height=Rank.loc[:, "Fold"], width=width, align='center', color=my_colors, tick_label=labels)
    ax2.tick_params(axis='both', which='major', labelsize=ticks_font_weight)

    if inset:
        if position != []:
            left, bottom, width, height = position
        else:
            left, bottom, width, height = [0.22, 0.4, 0.47, 0.5]
        ax3 = fig.add_axes([left, bottom, width, height])
        ax3, fig3, Rank3 = plot_quantiles_curve(singles_df, test_label, bins=100, low_quantile=0.9,
                                                top_quantile=1, figsize=figsize, font_size=font_size, plot=False,
                                                thresh_index=1, ticks_font_weight=ticks_font_weight, print_title=False,
                                                ax=ax3,
                                                inset=False, rel_fold=min_ratio_first_decile,
                                                in_inset=True)
    if in_inset:
        y_tick_labels = ['%0.1f' % x for x in np.arange(0, 0.81, 0.2)]
        y_tick_labels[0] = str(0)
        ax.set_yticks(np.arange(0, 0.81, 0.2))
        ax.set_yticklabels(y_tick_labels)
        x_labels = [91, 95, 100]
        ax.set_xticks([0 - 0.5, 4 - 0.5, 9 - 0.5])
        ax.set_xticklabels(x_labels)

    else:
        y_tick_labels = [x for x in np.arange(0, 0.21, 0.02)]
        y_tick_labels[0] = str(0)
        ax.set_yticks(np.arange(0, 0.21, 0.02))
        ax.set_yticklabels(y_tick_labels)
        ax.set_xlabel("Prediction quantile", fontsize=font_size)
        if y_label_text == None:
            ax.set_ylabel("Prevalence in quantile of \n" + test_label, fontsize=font_size)
        else:
            ax.set_ylabel("Prevalence in quantile of \n" + y_label_text, fontsize=font_size)
        ax2.set_ylabel("Prevalence fold in quantiles vs \n first decile precalence", fontsize=font_size)
        x_labels = [10, 50, 100]
        ax.set_xticks([0 - 0.5, 4 - 0.5, 9 - 0.5])
        ax.set_xticklabels(x_labels)

    if print_title:
        ax.set_title(test_label + "\n quantiles, folds of prevalence vs. mean prevalence of first decile",
                     fontsize=font_size + 6)
    plt.tight_layout()
    if quantile_path != []:
        plt.savefig(quantile_path, dpi=dpi, frameon=False)
    if plot:
        plt.show()
    return ax, fig, Rank


def calc_folders_Rank_df(folder_path, bins=10, low_quantile=0.1, top_quantile=1, save=False,
                         force_calc=False,num_of_files=None,minimal_diagnosed=1,recalc=False):
    Rank_df_list = []
    dirs = [x for x in os.listdir(folder_path) if (("LR" in x or "SA" in x or "Diabetes" in x)
                                                   and os.path.isdir(os.path.join(folder_path, x)))]
    name = os.path.basename(folder_path)
    res = 1. / bins
    quants_bins = [int(x * 100) / 100. for x in np.arange(low_quantile, top_quantile + res / 2, res)]
    for dir_name in dirs:
        print("working on:", dir_name)
        tmp_file_name=os.path.join(folder_path, dir_name,dir_name+"quantiles_rank_df.csv")
        if os.path.isfile(tmp_file_name) and not recalc:
            tmp_rank_df=pd.read_csv(tmp_file_name,index_col=0)
        else:
            if "Diabetes" in dir_name:
                tmp_rank_df = calc_gbdt_Rank_df(os.path.join(folder_path, dir_name), quants_bins,
                                                bins=bins,num_of_files=num_of_files,minimal_diagnosed=minimal_diagnosed)
            elif "LR" in dir_name or "SA" in dir_name:
                tmp_rank_df = calc_lr_Rank_df(os.path.join(folder_path, dir_name), quants_bins,
                                              bins=bins, force_calc=force_calc,num_of_files=num_of_files,
                                              minimal_diagnosed=minimal_diagnosed)
            tmp_rank_df.to_csv(os.path.join(folder_path, dir_name, dir_name + "quantiles_rank_df.csv"))
        if type(tmp_rank_df) != list:
            Rank_df_list.append(tmp_rank_df)
    rank_df_tot = pd.concat(Rank_df_list)
    if save:
        rank_df_tot.to_csv(os.path.join(folder_path, name + "_" + str(bins) + "_quantiles_rank_df.csv"), index=False)
    return rank_df_tot


def calc_gbdt_Rank_df(folder_name, quants_bins, bins,num_of_files=None,minimal_diagnosed=1):
    name = os.path.basename(folder_name)
    csv_file_name = os.path.join(folder_name, name + "_" + str(bins) + "_quantiles_rank_df.csv")
    try:
        Rank_df = pd.read_csv(csv_file_name, index_col=None)
    except:
        Rank_gbdt_list = []
        ci_path = os.path.join(folder_name, "Diabetes_Results", "CI")
        try:
            all_files = os.listdir(ci_path)
        except:
            return []
        ci_dict_names = [x for x in all_files if "CI_Dict" in x]
        if num_of_files is not None:
            ci_dict_names=ci_dict_names[:num_of_files]
        for f in ci_dict_names:
            with open(os.path.join(ci_path, f), 'rb') as fp:
                data_dict = pickle.load(fp)
            y_proba = data_dict["y_proba"]
            y_test = data_dict["y_val.values"].flatten()
            ci_tmp_df = pd.DataFrame({"y_pred": y_proba, "y_test": y_test})
            tmp_Rank_df = calc_Rank_df(ci_tmp_df, quants_bins,minimal_diagnosed=minimal_diagnosed)
            Rank_gbdt_list.append(tmp_Rank_df)
        Rank_df = pd.concat(Rank_gbdt_list)
        Rank_df["dir_name"] = name
        Rank_df.to_csv(csv_file_name, index=False)
    return Rank_df


def calc_lr_Rank_df(folder_name, quants_bins, bins, force_calc=False,num_of_files=None,minimal_diagnosed=1):
    name = os.path.basename(folder_name)
    csv_file_name = os.path.join(folder_name, name + "_" + str(bins) + "_quantiles_rank_df.csv")
    if os.path.isfile(csv_file_name) and not force_calc:
        Rank_df = pd.read_csv(csv_file_name, index_col=None)
    else:
        Rank_lr_df_list = []
        ci_path = os.path.join(folder_name, "CI")
        if os.path.isdir(ci_path):
            all_files = os.listdir(ci_path)
        else:
            return []
        ci_file_names = [x for x in all_files if "y_pred_results" in x]
        if num_of_files is not None:
            ci_file_names=ci_file_names[:num_of_files]
        for f in tqdm(ci_file_names):
            ci_tmp_df = pd.read_csv(os.path.join(ci_path, f), index_col=0)
            if "SA" in folder_name:
                ci_tmp_df.columns = ["y_pred", "y_test"]
            tmp_Rank_df = calc_Rank_df(ci_tmp_df, quants_bins,minimal_diagnosed=minimal_diagnosed)
            Rank_lr_df_list.append(tmp_Rank_df)
        Rank_df = pd.concat(Rank_lr_df_list)
        Rank_df["dir_name"] = name
        Rank_df.to_csv(csv_file_name, index=False)
    return Rank_df

def update_folds_tmp_rank_df(Rank_df,minimal_diagnosed=1):
    Rank_df.loc[:, "Ratio"] = Rank_df.loc[:, "Ratio"].fillna(0)

    if Rank_df.loc["1", "Diagnosed"]==0:
        Rank_df.loc["1", "Ratio"]=minimal_diagnosed/Rank_df.loc["1", "All"]
    else:
        Rank_df.loc["1", "Ratio"]=Rank_df.loc["1", "Diagnosed"] / Rank_df.loc["1", "All"]

    Rank_df["Fold"] = Rank_df.loc[:, "Ratio"] / Rank_df.loc["1", "Ratio"]
    Rank_df.index.name = "quantile number"
    Rank_df.reset_index(drop=False, inplace=True)
    return Rank_df

def test_rank_df(vals_df, boundaries,pred_col="y_pred",minimal_diagnosed=1):
    """
    vals_df is a dataframe containing "y_test" binary values and a pred_col which is the prediction scores
    This function bins the results to quantiles and compute its true means
    """
    "Called by compare_quantiles function"
    #     print("in test label:", test_label)
    vals_df = vals_df.sort_values(pred_col, ascending=False)
    Rank_df = pd.DataFrame()
    for ind in np.arange(len(boundaries)-1):
        low_bnd=boundaries[ind]
        upp_bnd=boundaries[ind+1]
        Rank_df.loc[np.str(ind + 1), "Lower Boundary"] = low_bnd
        Rank_df.loc[np.str(ind + 1), "upper Boundary"] = upp_bnd
        # if ind == 0:
        #     Rank_df.loc["1", "Diagnosed"] = vals_df.loc[vals_df[pred_col] <= upp_bnd, :].loc[:, 'y_test'].sum()
        #
        #     Rank_df.loc["1", "All"] =vals_df.loc[vals_df[pred_col] <= upp_bnd, :].loc[:, 'y_test'].count()
        #
        # else:
        Rank_df.loc[np.str(ind+1), "Diagnosed"] = vals_df.loc[((vals_df[pred_col] <= upp_bnd) &
                                                             (vals_df[pred_col] >low_bnd)), :].loc[:, 'y_test'].sum()
        Rank_df.loc[np.str(ind+1), "All"] = vals_df.loc[
                                            (vals_df[pred_col] >low_bnd) & (vals_df[pred_col] <=upp_bnd),:].loc[
                                            :, 'y_test'].count()

        Rank_df.loc[np.str(ind+1), "Ratio"] = Rank_df.loc[np.str(ind+1), "Diagnosed"] / Rank_df.loc[np.str(ind+1), "All"]
    Rank_df=update_folds_tmp_rank_df(Rank_df,minimal_diagnosed)
    return Rank_df


def calc_Rank_df(vals_df, quants_bins, prevalence_thresh_fold=None,pred_col="y_pred",minimal_diagnosed=1):
    """
    vals_df is a dataframe containing "y_test" binary values and a pred_col which is the prediction scores
    This function bins the results to quantiles and compute its true means
    """
    "Called by compare_quantiles function"
    #     print("in test label:", test_label)
    vals_df = vals_df.sort_values(pred_col, ascending=False)
    Quants = vals_df.loc[:, pred_col].quantile(quants_bins)
    Rank_df = pd.DataFrame()
    for ind, quant in enumerate(Quants.values):
        if ind == 0:
            Rank_df.loc["1", "Lower Boundary"] = 0
            Rank_df.loc["1", "upper Boundary"] = quant

            Rank_df.loc["1", "Diagnosed"] = vals_df.loc[vals_df[pred_col] <= quant, :].loc[:, 'y_test'].sum()

            Rank_df.loc["1", "All"] =vals_df.loc[vals_df[pred_col] <= quant, :].loc[:, 'y_test'].count()

            Rank_df.loc["1", "Ratio"] =float(Rank_df.loc[np.str(ind+1), "Diagnosed"]) / float(Rank_df.loc[np.str(ind+1), "All"])

        else:
            Rank_df.loc[np.str(int(ind+1)), "Lower Boundary"] = Quants.values[ind-1]
            Rank_df.loc[np.str(int(ind+1)), "upper Boundary"] = quant
            Rank_df.loc[np.str(int(ind+1)), "Diagnosed"] = vals_df.loc[((vals_df[pred_col] <= quant) &
                                                                 (vals_df[pred_col] >Quants.values[ind-1] )), :].loc[
                                                           :, 'y_test'].sum()
            Rank_df.loc[np.str(int(ind+1)), "All"] = vals_df.loc[((vals_df[pred_col] > Quants.values[ind-1]) & (vals_df[pred_col] <=
                                                                                          quant)),
                                              :].loc[
                                              :, 'y_test'].count()
            Rank_df.loc[np.str(int(ind+1)), "Ratio"] = Rank_df.loc[np.str(ind+1), "Diagnosed"] / Rank_df.loc[np.str(ind+1), "All"]
    Rank_df=update_folds_tmp_rank_df(Rank_df,minimal_diagnosed=minimal_diagnosed)
    return Rank_df


def calc_CI_percentile(folder_path, category="Fold", alpha=0.95, plot_hist=False,
                       calc_precentile=True, save_hist=[], dpi=200, prefix="10"):
    quant_ci_res_list = []
    name = os.path.basename(folder_path)
    Rank_df = pd.read_csv(os.path.join(folder_path, name + "_" + prefix + "_quantiles_rank_df.csv"),
                          index_col="dir_name")
    Rank_df = Rank_df.loc[Rank_df.loc[:, "quantile number"] == 10, :]
    files_names = list(set(Rank_df.index.values.tolist()))
    print(files_names)
    if plot_hist:
        fig, ax = plt.subplots(1, 1, figsize=(72, 64))
        Rank_df.reset_index().hist(column="Fold", bins=100, by="dir_name", ax=ax)
        plt.show()
        if type(save_hist) != list:
            plt.savefig(os.path.join(folder_path, name + prefix + "_quantiles_hist.jpg"), dpi=dpi)
            plt.savefig(os.path.join(folder_path, name + prefix + "_quantiles_hist.pdf"), dpi=dpi)
            print("saved to:", os.path.join(folder_path, name + prefix + "_quantiles_hist.pdf"))
    if calc_precentile:
        for file_name in files_names:
            metric_list = Rank_df.loc[file_name, category].values.tolist()
            metric_list = [x for x in metric_list if not pd.isnull(x)]
            p = ((1.0 - alpha) / 2.0) * 100
            lower = np.percentile(metric_list, p)
            p = (alpha + ((1.0 - alpha) / 2.0)) * 100
            upper = np.percentile(metric_list, p)
            mean = np.mean(metric_list)
            quant_ci = "{:.2f}".format(mean) + " (" + "{:.2f}".format(lower) + "-" + "{:.2f}".format(upper) + ")"
            tmp_df = pd.DataFrame({"File name": file_name,
                                   "quant_lower": [lower],
                                   'quant_mean': [mean],
                                   "quant_upper": upper,
                                   "quant_CI": quant_ci})
            tmp_df = tmp_df.set_index("File name", drop=True)
            quant_ci_res_list.append(tmp_df)

        CI_percentile_neto_df = pd.concat(quant_ci_res_list)
        CI_percentile_neto_df.to_csv(os.path.join(folder_path, name + "_" + prefix + "_quantiles_ci_neto.csv"),
                                     index=True)

        return CI_percentile_neto_df


def quant_update_CI_table(
        folder_path,
        path="/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/Imputed_screened_CI_Summary.csv",
        prefix="10"):
    base_name = os.path.basename(folder_path)
    CI_quant_path = os.path.join(folder_path, base_name + "_" + prefix + "_quantiles_ci_neto.csv")
    df = pd.read_csv(CI_quant_path, index_col="File name")
    ci_table = pd.read_csv(path, index_col="File name")
    new_cols = df.columns.values
    if new_cols[0] not in ci_table.columns:
        for col in new_cols:
            ci_table[col] = None
    df_file_names = [x.replace("_Diabetes", "") for x in df.index]
    ci_table.loc[df_file_names, new_cols] = df.loc[:, :].values
    ci_table = ci_table.loc[:, ['Label', 'Type', 'APS_lower', 'APS_mean', 'APS_median',
                                'APS_upper', 'AUROC_lower', 'AUROC_mean', 'AUROC_median',
                                'AUROC_upper', "quant_lower", 'quant_mean', "quant_upper", "quant_CI", 'Path']]
    ci_table.to_csv(path, index=True)
    return ci_table


def quant_update_folder_table(folder_path,
                              ci_path="/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/Imputed_screened_CI_Summary.csv"):
    #   labels_dict=create_labels_dict()
    CI_df = pd.read_csv(ci_path, index_col="Label")
    base_name = os.path.basename(folder_path)
    folder_table_path = os.path.join(folder_path, base_name + "_Summary_Table.csv")
    Folder_table = pd.read_csv(folder_table_path, index_col="Label")
    try:
        Folder_table.loc[:, "Quantiles fold"] = CI_df.loc[Folder_table.index, "quant_CI"]
    except:
        index_list_convert = [x for x in CI_df.index for y in Folder_table.index if y + " DT" == x]
        index_list = [y for x in CI_df.index for y in Folder_table.index if y + " DT" == x]
        Folder_table.loc[index_list, "Quantiles fold"] = CI_df.loc[index_list_convert, "quant_CI"].values
        Folder_table.to_csv(folder_table_path, index=True)
    return Folder_table


def plot_quantiles_sns(folders_list, ci=95, n_boots=1000, hue_order=None, font_scale=None,
                       kind="bar", x_labels=None, leg_title="Deciles' OR", leg_labels=None,
                       x_name="Quantile", y_name="Quantiles fold", fig_size=(30, 24),
                       fig_path=None, dpi=200, fontsize=72, y_scale=None,
                       leg_bbox_to_anchor=(0.5, 0.9), leg_loc="center_right", framealpha=0.5, frameon=True,
                       quantiles_ci_table=None):
    sns.set_style('whitegrid', {'legend.frameon': frameon})
    df = sns_load_files(folders_list)
    plot_cat(df, x_column="quantile number", y_column="Fold",
             category_column="dir_name", fig_size=fig_size, ci=ci, n_boots=n_boots,
             hue_order=hue_order, font_scale=font_scale, kind=kind,
             leg_title=leg_title, leg_labels=leg_labels, x_labels=x_labels,
             x_name=x_name, y_name=y_name, fig_path=fig_path, dpi=dpi, fontsize=fontsize,
             y_scale=y_scale, leg_bbox_to_anchor=leg_bbox_to_anchor, leg_loc=leg_loc, framealpha=framealpha,
             frameon=frameon,quantiles_ci_table=quantiles_ci_table)


def sns_load_files(folders_list,suff = "_10_quantiles_rank_df.csv"):
    df_list = []

    for folder in folders_list:
        base_folder_name = os.path.basename(folder)
        csv_file_name = os.path.join(folder, base_folder_name + suff)
        print("in sns_load_files")
        df_list.append(pd.read_csv(csv_file_name, index_col=None))

    df = pd.concat(df_list)
    print(df.head())
    return df


def plot_cat(df, font_scale, x_column, y_column, category_column, ax=None,
             fig_size=None, hue_order=None, n_boots=1000, ci=95, kind="bar", xlabels=None,
             leg_title=None, leg_labels=None, x_labels=None, x_name="Quantile",
             y_name="Quantiles fold", dpi=200, fig_path=None, fontsize=72, y_scale=None, framealpha=0.5,
             leg_bbox_to_anchor=(0.5, 0.9), leg_loc="center_right", frameon=True,quantiles_ci_table=None):
    #         color_csv=pd.read_csv("/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/colors.csv",
    #                               index_col="File name")
    if ax == None:
        fig, ax = plt.subplots(1, 1, figsize=fig_size)
    colors_df = pd.read_csv("/home/edlitzy/UKBB_Tree_Runs/For_article/colors.csv", index_col="File name")
    if hue_order == None:
        file_names_list = list(set(df.dir_name.values))
    else:
        file_names_list = hue_order
    orig_file_names_list=file_names_list
    file_names_list = [x.replace("_Diabetes", "") for x in file_names_list]
    colors_list = []
    for file_name in file_names_list:
        try:
            colors_list.append(colors_df.loc[file_name, "color"])
            print(file_name, ":", colors_df.loc[file_name, "color"])
        except:
            raise Exception(
                file_name,
                " was not found in the colors_df: \
                /home/edlitzy/UKBB_Tree_Runs/For_article/colors.csv")

    sns.set(font_scale=font_scale)
    sns.catplot(x=x_column, y=y_column, data=df, hue=category_column, capsize=0.2, ax=ax,
                hue_order=hue_order, n_boot=n_boots, ci=ci,
                kind=kind, palette=sns.color_palette(colors_list), height=fig_size[0],
                aspect=float(fig_size[1]) / fig_size[0])
    if y_scale == "log":
        ax.set_yscale("log")
    if x_labels != None:
        ax.set_xticklabels(x_labels)
    ax.legend(facecolor="white", frameon=frameon, loc=leg_loc, framealpha=framealpha,
              bbox_to_anchor=leg_bbox_to_anchor)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_ylim([df.loc[:, y_column].min(), df.loc[:, y_column].max()])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    for label in [ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(fontsize)
    leg = ax.get_legend()
    leg.set_title(leg_title)
    if quantiles_ci_table is not None:
        for t, l in zip(leg.texts, orig_file_names_list):
            mean_ci=str(int(np.round(quantiles_ci_table.loc[l,"quant_mean"],0)))
            lower_ci=str(int(np.round(quantiles_ci_table.loc[l,"quant_lower"],0)))
            upper_ci=str(int(np.round(quantiles_ci_table.loc[l,"quant_upper"],0)))
            t.set_text("x"+mean_ci+" ("+lower_ci+"-"+upper_ci+")")
    elif leg_labels is not None:
        for t, l in zip(leg.texts, leg_labels):
            t.set_text(l)  # Update labels
    plt.close(2)
    plt.tight_layout()
    if fig_path != None:
        plt.savefig(fig_path, dpi=dpi)
        print("Saved figure to:", fig_path)
    plt.show()


def Twof(number):
    return str("{0:.2f}".format(number))


def return_aps_summary(row):
    label = Twof(row["APS_mean"]) + " (" + Twof(row["APS_lower"]) + "-" + Twof(row["APS_upper"]) + ")"
    return label


def return_auroc_summary(row):
    label = Twof(row["AUROC_mean"]) + " (" + Twof(row["AUROC_lower"]) + "-" + Twof(row["AUROC_upper"]) + ")"
    return label


def calc_di_df(path):
    orig_ci = pd.read_csv(path)
    orig_ci.set_index = (orig_ci.columns[1])


def build_features_importance_df(folder_path, file_names_list, labels_dict, table_save_path):
    models_list = []
    for ind, path in enumerate(folder_path):
        cat = file_names_list[ind]
        CI_path = os.path.join(path, "CI")
        CI_Files = os.listdir(CI_path)
        models_files = [x for x in CI_Files if x.endswith(".sav")]
        models_paths = [os.path.join(CI_path, x) for x in models_files]
        use_cols_path = os.path.join(path, "use_cols.txt")
        with open(use_cols_path, "rb") as fp:  # Pickling
            use_cols = pickle.load(fp)
        use_cols = [x for x in use_cols if x != "eid"]
        for model_path in models_paths:
            clf = pickle.load(open(model_path, 'rb'))
            imp_coef = pd.DataFrame({"Covariates names": use_cols,
                                     "Covariate coefficient": clf.coef_.flatten(),
                                     "category": cat})
            bias_coef = pd.DataFrame({"Covariates names": "Bias",
                                      "Covariate coefficient": clf.intercept_,
                                      "category": cat})

            imp_coef = pd.concat([imp_coef, bias_coef])
            models_list.append(imp_coef)
    coeffs_table = pd.concat(models_list)
    coeffs_table.sort_values(by="Covariate coefficient", inplace=True)
    coeffs_table = coeffs_table
    labels_dict["Bias"] = "Bias"
    coeffs_table.loc[:, "Covariates names"] = coeffs_table.loc[:, "Covariates names"].map(
        labels_dict)
    coeffs_table.to_csv(table_save_path, index=False)
    return coeffs_table


class LR_Feature_importance:
    def __init__(self,
                 folder_path,
                 labels_file_path="/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/UKBIOBANK_Labels.csv",
                 figsize=(32, 24),
                 order=None,
                 hue_type="binary",
                 labels_dict=[],
                 dpi=200,
                 fig_path=[],
                 num_of_chars=18,
                 y_labels_rotation=0,
                 font_size=72,
                 space=1,
                 n_boots=100,
                 ci="sd",
                 positive_colors="red",
                 negative_colors="tomato",
                 linespacing=1,
                 fig=None,
                 ax=None,
                 file_names_list=None,
                 table_save_path=None,
                 hue_colors_dict={},
                 plot=True,
                 font_scale=5,
                 leg_labels=["4.5<HbA1c%<5.7", "5.7<HbA1c%<6.5"],
                 leg_title=None,
                 leg_pos=[0.9, 0.9],
                 show_values_on_bar=False,
                 remove_legend=False,
                 ci_summary_table_name="tmp_feature_imp.csv",
                 build_new=False,
                 used_labels_df_path=None):
        """
        plot_type would be:
            bar plot for type(Folder_path)==str or
            cat_plot for type(Folder_path)==list
        Folder path should contains the path to the folders whos features importance we want to present

        in case of a cat plot:
            category_name_list is required

        positive_colors, and negative_colors should be lists of colors (used only for barplot, not catplot)

        hue_colors_dict should be adictionary for the cat plot of the form:
            hue_colors_dict=dict(zip([folder_name1,folder_name2],[color1,color2])
        """
        sns.set_style('whitegrid', {'legend.frameon': True})
        self.used_labels_df_path = used_labels_df_path
        self.labels_df = None
        self.Labels_file_path = labels_file_path
        self.y_labels_rotation = y_labels_rotation
        self.n_boots = n_boots
        self.space = space  # distance of text from barplot
        self.folder_path = folder_path
        self.font_scale = font_scale
        self.figsize = figsize
        if ax == None:
            self.fig, self.ax = plt.subplots(1, 1, figsize=figsize)
        else:
            self.fig = fig
            self.ax = ax

        if type(fig_path) != list:
            self.fig_path = [fig_path]

        if file_names_list != None:
            self.file_names_list = file_names_list

        self.plot = plot
        self.hue_colors_dict = hue_colors_dict
        self.hue_type = hue_type
        self.labels_dict = labels_dict
        self.table_save_path = table_save_path
        self.LR_mean_coefficients_table_file_path = self.table_save_path[:-4] + "_mean.csv"
        self.fig_path = fig_path
        self.num_of_chars = num_of_chars
        self.ci = ci
        self.leg_labels = leg_labels
        self.leg_title = leg_title
        self.leg_pos = leg_pos
        self.negative_colors = list(negative_colors)
        self.positive_colors = list(positive_colors)
        self.linespacing = linespacing
        self.order = order
        self.font_size = font_size
        self.plot_type = self.plot_type()
        self.dpi = dpi
        self.show_values_on_bar = show_values_on_bar
        self.remove_legend = remove_legend
        self.features_ci_table = None
        self.ci_summary_table_name = os.path.join(
            "/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/Tables/", ci_summary_table_name)

        if not self.ci_summary_table_name.endswith(".csv"):
            self.ci_summary_table_name = self.ci_summary_table_name + ".csv"
        if build_new:
            if os.path.isfile(self.LR_mean_coefficients_table_file_path):
                os.remove(self.LR_mean_coefficients_table_file_path)
            else:
                print(self.LR_mean_coefficients_table_file_path, "does not exist")
            self.coeffs_table = self.build_features_importance_df()
            self.save_table()
            self.mean_coefficients_table_df = self.save_mean_coefficients_table()
        # if os.path.isfile(self.table_save_path):
        self.coeffs_table = pd.read_csv(self.table_save_path)
        self.mean_coefficients_table_df = pd.read_csv(self.LR_mean_coefficients_table_file_path,
                                                      index_col="Covariates names")

        # print("order:", self.order)
        if self.order != None:
            self.long_name_order = self.create_long_name_order()
        else:
            print("at self.long_name_order:")
            self.order = self.mean_coefficients_table_df.sort_values(
                by="Covariate coefficient", ascending=False).index.values.tolist()
            self.long_name_order = self.order
            # print("self.long_name_order:", self.long_name_order)
        #         print("plot_bar_plot()")
        self.calc_ci_table()
        self.plot_graph()
        # controls default text sizes

    def plot_type(self):
        if type(self.folder_path) == list:
            # print("plotting cat plot")
            self.plot_type = "cat_plot"
        elif type(self.folder_path) == str:
            # print("plotting bar plot")
            self.plot_type = "bar_plot"
            if len(self.positive_colors) < 1 or len(self.negative_colors) < 1:
                sys.exit("For bar plot, you should provide positive_colors str and negative_colors str")
        else:
            print("Folder_path should be either string for single bar plot or list for cat plot")
        return self.plot_type

    def get_order(self):
        return self.order

    def get_fig_ax_order(self):
        return self.fig, self.ax

    def plot_graph(self):
        if self.plot_type == "bar_plot":
            self.plot_bar(plot=self.plot)
        elif self.plot_type == "cat_plot":
            self.plot_cat(plot=self.plot)

    def plot_cat(self, plot=True):
        #         color_csv=pd.read_csv("/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/colors.csv",
        #                               index_col="File name")
        try:
            colors_list = [self.hue_colors_dict[x] for x in self.file_names_list]
        except:
            colors_list = ["b" for x in self.file_names_list]
        sns.set(font_scale=self.font_scale)
        ax_sns = sns.catplot(x="Covariate coefficient", y="Covariates names",
                             data=self.coeffs_table, hue="category", capsize=.2,
                             ax=self.ax, order=self.long_name_order,
                             hue_order=self.file_names_list, n_boot=self.n_boots, ci=self.ci,
                             kind="bar", palette=sns.color_palette(colors_list),
                             height=self.figsize[0],
                             aspect=float(self.figsize[1]) / self.figsize[0])
        plt.close(2)
        self.sns_plot_params_update()
        if self.show_values_on_bar:
            self.show_values_on_bars()
        plt.tight_layout()
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['top'].set_visible(False)
        self.save_fig()
        if plot:
            plt.show()
        else:
            return self.fig, self.ax

    def sns_plot_params_update(self):
        plt.rcParams.update({'font.size': self.font_size})
        self.short_labels = wrap_labels(self.long_name_order, self.num_of_chars)
        self.ax.set_yticklabels(self.short_labels, rotation=self.y_labels_rotation, linespacing=self.linespacing,
                                fontsize=self.font_size)
        self.ax.set_ylabel('')
        self.ax.set_xlabel('Logistic regression covariates coefficients', fontsize=self.font_size)
        self.ax.tick_params(labelsize=self.font_size)
        if self.remove_legend:
            self.ax.get_legend().remove()
        else:
            self.ax.legend(facecolor=None, frameon=False, loc='upper left')
            leg = self.ax.get_legend()
            leg.set_title(self.leg_title)

            for t, l in zip(leg.texts, self.leg_labels):
                t.set_text(l)  # Update labels

    def plot_bar(self, plot=True):
        self.hues = self.comp_hue()

        if self.hue_type == "binary":
            self.ax = sns.barplot(x="Covariate coefficient", y="Covariates names",
                                  data=self.coeffs_table, capsize=.2,
                                  palette=self.hues, saturation=self.saturation,
                                  ax=self.ax, order=self.long_name_order, n_boot=self.n_boots,
                                  ci=self.ci)
        else:
            self.ax = sns.barplot(x="Covariate coefficient", y="Covariates names",
                                  data=self.coeffs_table, capsize=.2,
                                  palette=sns.color_palette(self.color_pallete), saturation=self.saturation, ax=self.ax,
                                  order=self.long_name_order, n_boot=self.n_boots, ci=self.ci)
        print("Finished calculating barplot")
        self.show_values_on_bars()
        self.short_labels = wrap_labels(self.long_name_order, self.num_of_chars)
        self.ax.set_yticklabels(self.short_labels, rotation=self.y_labels_rotation, linespacing=self.linespacing)
        self.ax.yaxis.label.set_visible(False)
        plt.tight_layout()
        self.save_fig()
        if plot:
            plt.show()
        else:
            return self.fig, self.ax

    def save_mean_coefficients_table(self):
        """
        This function is used to order ht ecoefficients"""
        self.mean_coefficients_table_df = self.coeffs_table.loc[:,
                                          ["Covariates names", "Covariate coefficient"]
                                          ].groupby("Covariates names").mean()

        if type(self.table_save_path) != list:
            self.mean_coefficients_table_df.to_csv(self.table_save_path[:-4] + "_mean.csv", index=True)
        return self.mean_coefficients_table_df

    def comp_hue(self):
        coeffs_table = self.coeffs_table
        # print("self.long_name_order:", self.long_name_order)
        #         print("coeffs_table:",coeffs_table)
        if self.hue_type == "binary":
            self.hues = [
                self.positive_color if coeffs_table.loc[
                                           coeffs_table.loc[:,
                                           "Covariates names"] == x, "Covariate coefficient"].mean() > 0
                else self.negative_color for x in self.long_name_order]
        else:
            self.hues = [
                coeffs_table.loc[
                    coeffs_table.loc[
                    :, "Covariates names"] == x, "Covariate coefficient"].mean() for x in self.long_name_order]
            huemax = np.max(self.hues)
            huemin = np.min(self.hues)
            self.hues = [(x - huemin) / (huemax - huemin) for x in self.hues]
        return self.hues

    def build_features_importance_df(self):
        models_list = []
        for ind, path in enumerate(self.folder_path):
            cat = self.file_names_list[ind]
            CI_path = os.path.join(path, "CI")
            CI_Files = os.listdir(CI_path)
            models_files = [x for x in CI_Files if x.endswith(".sav")]
            models_paths = [os.path.join(CI_path, x) for x in models_files]
            use_cols_path = os.path.join(path, "use_cols.txt")
            with open(use_cols_path, "rb") as fp:  # Pickling
                use_cols = pickle.load(fp)
            use_cols = [x for x in use_cols if x != "eid"]
            for model_path in models_paths:
                clf = pickle.load(open(model_path, 'rb'))
                imp_coef = pd.DataFrame({"Covariates names": use_cols,
                                         "Covariate coefficient": clf.coef_.flatten(),
                                         "category": cat})
                bias_coef = pd.DataFrame({"Covariates names": "Bias",
                                          "Covariate coefficient": clf.intercept_,
                                          "category": cat})

                imp_coef = pd.concat([imp_coef, bias_coef])
                types = [type(x) for x in imp_coef["Covariates names"] if type(x) != str]
                # if types != []:
                #     print (types, " in ", model_path)
                models_list.append(imp_coef)
        coeffs_table = pd.concat(models_list)
        coeffs_table.sort_values(by="Covariate coefficient", inplace=True)
        self.coeffs_table = coeffs_table
        self.labels_dict["Bias"] = "Bias"
        self.add_covariate_name_todict_if_not_exists()
        self.coeffs_table.loc[:, "Covariates names"] = self.coeffs_table.loc[:, "Covariates names"].map(
            self.labels_dict)
        self.coeffs_table.to_csv(self.table_save_path, index=False)
        return self.coeffs_table

    def create_long_name_order(self, order):
        try:
            self.long_name_order = [self.labels_dict[x] for x in order]
        except:
            print("can't convert names in create_long_order()")
            self.long_name_order = order
        return self.long_name_order

    def show_values_on_bars(self):
        def _show_on_single_plot(ax):
            for p in ax.patches:
                #                 _x = p.get_x() + p.get_width() + self.space*np.sign(p.get_width())
                _x = p.get_x() - self.space * np.sign(p.get_width())
                _y = p.get_y() + p.get_height() * 3 / 4
                value = np.round(p.get_width(), 2)
                ax.text(_x, _y, str(value), ha="center")

        _show_on_single_plot(self.ax)

    def save_fig(self, fig_path=None):
        """
        fig_path is for external calling of the function. In case someone would like to save the figure
        to a new path
        """
        if fig_path != None:
            self.fig_path = fig_path
        if self.fig_path != []:
            plt.savefig(self.fig_path, dpi=self.dpi, frameon=False)
            print('Saved fig to:', self.fig_path)
        else:
            plt.savefig(self.fig_path[0], dpi=self.dpi, frameon=False)
            print('Saved fig to:', self.fig_path[0])

    def save_table(self):
        if self.table_save_path != None:
            self.coeffs_table.to_csv(self.table_save_path, index=False)
        print("Coeffs table saved to:", self.coeffs_table)

    def _summary_CI_col(self, row):
        print(row)
        return "{:.2f}".format(
            row["Mean"]) + " (" + "{:.2f}".format(row["CI 0.025"]) + ":" + "{:.2f}".format(
            row["CI 0.975"]) + ")"

    def calc_ci_table(self):
        res1 = self.coeffs_table.groupby("Covariates names").quantile([0.025, 0.5, 0.975], axis=0, numeric_only=True,
                                                                      interpolation='linear')
        res2 = pd.DataFrame(res1.unstack(level=-1).to_records()).set_index("Covariates names")
        res2.columns = ["CI 0.025", "Median", "CI 0.975"]
        res3 = self.coeffs_table.groupby("Covariates names").mean()
        res3.columns = ["Mean"]
        res4 = res2.join(res3)
        res4 = res4.loc[:, ["CI 0.025", "Mean", "CI 0.975", "Median"]]
        res4.loc[:, "Summary"] = res4.apply(self._summary_CI_col, axis=1)
        self.features_ci_table = res4
        res4.to_csv(self.ci_summary_table_name)

    def add_covariate_name_todict_if_not_exists(self, ):
        for col in self.coeffs_table.loc[:, "Covariates names"]:
            if col not in self.labels_dict.keys():
                print("Adding col", col, "to labels_dictionary, check or add col to : ", self.Labels_file_path)
                self.labels_dict[col] = col
        self.labels_df = pd.DataFrame(data=self.labels_dict.items())
        print("self.labels_df is:", self.labels_df)
        try:
            self.labels_df.to_csv(self.used_labels_df_path, index=False)
        except:
            print("Couldnt save to used_labels_df_path:", self.used_labels_df_path)


def simple_shap(model_folder_path, top_feat_num=10, n_rows=None,
                font_size=72, figsize=(30, 24), Shap_plot_path=None, plot=True,
                dpi=100, x_ticks_labels=None, y_labels_dict=None, xlabel_pos=0.4,
                ylabel_pos=0, num_of_chars=20, linespace=1, text=None,
                x_text=0.15, y_text=0.1, leg_loc=(0.15, 0.2), leg_title_pos=(0, 0),
                leg_title="Impact on prediction", color_positive="purple",
                color_negative="orange", variables_dict=None,
                external_labels=None,
                basic_path="/home/edlitzy/UKBB_Tree_Runs/For_article/", update=True):
    x_test_path = os.path.join(model_folder_path, "Diabetes_Results/Diabetestest_Data")
    if variables_dict == None:
        variables_df = pd.read(os.path.join(basic_path, "UKBIOBANK_Labels.csv"), index_col="Code")
        for key in variables_df.index.values:
            variables_dict[key] = variables_df.loc[key, "Label"]
    shap_path = os.path.join(model_folder_path, "Diabetes_Results/Diabetes_shap_values.csv")
    Shap_display = os.path.join(model_folder_path, "Diabetes_Results/Diabetes_X_display.csv")
    Feat_path = os.path.join(model_folder_path, "Diabetes_Results/Diabetes_DF_Features_List.csv")

    if not update:
        top_k2 = pd.read_csv(os.path.join(model_folder_path, "top_k2.csv"))
    else:
        with open(x_test_path, 'rb') as fp:
            Test_Data = pickle.load(fp)
        df_v = Test_Data["X_display"]
        shap_v = pd.read_csv(shap_path, index_col="eid")
        shap_v.drop("BiaShap", axis=1, inplace=True)
        corr_list = []
        feature_list = df_v.columns.values
        shap_v.columns = feature_list
        mut_ind = [x for x in df_v.index if x in shap_v.index]
        shap_v = shap_v.loc[mut_ind, :]
        df_v = df_v.loc[mut_ind, :]
        for i in feature_list:
            b = np.corrcoef(shap_v[i], df_v[i])[1][0]
            corr_list.append(b)
        corr_df = pd.concat([pd.Series(feature_list), pd.Series(corr_list)], axis=1).fillna(0)
        # Make a data frame. Column 1 is the feature, and Column 2 is the correlation coefficient
        corr_df.columns = ['Variable', 'Corr']
        signs = [color_positive if x > 0 else color_negative for x in corr_df['Corr']]
        corr_df['Sign'] = signs
        shap_abs = np.abs(shap_v)
        k = pd.DataFrame(shap_abs.mean()).reset_index()
        k.columns = ['Variable', 'SHAP_abs']
        k2 = k.merge(corr_df, left_on='Variable', right_on='Variable', how='inner')
        k2 = k2.sort_values(by='SHAP_abs', ascending=False)
        colorlist = k2['Sign']
        top_k2 = k2.iloc[:top_feat_num]
        top_k2.sort_values(by='SHAP_abs', inplace=True)
        top_k2.to_csv(os.path.join(model_folder_path, "top_k2.csv"), index=False)
    colorlist = top_k2['Sign']
    labels_list = [variables_dict[x] if x in variables_dict.keys() else x.split("_")[0] for x in top_k2.Variable]
    orig_name_list = [x for x in top_k2.Variable if x not in variables_dict.keys()]
    labels_dict = dict(zip(top_k2.Variable.values, labels_list))
    variables_dict.update(labels_dict)
    labels = [variables_dict[x] for x in top_k2.Variable]
    print("Labels:", labels)
    figsize = (30, 24)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax = top_k2.plot.barh(x="Variable", y='SHAP_abs', color=top_k2.loc[:, "Sign"].values,
                          legend=False, fontsize=font_size, ax=ax)
    ax.set_xlabel("SHAP feature importance", fontsize=font_size,
                  x=xlabel_pos, y=ylabel_pos)
    ax.set_facecolor("white")
    #     print x_ticks_labels
    if x_ticks_labels != None:
        ax.set_xticks(x_ticks_labels)
        x_ticks_labels = [str("{0:.2f}".format(x)) for x in x_ticks_labels]
        x_ticks_labels[0] = "0"
        ax.set_xticklabels(x_ticks_labels, fontsize=font_size)
    if type(external_labels) == list:
        y_labels = wrap_labels(external_labels, num_of_chars=num_of_chars)
    else:
        y_labels = wrap_labels(labels, num_of_chars=num_of_chars)
    ax.set_yticklabels(y_labels, fontsize=font_size, linespacing=linespace)
    if text != None:
        ax.text(x=x_text, y=y_text, s=text, fontsize=font_size)
    yaxislabel = ax.yaxis.get_label()
    yaxislabel.set_visible(False)
    positive_feat = mpatches.Patch(color=color_positive, label='Positive')
    negative__feat = mpatches.Patch(color=color_negative, label='Negative')
    leg = ax.legend(handles=[positive_feat, negative__feat], loc=leg_loc, fontsize=font_size, frameon=False)
    leg.set_title(title=leg_title, prop={'size': font_size})
    leg._legend_box.align = "left"
    leg.get_title().set_position(leg_title_pos)  # -10 is a guess
    plt.tight_layout()
    if Shap_plot_path != None:
        plt.savefig(Shap_plot_path, dpi=dpi, frameon=False)
    if plot:
        plt.show()
    return variables_dict, top_k2


def comp_roc_groups(df, comp_roc_path=None, plot=True, dpi=100, leg_len=20, labels_length=20, leg_pos=(0.02, 0.85),
                    ylim=(0.5, 1), barWidth=0.4, x_text=0.02, y_text=0.8, text="GDRS- German Diabetes Risk Score",
                    color_h="r", color_l="g", fig_size=(30, 24), high_recogniser="5.7<=HbA1c%<6.5",
                    low_recogniser="4<=HbA1c%<5.7",
                    replace_dict={}, ylabels=["0.5", "0.6", "0.7", "0.8", "0.9", "1"], font_size=16,
                    legend_font_size=16):
    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'
    bar_high_labels = [x for x in df.index.values if high_recogniser in x]
    bar_low_labels = [x for x in df.index.values if low_recogniser in x]
    print ("bar_high_labels:", bar_high_labels)
    print ("bar_low_labels:", bar_low_labels)

    bar_high = df.loc[bar_high_labels, "AUC"].values
    bar_low = df.loc[bar_low_labels, "AUC"].values
    r1 = np.arange(len(bar_low)) - barWidth / 2
    r2 = [x + barWidth for x in r1]
    ax.bar(r1, bar_low, color=color_l, width=barWidth, edgecolor='white', label='4<=%HbA1c<5.7')
    ax.bar(r2, bar_high, color=color_h, width=barWidth, edgecolor='white', label='5.7<=%HbA1c<6.5')
    ax.set_ylim(ylim)
    ax.tick_params(axis='both', which='major', labelsize=font_size)

    for key in replace_dict:
        print (key)
        print (replace_dict[key])
        bar_high_labels = [x.replace(key, replace_dict[key]) for x in bar_high_labels]
    labels = wrap_labels(bar_high_labels, labels_length)
    ax.set_xticks([r for r in range(len(bar_high))])
    ax.set_xticklabels(labels, rotation='vertical', fontsize=font_size)
    ax.set_yticks([float(x) for x in ylabels])
    ax.set_yticklabels(ylabels, fontsize=font_size)
    ax.set_ylabel('AUROC', fontsize=font_size)
    ax.text(x=x_text, y=y_text, s=text, fontsize=font_size)
    # Create legend & Show graphic
    handles, labels = ax.get_legend_handles_labels()
    labels = wrap_labels(labels, leg_len)
    leg = ax.legend(reversed(handles), reversed(labels), loc=leg_pos,
                    fontsize=legend_font_size, fancybox=False, labelspacing=0.15, frameon=False)

    plt.tight_layout()

    if comp_roc_path != None:
        plt.savefig(comp_roc_path, dpi=dpi, frameon=False)
    if plot:
        plt.show()
    return ax


def comp_aps_groups(df, comp_aps_path=None, fig_size=(30, 24), plot=True, dpi=100, labels_length=15,
                    leg_pos1=(0, 0.87), leg_pos2=(0, 0.8), barWidth=0.4, x_text=-0.55, y_text=0.45,
                    text=[], color_h="r", color_l="g", linespace=1, high_recogniser="5.7<=HbA1c%<6.5",
                    low_recogniser="4<=HbA1c%<5.7", replace_dict={}, yticks_high=None, yticks_low=None, font_size=16):
    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'
    proportion = 1.5
    bar_high_labels = [x for x in df.index.values if high_recogniser in x]
    bar_low_labels = [x for x in df.index.values if low_recogniser in x]

    bar_high = df.loc[bar_high_labels, "APS"].values
    #     print "bar_high:", df.loc[bar_high_labels,"APS"]
    bar_low = df.loc[bar_low_labels, "APS"].values
    #     print "bar_low:", df.loc[bar_low_labels,"APS"]
    xticks = np.arange(len(bar_high))
    r1 = [x + barWidth / 2 for x in xticks]
    r2 = [x - barWidth / 2 for x in xticks]

    # r2 = [x + spacing+len(bar_high) for x in r1]

    #     colors=[color_dict[x] for x in bar_high_labels]

    # ax.bar(r1, bar_low, color=color_dict['Minimal features, HbA1c% 4-5.6% -LR'], width=barWidth, edgecolor='white', label='%HbA1C 4-5.6%')
    #     ax.bar(r2, bar_high, color=color_dict['Minimal features, HbA1c%>5.6% -LR'], width=barWidth, edgecolor='white', label='%HbA1C>5.6%')
    ax.bar(r1, bar_high, color=color_h,
           width=barWidth, label='5.7<=%HbA1C<6.5')
    ax2 = ax.twinx()
    # ax2.set_ylim(0,1.05*np.max(np.array(bar_low)))
    ax2.bar(r2, bar_low, color=color_l,
            width=barWidth, label='4<=%HbA1C<5.7')

    ax.tick_params(axis='both', which='major', labelsize=font_size)
    # ax.set_xlabel('group', fontweight='bold')

    for key in replace_dict:
        bar_high_labels = [x.replace(key, replace_dict[key]) for x in bar_high_labels]

    labels = wrap_labels(bar_high_labels, labels_length)

    # ax.set_xticks([[r for r in range(len(bar_high))])
    ax.set_xticks(xticks)
    ax.set_xticklabels(labels, rotation='vertical', fontsize=font_size, linespacing=linespace)

    if yticks_high == None:
        yticks_high = np.arange(0, np.max(bar_high), 0.1)
    ax.set_yticks([float(x) for x in yticks_high])
    ax.set_yticklabels(yticks_high, fontsize=font_size)
    ax.set_ylabel('APS (%HbA1c 5.7-6.5%)', fontsize=font_size)
    ax.legend(loc=leg_pos1, fontsize=font_size, frameon=False)
    if type(text) != list:
        ax.text(x=x_text, y=y_text, s=text, fontsize=font_size)

    if yticks_low == None:
        yticks_low = np.arange(0, np.max(bar_low), 0.02)
    ax2.set_yticks([float(x) for x in yticks_low])
    ax2.set_yticklabels(yticks_low, fontsize=font_size)
    ax2.legend(loc=leg_pos2, fontsize=font_size, frameon=False)
    ax2.set_ylabel('APS (%HbA1c 4-5.7%)', fontsize=font_size)
    ax2.tick_params(axis='both', which='major', labelsize=font_size)

    leg = ax.get_legend()
    leg.legendHandles[0].set_color(color_h)
    leg2 = ax2.get_legend()
    leg2.legendHandles[0].set_color(color_l)
    plt.tight_layout()
    if comp_aps_path != None:
        #         print("Saving to:",comp_aps_path)
        plt.savefig(comp_aps_path, dpi=dpi, frameon=False)
    if plot:
        plt.show()
    return ax


def load_lr_model(filename, Rel_Feat_Names, title="Feature importance using Lasso Model", figsize=(16, 9),
                  num_feat=10):
    #    filename="/home/edlitzy/UKBB_Tree_Runs/For_article/compare_GDRS/LR_Non_lab_non_diet/LR_Model.sav"
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    clf = joblib.load(filename)
    imp_coef = pd.DataFrame({"Covariates names": Rel_Feat_Names, "Covariate coefficient": clf.coef_.flatten()})
    imp_coef_sorted = imp_coef.sort_values(by="Covariate coefficient", ascending=True)
    imp_coef_sorted.set_index("Covariates names", inplace=True)
    top_feat = list(np.arange(0, num_feat))
    bot_feat = list(np.arange(-num_feat, 0))
    tot_feat = top_feat + bot_feat
    top_20_coeff = imp_coef_sorted.iloc[tot_feat, :]
    mpl.rcParams['figure.figsize'] = (8.0, 10.0)
    top_20_coeff.plot.barh(figsize=figsize)
    ax.set_title(title, fontsize=18)
    ax.set_yticklabels(top_20_coeff.index.values, fontsize=18)
    ax.set_xlabel("Covariates coefficients", fontsize=18)
    # plt.ylabel("Covariates names",fontsize=18)
    return fig, ax


def plot_calibration_curve(df, files_list=None, path=None, nbins=10, splits=3, fig_size=(16, 9), colors_list=None,
                           pallete="viridis", colors_dict=[],
                           plot_hist=True, fontsize=16, leg_dict=[], x_text=0, y_text=1,
                           x_ticks=["0", "0.2", "0.4", "0.6", "0.8", "1"],
                           y_ticks=["0", "0.2", "0.4", "0.6", "0.8", "1"],
                           xlabel="predicted probability", ylabel="True probability", lw=2, dpi=200, plot=True,
                           marker="o", markersize=12, range_dict={}, bins_dict={}, splits_dict={}, nbins_hist=20,
                           y_hist_ticks=["10", "100", "1000"], xlim=0.8, ylim=0.8, hist_alpha=1, leg_loc="best",
                           hist_bar_pos=0.5, hist_bar_width=0.8, font_size=16):
    """Plot calibration curve for est w/o and with calibration. """
    if type(files_list) != list:
        files_list = df.index.values
    if type(colors_dict) == list:
        color_pallete = cm.get_cmap('viridis', len(files_list))
        color_dict = {}
        colors_list = color_pallete(np.linspace(0, 1, len(files_list)))
        colors_dict = dict(zip(files_list, colors_list))
    if type(leg_dict) == list:
        leg_dict = dict(zip(files_list, files_list))

    fig, ax1 = plt.subplots(1, 1, figsize=fig_size)

    ax1.plot([0, 10], [0, 1], "k:", lw=lw)

    plt.rc('font', size=fontsize)  # controls default text sizes
    plt.rc('axes', titlesize=fontsize)  # fontsize of the axes title
    plt.rc('axes', labelsize=fontsize)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=fontsize)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=fontsize)  # fontsize of the tick labels
    plt.rc('legend', fontsize=fontsize)  # legend fontsize
    plt.rc('figure', titlesize=fontsize)  # fontsize of the figure title
    hist_dict = {}
    results_dict = {}
    for ind, label_name in enumerate(files_list):
        print("****", label_name, "*******")
        try:
            top_range = range_dict[label_name]
        except:
            top_range = 1
        try:
            n_bins = bins_dict[label_name]
        except:
            n_bins = nbins
        try:
            n_splits = splits_dict[label_name]
        except:
            n_splits = splits
        try:
            leg_val = leg_dict[label_name]
        except:
            leg_val = label_name

        if type(label_name) != str:
            print("Label name:", label_name, " is not astring, might be nan")
            continue
        with open(df.loc[label_name, "y_files"], 'rb') as fp_test:
            y_test = pickle.load(fp_test)
        with open(df.loc[label_name, "predictions_files"], 'rb') as fp_pred:
            y_pred = pickle.load(fp_pred)

        fraction_of_positives, mean_predicted_value, y_test_array, y_cal_prob_array = monotonic_calibration(
            y_test, y_pred, splits=splits, nbins=n_bins)
        print("n_bins:", n_bins)
        results_dict[label_name] = pd.DataFrame(index=np.arange(len(fraction_of_positives)),
                                                columns=["True pos fraction", "pred pos fraction"])
        results_dict[label_name].loc[:, "True pos fraction"] = fraction_of_positives
        results_dict[label_name].loc[:, "pred pos fraction"] = mean_predicted_value
        clf_score = brier_score_loss(y_test_array, y_cal_prob_array,
                                     pos_label=np.max(y_test_array))
        hist_dict[label_name] = y_cal_prob_array
        low_mean_calibrated_value = [x for x in mean_predicted_value if x <= top_range]
        high_mean_calibrated_value = [x for x in mean_predicted_value if x >= top_range]
        #         print "low_mean_calibrated_value:",low_mean_calibrated_value
        if len(low_mean_calibrated_value) > 0:
            low_fraction_of_positives = fraction_of_positives[:len(low_mean_calibrated_value)]
            stretch_low_mean_calibrated_value = [10 * x for x in low_mean_calibrated_value]
            ax1.plot(stretch_low_mean_calibrated_value, low_fraction_of_positives, color=colors_dict[label_name],
                     linestyle="-", lw=lw, marker=marker, markersize=markersize, label=leg_val + "-"
                                                                                       + "{:.3f}".format(clf_score))

            print("calibrated ", label_name, " brier score: ", clf_score)
    #             print("y_cal_prob_array of:",label_name," is:",y_cal_prob_array)

    #     ax1.set_xlim(0,xlim)

    if plot_hist:
        ax2 = ax1.twinx()
        x = np.arange(0, (nbins_hist + 1) / 10, 0.1)
        y = pd.DataFrame(index=x, columns=hist_dict.keys())
        for key in hist_dict.keys():
            tmp_hist = np.histogram(hist_dict[key], bins=nbins_hist)
            y.loc[:, key] = tmp_hist[0]
        colors = [colors_dict[ind] for ind in y.columns]
        y.plot(kind='bar', color=colors, legend=False, alpha=0.3, rot=0, ax=ax2, use_index=True, position=hist_bar_pos,
               width=hist_bar_width)
        ax2.set_ylabel("Participants/bin")
        ax2.set_yscale('log')
        y_hist_ticks = [1, 10, 100, 1000, 10000]
        y_hist_tick_labels = [str('%.e' % ind)[:2] + str('%.E' % ind)[-1] for ind in y_hist_ticks]
        ax2.set_yticks(y_hist_ticks)
        ax2.set_yticklabels(y_hist_tick_labels, fontsize=font_size)

    ax1.set_ylabel(ylabel)
    ax1.set_xlabel(xlabel)
    ax1.legend(loc=leg_loc, fontsize=fontsize, frameon=False, framealpha=0, title="Brier score")
    ax1.set_xticks([10 * float(x) for x in x_ticks])
    ax1.set_yticks([float(x) for x in y_ticks])
    ax1.set_xticklabels(x_ticks)
    ax1.set_xlim(-0.4, xlim * 10)
    ax1.set_ylim(0, ylim)
    plt.tight_layout()
    if path != None:
        print("Saving to:", path)
        plt.savefig(path, dpi=dpi, frameon=False)
    if plot:
        plt.show()
    return results_dict


def monotonic_calibration(y_test, y_pred, splits, nbins):
    X = np.array(y_pred)
    y = np.array(y_test)
    ir = IsotonicRegression()
    skf = StratifiedKFold(n_splits=splits, random_state=0, shuffle=False)
    y_true_list = []
    y_cal_prob_list = []
    for train_index, test_index in skf.split(X, y):
        # print "test index shape",test_index.shape
        # print "train index shape", train_index.shape

        X_Cal_train, X_Cal_test = X[train_index], X[test_index]
        y_Cal_train, y_Cal_test = y[train_index], y[test_index]

        ir.fit(X_Cal_train, y_Cal_train)
        p_calibrated = ir.transform(X_Cal_test)
        y_cal_prob_list = np.append(y_cal_prob_list, p_calibrated)
        y_true_list = np.append(y_true_list, y_Cal_test)

    y_true_array = np.array(y_true_list)
    y_cal_prob_array = np.array(y_cal_prob_list)
    nans_list = np.argwhere(np.isnan(y_cal_prob_array))
    y_cal_prob_array = np.delete(y_cal_prob_array, nans_list)
    y_true_array = np.delete(y_true_array, nans_list)
    frac_pos, frac_neg = calibration_curve(y_true_array, y_cal_prob_array, n_bins=nbins)

    return frac_pos, frac_neg, y_true_array, y_cal_prob_array


def get_key_name(val, dictionary):
    for key in dictionary.keys():
        if dictionary[key] == val:
            return key
    print("No key for: ", val, " Dictionary is: ", dictionary)


def plot_legend(figsize, labels_list, colors_list, ncolumns=1):
    """
    This funcrtion gets nrows ncolumns,figsize and label_color_dictionar which is a dictionary of labels and colors
    Plots, returns saves a legend as a fig
    """
    # Create a color palette
    palette = dict(zip(labels_list, colors_list))
    # Create legend handles manually
    handles = [mpl.patches.Patch(color=palette[x], label=x) for x in palette.keys()]
    # Create legend
    leg = plt.legend(handles=handles, ncol=ncolumns)
    # Get current axes object and turn off axis
    plt.gca().set_axis_off()
    plt.show()
    return leg


def net_benefit_CI(df, files_list=None, path=None, leg_dict=[], figsize=(16, 9), dpi=200,
                   plot=True, fontsize=16, x_ticks=["0", "0.2", "0.4", "0.6", "0.8", "1"], colors_dict=[], ci_range=10,
                   y_ticks_labels=None):
    """Plot calibration curve for est w/o and with calibration. """
    net_benefit_list = []
    Pt_start = 0
    Pt_stop = 1
    Pt_step = 0.001

    if type(files_list) != list:
        files_list = df.index.values
    #     print("files_list:",files_list)
    if type(colors_dict) == list:
        color_dict = np.load("/home/edlitzy/UKBB_Tree_Runs/For_article/color_dict.npy").tolist()
    #     print("colors_dict: ",colors_dict)

    if type(leg_dict) == list:
        leg_dict = dict(zip(files_list, files_list))
    #     print("leg_dict:",leg_dict)

    plt.rc('font', size=fontsize)  # controls default text sizes
    plt.rc('axes', titlesize=fontsize)  # fontsize of the axes title
    plt.rc('axes', labelsize=fontsize)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=fontsize)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=fontsize)  # fontsize of the tick labels
    plt.rc('legend', fontsize=fontsize)  # legend fontsize
    plt.rc('figure', titlesize=fontsize)  # fontsize of the figure title
    for ind in np.arange(1, ci_range):
        net_benefit_list.append(net_benefit_single(df, leg_dict=leg_dict, nbins=10, splits=3, file_ind=ind))
    net_benefit = pd.concat(net_benefit_list)
    #     print net_benefit.head()
    fig, ax1 = plt.subplots(1, 1, figsize=figsize)
    labal_list = []
    #     print("net_benefit:",net_benefit)
    net_benefit.index.rename("pt", inplace=True)
    net_benefit = net_benefit.reset_index()
    for label in net_benefit.columns.values:
        if label != "pt":
            #             print("label",label)
            #             print(net_benefit[label])
            label_in_df = get_key_name(label, leg_dict)
            label_in_df
            #             print("label_in_df:",label_in_df)
            #             print("net_benefit:",net_benefit)
            print("color_dict[label_in_df]", color_dict)
            ax1 = sns.lineplot(x="pt", y=label, data=net_benefit, ax=ax1, n_boot=1000, color=color_dict[label_in_df])
            short_label = label.replace("LR Singles ", "").replace(" with whr", "").replace("BDT Baseline ", "")
            labal_list.append(mpatches.Patch(color=color_dict[label_in_df], label=short_label))
    net_benefit.set_index("pt", inplace=True)
    limits_df = net_benefit.loc[:, :].drop("Test all", axis=1)
    print ("limits_df.max():", limits_df.max())

    xticks = [float(x) for x in x_ticks]
    ax1.set_xlim(0, max(xticks))
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(x_ticks)
    ax1.set_xlabel("probability threshold")

    ax1.set_ylim(max(-0.005, limits_df.min().min()), limits_df.max().max())
    ax1.set_ylabel("Benefit")
    ax1.legend(handles=labal_list)
    if y_ticks_labels != None:
        yticks = [float(x) for x in y_ticks_labels]
        ax1.set_yticks(yticks)
        ax1.set_yticklabels(y_ticks_labels, rotation=90)

    plt.tight_layout()
    if path != None:
        print("Saving to:", path)
        plt.savefig(path, dpi=dpi, frameon=False)
    if plot:
        plt.show()
    return net_benefit, net_benefit_list


def net_benefit_single(df, leg_dict, files_list=None, nbins=10, splits=10, file_ind=10):
    """Plot calibration curve for est w/o and with calibration. """
    cal_res_dict = {}
    Pt_start = 0
    Pt_stop = 0.9
    Pt_step = 0.002
    Net_Benefit_df = pd.DataFrame(index=np.arange(Pt_start, Pt_stop, Pt_step))

    if type(files_list) != list:
        files_list = df.index.values
    #     print("files_list:",files_list)

    for ind, label in enumerate(files_list):
        cal_res_dict[label] = pd.DataFrame()
        #         print ("ind:",ind)
        #         print("label:",label)
        files_hyper_path = df.loc[label, "y_files"]
        if os.path.split(files_hyper_path)[0].endswith("CI"):
            files_path = os.path.split(files_hyper_path)[0]
            file_type = "csv"
        else:
            files_path = os.path.join(os.path.split(files_hyper_path)[0], "CI")
            file_type = "pkl"
        tmp_files_list = [os.path.join(files_path, x) for x in os.listdir(files_path)]

        tmp_file_path = [os.path.join(files_path, x) for x in os.listdir(files_path)
                         if (x.endswith("_" + str(int(file_ind))) or
                             (x.endswith("y_pred_results_" + str(int(file_ind)) + ".csv")))][0]

        #         print tmp_file_path
        if file_type == "csv":
            tmp_file = pd.read_csv(tmp_file_path, index_col=0)
            #             print tmp_file
            columns = tmp_file.columns.values
            pred_col = [x for x in columns if "y_pred" in x][0]
            test_col = [x for x in columns if "y_test" in x][0]

            y_pred = tmp_file.loc[:, pred_col].values
            y_test = tmp_file.loc[:, test_col].values

        else:
            with open(tmp_file_path, 'rb') as fp_test:
                tmp_file = pickle.load(fp_test)
            y_test = tmp_file["y_val.values"].flatten()
            y_pred = tmp_file["y_proba"]
        #         print("******** Label:",label," **************")

        X = np.array(y_pred)
        y = np.array(y_test)

        ir = IsotonicRegression()
        skf = StratifiedKFold(n_splits=splits, random_state=0, shuffle=False)
        y_true_list = []
        y_cal_prob_list = []
        for train_index, test_index in skf.split(X, y):
            X_Cal_train, X_Cal_test = X[train_index], X[test_index]
            y_Cal_train, y_Cal_test = y[train_index], y[test_index]
            ir.fit(X_Cal_train, y_Cal_train)

            p_calibrated = ir.transform(X_Cal_test)
            y_cal_prob_list = np.append(y_cal_prob_list, p_calibrated)
            y_true_list = np.append(y_true_list, y_Cal_test)

        cal_res_dict[label].loc[:, "Orig_prob_nbins_" + str(int(ind))] = X
        cal_res_dict[label].loc[:, "Cal_prob_nbins_" + str(int(ind))] = y_cal_prob_list
        cal_res_dict[label].loc[:, "y_true_nbins_" + str(int(ind))] = y_true_list

        True_pos_list = []
        False_pos_list = []
        N = len(y_cal_prob_list)

        true_pos_all = np.sum(y_true_list)
        false_pos_all = len(y_true_list) - true_pos_all
        for pt in np.arange(Pt_start, Pt_stop, Pt_step):
            Y_df = cal_res_dict[label]
            Pos_cal = Y_df.loc[Y_df.loc[:, "Cal_prob_nbins_" + str(int(ind))] >= pt, "y_true_nbins_" + str(int(ind))]
            if Pos_cal.count() != 0:
                True_Pos_cal = Pos_cal.sum()
                False_Pos_cal = Pos_cal.shape[0] - True_Pos_cal
            else:
                True_Pos_cal = 0
                False_Pos_cal = 0
            Net_Benefit_df.loc[pt, leg_dict[label]] = (
                    (True_Pos_cal / N) - (False_Pos_cal / N) * (pt / (1 - pt)))

    for pt in np.arange(Pt_start, Pt_stop, Pt_step):
        Net_Benefit_df.loc[pt, "Test all"] = (
                (true_pos_all / N) - (false_pos_all / N) * (pt / (1 - pt)))
        Net_Benefit_df.loc[pt, "Do not test"] = 0
    return Net_Benefit_df


def plot_legend(figsize, colors_dict, ncolumns=1, font_size=16,
                fig_path="/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/figures/figure1/colors_legend.jpg",
                dpi=200, leg_text="Legend colors are:"):
    """
    This funcrtion gets nrows ncolumns,figsize and label_color_dictionar which is a dictionary of labels and colors
    Plots, returns saves a legend as a fig
    """
    # Create a color palette
    palette = colors_dict
    # Create legend handles manually
    handles = [mpl.patches.Patch(color=palette[x], label=x) for x in palette.keys()]
    # Create legend
    legend = plt.legend(handles=handles, ncol=ncolumns, fontsize=font_size, frameon=False)
    if leg_text != None:
        legend.set_title(leg_text, prop={'size': font_size})

    # Get current axes object and turn off axis
    fig = legend.figure
    plt.gca().set_axis_off()
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

    fig.savefig(fig_path, dpi=200, bbox_inches=bbox)
    print("Save legend figure to:", fig_path)
    #     plt.savefig(fig_path,dpi=dpi)
    plt.show()
    return legend


def get_latest_col_value(row, code):
    if code == "30750":
        return row[code + "-1.0"]
    elif ~np.isnan(row[code + "-2.0"]) and row[code + "-2.0"] >= 0:
        return row[code + "-2.0"]
    else:
        return row[code + "-1.0"]


def extract_latest_values(df, encodings_list):
    for code in encodings_list:
        df[code + "-3.0"] = df.apply(get_latest_col_value, code=code, axis=1)
    return df


def convert_hba1c_mmol_mol_2_percentage(row):
    for col in row.columns:
        try:
            row[col] = 0.0915 * row[col] + 2.15
        except:
            row[col] = None
    return row


def find_code(row, code):
    values = row.values
    check_list = [1 if x == code else 0 for x in values]
    return np.sum(np.array(check_list))


def return_clean_data(df_to_filter_path, All_disease_data_df, metformin=1140884600, Insulin=3):
    df_to_filter = pd.read_csv(df_to_filter_path, index_col="eid")
    data_index = df_to_filter.index
    df_disease_data = All_disease_data_df.loc[data_index, :]
    df_disease_data["Metformin"] = df_disease_data.apply(func=find_code, code=metformin, axis=1)
    df_disease_data["Insulin"] = df_disease_data.apply(func=find_code, code=Insulin, axis=1)
    df_disease_data["met_or_ill"] = df_disease_data["Insulin"] + df_disease_data["Metformin"]
    healthy_index = df_disease_data.loc[df_disease_data.loc[:, "met_or_ill"] == 0, :].index
    df_healthy = df_to_filter.loc[healthy_index, :]
    df_healthy.to_csv(df_to_filter_path, index=True)
    df_to_filter.to_csv(df_to_filter_path[:-4] + "w_diab_medicine.csv", index=True)
    return df_healthy, df_disease_data["met_or_ill"], df_to_filter


def conv_hba1c(row):
    HbA1c = 0.09148 * row["30750-0.0"] + 2.152
    return HbA1c


def conv_a1c(val):
    return 0.09148 * val + 2.152


def filter_hba1c(path):
    DF = pd.read_csv(path, index_col="eid")
    DF["HbA1c%"] = DF.apply(conv_hba1c, axis=1)
    print("Number of participants with HbA1c%>6.5%: ", DF.loc[DF["HbA1c%"] > 6.5, "HbA1c%"].count())
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    DF.loc[DF["HbA1c%"] > 6.5, "HbA1c%"].hist()
    ax.set_title("Patricipants with Hba1c>6.5%")
    DF_clean = DF.loc[DF["HbA1c%"] < 6.5, :]
    #     if save:
    #         DF.to_csv(os.path.join(bu_save_to_path),index=True)
    #         DF_clean.to_csv(path,index=True)
    return ax, DF, DF_clean


def plot_legend(colors_dict, ncolumns=1, font_size=16,
                fig_path="/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/figures/figure1/colors_legend.jpg",
                leg_text="Legend colors are:"):
    """
    This funcrtion gets nrows ncolumns,figsize and label_color_dictionar which is a dictionary of labels and colors
    Plots, returns saves a legend as a fig
    """
    # Create a color palette
    palette = colors_dict
    # Create legend handles manually
    handles = [mpl.patches.Patch(color=palette[x], label=x) for x in palette.keys()]
    # Create legend
    legend = plt.legend(handles=handles, ncol=ncolumns, fontsize=font_size, frameon=False)
    if leg_text != None:
        legend.set_title(leg_text, prop={'size': font_size})

    # Get current axes object and turn off axis
    fig = legend.figure
    plt.gca().set_axis_off()
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

    fig.savefig(fig_path, dpi=200, bbox_inches=bbox)
    print("Save legend figure to:", fig_path)
    #     plt.savefig(fig_path,dpi=dpi)
    plt.show()
    return legend


def find_code(row, code):
    values = row.values
    check_list = []
    check_list = [1 if x == code else 0 for x in values]
    return np.sum(np.array(check_list))


def get_latest_col_value(row, code):
    if code == "30750":
        return row[code + "-1.0"]
    elif ~np.isnan(row[code + "-2.0"]) and row[code + "-2.0"] >= 0:
        return row[code + "-2.0"]
    else:
        return row[code + "-1.0"]


# In[113]:


def extract_latest_values(df, encodings_list):
    for code in encodings_list:
        df[code + "-3.0"] = df.apply(get_latest_col_value, code=code, axis=1)
    return df


# In[318]:


def convert_hba1c_mmol_mol_2_percentage(row):
    for col in row.columns:
        try:
            row[col] = 0.0915 * row[col] + 2.15
        except:
            row[col] = None
    return row


def return_clean_data(df_to_filter_path, All_disease_data_df, metformin=1140884600, Insulin=3):
    df_to_filter = pd.read_csv(df_to_filter_path, index_col="eid")
    data_index = df_to_filter.index
    df_disease_data = All_disease_data_df.loc[data_index, :]
    df_disease_data["Metformin"] = df_disease_data.apply(func=find_code, code=metformin, axis=1)
    df_disease_data["Insulin"] = df_disease_data.apply(func=find_code, code=Insulin, axis=1)
    df_disease_data["met_or_ill"] = df_disease_data["Insulin"] + df_disease_data["Metformin"]
    healthy_index = df_disease_data.loc[df_disease_data.loc[:, "met_or_ill"] == 0, :].index
    df_healthy = df_to_filter.loc[healthy_index, :]
    df_healthy.to_csv(df_to_filter_path, index=True)
    df_to_filter.to_csv(df_to_filter_path[:-4] + "w_diab_medicine.csv", index=True)
    return df_healthy, df_disease_data["met_or_ill"], df_to_filter


def conv_hba1c(row):
    HbA1c = 0.09148 * row["30750-0.0"] + 2.152
    return HbA1c


def conv_a1c(val):
    return 0.09148 * val + 2.152


def filter_hba1c(path):
    DF = pd.read_csv(path, index_col="eid")
    DF["HbA1c%"] = DF.apply(conv_hba1c, axis=1)
    print("Number of participants with HbA1c%>6.5%: ", DF.loc[DF["HbA1c%"] > 6.5, "HbA1c%"].count())
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    DF.loc[DF["HbA1c%"] > 6.5, "HbA1c%"].hist()
    ax.set_title("Patricipants with Hba1c>6.5%")
    DF_clean = DF.loc[DF["HbA1c%"] < 6.5, :]
    #     if save:
    #         DF.to_csv(os.path.join(bu_save_to_path),index=True)
    #         DF_clean.to_csv(path,index=True)
    return ax, DF, DF_clean


def find_code(row, code):
    values = row.values
    check_list = []
    check_list = [1 if x == code else 0 for x in values]
    return np.sum(np.array(check_list))
