from Article_files.UKBB_article_plots_functions import *
# # Scoreboard plots

# ## Anthropometrics WOE LR Features importance
runs_dict={"Anthro_debug":"LR_Anthro_scoreboard_debug",
           "Anthro":"LR_Anthro_scoreboard",
           "Anthro_explore":"LR_Anthro_scoreboard_explore",
           "Five_blood_tests":"LR_Five_blood_tests_scoreboard",
           "Five_blood_tests_explore":"LR_Five_blood_tests_scoreboard_explore"}

scoreboard_type = "Anthro_debug"
run_name=runs_dict[scoreboard_type]
base_path= "/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened" #For seldom categories
scoreboard_path=os.path.join(base_path,"Scoreboards")
figs_path=os.path.join(base_path,"figures","figure1")
tables_path=os.path.join(base_path,"Tables")
CI_Results_path="/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/Imputed_screened_CI_Summary.csv"
conversion_files_obj=CI_Summary_table_object(
    root_path=base_path,
    save_to_file=True,
    update=True,
    save_prefix="Imputed_screened")
conversion_files_df=conversion_files_obj.CI_Summary_table
labels_dict=conversion_files_obj.UKBIOBANK_dict
figsize=(30,24)
font_size=24
num_feat = 100
# labels_dict = upload_ukbb_dict()
file_names_list=[run_name]
Folder_path = [os.path.join(scoreboard_path, run_name)]
table_save_path=os.path.join(tables_path,run_name+"_features_importance_summary.csv")
figpath=os.path.join(figs_path,run_name+"_features_importance.png")
hue_colors_dict={"LR_Antro_neto_whr":"lime"}
# order=["30750-0.0","30250-0.0","21003-4.0","30730-0.0","30870-0.0",
#                        "31-0.0_0.0","21003-3.0","30760-0.0","Bias"]

# coeffs_table=pd.read_csv("/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/Tables/LR_Anthro_scoreboard_features_importance.csv")
coeffs_table=pd.read_csv(table_save_path)

# dirs_names=['LR_Anthro_scoreboard', 'LR_Five_blood_tests_scoreboard']
# results_directory_name="Diabetes_Results"
dirs_names=file_names_list
df, precision_list, recall_list, fpr_list, tpr_list = calc_roc_aps_lists(
    directories_path=scoreboard_path,dirs_names=dirs_names,conversion_files_df=conversion_files_df)

ci_dict_path=conversion_files_obj.get_specific_ci_path_dict(df.index.values)
sort=True
# color_dict=dict(zip(df.sort_values(by="AUC",ascending=True).index.values,cmap))
txt_font_size=font_size-16
font_size=72
# order=["30750-0.0","30250-0.0","21003-4.0","30730-0.0","30870-0.0",
#                        "31-0.0_0.0","21003-3.0","30760-0.0","Bias"]
coeffs_table_obj = LR_Feature_importance(
    folder_path=Folder_path,
    positive_colors="green",
    negative_colors="lime",
    linespacing=1,
    plot=True,
    font_scale=5,
    leg_labels=["Anthropometrics"],
    leg_title=None,
    leg_pos=[0.9, 0.9],
    hue_type="binary",
    figsize=figsize,
    font_size=font_size,
    n_boots=10,
    space=0.2,
    fig_path=figpath,
    labels_dict=labels_dict,
    table_save_path=table_save_path,
    hue_colors_dict=hue_colors_dict,
    file_names_list=file_names_list,
    show_values_on_bar=True,
    remove_legend=True,
    ci_summary_table_name="Anthropometrics_features.csv"
)
print("Wait")