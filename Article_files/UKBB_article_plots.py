from UKBB_article_plots_config import *
from UKBB_article_plots_functions import *

conversion_files_obj = CI_Summary_table_object(
    root_path="/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/",
    save_to_file=True,
    update=True,
    save_prefix="Imputed_screened")
conversion_files_df = conversion_files_obj.get()
ci_path_dict = conversion_files_obj.get_ci_path_dict()
color_dict = conversion_files_obj.color_dict
variables_dict = conversion_files_obj.get_UKBB_dict()
figsize = (30, 24)
ticks_font_size = 72
font_size = 72
legend_font_size = 72
label_size = 72
legend_len = 20
sort = True
color = "Reds"

# ## Plotting definitions


all_vals = [x for x in conversion_files_obj.get().index.values if "All" in x]
CI_Results_path = "/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/Imputed_screened_CI_Summary.csv"
directories_path = "/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/New_Singles/"  # For seldom categories
all_directories = [x for x in os.listdir(directories_path) if
                   (x != "shap_folder" and x != "LR_comparison" and
                    os.path.isdir(os.path.join(directories_path, x)))]

dirs_names = ['Anthropometry_Diabetes', 'Five_Blood_Tests_Diabetes', 'All_Singles_Diabetes']

singles_df, precision_list, recall_list, fpr_list, tpr_list = calc_roc_aps_lists(
    directories_path=directories_path, dirs_names=dirs_names, conversion_files_df=conversion_files_obj.get(),
    CI_Results_path=CI_Results_path)

quantile_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/figures/fig3_quantile_gbdt.png"
test_labels = ['BDT Singles Anthropometry', 'BDT Singles Blood tests', "BDT Singles All"]
leg_labels = ['Anthropometrics', 'Blood Tests', "All features"]
cmap = sns.color_palette(color, singles_df.shape[0])
color_dict = dict(zip(singles_df.sort_values(by="AUC", ascending=True).index.values, cmap))
color_dict['BDT Singles All'] = "blue"
color_dict['BDT Singles Blood tests'] = 'red'
color_dict['BDT Singles Anthropometry'] = 'green'

ci_dict_path = {}
ci_dict_path['BDT Singles All'] = "/home/edlitzy/UKBB_Tree_Runs/For_article/Addings/All_Diabetes/Diabetes_Results/CI/"
ci_dict_path[
    'BDT Singles Blood tests'] = "/home/edlitzy/UKBB_Tree_Runs/For_article/Singles/Blood_Tests_Diabetes/Diabetes_Results/CI/"
ci_dict_path[
    'BDT Singles Anthropometry'] = "/home/edlitzy/UKBB_Tree_Runs/For_article/Singles/Anthropometry_Diabetes/Diabetes_Results/CI/"

figsize = (30, 24)
ticks_font_size = 72
font_size = 72
legend_font_size = 72
label_size = 72
sort = True
labels = singles_df.index.values

pr_ci_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/figures/figure3/Fig3_pr_ci_min_gbdt.png"
legend_dict = {"BDT Singles All": "All features", 'BDT Singles Blood tests': "Blood tests",
               'BDT Singles Anthropometry': "Anthropometry"}
ax = plot_aps_ci(df=singles_df, ci_dict_path=ci_dict_path, color_dict=color_dict,
                 lw=0.2, leg_pos=(0.5, 0.6), alpha=0.5,
                 num_of_files=None, xlabels=["0", "0.2", "0.4", "0.6", "0.8", "1"],
                 ylabels=["0", "0.12", "0.2", "0.4", "0.6", "0.65", "0.8", "1"],
                 xy_lines=[(0.2, 0.65), (0.20, 0.12)], lines_color=['green', 'red'], pr_ci_path=pr_ci_path,
                 short_labels=False, num_of_chars=20, figsize=figsize, fontsize=font_size, legend_dict=legend_dict)

# ## plot ROC


directories_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/Two_hot/"  # For seldom categories
all_directories = [x for x in os.listdir(directories_path) if
                   (x != "shap_folder" and x != "LR_comparison" and
                    os.path.isdir(os.path.join(directories_path, x)))]

dirs_names = ['imp_Age_and_Sex_Diabetes', 'imp_All_Diabetes',
              'imp_BP_and_HR_Diabetes', 'imp_Diet_Diabetes', 'imp_Blood_Tests_Diabetes',
              'imp_All_No_A1c_No_Gluc_Diabetes', 'imp_All_No_BT_Diabetes',
              'imp_Anthropometry_Diabetes', 'imp_AnS_Genetics_Diabetes',
              'imp_Lifestyle_and_physical_activity_Diabetes', 'imp_HbA1c_Diabetes']

singles_df, precision_list, recall_list, fpr_list, tpr_list = calc_roc_aps_lists(directories_path=directories_path,
                                                                                 dirs_names=dirs_names)
figsize = (30, 24)
ticks_font_size = 72
font_size = 72
legend_font_size = 58
label_size = 24
legend_len = 20
sort = True
color = 'plasma'
cmap = sns.color_palette(color, singles_df.shape[0])
color_dict = dict(zip(singles_df.sort_values(by="AUC", ascending=True).index.values, cmap))
sum_figsize = (30, 25)
roc_auc_sum_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/Figs_for_article_rev2/roc_auc_sum_all.png"

figsize = (30, 24)
font_size = 72
num_feat = 10
labels_dict = upload_ukbb_dict()
file_names_list = ['LR_Five_Blood_Tests']
Folder_path = [
    "/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/New_Singles_LR/LR_Five_Blood_Tests"]
table_save_path = "/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/Tables/LR_Five_Blood_Tests_features_importance.csv"
figpath = "/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/figures/figure3/LR_Five_Blood_Tests.png"
hue_colors_dict = {"LR_Five_Blood_Tests": "red"}
# order=["30750-0.0","30250-0.0","21003-4.0","30730-0.0","30870-0.0",
#                        "31-0.0_0.0","21003-3.0","30760-0.0","Bias"]
coeffs_table_obj = LR_Feature_importance(
    folder_path=Folder_path,
    linespacing=1,
    plot_type="bar_plot",
    plot=True,
    font_scale=7,
    leg_labels=["Five\nblood\ntests"],
    leg_title=None,
    leg_pos=[0.9, 0.9],
    hue_type="binary",
    figsize=figsize,
    font_size=font_size,
    n_boots=10,
    space=0.35,
    fig_path=figpath,
    labels_dict=labels_dict,
    table_save_path=table_save_path,
    hue_colors_dict=hue_colors_dict,
    file_names_list=file_names_list,
    show_values_on_bar=True, remove_legend=True, ci_summary_table_name="five_blood_fetures.csv"
)

coeffs_table_obj.coeffs_table.groupby("Covariates names").mean()

directories_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/New_Baseline_compare/"
CI_Results_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened_CI_Summary.csv"
dirs_names = ['LR_Antro_neto_whr', 'LR_Blood_Tests', 'LR_Five_Blood_Tests', "LR_Finrisc"]
results_directory_name = "Diabetes_Results"
df, precision_list, recall_list, fpr_list, tpr_list = calc_roc_aps_lists(
    directories_path=directories_path, dirs_names=dirs_names, conversion_files_df=conversion_files_df)
leg_dict = {"Anthropometry with WHR LR": "Anthropometry", "Five blood tests LR": "Five Blood tests",
            "Blood Tests LR": "Blood Tests", "LR_Finrisc": "FINDRISC"}
path = "/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/figures/figure3/calibration.png"
figsize = (30, 24)
font_size = 72
range_dict = {"Anthropometry with WHR LR": 1.1, "Five blood tests LR": 1.1}
bins_dict = {"Anthropometry with WHR LR": 6, "Five blood tests LR": 6}
splits_dict = {"Anthropometry with WHR LR": 2, "Five blood tests LR": 2}
res_dict = plot_calibration_curve(df=df, nbins=10, splits=2,
                                  colors_dict=color_dict, plot_hist=True,
                                  fontsize=font_size, fig_size=figsize, leg_dict=leg_dict, x_text=0, y_text=0.8,
                                  x_ticks=["0", "0.1", "0.2", "0.3", "0.4"], y_ticks=["0", "0.1", "0.2", "0.3", "0.4"],
                                  xlabel="predicted probability", ylabel="True probability", lw=5, path=path, dpi=50,
                                  plot=True,
                                  marker="o", markersize=24, range_dict=range_dict, bins_dict=bins_dict,
                                  splits_dict=splits_dict,
                                  y_hist_ticks=["1", "10", "100", "1000", "10000"], xlim=1, ylim=1, hist_alpha=0.2,
                                  nbins_hist=10,
                                  leg_loc=(0.18, 0.69), hist_bar_pos=0.5, hist_bar_width=0.6)

range_dict = {"Anthropometry with WHR LR": 1, "Five blood tests LR": 1}
bins_dict = {"Anthropometry with WHR LR": 5, "Five blood tests LR": 5}
splits_dict = {"Anthropometry with WHR LR": 5, "Five blood tests LR": 5}
hist_dict = plot_calibration_curve(df=df, nbins=10, splits=2,
                                   colors_dict=color_dict, plot_hist=True,
                                   fontsize=16, fig_size=figsize, leg_dict=leg_dict, x_text=0, y_text=0.8,
                                   x_ticks=["0", "0.1", "0.2", "0.3", "0.4"], y_ticks=["0", "0.1", "0.2", "0.3", "0.4"],
                                   xlabel="predicted probability", ylabel="True probability", lw=5, path=path, dpi=50,
                                   plot=True,
                                   marker="o", markersize=24, range_dict=range_dict, bins_dict=bins_dict,
                                   splits_dict=splits_dict,
                                   y_hist_ticks=["1", "10", "100", "1000", "10000"], xlim=1, ylim=1, hist_alpha=0.2,
                                   nbins_hist=10,
                                   leg_loc=(0.18, 0.69), hist_bar_pos=0.5, hist_bar_width=0.6)

fig3 = plt.figure(constrained_layout=True, figsize=(7, 7))
gs = gridspec.GridSpec(ncols=2, nrows=2, wspace=0.1, hspace=0)
aspect = 'auto'

f3_ax1 = fig3.add_subplot(gs[:, 0])
Table3_path = figure1_pdf_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/figures/figure3/singles table.PNG"

plot_ax(Table3_path, f3_ax1, "A", aspect='equal', pos=(-0.04, 1))
# ,pos=(-0.02,1))

f3_ax2 = fig3.add_subplot(gs[0, 1])
calibration_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/figures/figure3/isotonic_5_brier_calibration.png"
plot_ax(calibration_path, f3_ax2, "B", aspect=aspect, pos=(0.05, 0.96))

f3_ax3 = fig3.add_subplot(gs[1, 1])
Feat_imp_LR_fig3_path = "/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/figures/figure3/LR_Five_Blood_Tests.png"
plot_ax(Feat_imp_LR_fig3_path, f3_ax3, "C", aspect=aspect, pos=(0.05, 0.96))

figure3_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/figures/final_figures/figure3.jpg"
figure3_pdf_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/figures/final_figures/figure3.pdf"
plt.tight_layout()

# plt.savefig(figure1_path,dpi=600)
plt.savefig(figure3_path, bbox_inches='tight', dpi=1000)
plt.savefig(figure3_pdf_path, bbox_inches='tight', dpi=1000)

plt.show()

addings_directories_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/New_Addings/"  # For seldom categories
adding__directories = [x for x in os.listdir(addings_directories_path) if
                       (x != "shap_folder" and x != "LR_comparison" and
                        os.path.isdir(os.path.join(addings_directories_path, x)))]

df = conversion_files_obj.get()

adding_dirs_names = ['HbA1c_Diabetes',
                     'Five_Blood_Tests_Diabetes',
                     'Blood_Tests_Diabetes',
                     'A1_BT__Anthro_Diabetes',
                     'All_Diabetes', ]

adding_df, precision_list, recall_list, fpr_list, tpr_list = calc_roc_aps_lists(
    conversion_files_df=conversion_files_df,
    directories_path=addings_directories_path, dirs_names=adding_dirs_names)

test_labels = adding_df.index.values
xlabels = [str(x) for x in np.arange(1, 11)]
colors_df = pd.read_csv("/home/edlitzy/UKBB_Tree_Runs/For_article/colors.csv", index_col="File name")
colors_df
folder_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/New_Addings/"

folders_basename_list = ['HbA1c_Diabetes', 'Five_Blood_Tests_Diabetes',
                         'Blood_Tests_Diabetes', 'A1_BT__Anthro_Diabetes', "All_Diabetes"
                         ]
fig_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/figures/figure4/Quantlites_baseline_compare.jpg"
folders_list = [os.path.join(folder_path, x) for x in folders_basename_list]
hue_order = folders_basename_list
plot_quantiles_sns(folders_list, ci="sd", n_boots=1, hue_order=hue_order, font_scale=7,
                   kind="bar",
                   x_labels=xlabels,
                   leg_labels=["2)HbA1c%", "3)Five blood tests",
                               "4)Full Blood tests", "5)+Anthropometrics",
                               "16)+DNA sequencing"
                               ],
                   x_name="Quantile", y_name="Deciles fold ratio",
                   fig_path=fig_path,
                   dpi=200, fontsize=font_size)

n_rows = None
path = '/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/New_Singles/All_No_A1c_No_Gluc_Diabetes/'
Shap_plot_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/figures/figure4/Shap_all_no_A1c.png"
# y_labels_dict={9:"Gamma GT",5:"BMI",4:"Alanine AT"}
y_labels_dict = ()
variables_dict = conversion_files_obj.get_UKBB_dict()
simple_shap(
    plot=True,
    num_of_chars=18,
    model_folder_path=path,
    top_feat_num=10,
    n_rows=None,
    font_size=font_size,
    figsize=figsize,
    Shap_plot_path=Shap_plot_path,
    x_ticks_labels=[0, 0.1, 0.2, 0.3],
    xlabel_pos=0.45,
    ylabel_pos=0,
    y_labels_dict=y_labels_dict,
    linespace=0.9,
    text=None,
    x_text=0.15,
    y_text=2,
    leg_loc=(0.45, 0.01),
    leg_title_pos=(0, 0),
    leg_title="All Feature's\nimportance\nw/o A1c/Glucose",
    color_negative="dodgerblue",
    color_positive="darkblue",
    variables_dict=variables_dict, update=True)

# ## Figure 4 summary

figure4_pdf_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/figures/final_figures/fig_4_addings.pdf"
figure4_png_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/figures/final_figures/fig_4_addings.png"
fig2 = plt.figure(constrained_layout=True, figsize=(7, 7))
gs = gridspec.GridSpec(ncols=2, nrows=2, wspace=0.01, hspace=0.02)
aspect = 'auto'
Table2_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/figures/figure4/Table4 addings.PNG"
f2_ax1 = fig2.add_subplot(gs[0:, 0])
plot_ax(Table2_path, f2_ax1, "A", aspect='equal', pos=(-0.04, 1.02))

f2_ax2 = fig2.add_subplot(gs[0, 1])
shap_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/figures/figure4/Quantlites_baseline_compare.jpg"
plot_ax(shap_path, f2_ax2, "B", aspect=aspect, pos=(0.06, 1))

f2_ax3 = fig2.add_subplot(gs[1, 1])
quantile_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/figures/figure4/Shap_all_no_A1c.png"
plot_ax(quantile_path, f2_ax3, "C", aspect=aspect, pos=(0.06, 1))

plt.tight_layout()
plt.savefig(figure4_pdf_path, bbox_inches='tight', dpi=800)
plt.savefig(figure4_png_path, bbox_inches='tight', dpi=800)
plt.show()

# # Figure 1

directories_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/New_Baseline_compare/"  # For seldom categories
CI_Results_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/Imputed_screened_CI_Summary.csv"
conversion_files_df = pd.read_csv(CI_Results_path, index_col="File name")
dirs_names = ['LR_Five_Blood_Tests', 'All_Diabetes', 'SA_GDRS', 'LR_Finrisc',
              'LR_Antro_neto_whr']
results_directory_name = "Diabetes_Results"
df, precision_list, recall_list, fpr_list, tpr_list = calc_roc_aps_lists(
    directories_path=directories_path, dirs_names=dirs_names, conversion_files_df=conversion_files_df)
ci_dict_path = conversion_files_obj.get_specific_ci_path_dict(df.index.values)
sort = True
txt_font_size = font_size - 16

# ## APS

# ### plot PR CI plot

df = df.sort_values(by="APS")
pr_ci_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/figures/figure1/fig_1_aps_ci.png"
color_dict = conversion_files_obj.load_color_dict()[0]
labels = df.index.values
ax = plot_aps_ci(df=df, ci_dict_path=ci_dict_path, color_dict=color_dict, lw=0.15, leg_pos=(0.45, 0.42),
                 alpha=0.25, num_of_files=None, xlabels=["0", "0.2", "0.4", "0.6", "0.8", "1"],
                 ylabels=["0", "0.25", "0.5", "0.75", "1"],
                 xy_lines=None, lines_color=['green', 'yellow'],
                 pr_ci_path=pr_ci_path, figsize=figsize, fontsize=font_size,
                 short_labels=True, x_txt=0.23,
                 y_txt=0.95)

roc_ci_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/figures/figure1/fig_1_roc_ci.png"
plot_roc_ci(df=df, ci_dict_path=ci_dict_path, color_dict=color_dict, lw=0.2, leg_pos=(0.44, 0.01), alpha=0.3,
            num_of_files=None, xlabels=["0", "0.25", "0.5", "0.75", "1"], roc_ci_path=roc_ci_path,
            num_of_chars=20, x_txt=0.6, y_txt=0.55, figsize=figsize,
            fontsize=font_size, facecolor="white")

xlabels = [str(x) for x in np.arange(1, 11)]
colors_df = pd.read_csv("/home/edlitzy/UKBB_Tree_Runs/For_article/colors.csv", index_col="File name")
colors_df
folder_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/New_Baseline_compare"
fig_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/figures/figure1/Quantlites_baseline_compare.jpg"
print(os.listdir(folder_path))

xlabels = [str(x) for x in np.arange(1, 11)]
colors_df = pd.read_csv("/home/edlitzy/UKBB_Tree_Runs/For_article/colors.csv", index_col="File name")

folder_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/New_Baseline_compare"
fig_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/figures/figure1/Quantlites_baseline_compare.jpg"
print(os.listdir(folder_path))
# all_dirs=['A1c_strat', 'New_Addings', 'New_Singles', 'New_Singles_LR']
folders_basename_list = ['SA_GDRS', 'LR_Finrisc', 'LR_Antro_neto_whr', 'LR_Five_Blood_Tests',
                         "All_Diabetes"]
folders_list = [os.path.join(folder_path, x) for x in folders_basename_list]
hue_order = folders_basename_list
plot_quantiles_sns(folders_list, ci="sd",
                   n_boots=1,
                   hue_order=hue_order,
                   font_scale=7,
                   kind="bar",
                   x_labels=xlabels,
                   leg_labels=["x7(2.8-18.5)", "x26(10-41)", "x32(15-42)", "x59(27-75)", "x65(49-73)"],
                   x_name="Quantile",
                   y_name="Deciles fold ratio",
                   fig_path=fig_path,
                   dpi=200,
                   fontsize=font_size,
                   fig_size=figsize, y_scale="log",
                   leg_title="Deciles' OR", leg_bbox_to_anchor=(0.48, 0.75), leg_loc='center right',
                   framealpha=0.7,
                   frameon=True, )

# ## Feature importance

# ### Anthropometry


figsize = (30, 24)
font_size = 72
num_feat = 10
labels_dict = upload_ukbb_dict()
file_names_list = ['LR_Antro_neto_whr']
Folder_path = [
    "/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/New_Singles_LR/LR_Antro_neto_whr"]
table_save_path = "/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/Tables/LR_Antro_neto_whr_features_importance.csv"
figpath = "/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/figures/figure1/LR_Anthro_neto_whr.png"
hue_colors_dict = {"LR_Antro_neto_whr": "green"}
# order=["30750-0.0","30250-0.0","21003-4.0","30730-0.0","30870-0.0",
#                        "31-0.0_0.0","21003-3.0","30760-0.0","Bias"]
# plot_type = "bar_plot",
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

# ### Blood tests stratification

# In[2344]:


# color_dict=dict(zip(df.sort_values(by="AUC",ascending=True).index.values,cmap))
Folder_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/Baseline_compare/LR_Antro_neto_whr/"
model_path = os.path.join(Folder_path, "LR_Model.sav")
use_cols_path = os.path.join(Folder_path, "use_cols.txt")
with open(use_cols_path, "rb") as fp:  # Pickling
    use_cols = pickle.load(fp)
use_cols = [x for x in use_cols if x != "eid"]
lr_model = pickle.load(open(model_path, 'rb'))

# color='plasma'
# cmap = sns.color_palette(color, df.shape[0])
labels = df.index.values
# color_dict=dict(zip(df.sort_values(by="AUC",ascending=True).index.values,cmap))


n_rows = None
path = "/home/edlitzy/UKBB_Tree_Runs/For_article/compare_GDRS/Antro_neto_whr_Diabetes/"
Shap_plot_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/Figs_for_article_rev2/Fig1_Shap_Anthro_whr.png"
# y_labels_dict={
#     9:"Body mass index (BMI)",
#     8:"Waist circumfeZrence",
#     7:"Very happy with own health",
#     6:"Age at last visit",
#     5:"Weight",
#     4:"Years between visits",
#     3:"Systolic blood pressure",
#     2:"Ilnesses of siblings not listed",
#     1:"Diastolic blood pressure",
#     0:"Extremely happy with own health"}
n_rows = None
path = '/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/New_Singles/All_No_A1c_No_Gluc_Diabetes/'
Shap_plot_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/figures/figure4/Shap_all_no_A1c.png"
y_labels_dict = {9: "Gamma GT", 5: "BMI", 4: "Alanine AT"}
variables_dict = conversion_files_obj.get_UKBB_dict()
simple_shap(
    plot=True,
    num_of_chars=18,
    model_folder_path=path,
    top_feat_num=10,
    n_rows=None,
    font_size=font_size,
    figsize=figsize,
    Shap_plot_path=Shap_plot_path,
    x_ticks_labels=[0, 0.1, 0.2, 0.3],
    xlabel_pos=0.45,
    ylabel_pos=0,
    y_labels_dict=y_labels_dict,
    linespace=0.9,
    text=None,
    x_text=0.15,
    y_text=2,
    leg_loc=(0.5, 0.01),
    leg_title_pos=(0, 0),
    leg_title="All Feature's\nimportance\nw/o A1c/Glucose",
    color_negative="royalblue",
    color_positive="darkblue",
    variables_dict=variables_dict)

# ## Plot 1 summary


fig1 = plt.figure(constrained_layout=True, figsize=(7, 10))
gs = gridspec.GridSpec(ncols=4, nrows=31, wspace=0.1, hspace=0.2)
aspect = 'auto'

f1_ax1 = fig1.add_subplot(gs[0:10, 0:2])
roc_path = "/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/figures/figure1/fig_1_roc_ci.png"
plot_ax(roc_path, f1_ax1, "A", aspect=aspect, pos=(-0.02, 1))

APS_path = "/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/figures/figure1/fig_1_aps_ci.png"
f1_ax2 = fig1.add_subplot(gs[0:10, 2:])
plot_ax(APS_path, f1_ax2, "B", aspect=aspect, pos=(0.02, 1))

f1_ax3 = fig1.add_subplot(gs[10:20, :2])
quantile_path = "/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/figures/figure1/Quantlites_baseline_compare.jpg"
plot_ax(quantile_path, f1_ax3, "C", aspect=aspect, pos=(-0.02, 1))

f1_ax4 = fig1.add_subplot(gs[10:20, 2:])
Anthro_Feature_importance_path = "/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/figures/figure1/LR_Anthro_neto_whr.png"
plot_ax(Anthro_Feature_importance_path, f1_ax4, "D", aspect=aspect, pos=(0.02, 1))

f1_ax5 = fig1.add_subplot(gs[20:30, :2])
BT_Feature_importance_path = "/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/figures/figure3/LR_Five_Blood_Tests.png"
plot_ax(BT_Feature_importance_path, f1_ax5, "E", aspect=aspect, pos=(0.02, 1))

f1_ax6 = fig1.add_subplot(gs[20:30, 2:])
calibration_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/figures/figure3/isotonic_5_brier_calibration.png"
plot_ax(calibration_path, f1_ax6, "F", aspect=aspect, pos=(0.02, 1))

f1_ax7 = fig1.add_subplot(gs[30:, :])
legened_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/figures/figure1/colors_legend.jpg"
plot_ax(legened_path, f1_ax7, None, aspect=aspect, pos=(0.02, 1))

figure1_pdf_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/figures/final_figures/fig_1.1.pdf"
figure1_png_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/figures/final_figures/fig_1.1.png"

plt.tight_layout()

# plt.savefig(figure1_path,dpi=600)
plt.savefig(figure1_pdf_path, bbox_inches='tight', dpi=500, frameon=False)
plt.savefig(figure1_png_path, bbox_inches='tight', dpi=500, frameon=False)
plt.show()

# # Updated figure 2

from collections import OrderedDict

labels_colors_dict = OrderedDict([("GDRS", "gray"), ("FINDRISC", "gold"), ("Anthropometry", "green"),
                                  ("Five blood tests", "red"), ("All features", "blue")])
fig_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/figures/figure1/colors_legend.jpg"
plot_legend(figsize=(7, 1), fig_path=fig_path, colors_dict=labels_colors_dict, ncolumns=5, font_size=12,
            leg_text=None)
fig1 = plt.figure(constrained_layout=True, figsize=(7, 10))
gs = gridspec.GridSpec(ncols=4, nrows=32, wspace=0.1, hspace=0.2)
aspect = 'auto'

f1_ax1 = fig1.add_subplot(gs[0:10, 0:2])
roc_path = "/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/figures/figure1/fig_1_roc_ci.png"
plot_ax(roc_path, f1_ax1, "A", aspect=aspect, pos=(-0.02, 1))

APS_path = "/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/figures/figure1/fig_1_aps_ci.png"
f1_ax2 = fig1.add_subplot(gs[0:10, 2:])
plot_ax(APS_path, f1_ax2, "B", aspect=aspect, pos=(0.02, 1))

f1_ax3 = fig1.add_subplot(gs[10:20, :2])
quantile_path = "/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/figures/figure1/Quantlites_baseline_compare.jpg"
plot_ax(quantile_path, f1_ax3, "C", aspect=aspect, pos=(-0.02, 1))

f1_ax4 = fig1.add_subplot(gs[10:20, 2:])
Anthro_Feature_importance_path = "/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/figures/figure1/LR_Anthro_neto_whr.png"
plot_ax(Anthro_Feature_importance_path, f1_ax4, "D", aspect=aspect, pos=(0.02, 1))

f1_ax5 = fig1.add_subplot(gs[20:30, :2])
BT_Feature_importance_path = "/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/figures/figure3/LR_Five_Blood_Tests.png"
plot_ax(BT_Feature_importance_path, f1_ax5, "E", aspect=aspect, pos=(-0.02, 1))

f1_ax6 = fig1.add_subplot(gs[20:30, 2:])
calibration_path = "/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/figures/figure3/isotonic_5_brier_calibration.png"
plot_ax(calibration_path, f1_ax6, "F", aspect=aspect, pos=(0.02, 1))

f1_ax7 = fig1.add_subplot(gs[30:, :])
legened_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/figures/figure1/colors_legend.jpg"
plot_ax(legened_path, f1_ax7, None, aspect=aspect, pos=(0.02, 1))

figure1_pdf_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/figures/final_figures/fig_1.1.pdf"
figure1_png_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/figures/final_figures/fig_1.1.png"

plt.tight_layout()

# plt.savefig(figure1_path,dpi=600)
plt.savefig(figure1_pdf_path, bbox_inches='tight', dpi=500, frameon=False)
plt.savefig(figure1_png_path, bbox_inches='tight', dpi=500, frameon=False)
plt.show()

conversion_files_df

# In[1414]:


os.listdir("/home/edlitzy/UKBB_Tree_Runs/For_article/A1c_strat/")
directories_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/A1c_strat/"  # For seldom categories
CI_Results_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/Imputed_screened_CI_Summary.csv"
# directories_path="/home/edlitzy/UKBB_Tree_Runs/For_article/Adding/" #For adding categoriesdirectories_path="/home/edlitzy/UKBB_Tree_Runs/For_article/Two_hot_Add/" #For adding categories
results_directory_name = "Diabetes_Results"
df, precision_list, recall_list, fpr_list, tpr_list = calc_roc_aps_lists(
    directories_path=directories_path, conversion_files_df=conversion_files_df)

color_h = "magenta"
color_l = "cyan"
text = "GDRS- German diabetes risk score\nFINDRISC-Finnish diabetes risk score"

replace_dict = {" 5.7<=HbA1c%<6.5": "",
                " 4<=HbA1c%<5.7": "",
                "Anthropometry": "Anthro- pometrics"}

comp_roc_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/figures/figure2/fig_2_roc_groups.jpg"

comp_roc_groups(
    df,
    comp_roc_path=comp_roc_path,
    plot=True,
    dpi=100,
    leg_len=20,
    labels_length=10,
    leg_pos=(0.02, 0.71),
    ylim=(0.5, 0.9),
    barWidth=0.4,
    x_text=-0.5,
    y_text=0.86,
    text=text,
    color_h=color_h,
    color_l=color_l,
    fig_size=figsize,
    replace_dict=replace_dict,
    ylabels=["0.5", "0.6", "0.7", "0.8", "0.9"],
    high_recogniser='5.7<=HbA1c%<6.5',
    low_recogniser='4<=HbA1c%<5.7',

)

# ## fig 2 comp_aps_groups

# In[1449]:


comp_aps_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/figures/figure2/Fig_2_comp_aps.png"
# text="GDRS- German diabetes risk score\nBDT-Boosting decision trees"
comp_aps_groups(df, comp_aps_path=comp_aps_path, plot=True, dpi=100, leg_len=20, labels_length=10,
                leg_pos1=(0, 0.88), leg_pos2=(0, 0.81), barWidth=0.4, x_text=-0.55, y_text=0.45,
                color_h="magenta", color_l="cyan",
                fig_size=figsize, replace_dict=replace_dict, linespace=1,
                yticks_high=["0", "0.1", "0.2", "0.3", "0.4", "0.5"],
                yticks_low=["0", "0.02", "0.04", "0.06", "0.08"],
                high_recogniser='5.7<=HbA1c%<6.5',
                low_recogniser='4<=HbA1c%<5.7', )

# ## Features importance figure 2

# ### Anthropometry

# In[591]:


folder_names = ["LR_Strat_L20_H39_Antro_neto_whr", "LR_Strat_L39_Antro_neto_whr"]
hue_colors_dict = dict(zip(folder_names, ["cyan", "magenta"]))
hue_colors_dict
num_feat = 5
labels_dict = UKBB_labels_dict
base_path = "/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/A1c_strat/"
folder_names = ["LR_Strat_L20_H39_Antro_neto_whr", "LR_Strat_L39_Antro_neto_whr"]
folder_path = [os.path.join(base_path, x) for x in folder_names]
# order=["30750-0.0","30250-0.0","21003-4.0","30730-0.0","30870-0.0",
#                        "31-0.0_0.0","21003-3.0","30760-0.0","Bias"]
coeffs_table_obj = LR_Feature_importance(
    folder_path=folder_path,
    color_pallete="Greens",
    hue_type="binary",
    labels_dict=UKBB_labels_dict,
    num_of_chars=14,
    table_save_path="/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/Tables/LR_Strat_Anthro_neto_whr_feature_importance.csv",
    fig_path="/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/figures/figure1/LR_Strat_Anthro_neto_whr.png",
    figsize=figsize,
    font_size=font_size,
    n_boots=10,
    space=0.15,
    hue_colors_dict=dict(zip(folder_names, ["cyan", "magenta"])),
    linespacing=1,
    plot_type="cat_plot",
    file_names_list=folder_names,
    font_scale=6.5,
    leg_labels=["4%-5.6%", "5.7%-6.5%"],
    leg_title="Anthropometry\nHbA1c stratified",
    leg_pos=(0.75, 0.1))

# ### Blood tests

# In[2349]:


base_path = "/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/A1c_strat/"
os.listdir(base_path)

hue_colors_dict = dict(zip(folder_names, ["cyan", "magenta"]))
hue_colors_dict
num_feat = 5
labels_dict = UKBB_labels_dict
base_path = "/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/A1c_strat/"
folder_names = ["LR_Strat_L20_H39_Five_Blood_Tests", "LR_Strat_L39_Five_Blood_Tests"]
folder_path = [os.path.join(base_path, x) for x in folder_names]
# order=["30750-0.0","30250-0.0","21003-4.0","30730-0.0","30870-0.0",
#                        "31-0.0_0.0","21003-3.0","30760-0.0","Bias"]
coeffs_table_obj = LR_Feature_importance(
    folder_path=folder_path,
    color_pallete="Greens",
    hue_type="binary",
    labels_dict=UKBB_labels_dict,
    num_of_chars=14,
    table_save_path="/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/Tables/LR_Strat_5BT_feature_importance.csv",
    fig_path="/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/figures/figure1/LR_Strat_5BT.png",
    figsize=figsize,
    font_size=font_size,
    n_boots=10,
    space=0.15,
    hue_colors_dict=dict(zip(folder_names, ["cyan", "magenta"])),
    linespacing=1,
    plot_type="cat_plot",
    file_names_list=folder_names,
    font_scale=6.5,
    leg_labels=["4%-5.6%", "5.7%-6.5%"],
    leg_title="Five Blood tests\nHbA1c stratified",
    leg_pos=(0.75, 0.1))

# ## Figure 2 summary


fig2 = plt.figure(constrained_layout=True, figsize=(7, 10))
gs = gridspec.GridSpec(ncols=2, nrows=3, wspace=0.1, hspace=0.1)
aspect = 'auto'

f2_ax1 = fig2.add_subplot(gs[0, 0])
comp_roc_path = comp_roc_path
plot_ax(comp_roc_path, f2_ax1, "A", aspect=aspect, pos=(-0.02, 1))

comp_aps_path = comp_aps_path
f2_ax2 = fig2.add_subplot(gs[0, 1])
plot_ax(comp_aps_path, f2_ax2, "B", aspect=aspect, pos=(0.02, 1))

f2_ax3 = fig2.add_subplot(gs[2, 0])
Feature_importance_Anthro = "/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/figures/figure1/LR_Strat_Anthro_neto_whr.png"
# Shap_plot_high_path
plot_ax(Feature_importance_Anthro, f2_ax3, "D", aspect=aspect, pos=(-0.02, 1))

f2_ax4 = fig2.add_subplot(gs[2, 1])
Feature_importance_5BT = "/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/figures/figure1/LR_Strat_5BT.png"
# Shap_plot_low_path
plot_ax(Feature_importance_5BT, f2_ax4, "E", aspect=aspect, pos=(0.02, 1))

f2_ax5 = fig2.add_subplot(gs[1, :])
a1c_strat_summary_table = "/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/figures/figure2/strat table.PNG"
# Shap_plot_low_path
plot_ax(a1c_strat_summary_table, f2_ax5, "C", aspect='equal', pos=(-0.11, 1))

figure2_pdf_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/figures/final_figures/fig_2_strat.pdf"
figure2_png_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/figures/final_figures/fig_2_strat.png"

plt.tight_layout()

# plt.savefig(figure1_path,dpi=600)
plt.savefig(figure2_pdf_path, bbox_inches='tight', dpi=1000, frameon=False)
plt.savefig(figure2_png_path, bbox_inches='tight', dpi=1000, frameon=False)
plt.show()

# # Calibration plots

# ## Net benefit

# In[73]:


directories_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/Baseline_compare/"  # For seldom categories
CI_Results_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/Articles_CI_Summary.csv"
conversion_files_df = pd.read_csv(CI_Results_path, index_col="File name")

directories_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/Baseline_compare/"  # For seldom categories
CI_Results_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/Articles_CI_Summary.csv"
conversion_files_df = pd.read_csv(CI_Results_path, index_col="File name")
# dirs_names=['LR_Antro_neto_whr', 'ALL_BDT_Diabetes','LR_Blood_Tests']
# directories_path="/home/edlitzy/UKBB_Tree_Runs/For_article/Adding/" #For adding categoriesdirectories_path="/home/edlitzy/UKBB_Tree_Runs/For_article/Two_hot_Add/" #For adding categories
results_directory_name = "Diabetes_Results"
df, precision_list, recall_list, fpr_list, tpr_list = calc_roc_aps_lists(
    directories_path=directories_path)
# df= calc_roc_aps_lists(directories_path=directories_path)
# print(df)


# In[447]:


leg_dict = {"GDRS": "GDRS", "Finrisc": "FINDRISC", "Anthropometry with whr": "Anthropometry",
            "LR Singles Blood Tests": "Blood Tests", "BDT Baseline All": "All features", "Test all": "Test all",
            "Do not test": "Do not test"}

color_dict = np.load("/home/edlitzy/UKBB_Tree_Runs/For_article/color_dict.npy").tolist()

leg_dict = {"GDRS": "GDRS", "Finrisc": "FINDRISC", "Anthropometry with whr": "Anthropometry",
            "LR Singles Blood Tests": "Blood tests", "BDT Baseline All": "All features"}

net_benefit, net_benefit_list = net_benefit_CI(df, ci_range=100,
                                               path="/home/edlitzy/UKBB_Tree_Runs/For_article/figures/Calibrations/Net_benefit.jpg",
                                               fontsize=font_size, figsize=figsize, lw=5,
                                               y_ticks_labels=["0", "0.01", "0.02"],
                                               x_ticks=['0', '0.2', '0.4', '0.6', '0.8'], splits=30, leg_dict=leg_dict)

fig_cal = plt.figure(constrained_layout=True, figsize=(7, 3))
gs = gridspec.GridSpec(ncols=2, nrows=1, wspace=0.1, hspace=0)
aspect = 'auto'

fcal_ax1 = fig_cal.add_subplot(gs[0, 0])
cal_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/figures/Calibrations/Calibration.jpg"
plot_ax(cal_path, fcal_ax1, "A", aspect=aspect, pos=(0.02, 1))

fcal_ax2 = fig_cal.add_subplot(gs[0, 1])
net_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/figures/Calibrations/Net_benefit.jpg"
plot_ax(net_path, fcal_ax2, "B", aspect=aspect, pos=(-0.02, 1))

figure1_pdf_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/figures/final_figures/Cal.pdf"
figure1_png_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/figures/final_figures/cal.png"

plt.tight_layout()

plt.savefig(figure1_pdf_path, bbox_inches='tight', dpi=1000, frameon=False)
plt.savefig(figure1_png_path, bbox_inches='tight', dpi=1000, frameon=False)
plt.show()

update_labels_dict = {"BDT Singles ": "", "BDT Addings ": "", "LR Singles ": ""}

directories_path = "/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/"

conversion_files_obj = CI_Summary_table_object(
    root_path="/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/",
    save_to_file=True,
    save_prefix="Imputed_screened")

directory = "/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/"
CI_table_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/Imputed_screened_CI_Summary.csv"
Quant_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/Tables"

# In[68]:


tables_list = []
dirs = ["Scoreboards"]
for curr_dir in dirs:
    direct = os.path.join(directory, curr_dir)
    tmp_table = create_folder_table(directories_path=direct, Labels=None,
                                    CI_Results_path="/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/",
                                    Save_to_Table_path=Quant_path,
                                    calc_quantiles=True, update_labels_dict=[])
    tables_list.append(tmp_table.get())

summary_table = pd.concat(tables_list)
summary_table.sort_values(by="AUROC_mean", inplace=True, ascending=False)
summary_table.to_csv(os.path.join(Quant_path, "Quant_CI_Summary_table.csv"), index=True)

Val_summary_table = CI_Summary_table_object(
    root_path="/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/explore_val/",
    save_to_file=False,
    save_prefix="Explore_val_")
Val_summary_df = Val_summary_table.get()
Val_summary_df.sort_values(by="AUROC_median", inplace=True, ascending=False)
Val_summary_df.to_csv("/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/explore_val/Explore_Summary_table.csv",
                      index=True)

CI_Results_path = "/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/Imputed_screened_CI_Summary.csv"
CI_summary_df = pd.read_csv(CI_Results_path, index_col="File name")
df0 = create_folder_table(directories_path="/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/New_Singles_LR/",
                          conversion_files_df=CI_summary_df,
                          calc_quantiles=True, update_labels_dict={" LR": ""})
df0.to_csv("/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/Tables/LR_Single.csv")
CI_Results_path = '/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/Imputed_screened_CI_Summary.csv'

df1 = create_folder_table(directories_path="/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/New_Singles/",
                          calc_quantiles=True,
                          conversion_files_df=CI_summary_df,
                          update_labels_dict={" DT": ""})
df1.to_csv("/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/Tables/Singles.csv")
df2 = create_folder_table(directories_path="/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/New_Addings/",
                          calc_quantiles=True,
                          conversion_files_df=CI_summary_df)
df2.to_csv("/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/Tables/Adding.csv")
df3 = create_folder_table(
    directories_path="/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/New_Baseline_compare/",
    calc_quantiles=True,
    conversion_files_df=CI_summary_df)
df3.to_csv("/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/Tables/Baseline_compare.csv")

df4 = create_folder_table(directories_path="/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/A1c_strat/",
                          calc_quantiles=True,
                          conversion_files_df=CI_summary_df)
df4.to_csv("/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/Tables/A1c_strat.csv")

Shap_plot_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/Figs_for_article_rev2/Fig3_Shap_Min_gbdt.png"
simple_shap(model_folder_path=path, top_feat_num=10, n_rows=1000,
            font_size=font_size, figsize=figsize, Shap_plot_path=Shap_plot_path,
            x_ticks_labels=[0, 0.1, 0.2, 0.3],
            y_labels_dict={9: "Gamma GT", 5: "BMI", 4: "Alanine AT"}, xlabel_pos=0.45,
            ylabel_pos=0)
