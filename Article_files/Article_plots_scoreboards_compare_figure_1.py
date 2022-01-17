from UKBB_article_plots_config import *
from UKBB_article_plots_functions import *
#%%
conversion_files_obj = CI_Summary_table_object(
    root_path="/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/Scoreboards/Scoreboards_Baseline_compare/",
    save_to_file=True,
    update=True,
    save_prefix="Imputed_screened")
#%%
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

labels_dict = upload_ukbb_dict()

# # Figure 1

directories_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/Scoreboards_Baseline_compare/Scoreboards_Baseline_compare/"  # For seldom categories
CI_Results_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/Scoreboards_Baseline_compare/Imputed_screened_CI_Summary.csv"
conversion_files_df = pd.read_csv(CI_Results_path, index_col="File name")
dirs_names = ['SA_GDRS', 'LR_Finrisc','LR_Anthro_scoreboard_explore/']
results_directory_name = "Diabetes_Results"
df, precision_list, recall_list, fpr_list, tpr_list = calc_roc_aps_lists(
    directories_path=directories_path, dirs_names=dirs_names, conversion_files_df=conversion_files_df)
ci_dict_path = conversion_files_obj.get_specific_ci_path_dict(df.index.values)
sort = True
txt_font_size = font_size - 16

# ## APS

# ### plot PR CI plot

df = df.sort_values(by="APS")
pr_ci_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/figures/scoreboards_compare/scoreboards_compare_1_aps_ci.png"
color_dict = conversion_files_obj.load_color_dict()[0]
labels = df.index.values
ax = plot_aps_ci(df=df, ci_dict_path=ci_dict_path, color_dict=color_dict, lw=0.15, leg_pos=(0.45, 0.42),
                 alpha=0.25, num_of_files=None, xlabels=["0", "0.2", "0.4", "0.6", "0.8", "1"],
                 ylabels=["0", "0.25", "0.5", "0.75", "1"],
                 xy_lines=None, lines_color=['green', 'yellow'],
                 pr_ci_path=pr_ci_path, figsize=figsize, fontsize=font_size,
                 short_labels=True, x_txt=0.23,
                 y_txt=0.95)

roc_ci_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/figures/scoreboards_compare/scoreboards_compare_roc_ci.png"
plot_roc_ci(df=df, ci_dict_path=ci_dict_path, color_dict=color_dict, lw=0.2, leg_pos=(0.44, 0.01), alpha=0.3,
            num_of_files=None, xlabels=["0", "0.25", "0.5", "0.75", "1"], roc_ci_path=roc_ci_path,
            num_of_chars=20, x_txt=0.6, y_txt=0.55, figsize=figsize,
            fontsize=font_size, facecolor="white")

xlabels = [str(x) for x in np.arange(1, 11)]
colors_df = pd.read_csv("/home/edlitzy/UKBB_Tree_Runs/For_article/colors.csv", index_col="File name")
colors_df
folder_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/New_Baseline_compare"
fig_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/figures/scoreboards_compare/Quantlites_baseline_compare.jpg"
print((os.listdir(folder_path)))

xlabels = [str(x) for x in np.arange(1, 11)]
colors_df = pd.read_csv("/home/edlitzy/UKBB_Tree_Runs/For_article/colors.csv", index_col="File name")

folder_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/New_Baseline_compare"
fig_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/figures/scoreboards_compare/Quantlites_baseline_compare.jpg"
print((os.listdir(folder_path)))
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
figpath = "/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/figures/scoreboards_compare/LR_Anthro_neto_whr.png"
hue_colors_dict = {"LR_Antro_neto_whr": "green"}
# order=["30750-0.0","30250-0.0","21003-4.0","30730-0.0","30870-0.0",
#                        "31-0.0_0.0","21003-3.0","30760-0.0","Bias"]
coeffs_table_obj = LR_Feature_importance(
    folder_path=Folder_path,
    positive_colors="green",
    negative_colors="lime",
    linespacing=1,
    plot_type="bar_plot",
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

# # Updated figure scoreboards_compare

from collections import OrderedDict

labels_colors_dict = OrderedDict([("GDRS", "gray"), ("FINDRISC", "gold"), ("Anthropometry", "green"),
                                  ("Five blood tests", "red"), ("All features", "blue")])
fig_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/figures/scoreboards_compare/colors_legend.jpg"
plot_legend(figsize=(7, 1), fig_path=fig_path, colors_dict=labels_colors_dict, ncolumns=5, font_size=12,
            leg_text=None)
fig1 = plt.figure(constrained_layout=True, figsize=(7, 10))
gs = gridspec.GridSpec(ncols=4, nrows=32, wspace=0.1, hspace=0.2)
aspect = 'auto'

f1_ax1 = fig1.add_subplot(gs[0:10, 0:2])
roc_path = "/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/figures/scoreboards_compare/scoreboards_compare_roc_ci.png"
plot_ax(roc_path, f1_ax1, "A", aspect=aspect, pos=(-0.02, 1))

APS_path = "/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/figures/scoreboards_compare/scoreboards_compare_aps_ci.png"
f1_ax2 = fig1.add_subplot(gs[0:10, 2:])
plot_ax(APS_path, f1_ax2, "B", aspect=aspect, pos=(0.02, 1))

f1_ax3 = fig1.add_subplot(gs[10:20, :2])
quantile_path = "/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/figures/scoreboards_compare/Quantlites_baseline_compare.jpg"
plot_ax(quantile_path, f1_ax3, "C", aspect=aspect, pos=(-0.02, 1))

f1_ax4 = fig1.add_subplot(gs[10:20, 2:])
Anthro_Feature_importance_path = "/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/figures/scoreboards_compare/LR_Anthro_neto_whr.png"
plot_ax(Anthro_Feature_importance_path, f1_ax4, "D", aspect=aspect, pos=(0.02, 1))

f1_ax5 = fig1.add_subplot(gs[20:30, :2])
BT_Feature_importance_path = "/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/figures/figure3/LR_Five_Blood_Tests.png"
plot_ax(BT_Feature_importance_path, f1_ax5, "E", aspect=aspect, pos=(-0.02, 1))

f1_ax6 = fig1.add_subplot(gs[20:30, 2:])
calibration_path = "/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/figures/scoreboards_compare/isotonic_5_brier_calibration.png"
plot_ax(calibration_path, f1_ax6, "F", aspect=aspect, pos=(0.02, 1))

f1_ax7 = fig1.add_subplot(gs[30:, :])
legened_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/figures/scoreboards_compare/colors_legend.jpg"
plot_ax(legened_path, f1_ax7, None, aspect=aspect, pos=(0.02, 1))

scoreboards_compare_pdf_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/figures/final_figures/scoreboards_compare.pdf"
scoreboards_compare_png_path = "/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/figures/final_figures/scoreboards_compare.png"

plt.tight_layout()

# plt.savefig(scoreboards_compare_path,dpi=600)
plt.savefig(scoreboards_compare_pdf_path, bbox_inches='tight', dpi=500, frameon=False)
plt.savefig(scoreboards_compare_png_path, bbox_inches='tight', dpi=500, frameon=False)
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
#%%
