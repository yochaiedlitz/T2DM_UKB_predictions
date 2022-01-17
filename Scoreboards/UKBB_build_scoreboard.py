from Article_files.UKBB_article_plots_functions import *
from LR_CI import LR_CI
from UKBB_Scoreboard_functions import *
# train_val_test_type - Old to use files that were used before the update with ScoreBoard or Updated to use the files that were updated
"""
This script is building and running the ScoreBoard model
"""
# # Anthro,Five_blood,Anthro_explore,Five_blood_explore
# Five_blood_scoreboard_object = ScoreBoard(scoreboard_type="Five_blood_explore",
#                                           build_new_database=True,
#                                           save_database=True,
#                                           train_val_test_type="Old",
#                                           base_path="/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/",
#                                           save_to_scoreboards_basic_path="/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/Scoreboards/WOE_computation",
#                                           pdo=10,
#                                           max_or=1000
#                                           ) #Anthro, Five_blood
Anthro_scoreboard_object = ScoreBoard(scoreboard_type="Anthro_explore",
                                      build_new_database=True,
                                      save_database=True,
                                      train_val_test_type="Updated",
                                      base_path="/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/",
                                      save_to_scoreboards_basic_path="/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/Scoreboards/WOE_computation",
                                      pdo=10,
                                      max_or=50
                                      )
# LR_CI(Five_blood_scoreboard_object.run_name)
# plot_woe_graphs(Anthro_scoreboard_object.woe_dict, Anthro_scoreboard_object.save_to_folder)
#
# # LR_CI(Anthro_scoreboard_object.run_name)
# # plot_woe_graphs(Anthro_scoreboard_object.woe_dict, Anthro_scoreboard_object.save_to_folder)
# #
# from Article_files.UKBB_article_plots_functions import *
#
# # #Calculate the Feature importance for the anthropometrics
#
# labels_dict = upload_ukbb_dict()
# figsize=(30,24)
# font_size=24
# num_feat = 20
#
# Five_blood_tests_coeffs_table_obj = LR_Feature_importance(
#     folder_path="/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/Scoreboards/LR_Five_blood_tests_scoreboard/",
#     plot_type="bar_plot",
#     plot=True,
#     font_scale=3,
#     leg_labels=["Five blood tests ScoreBoard"],
#     leg_title=None,
#     leg_pos=[0.9, 0.9],
#     hue_type="binary",
#     figsize=figsize,
#     font_size=font_size,
#     n_boots=10,
#     space=0.2,
#     figpath="/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/figures/figure1/LR_Five_blood_tests_scoreboard_features_importance.png",
#     labels_dict=labels_dict,
#     table_save_path="/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/Tables/LR_Five_blood_tests_scoreboard_importance.csv",
#     hue_colors_dict = {"LR_Five_blood_tests_scoreboard":"red"},
#     file_names_list=["LR_Five_blood_tests_scoreboard"],
#     show_values_on_bar=True,
#     remove_legend=True,
#     ci_summary_table_name="Scoreboared_five_blood_fetures_importance_summary.csv")
