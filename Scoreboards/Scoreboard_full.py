from UKBB_Scoreboard_functions import *
from LR_CI import LR_CI
# from Scoreboards.Measure_binning_performance import calculate_scoreboards_bins_scores
# from Scoreboards.calculate_scoreboards_ci import calculate_scoreboards_ci

# train_val_test_type - Old to use files that were used before the update with ScoreBoard or Updated to use the files that were updated
"""
This script is building and running the ScoreBoard model
The reason there are both run_name and ancestor_run_name is this:
The scoreboard runs should not be 
"""
#Anthro,#"Five_blood""#,No_reticulocytes_scoreboard#
#Five_blood,Anthro_explore,Five_blood_explore,#Anthro_explore
#"LR_explore_No_reticulocytes_scoreboard"#LR_explore_No_reticulocytes_scoreboard#LR_Anthro_scoreboard_explore"
#"Anthro_debug",Five_blood_debug
#LR_Strat_H39_Antro_neto_whr_scoreboard#"LR_Strat_L39_Antro_neto_whr_scoreboard"
# #"LR_Strat_L39_Four_BT_SB_revision","LR_Strat_L39_Antro_SB_revision"
# LR_Strat_L20_H39_Antro_SB_revision
#"LR_Strat_H39_Four_BT_scoreboard_custom"
scoreboard_type = "LR_Strat_L20_H39_Antro_neto_whr_scoreboard"
debug=False
calc_new_training=True
force_update_LR=True


leg_dict = {"LR_Five_blood_tests_scoreboard": "Five blood tests ScoreBoard",
            "LR_Anthro_scoreboard": "Anthropometric ScoreBoard",
            "LR_Five_blood_tests_scoreboard_explore": "Five blood tests ScoreBoard explore",
            "LR_Anthro_scoreboard_explore": "LR Anthro ScoreBoard explore",
            "LR_Five_blood_tests_scoreboard_debug": "Five blood tests ScoreBoard debug",
            "LR_Anthro_scoreboard_debug": "LR Anthro ScoreBoard debug",
            "LR_No_reticulocytes_scoreboard": "Four blood tests Score board",
            "LR_Strat_L39_Four_BT_SB_revision": "Normal glucose four blood tests score board",
            "LR_Strat_L20_H39_Antro_SB_revision": "Normal glucose anthropometry scoreboard",
            "LR_explore_No_reticulocytes_scoreboard": "explore No reticulocytes scoreboard",
            "LR_Strat_H39_Four_BT_scoreboard_custom":"Strat H39 Four BT scoreboard custom",
            "LR_Strat_L20_H39_Four_Blood_Tests_scoreboard": "Strat L20 H39 Four blood tests scoreboard",
            "LR_Strat_H39_Antro_neto_whr_scoreboard":"Strat H39 Antro_neto_whr scoreboard custom",
            "LR_Strat_L20_H39_Antro_neto_whr_scoreboard":"Strat L20 H39 Antro_neto_whr scoreboard",
            "LR_Strat_L39_Antro_neto_whr_scoreboard":"Strat L39 Antro neto whr scoreboard custom",
            "LR_Strat_L39_Four_Blood_Tests_scoreboard_custom":"Strat L39 four blood tests scoreboard custom"}
save_to_scoreboard_basic_path="/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Revision_runs/stratified/" # "/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/Scoreboards/"
base_path="/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Revision_runs/"
force_update_ancetor_LR=False
recover_phase=1
print("working on:",scoreboard_type,", debug:", debug)
#phase_1
#Build the scoreboard WOE
if recover_phase<=1:
    # save_to_scoreboards_basic_path="/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/
    # Scoreboards/WOE_computation",
    scoreboard = ScoreBoard(scoreboard_type=scoreboard_type,
                            build_new_database=True,save_database=False,
                            train_val_test_type="Updated",
                            base_path=base_path,
                            save_to_scoreboards_basic_path=save_to_scoreboard_basic_path,
                            pdo=10,
                            max_or=1000,debug=debug,min_num_of_ills=1,
                            force_update_LR=force_update_LR,
                            force_update_ancestor_LR=force_update_ancetor_LR,
                            leg_dict=leg_dict)
    print("Saving ScoreBoard to ", scoreboard.save_to_folder)
    with open(os.path.join(scoreboard.save_to_folder, "scoreboard_object_" + scoreboard_type+".pkl"), 'wb') as fp:
        pickle.dump(scoreboard, fp)
else:
    try:
        scoreboard = recover_scoreboard(scoreboard_type,
                                        save_to_scoreboard_basic_path,
                                        base_path)
    except:
        sys.exit("Couldn't recover scoreboard:", scoreboard_type)
#Phase_2
#Build the LR model based on the
if recover_phase<=2:
    print("started_phase_2")
    LR_CI(run_name=scoreboard.run_name, force_update=True,
          run_object=scoreboard.run_object,
          calc_new_training=calc_new_training,debug=debug)
    with open(os.path.join(scoreboard.save_to_folder, "scoreboard_object_" + scoreboard_type+".pkl"), 'wb') as fp:
        pickle.dump(scoreboard, fp)
    print("Finished_phase_2")

#Phase_3
if recover_phase<=3:
    print("started_phase_3")
    scoreboard.calculate_scoreboards_bins_scores(
        build_new_feature_importance=True,leg_dict=leg_dict,figsize=(30, 24),font_size=36)
    print("Finished_phase_3")
    with open(os.path.join(scoreboard.save_to_folder, "scoreboard_object_" + scoreboard_type+".pkl"), 'wb') as fp:
        pickle.dump(scoreboard, fp)
#phase_4
if recover_phase<=4:
    print("started_phase_4")
    scoreboard.calculate_scoreboards_ci(force_recalculate_ci=True)
    with open(os.path.join(scoreboard.save_to_folder, "scoreboard_object_" + scoreboard_type+".pkl"), 'wb') as fp:
        pickle.dump(scoreboard, fp)
    print("Finished")