from UKBB_Scoreboard_functions import *
from LR_CI import LR_CI
# from Scoreboards.Measure_binning_performance import calculate_scoreboards_bins_scores
# from Scoreboards.calculate_scoreboards_ci import calculate_scoreboards_ci

# train_val_test_type - Old to use files that were used before the update with ScoreBoard or Updated to use the files that were updated
"""
This script is building and running the ScoreBoard model
"""# # Anthro,Five_blood,Anthro_explore,Five_blood_explore,"Anthro_debug",Five_blood_debug
scoreboard_type = "Five_blood"#"Five_blood"#"Anthro"#"Five_blood"
debug=False
calc_new_training=True
recover_phase=1
print("working on:",scoreboard_type,", debug:", debug)
#phase_1
#Build the scoreboard WOE
if recover_phase<=1:
    # save_to_scoreboards_basic_path="/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/
    # Scoreboards/WOE_computation",
    scoreboard = ScoreBoard(scoreboard_type=scoreboard_type,
                                build_new_database=True,save_database=True,
                                train_val_test_type="Updated",
                                base_path="/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/",
                                save_to_scoreboards_basic_path="/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/"
                                                               "Imputed_screened/Scoreboards/",
                                pdo=10,
                                max_or=1000,debug=debug,min_num_of_ills=1)
    print("Saving ScoreBoard to ", scoreboard.save_to_folder)
    with open(os.path.join(scoreboard.save_to_folder, "scoreboard_object_" + scoreboard_type+".pkl"), 'wb') as fp:
        pickle.dump(scoreboard, fp)
else:
    try:
        scoreboard = recover_scoreboard(scoreboard_type)
    except:
        print("Couldn't recover scoreboard:", scoreboard_type)
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
    leg_dict={"LR_Five_blood_tests_scoreboard": "Five blood tests ScoreBoard",
              "LR_Anthro_scoreboard": "Anthropometric ScoreBoard",
              "LR_Five_blood_tests_scoreboard_explore": "Five blood tests ScoreBoard explore",
              "LR_Anthro_scoreboard_explore": "LR Anthro ScoreBoard explore",
              "LR_Five_blood_tests_scoreboard_debug": "Five blood tests ScoreBoard debug",
              "LR_Anthro_scoreboard_debug": "LR Anthro ScoreBoard debug"
                                          }
    scoreboard.calculate_scoreboards_bins_scores(build_new_feature_importance=True,
                                      leg_dict=leg_dict,
                                      figsize=(30, 24), font_size=36)
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