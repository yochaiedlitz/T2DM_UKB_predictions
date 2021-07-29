
from Article_files.UKBB_article_plots_functions import *


def measure_binning_performance(file_name="LR_Five_blood_tests_scoreboard_explore",
                                build_new_feature_importance=True,
                                leg_dict={"LR_Five_blood_tests_scoreboard":"Five blood tests ScoreBoard","LR_Anthro_scoreboard":"Anthropometric ScoreBoard",
                                          "LR_Five_blood_tests_scoreboard_explore":"Five blood tests ScoreBoard explore",
                                          "LR_Anthro_scoreboard_explore":"LR Anthro ScoreBoard explore"},
                                figsize=(30, 24),font_size = 36):
    # #Calculate the Feature importance for the anthropometrics
    #file_name==run_name ... ex. 'LR_Five_blood_tests_scoreboard'
    
    ci_summary_table_name = file_name + "_features_importance_summary.csv"
    woe_dict_path="/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/Scoreboards/WOE_computation/"+file_name+"/woe_dict.pkl"
    woe_csv_path="/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/Scoreboards/WOE_computation/"+file_name+"/woe_csv.csv"
    # used_labels_df_path="/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/Scoreboards/WOE_computation/"+file_name+"/used_labels_df.csv"
    labels_dict,UKBIOBANK_labels_df = upload_ukbb_dict()
    file_names_list = [file_name]
    Folder_path = [os.path.join("/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/Scoreboards/",
                                file_name,"")]
    table_save_path = os.path.join("/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/Tables/",
                                   file_name+"_importance.csv")
    figpath =os.path.join( "/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/figures/Scoreboard",
                           file_name+"_features_importance.png")
    
    lr_features_coeff_pkl_path = os.path.join("/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/Scoreboards/",
                                              file_name,"coeffs_table_obj.pkl")
    fbt_hue_colors_dict = {"LR_Five_blood_tests_scoreboard": "red","LR_Anthro_scoreboard":"green"}
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
            leg_labels=leg_dict[file_name],
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
            hue_colors_dict=fbt_hue_colors_dict,
            file_names_list=file_names_list,
            show_values_on_bar=True,
            remove_legend=True,
            ci_summary_table_name=ci_summary_table_name,build_new=build_new_feature_importance
        )
        with open(lr_features_coeff_pkl_path, "wb") as fp:  # Pickling
            pickle.dump(lr_features_coeff_obj, fp)

    print("coeffs_table_obj.long_name_order: ", lr_features_coeff_obj.long_name_order)
    print("coeffs_table_obj.mean_coefficients_table_df : ", lr_features_coeff_obj.mean_coefficients_table_df)
    categories=[re.split(' <| >| :',x)[0] for x in lr_features_coeff_obj.mean_coefficients_table_df.index]
    lr_features_coeff_obj.mean_coefficients_table_df["category"]=categories
    mean_coefficients_table_df=lr_features_coeff_obj.mean_coefficients_table_df
    
    print ("mean_coefficients_table_df:",mean_coefficients_table_df)
    with open(woe_dict_path,"rb") as fp:
        woe_dict=pickle.load(fp)
    
    n=len(mean_coefficients_table_df.loc[:,"category"].unique())-1 #Number of independent variablles in the model
    bias=mean_coefficients_table_df.loc["Bias","Covariate coefficient"]
    
    pdo=10
    max_or=50
    Factor=pdo/np.log(2)
    Offset=100-Factor*np.log(max_or)
    
    print('Factor:',round(Factor,2),'Offset:',round(Offset,2))
    woe_list=[]
    for category in mean_coefficients_table_df.index:
        print("category:", category)
        category_features=[]
        feat_coeff=mean_coefficients_table_df.loc[category,"Covariate coefficient"]
        print("feat_coeff:", feat_coeff)
        if category!="Bias":
            for ind,feature in enumerate(woe_dict[category].index):
                category_features.append(feature)
                woe_i=woe_dict[category].loc[feature, "WOE"]
                woe_dict[category].loc[feature, "Score"]=int((woe_i*feat_coeff+bias/n)*Factor + Offset/n)
                if ind==0:
                    min_feat=woe_dict[category].loc[feature, "Score"]
                else:
                    if woe_dict[category].loc[feature, "Score"]<min_feat:
                        min_feat=woe_dict[category].loc[feature, "Score"]
            woe_dict[category].loc[category_features, "Score"]=woe_dict[category].loc[category_features, "Score"]-min_feat
            woe_list.append(woe_dict[category])
    woe_df=pd.concat(woe_list)
    woe_df.to_csv(woe_csv_path)
    print ("woe_df:",woe_df)