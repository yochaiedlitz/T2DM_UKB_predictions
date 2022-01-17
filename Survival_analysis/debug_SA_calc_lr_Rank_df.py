
from UKBB_article_plots_functions import calc_lr_Rank_df
import pandas as pd
import os
import numpy as np

if __name__=="__main__":
    base_path="/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Revision_runs/results_folder"
    dir_name="SA_Antro_neto_whr"
    folder_path=os.path.join(base_path,dir_name)
    save=True
    force_calc=True
    num_of_files=1000
    minimal_diagnosed=1
    bins=10
    low_quantile=0.1
    top_quantile=1
    recalc=True
    Rank_df_list = []
    name = os.path.basename(folder_path)
    res = 1. / bins
    quants_bins = [int(x * 100) / 100. for x in np.arange(low_quantile, top_quantile + res / 2, res)]
    tmp_file_name=os.path.join(base_path, dir_name,dir_name+"quantiles_rank_df.csv")
    calc_lr_Rank_df(folder_name=folder_path,quants_bins=quants_bins,bins=bins,
                    force_calc=force_calc,num_of_files=num_of_files,minimal_diagnosed=minimal_diagnosed)
