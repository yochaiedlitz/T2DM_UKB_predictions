
import sys
# /home/edlitzy/PycharmProjects/UKBB
print(sys.path)
from UKBB_article_plots_config import *
from UKBB_article_plots_functions import *
from LabData import config_global as config
from LabUtils.addloglevels import sethandlers
from LabQueue.qp import fakeqp
qp = config.qp
qp=fakeqp

singles_gbdt_path="/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Revision_runs/Singles_GBDT"

conversion_files_obj = CI_Summary_table_object(
    root_path=singles_gbdt_path,
    save_to_file=True,
    update=True,
    save_prefix="Singles_GBDT")

conversion_files_df = conversion_files_obj.get()
ci_path_dict = conversion_files_obj.get_ci_path_dict()
color_dict = conversion_files_obj.color_dict
variables_dict = conversion_files_obj.get_UKBB_dict()

labels_dict = upload_ukbb_dict()


results_directory_name = "Diabetes_Results"

df, precision_list, recall_list, fpr_list, tpr_list = calc_roc_aps_lists(
    directories_path=singles_gbdt_path, conversion_files_df=conversion_files_df)
print(df)
print("Wait")

