import os

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from scipy.stats import chi2

plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
mpl.rcParams['axes.facecolor'] = 'w'  # Or any suitable colour...


def try_except(range_dict, bins_dict, nbins, splits_dict, splits, leg_dict, label_name):
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
    return top_range, n_bins, n_splits, leg_val


def hosner_leveshow(observed_expected_N_df,dof=8):
    Observed_true_list=observed_expected_N_df.loc[:, "true_mean"].values
    Expected_true_list=observed_expected_N_df.loc[:, "pred_mean"].values
    N_list=observed_expected_N_df.loc[:,"num_participants"].values
    hl_total = 0
    dof_p2=0
    for ind, observed_p in enumerate(Observed_true_list):
        print(("ind:",ind))
        N=N_list[ind]
        expected_p=Expected_true_list[ind]
        expected_N=N*expected_p
        observed_N=N*observed_p
        expected_not_N=N*(1-expected_p)
        observed_not_N=N*(1-observed_p)
        acc_pos_tmp_score = ((observed_N -expected_N) ** 2) / (expected_N*(1-expected_N/N))
        # acc_pos_tmp_score = (((observed_N - expected_N) ** 2) / expected_N)+\
        #                     (((observed_not_N - expected_not_N) ** 2) / expected_not_N)
        print(("ind:",ind,", pos tmp score is: ",acc_pos_tmp_score))

        if not np.isnan(acc_pos_tmp_score):
            hl_total=hl_total+acc_pos_tmp_score
            dof_p2+=1
    print(("acc_hl_score is:", hl_total))
    dof=dof_p2-2
    print(("dof:",dof))
    if dof>=0:
        p_value=chis2_p_value(hl_total,dof)
        print(("p_value:",p_value))
        return hl_total, p_value, dof
    else:
        print("dof must be greater then 0:")
        exit()



def chis2_p_value(x, dof=9):
    return 1-chi2.cdf(x,dof)

def plot_calibration_curve(df, files_list=None, path=None, nbins=10, splits=3, fig_size=(16, 9), colors_list=None,
                           pallete="viridis", colors_dict=[],
                           plot_hist=True, fontsize=72, leg_dict=[], x_text=0, y_text=1,
                           x_ticks=["0", "0.1", "0.2", "0.4", "0.6", "0.8", "1"],
                           y_ticks=["0", "0.1", "0.2", "0.4", "0.6", "0.8", "1"],
                           xlabel="predicted probability", ylabel="True probability", lw=2, dpi=200, plot=True,
                           marker="o", markersize=12, range_dict={}, bins_dict={}, splits_dict={}, nbins_hist=10,
                           y_hist_ticks=["10", "100", "1000"], xlim=0.8, ylim=0.8, hist_alpha=1, leg_loc="best",
                           hist_bar_pos=0.5, hist_bar_width=0.8, calib_type="isotonic", num_of_files=1000,
                           labels_order=None,
                           Save_to_folder_path="/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/",
                           load_saved_data=False):
    """Plot calibration curve for est w/o and with calibration. """
    if type(files_list) != list:
        files_list = df.index.values
    if type(labels_order) != list:
        labels_order = [leg_dict[key] for key in files_list]
    if type(colors_dict) == list:
        color_pallete = cm.get_cmap('viridis', len(files_list))
        color_dict = {}
        colors_list = color_pallete(np.linspace(0, 1, len(files_list)))
        colors_dict = dict(list(zip(files_list, colors_list)))
    if type(leg_dict) == list:
        leg_dict = dict(list(zip(files_list, files_list)))

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
    results_df_dict = {}

    for file_ind, label_name in enumerate(files_list):
        print(("****", label_name, "*******"))
        top_range, n_bins, n_splits, leg_val = try_except(range_dict, bins_dict, nbins, splits_dict, splits, leg_dict,
                                                          label_name)
        base_path = os.path.dirname(df.loc[label_name, "y_files"])
        files = [os.path.join(base_path, x) for x in os.listdir(base_path) if
                 x.startswith("y_pred_results_") and x.endswith("csv")]
        y_test_calib_list = []
        y_pred_calib_list = []
        if load_saved_data:
            results_df_dict[label_name] = pd.read_csv(
                os.path.join(Save_to_folder_path, label_name + "_" + calib_type + "_calibration.csv"))
            mean_predicted_value = results_df_dict[label_name].loc[:, "pred_mean"]
            fraction_of_positives = results_df_dict[label_name].loc[:, "true_mean"]
        else:
            for file_path_ind, y_file_path in enumerate(files):
                if file_path_ind % 200 == 0:
                    print(file_path_ind)
                if file_path_ind == num_of_files:
                    break
                y_file = pd.read_csv(y_file_path, index_col=None)
                y_test = y_file.loc[:, "y_test"].values
                y_pred = y_file.loc[:, "y_pred"].values

                y_test_array, y_cal_prob_array = perform_calibration(y_test, y_pred, splits, calib_type)

                y_test_calib_list.extend(y_test_array.tolist())
                y_pred_calib_list.extend(y_cal_prob_array.tolist())

            results_df_dict[label_name] = self_calibration_curve(np.array(y_test_calib_list),
                                                                 np.array(y_pred_calib_list),
                                                                 bins=n_bins)

            mean_predicted_value = results_df_dict[label_name].loc[:, "pred_mean"]
            fraction_of_positives = results_df_dict[label_name].loc[:, "true_mean"]
            clf_score = brier_score_loss(np.array(y_test_calib_list), np.array(y_pred_calib_list),
                                         pos_label=np.max(y_test_array))
            if file_ind == 0:
                rel_clf_score = clf_score
            brier_skill_score = 1 - clf_score / rel_clf_score
            hist_dict[label_name] = y_pred_calib_list

            results_df_dict[label_name].loc[:,
            "Deviation pred from True"] = 1 - mean_predicted_value / fraction_of_positives
            results_df_dict[label_name].loc[1, "Brier skill score"] = brier_skill_score
            results_df_dict[label_name].loc[1, "Brier score"] = clf_score
            hosner_leveshow_val,hosner_leveshow_p,hosner_leveshow_dof=hosner_leveshow(results_df_dict[label_name])
            results_df_dict[label_name].loc[1, "hosner_leveshow_p"] = hosner_leveshow_p
            results_df_dict[label_name].loc[1, "hosner_leveshow_val"] = hosner_leveshow_val
            results_df_dict[label_name].loc[1, "hosner_leveshow_dof"] = hosner_leveshow_dof
            results_df_dict[label_name].to_csv(os.path.join(Save_to_folder_path,
                                                            label_name + "_" + calib_type + "_calibration.csv"),
                                               index=False)
            print(("***************saved:", os.path.join(base_path, label_name + "_calibration.csv")))




        if load_saved_data:
            hist_dict = np.load(os.path.join(Save_to_folder_path, calib_type + "calib_hist_dict.npy"),
                                allow_pickle=True).item()
            brier_skill_score = results_df_dict[label_name].loc[1, "Brier skill score"]
            clf_score = results_df_dict[label_name].loc[1, "Brier score"]
            hosner_leveshow_p=results_df_dict[label_name].loc[0, "hosner_leveshow_p"]
            hosner_leveshow_val=results_df_dict[label_name].loc[0, "hosner_leveshow_val"]
            hosner_leveshow_dof=results_df_dict[label_name].loc[0, "hosner_leveshow_dof"]
            print(("hosner_leveshow_p: ",hosner_leveshow_p,
                   ", hosner_leveshow_val: ",hosner_leveshow_val,
                   ", hosner_leveshow_dof: ",hosner_leveshow_dof))

        else:
            np.save(os.path.join(Save_to_folder_path, calib_type + "calib_hist_dict"), hist_dict)

        low_mean_calibrated_value = [x for x in mean_predicted_value if x <= top_range]
        high_mean_calibrated_value = [x for x in mean_predicted_value if x >= top_range]
        #         print "low_mean_calibrated_value:",low_mean_calibrated_value
        if len(low_mean_calibrated_value) > 0:
            low_fraction_of_positives = fraction_of_positives[:len(low_mean_calibrated_value)]
            stretch_low_mean_calibrated_value = [10 * x for x in low_mean_calibrated_value]
            ax1.plot(stretch_low_mean_calibrated_value, low_fraction_of_positives, color=colors_dict[label_name],
                     linestyle="-", lw=lw, marker=marker, markersize=markersize,
                     label=leg_val + "-" + "{:.3f}".format(np.round(clf_score, 3)))

            print(("***calibrated ", label_name, "Brier skill score: ", brier_skill_score))
            print(("***calibrated ", label_name, "Brier score: ", clf_score))

    #             print("y_cal_prob_array of:",label_name," is:",y_cal_prob_array)

    #     ax1.set_xlim(0,xlim)


    if plot_hist:
        ax2 = ax1.twinx()
        x = np.arange(0, (nbins_hist + 1) / 10, 0.1)
        y = pd.DataFrame(index=x, columns=list(hist_dict.keys()))
        for key in list(hist_dict.keys()):
            tmp_hist = np.histogram(hist_dict[key], bins=nbins_hist)
            y.loc[:, key] = tmp_hist[0]
        colors = [colors_dict[ind] for ind in y.columns]
        y.plot(kind='bar', color=colors, legend=False, alpha=0.3, rot=0, ax=ax2, use_index=True, position=hist_bar_pos,
               width=hist_bar_width)
        ax2.set_ylabel("Participants/bin", fontsize=fontsize)
        ax2.set_yscale('log')
        y_hist_ticks = [10, 100, 1000, 10000, 100000, 1000000, 10000000]
        y_hist_tick_labels = [str('%.e' % ind)[:2] + str('%.E' % ind)[-1] for ind in y_hist_ticks]
        ax2.set_yticks(y_hist_ticks)
        ax2.set_yticklabels(y_hist_tick_labels, fontsize=fontsize)
        ax2.spines['top'].set_visible(False)
    ax1.set_ylabel(ylabel, fontsize=fontsize)
    ax1.set_xlabel(xlabel, fontsize=fontsize)
    handles, labels = ax1.get_legend_handles_labels()
    handles = [handles[ind] for x in labels_order for ind, label in enumerate(labels) if label.startswith(x)]
    labels = [labels[ind] for x in labels_order for ind, label in enumerate(labels) if label.startswith(x)]
    ax1.legend(handles, labels, loc=leg_loc, fontsize=fontsize, frameon=False, framealpha=0, title="Brier score")
    ax1.set_xticks([10 * float(x) for x in x_ticks])
    ax1.set_yticks([float(x) for x in y_ticks])
    ax1.set_yticklabels(y_ticks, fontsize=fontsize)
    ax1.set_xticklabels(x_ticks, fontsize=fontsize)
    ax1.set_xlim(-0.4, xlim * 10)
    ax1.set_ylim(0, ylim)
    ax1.spines['top'].set_visible(False)

    plt.tight_layout()
    if path != None:
        print(("Saving to:", path))
        plt.savefig(path, dpi=dpi, frameon=False)
    if plot:
        plt.show()
    return results_df_dict


def perform_calibration(y_test, y_pred, splits, calib_type):
    X = np.array(y_pred)
    y = np.array(y_test)
    if calib_type == "isotonic":
        ir = IsotonicRegression()
    elif calib_type == "gausian":
        ir = GaussianNB()
    # Calibrated with sigmoid calibration
    # if calib_type=="svc":
    #     ir = LinearSVC(max_iter=10000)

    skf = StratifiedKFold(n_splits=splits, random_state=None, shuffle=False)
    y_true_list = []
    y_cal_prob_list = []
    for train_index, test_index in skf.split(X, y):

        X_Cal_train, X_Cal_test = X[train_index], X[test_index]
        y_Cal_train, y_Cal_test = y[train_index], y[test_index]
        if calib_type == "gausian":
            X_Cal_train = X_Cal_train.reshape(-1, 1)
            X_Cal_test = X_Cal_test.reshape(-1, 1)
            ir.fit(X_Cal_train, y_Cal_train)
            p_calibrated = ir.predict_proba(X_Cal_test)

        else:
            ir.fit(X_Cal_train, y_Cal_train)
            p_calibrated = ir.transform(X_Cal_test)

        y_cal_prob_list.extend(p_calibrated)
        y_true_list.extend(y_Cal_test)

    y_true_array = np.array(y_true_list)
    y_cal_prob_array = np.array(y_cal_prob_list)
    nans_list = np.argwhere(np.isnan(y_cal_prob_array))
    y_cal_prob_array = np.delete(y_cal_prob_array, nans_list)
    y_true_array = np.delete(y_true_array, nans_list)
    return y_true_array, y_cal_prob_array


def self_calibration_curve(y_true_array, y_cal_prob_array, bins):
    calibration_df = pd.DataFrame(index=np.arange(1, len(bins), 1),
                                  columns=["num_participants", "pred_mean", "true_mean"])
    for ind, upper in enumerate(bins):
        if ind > 0:
            calibration_df.loc[ind, :] = mean_pred_prob(y_cal_prob_array, y_true_array, bins[ind - 1], bins[ind])
    return calibration_df


def mean_pred_prob(pred_list, true_list, lower, upper):
    # x for x in list1 is same as traversal in the list
    # the if condition checks for the number of numbers in the range
    # lower to upper
    # the return is stored in a list
    # whose length is the answer
    pred_array = np.array([x for x in pred_list if lower <= x and x <= upper])
    true_array = np.array([x for ind, x in enumerate(true_list) if lower <= pred_list[ind] and pred_list[ind] <= upper])
    pred_mean = pred_array.mean()
    try:
        true_mean = float(true_array.sum()) / true_array.shape[0]
        num_participants = true_array.shape[0]
    except:
        true_mean = np.nan
        # print("****got empty list")
        # print("Lower:",lower,", upper:",upper)
        # print("pred_array  is:",pred_array)
        # print("true_array  is:",true_array)
        num_participants = 0

    return num_participants, pred_mean, true_mean


def load_color_dict(path):
    color_dict = {}
    colors_csv = pd.read_csv(path, index_col="File name")
    for key in colors_csv.loc[:, "Label"]:
        color_dict[key] = colors_csv.loc[colors_csv.loc[:, "Label"] == key, "color"].values[0]
    return color_dict


def main():
    cv = 5
    num_of_files = 1001
    load_saved_data = False
    leg_dict = {"Anthropometry with WHR LR": "Anthropometry", "Five blood tests LR": "Five blood tests",
                "Blood Tests LR": "Blood tests", "FINDRISC LR": "FINDRISC"}
    # color_dict=conversion_files_obj.color_dict
    Save_to_folder_path = "/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/Imputed_screened/New_Baseline_compare"
    labels_order = ["Five blood tests", "Anthropometry", "Blood tests", "FINDRISC"]
    files_list = ["FINDRISC LR", "Anthropometry with WHR LR", "Five blood tests LR", "Blood Tests LR"]
    figsize = (30, 24)
    font_size = 72
    df = pd.read_csv("/home/edlitzy/dbg/df.csv", index_col="labels")
    df = df.loc[["Five blood tests LR", "Anthropometry with WHR LR", "Blood Tests LR", "FINDRISC LR"], :]
    print(df)
    bins_vec = np.arange(0,1.05,0.1)
    bins_vec[0]=0
    print(bins_vec)
    range_dict = {"Anthropometry with WHR LR": 1.1, "Five blood tests LR": 1.1,
                  "Blood Tests LR": 1.1, "FINDRISC LR": 1.1}
    bins_dict = {"Anthropometry with WHR LR": bins_vec,
                 "Five blood tests LR": bins_vec,
                 "Blood Tests LR": bins_vec,
                 "FINDRISC LR": bins_vec}
    splits_dict = {"Anthropometry with WHR LR": cv, "Five blood tests LR": cv,
                   "Blood Tests LR": cv, "LR_Finrisc": cv, "SA_GDRS": cv}
    color_dict = load_color_dict("/net/mraid08/export/jafar/Yochai/UKBB_Runs/For_article/colors.csv")

    for calib_type in ["isotonic"]:
        # print("calib_type:",calib_type)
        path = "/home/edlitzy/UKBB_Tree_Runs/For_article/Imputed_screened/figures/figure3/" + calib_type + "_" + str(
            cv) + "_brier_calibration.png"
        res_dict = plot_calibration_curve(df=df, nbins=10, splits=cv, files_list=files_list,
                                          colors_dict=color_dict, plot_hist=True,
                                          fontsize=font_size, fig_size=figsize, leg_dict=leg_dict, x_text=0, y_text=0.8,
                                          x_ticks=["0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9",
                                                   "1"],
                                          y_ticks=["0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9",
                                                   "1"],
                                          xlabel="predicted probability", ylabel="True probability", lw=5, path=path,
                                          dpi=50, plot=False,
                                          marker="o", markersize=24, range_dict=range_dict, bins_dict=bins_dict,
                                          splits_dict=splits_dict,
                                          y_hist_ticks=["10", "100", "1000", "10000", "100000", "1000000"], xlim=1,
                                          ylim=1,
                                          leg_loc=(0.17, 0.63), hist_bar_pos=0.5, hist_bar_width=0.6,
                                          calib_type=calib_type, nbins_hist=10, num_of_files=num_of_files,
                                          labels_order=labels_order, Save_to_folder_path=Save_to_folder_path,
                                          load_saved_data=load_saved_data)

    print(res_dict)


if __name__ == "__main__":
    main()
