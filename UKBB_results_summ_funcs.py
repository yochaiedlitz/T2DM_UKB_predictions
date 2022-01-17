"""
This package contains functions for plotting the results for UKBB Diabetes article
"""
import pandas as pd
import numpy as np
import os
import UKBB_Func
import pickle
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from matplotlib import rcParams as rc
import seaborn as sns
import pylustrator
pylustrator.start()


def get_cmap(n, name='inferno'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def sort_scores(y_files, predictions_files, labels, test="AUC"):
    y_scores_df = pd.DataFrame({
        "y_files": y_files,
        "predictions_files": predictions_files,
        "labels": labels
    })
    y_scores_df = y_scores_df.set_index("labels")
    for ind, label in enumerate(labels):
        with open(predictions_files[ind], 'rb') as fp:
            y_pred_val = pickle.load(fp)
        with open(y_files[ind], 'rb') as fp:
            y_test_val = pickle.load(fp)
        y_scores_df.loc[label, "AUC"] = roc_auc_score(y_test_val, y_pred_val)
        y_scores_df.loc[label, "APS"] = average_precision_score(
            y_test_val, y_pred_val)
    return y_scores_df


def wrap_labels(labels, num_of_words):
    new_labels = []
    for x in labels:
        y = x.split()
        if len(y) > num_of_words:
            y = " ".join(y[0:num_of_words]) + "\n" + " ".join(y[num_of_words:])
        else:
            y = " ".join(y)
        new_labels.append(y)
    return new_labels


def calc_precision_recall_list(df, sort=True):
    #How can be "simple" or "relative"
    precision_list = []
    recall_list = []
    if sort:
        df = df.sort_values('APS')
    y_files = df.y_files.values
    predictions_files = df.predictions_files.values
    labels = df.index.values
    APS = df.APS.values
    n = len(labels)
    if (all(len(x) == n for x in [y_files, predictions_files])):
        for ind, label in enumerate(labels):
            with open(df.predictions_files.values[ind], 'rb') as fp:
                y_pred_val = pickle.load(fp)
            with open(df.y_files.values[ind], 'rb') as fp:
                y_test_val = pickle.load(fp)
            precision, recall, _ = precision_recall_curve(
                y_test_val, y_pred_val)
            precision_list.append(precision)
            recall_list.append(recall)
    return precision_list, recall_list


def calc_roc_list(df, sort=True):
    #How can be "simple" or "relative"
    fpr_list = []
    tpr_list = []
    if sort:
        df = df.sort_values('AUC')
    y_files = df.y_files.values
    predictions_files = df.predictions_files.values
    labels = df.index.values
    AUC = df.AUC.values
    n = len(labels)
    if (all(len(x) == n for x in [y_files, predictions_files])):
        for ind, label in enumerate(labels):
            with open(df.predictions_files.values[ind], 'rb') as fp:
                y_pred_val = pickle.load(fp)
            with open(df.y_files.values[ind], 'rb') as fp:
                y_test_val = pickle.load(fp)
            fpr, tpr, _ = roc_curve(y_test_val, y_pred_val)
            fpr_list.append(fpr)
            tpr_list.append(tpr)
    return fpr_list, tpr_list


def plot_ROC_curves(fpr_list, tpr_list, df, ax=None, color="cool", legend_length=3, font_size=16, sort=True):
    ax = ax or plt.gca()
    if sort:
        df = df.sort_values('AUC')
    y_files = df.y_files.values
    predictions_files = df.predictions_files.values
    labels = df.index.values
    new_labels = wrap_labels(labels, num_of_words=legend_length)
    AUC = df.AUC.values
    n = len(labels)
    lw = 2
    if all(len(x) == n for x in [y_files, predictions_files]):
        cmap = sns.color_palette(color, df.shape[0])
        for ind, label in enumerate(new_labels):
            fpr = fpr_list[ind]
            tpr = tpr_list[ind]
            ax.plot(
                fpr,
                tpr,
                c=cmap[ind],
                lw=lw,
                label=label + ' AUC={0:0.2f}'.format(AUC[ind]))

        ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=font_size)
        ax.set_ylabel('True Positive Rate', fontsize=font_size)
        ax.legend(loc='best')
        plt.tight_layout()
    else:
        print("all input lists should be same length, y_test_val_list:",
              len(y_files), ", y_pred_val_list:", len(predictions_files),
              ", labels:", len(labels))
    return ax


def plot_APS_curves(precision_list, recall_list, df,ax=None, color="cool", legend_length=3, font_size=16, sort=True):
    #How can be "simple" or "relative"
    ax = ax or plt.gca()
    if sort:
        df = df.sort_values('APS')
    else:
        df = df
    y_files = df.y_files.values
    predictions_files = df.predictions_files.values
    labels = df.index.values
    new_labels = wrap_labels(labels, num_of_words=legend_length)
    APS = df.APS.values
    n = len(labels)
    lw = 2
    if (all(len(x) == n for x in [y_files, predictions_files])):
        cmap = sns.color_palette(color, df.shape[0])
        for ind, label in enumerate(new_labels):
            precision = precision_list[ind]
            recall = recall_list[ind]
            ax.step(
                recall,
                precision,
                c=cmap[ind],
                where='post',
                label=label + ' APS={0:0.2f}'.format(APS[ind]))

        ax.set_ylim([0.0, 1.05 * max(precision)])
        axi = ax.twinx()
        # set limits for shared axis
        axi.set_ylim(ax.get_ylim())
        # set ticks for shared axis
        relative_ticks = []
        label_format = '%.1f'
        for tick in ax.get_yticks():
            tick = tick / precision[0]
            relative_ticks.append(label_format % (tick, ))
        axi.set_yticklabels(relative_ticks)
        axi.set_ylabel('Precision fold', fontsize=font_size)
        ax.set_xlim([0.0, 1.0])
        ax.set_xlabel('Recall', fontsize=font_size)
        ax.set_ylabel('Precision', fontsize=font_size)
        ax.legend(loc='best')
        plt.tight_layout()
    else:
        print("all input lists should be same length, y_test_val_list:",
              len(y_files), ", y_pred_val_list:", len(predictions_files),
              ", labels:", len(labels))
        print("please make sure that how==simple or how==relative")
    return ax


def plot_aps_summary(df, ax=None, font_size=16, color="cool", sort=True):
    if ax:
        ax = ax
    else:
        ax = plt.gca()
    my_colors = sns.color_palette(color, df.shape[0])
    color_list = my_colors
    if sort:
        aps_df = df.sort_values(["APS"]).loc[:, "APS"]
    else:
        aps_df = df.loc[:, "APS"]
    labels = aps_df.index.values
    new_labels = wrap_labels(labels, num_of_words=3)
    ax.bar(new_labels, height=aps_df.values, align='center', color=color_list)

    ax.xaxis.label.set_visible(False)
    ax.set_ylabel("Average precision score", fontsize=font_size)
    #     ax.set_title('Average precision recall summary of adding features',fontsize=font_size+2)
    ax.tick_params(labelrotation=90, labelsize=font_size - 4)
    return ax


def plot_roc_auc_summary(df, ax=None, font_size=16, color="cool", sort=True):
    if ax:
        ax = ax
    else:
        ax = plt.gca()
    values = df.sort_values(["AUC"]).loc[:, "AUC"].values
    my_colors = sns.color_palette(color, df.shape[0])
    color_list = my_colors
    if sort:
        auc_df = df.sort_values(["AUC"]).loc[:, "AUC"]
    else:
        auc_df = df.loc[:, "AUC"]
    labels = auc_df.index.values
    new_labels = wrap_labels(labels, num_of_words=3)
    x = np.arange(len(new_labels))  # the label locations
    width = 0.8
    ax.bar(
        left=x - width / 2,
        height=auc_df.values,
        width=width,
        align='center',
        color=color_list,
        tick_label=new_labels)
    ax.xaxis.label.set_visible(False)
    ax.set_ylabel(
        "Area under curve (AUC) of the ROC curves", fontsize=font_size)
    ax.set_ylim([0.5, 1.05 * max(values)])
    ax.tick_params(labelrotation=90, labelsize=font_size - 2)
    return ax


def plot_summary_curves(df, precision_list, recall_list, fpr_list, tpr_list, figsize=(25, 16), dpi=1200,
                        color_palette="Set1", file_name="temp", legend_len=4, font_size=24, legend_size=18, sort=True):
    file_name = os.path.join("/home/edlitzy/UKBB_Tree_Runs/For_article/plots/",
                             file_name)
    fig = plt.gcf()
    rc.update({'font.size': font_size})
    plt.rc('legend', fontsize=legend_size)
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    ax1 = plot_roc_auc_summary(df, ax1, font_size, color_palette, sort=sort)
    ax1.text(
        -0.04,
        1,
        'A',
        transform=ax1.transAxes,
        fontsize=font_size,
        fontweight='bold',
        va='top',
        ha='right')

    ax2 = plot_aps_summary(df, ax2, font_size, color_palette, sort=sort)
    ax2.text(
        -0.04,
        1,
        'B',
        transform=ax2.transAxes,
        fontsize=font_size,
        fontweight='bold',
        va='top',
        ha='right')

    ax3 = plot_APS_curves(
        precision_list,
        recall_list,
        df,
        ax3,
        color_palette,
        legend_length=legend_len,
        font_size=font_size,
        sort=sort)
    ax3.text(
        -0.04,
        1,
        'C',
        transform=ax3.transAxes,
        fontsize=font_size,
        fontweight='bold',
        va='top',
        ha='right')

    ax4 = plot_ROC_curves(
        fpr_list,
        tpr_list,
        df,
        ax4,
        color_palette,
        legend_length=legend_len,
        font_size=font_size,
        sort=sort)
    ax4.text(
        -0.04,
        1,
        'D',
        transform=ax4.transAxes,
        fontsize=font_size,
        fontweight='bold',
        va='top',
        ha='right')

    plt.tight_layout()
    plt.savefig(file_name, dpi=dpi)
    plt.show()


def plot_quantiles_curve(test_label="Blood Tests",bins=100,low_quantile=0.8,top_quantile=1,figsize=(16,9)):
    with open(singles_df.loc[test_label,"predictions_files"], 'rb') as fp:
        y_pred_val=pickle.load(fp)
    with open(singles_df.loc[test_label,"y_files"], 'rb') as fp:
        Blood_Tests_test_val=pickle.load(fp)
    vals_df = pd.DataFrame(data={"Y_test": y_test_val, "Y_Pred": y_pred_val})
    res=1./bins
    quants_bins=[int(x*100)/100. for x in np.arange(low_quantile, top_quantile+res/2, res)]
    vals_df = vals_df.sort_values("Y_Pred", ascending=False)
    Quants=pd.DataFrame()
    Quants = vals_df.loc[:, "Y_Pred"].quantile(quants_bins)
    Rank = pd.DataFrame()
    for ind, quant in enumerate(Quants.values[:-1]):
        print quant
        Rank.loc[np.str(ind), "Diagnosed"] = vals_df.loc[((vals_df["Y_Pred"] <= Quants.values[ind + 1]) &                                                      (vals_df["Y_Pred"] > quant))].loc[:,
                                             'Y_test'].sum()
        Rank.loc[np.str(ind), "All"] = vals_df.loc[((vals_df["Y_Pred"] > quant) &                                                (vals_df["Y_Pred"] <= Quants.values[ind + 1]))].loc[:, 'Y_test'].count()
        Rank.loc[np.str(ind), "Ratio"] = Rank.loc[np.str(ind), "Diagnosed"] / Rank.loc[np.str(ind), "All"]
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.bar(Rank.index.values, Rank.loc[:, "Ratio"])
    labels = [str(int(100*x)) for x in np.arange(low_quantile+res, top_quantile+res/2, res)]
    ax.set_xticklabels(labels)
    # ax.set_xlim(low_quantile+res, top_quantile)
    # ax.set_title('Precentage of Ill Vs. Prediction quantile with:'+str(bins)+" bins")
    ax.set_xlabel("Prediction quantile")
    ax.set_ylabel("Prevalence in quantile")
    plt.tight_layout()
    plt.show()
    print Rank

