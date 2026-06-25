import os
# import matplotlib
import matplotlib.pyplot as plt
import math
import numpy as np
import csv
import seaborn as sns
from sklearn.metrics import r2_score
import argparse
import shutil
import pandas as pd

parser = argparse.ArgumentParser(description="Statistics and graphs generation script")
parser.add_argument("-f", "--per-fold", action="store_true", help="Generate statistics per fold")

# csv_path = "./results_fold_1/predicted_metrics/predicted_metrics.csv"
os.makedirs("./plots", exist_ok=True)

def compute_statistics(y_unordered):
    y_ordered = sorted(y_unordered)
    n = len(y_unordered)
    average = sum(y_ordered)/ n

    middle = n // 2
    if n % 2 == 0:
        median = (y_ordered[middle-1] + y_ordered[middle])/2
    else:
        median = y_ordered[middle]

    total_variance = 0
    for y in y_ordered:
        variance = (y - average)**2
        total_variance += variance
    variance = total_variance/n
    standard_deviantion = math.sqrt(variance)
    return average, median, variance, standard_deviantion

def compute_r_squared(X, Y, average):
    sum_squares = 0
    sum_residues = 0
    for i, x in enumerate(X):
        sum_squares += (x-average)**2
        sum_residues += (x-Y[i])**2
    return 1 - (sum_residues/sum_squares)

def get_species(image_name):
    lower_case = image_name.lower()
    if "tomato" in lower_case:
        return "tomato"
    if "eucalyptus" in lower_case:
        return "eucalyptus"
    if "coffee" in lower_case or "cafe" in lower_case:
        return "coffee"
    if "goiaba" in lower_case or lower_case.startswith("guava"):
        return "guava"
    if "pumpkin" in lower_case:
        return "orange"
    if "orange" in lower_case:
        return "pumpkin"
    if "lemon" in lower_case or "limao" in lower_case:
        return "lemon"
    if "bean_pat" in lower_case or "bean_leaflet" in lower_case or lower_case.startswith("bean"):
        return "bean"
    if "castor_bean" in lower_case or lower_case.startswith("castor bean"):
        return "castor-bean"
    if "bell_pepper" in lower_case or lower_case.startswith("bell pepper"):
        return "bell-pepper"
    if "soybean" in lower_case:
        return "soybean"
    if "corn" in lower_case:
        return "corn"
    if "cotton" in lower_case:
        return "cotton"
    if "multi" in lower_case:
        return "multi-specie"
    print(f"Error: species not known of image {image_name}")
    return "unkown"

def get_linear_regression(data_frame, x = 'real', y = 'annot'):
    x_list = data_frame[x].to_list()
    y_list = data_frame[y].to_list()
    (m, b) = np.polyfit(x_list, y_list, 1)
    return m, b

def plot_linear_regression(data_frame, max, min, x = 'real', y = 'annot', label = 'annotations', ax = None, figsize = (8,6)):
    color = 'b'
    marker = 'x'
    if label != 'annotations':
        color = 'r'
        marker = 'o'

    m, b = get_linear_regression(data_frame, x, y)

    if ax == None:
        ax = data_frame.plot.scatter(x=x, y=y, alpha=0.5, marker=marker, color=color, label=label, figsize=figsize)
    else:
        data_frame.plot.scatter(x=x, y=y, alpha=0.5, marker=marker, color=color, label=label, figsize=figsize, ax = ax)

    plt.plot(np.array([min, max]), np.array([m*min + b, m*max + b]), color=color, linestyle='dashed')
    return m, b, ax

def draw_scatter_plot(dataframe, x = 'real', y_annotated = 'annot', y_annotated_RER = 'annot_RER', y_predicted = 'pred', y_predicted_RER = 'pred_RER', dir = "./plots", metric = 'area', fold = 1, save = True, gen_simple = True):
    filtered_df = dataframe[(dataframe['metric']==metric) & (dataframe[x]>0)]
    if fold > 0:
        filtered_df = filtered_df[filtered_df['fold']==fold]
        os.makedirs(dir + f"/fold_{fold}/" + metric, exist_ok=True)
    else:
        os.makedirs(dir + "/all_folds/" + metric, exist_ok=True)
    if filtered_df.empty:
        return

    max_x = filtered_df[x].max() + 10
    max_y = max(filtered_df[y_annotated].max(), filtered_df[y_predicted].max())
    min_x = filtered_df[x].min()
    min_y = min(filtered_df[y_annotated].min(), filtered_df[y_predicted].min())

    ### Annotated
    # m_annot, b_annot, ax = plot_linear_regression(filtered_df, max_x, min_x, x, y_annotated)

    # mean, median, variance, standard_deviation = compute_statistics(filtered_df[y_annotated].to_list())
    # RER_mean, RER_median, _, RER_standard_deviation = compute_statistics(filtered_df[y_annotated_RER].to_list())

    # r_squared_annot = None
    # if filtered_df[x].size > 2:
    #     r_squared_annot = r2_score(filtered_df[x].to_list(), filtered_df[y_annotated].to_list())

    # if save:
    #     with open(dir + "/statistics_fix.csv", "a", newline='') as csvfile:
    #         spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #         spamwriter.writerow([fold, "all", metric.title(), "Annotation", m_annot, b_annot, r_squared_annot if r_squared_annot!= None else 'Nava', mean, median, variance, standard_deviation, RER_mean, RER_median, RER_standard_deviation])

    ### Predicted
    m, b, ax = plot_linear_regression(filtered_df, max_x, min_x, x=x, y=y_predicted, label='predictions')

    mean, median, variance, standard_deviation = compute_statistics(filtered_df[y_predicted].to_list())
    RER_mean, RER_median, _, RER_standard_deviation = compute_statistics(filtered_df[y_predicted_RER].to_list())

    r_squared = None
    if filtered_df[x].size > 2:
        r_squared = r2_score(filtered_df[x].to_list(), filtered_df[y_predicted].to_list())

    if save:
        with open(dir + "/statistics_fix.csv", "a", newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow([fold, "all", metric.title(), "Prediction", m, b, r_squared if r_squared!= None else 'Nava', mean, median, variance, standard_deviation, RER_mean, RER_median, RER_standard_deviation])

    # Label
    ax.set_ylabel("estimated")

    # Legend
    #line1, = ax.plot([], label=f'Annotations\nm = {m_annot:.2f}, b = {b_annot:.2f}\nR2 = {r_squared_annot:.4f}', marker='x', color='b', linestyle='')
    line2, = ax.plot([], label=f'Predictions\nm = {m:.2f}, b = {b:.2f}\nR2 = {r_squared:.4f}', marker='o', color='r', linestyle='')
    #first_legend = ax.legend(handles=[line1], loc='upper left')
    #ax.add_artist(first_legend)
    ax.legend(handles=[line2], loc='center left', bbox_to_anchor=(0.0, 0.8))

    plt.tight_layout()

    if fold > 0:
        ax.set_title(metric.title() + f" estimation linear regression fold {fold}")
        plt.savefig(dir + f"/fold_{fold}/" + metric + "/scatter_plot.pdf")
    else:
        ax.set_title(metric.title() + " estimation linear regression all folds")
        plt.savefig(dir + "/all_folds/" + metric + "/scatter_plot.pdf")

    if gen_simple:
        ax.set_title(metric.title() + " - DeepLabV3+", fontsize=30)
        ax.tick_params(axis='both', labelsize = 20)
        ax.set_ylabel("")
        ax.set_xlabel("")
        #first_legend.set_visible(False)
        ax.legend().set_visible(False)

        plt.tight_layout()

        if fold > 0:
            plt.savefig(dir + f"/fold_{fold}/" + metric + "/scatter_plot_simple.pdf")
        else:
            plt.savefig(dir + "/all_folds/" + metric + "/scatter_plot_simple.pdf")
    plt.close()

def draw_scatter_subplots(dataframe, categories, x = 'real', y_annotated = 'annot', y_annotated_RER = 'annot_RER', y_predicted = 'pred', y_predicted_RER = 'pred_RER', dir = "./plots", metric = 'area', fold = 1, group = 'species', save = True):
    figsize = (20, 6)

    if fold > 0:
        os.makedirs(dir + f"/fold_{fold}/" + metric, exist_ok=True)
    else:
        figsize = (20, 12)
        os.makedirs(dir + "/all_folds/" + metric, exist_ok=True)

    ncols = 3
    nrows = math.ceil(len(categories)/ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    if fold > 0:
        fig.suptitle(metric.title() + " estimation linear regression per " + group + f" fold {fold}")
    else:
        fig.suptitle(metric.title() + " estimation linear regression per " + group + " all folds")
    axes = axes.flat

    for i, category in enumerate(categories):
        filtered_df = dataframe[(dataframe[group] == category) & (dataframe['metric'] == metric) & (dataframe[x]>0)]
        if fold > 0:
            filtered_df = filtered_df[filtered_df['fold'] == fold]
        if filtered_df.empty:
            continue

        axes[i].set_title(category)

        max_x = filtered_df[x].max() + 10
        max_y = max(filtered_df[y_annotated].max(), filtered_df[y_predicted].max())
        min_x = filtered_df[x].min()
        min_y = min(filtered_df[y_annotated].min(), filtered_df[y_predicted].min())

        outliers_df = filtered_df[filtered_df['is_outlier']==True]
        non_outliers_df = filtered_df[filtered_df['is_outlier']==False]

        ### Annotated
        # (m, b) = np.polyfit(filtered_df[x].to_list(), filtered_df[y_annotated].to_list(), 1)
        # outliers_df.plot.scatter(x=x, y=y_annotated, alpha=0.5, marker='x', color='b', label="annotations outliers", ax=axes[i])
        # non_outliers_df.plot.scatter(x=x, y=y_annotated, alpha=0.5, s=5, marker='o', color='b', label="annotations", ax=axes[i])
        # axes[i].plot(np.array([min_x, max_x]), np.array([m*min_x + b, m*max_x + b]), color = "b", linestyle='dashed', label="annotations linear regression")

        # mean, median, variance, standard_deviation = compute_statistics(filtered_df[y_annotated].to_list())
        # RER_mean, RER_median, _, RER_standard_deviation = compute_statistics(filtered_df[y_annotated_RER].to_list())

        # r_squared = None
        # if filtered_df[x].size > 2:
        #     r_squared = r2_score(filtered_df[x].to_list(), filtered_df[y_annotated].to_list())

        # if save:
        #     with open(dir + "/statistics_fix.csv", "a", newline='') as csvfile:
        #         spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        #         spamwriter.writerow([fold, f"{group} {category}", metric.title(), "Annotation", m, b, r_squared if r_squared!= None else 'Nava', mean, median, variance, standard_deviation, RER_mean, RER_median, RER_standard_deviation])

        ### Predicted
        (m, b) = np.polyfit(filtered_df[x].to_list(), filtered_df[y_predicted].to_list(), 1)
        outliers_df.plot.scatter(x=x, y=y_predicted, alpha=0.5, marker='x', color='r', label="predictions", ax=axes[i])
        non_outliers_df.plot.scatter(x=x, y=y_predicted, alpha=0.5, s=5, marker='o', color='r', label="predictions", ax=axes[i])
        axes[i].plot(np.array([min_x, max_x]), np.array([m*min_x + b, m*max_x + b]), color = "r", linestyle='dashed', label="predictions linear regression")

        mean, median, variance, standard_deviation = compute_statistics(filtered_df[y_predicted].to_list())
        RER_mean, RER_median, _, RER_standard_deviation = compute_statistics(filtered_df[y_predicted_RER].to_list())

        r_squared = None
        if filtered_df[x].size > 2:
            r_squared = r2_score(filtered_df[x].to_list(), filtered_df[y_predicted].to_list())

        if save:
            with open(dir + "/statistics_fix.csv", "a", newline='') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                spamwriter.writerow([fold, f"{group} {category}", metric.title(), "Prediction", m, b, r_squared if r_squared!= None else 'Nava', mean, median, variance, standard_deviation, RER_mean, RER_median, RER_standard_deviation])
    plt.tight_layout()
    if fold > 0:
        plt.savefig(dir + f"/fold_{fold}/" + metric + "/scatter_plot_" + group + ".pdf")
    else:
        plt.savefig(dir + "/all_folds/" + metric + "/scatter_plot_" + group + ".pdf")
    plt.close()
    
def draw_multiple_scatter_plots(dataframe, categories, x = 'real', y_annotated = 'annot', y_annotated_RER = 'annot_RER', y_predicted = 'pred', y_predicted_RER = 'pred_RER', dir = "./plots", metric = 'area', fold = 1, group = 'species', save = True, gen_simple = False):
    if fold > 0:
        os.makedirs(dir + f"/fold_{fold}/{metric}/{group}", exist_ok=True)
    else:
        os.makedirs(dir + f"/all_folds/{metric}/{group}", exist_ok=True)

    figsize = (8, 6)

    for i, category in enumerate(categories):
        filtered_df = dataframe[(dataframe[group] == category) & (dataframe['metric'] == metric) & (dataframe[x]>0)]
        if filtered_df.empty:
            continue

        max_x = filtered_df[x].max() + 10
        max_y = max(filtered_df[y_annotated].max(), filtered_df[y_predicted].max())
        min_x = filtered_df[x].min()
        min_y = min(filtered_df[y_annotated].min(), filtered_df[y_predicted].min())

        ### Annotated
        # m_annot, b_annot, ax = plot_linear_regression(filtered_df, max_x, min_x, x=x, y=y_annotated)
        # mean, median, variance, standard_deviation = compute_statistics(filtered_df[y_annotated].to_list())
        # RER_mean, RER_median, _, RER_standard_deviation = compute_statistics(filtered_df[y_annotated_RER].to_list())

        # r_squared_annot = None
        # line1, = ax.plot([], label=f'Annotations\nm = {m_annot:.2f}, b = {b_annot:.2f}\nR2 = None',
        #              marker='x', color='b', linestyle='')
        # if filtered_df[x].size > 2:
        #     r_squared_annot = r2_score(filtered_df[x].to_list(), filtered_df[y_annotated].to_list())
        #     line1, = ax.plot([], label=f'Annotations\nm = {m_annot:.2f}, b = {b_annot:.2f}\nR2 = {r_squared_annot:.4f}',
        #                  marker='x', color='b', linestyle='')

        # if save:
        #     with open(dir + "/statistics_fix.csv", "a", newline='') as csvfile:
        #         spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        #         spamwriter.writerow([fold, f"{group} {category}", metric.title(), "Annotation", m_annot, b_annot, r_squared_annot if r_squared_annot!= None else 'Nava', mean, median, variance, standard_deviation, RER_mean, RER_median, RER_standard_deviation])

        ### Predicted
        m, b, ax = plot_linear_regression(filtered_df, max_x, min_x, x=x, y=y_predicted, label='predictions')

        mean, median, variance, standard_deviation = compute_statistics(filtered_df[y_predicted].to_list())
        RER_mean, RER_median, _, RER_standard_deviation = compute_statistics(filtered_df[y_predicted_RER].to_list())

        r_squared = None
        line2, = ax.plot([], label=f'Predictions\nm = {m:.2f}, b = {b:.2f}\nR2 = None', marker='o',
                     color='r', linestyle='')
        if filtered_df[x].size > 2:
            r_squared = r2_score(filtered_df[x].to_list(), filtered_df[y_predicted].to_list())
            line2, = ax.plot([], label=f'Predictions\nm = {m:.2f}, b = {b:.2f}\nR2 = {r_squared:.4f}', marker='o',
                         color='r', linestyle='')

        if save:
            with open(dir + "/statistics_fix.csv", "a", newline='') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                spamwriter.writerow([fold, f"{group} {category}", metric.title(), "Prediction", m, b, r_squared if r_squared!= None else 'Nava', mean, median, variance, standard_deviation, RER_mean, RER_median, RER_standard_deviation])

        # Label
        ax.set_ylabel("estimated")

        #first_legend = ax.legend(handles=[line1], loc='upper left')
        #ax.add_artist(first_legend)
        #ax.legend(handles=[line2], loc='center left', bbox_to_anchor=(0.0, 0.8))

        if fold > 0:
            filtered_df = filtered_df[filtered_df['fold'] == fold]
            ax.set_title(metric.title() + f" estimation linear regression per {group}: {category} fold {fold}")
        else:
            ax.set_title(metric.title() + f" estimation linear regression per {group}: {category} all folds")

        plt.tight_layout()

        if fold > 0:
            plt.savefig(dir + f"/fold_{fold}/{metric}/{group}/scatter_plot_" + str(category).replace(".", "-") + ".pdf")
        else:
            plt.savefig(dir + f"/all_folds/{metric}/{group}/scatter_plot_" + str(category).replace(".", "-") + ".pdf")

        if gen_simple:
            ax.set_title(metric.title() + f" per {group} - DeepLabV3+", fontsize=30)
            ax.tick_params(axis='both', labelsize=20)
            ax.set_ylabel("")
            ax.set_xlabel("")
            #first_legend.set_visible(False)
            ax.legend().set_visible(False)

            plt.tight_layout()

            if fold > 0:
                plt.savefig(
                    dir + f"/fold_{fold}/{metric}/{group}/scatter_plot_" + str(category).replace(".", "-") + "_simple.pdf")
            else:
                plt.savefig(
                    dir + f"/all_folds/{metric}/{group}/scatter_plot_" + str(category).replace(".", "-") + "_simple.pdf")

        plt.close()

def draw_histogram(dataframe, y_annotated = 'annot_RER', y_predicted = 'pred_RER', dir = "./plots", metric = 'area', fold = 1, save = True):
    filtered_df = dataframe[(dataframe['metric'] == metric) & (dataframe['real'] > 0)]
    if fold > 0:
        filtered_df = filtered_df[filtered_df['fold'] == fold]
        os.makedirs(dir + f"/fold_{fold}/" + metric, exist_ok=True)
    else:
        os.makedirs(dir + "/all_folds/" + metric, exist_ok=True)
    if filtered_df.empty:
        return

    histogram_x = 'estimation'
    if 'RER' in y_annotated or 'RER' in y_predicted:
        if 'method' in y_annotated or 'method' in y_predicted:
            histogram_x = 'method RER'
        else:
            histogram_x = 'RER'

    plt.figure(figsize=(8, 6))
    if fold > 0:
        plt.title(metric.title() + f" {histogram_x} histogram fold {fold}")
    else:
        plt.title(metric.title() + f" {histogram_x} histogram all folds")

    y_annotated_RER_sorted = sorted(filtered_df[y_annotated].to_list())
    y_predicted_RER_sorted = sorted(filtered_df[y_predicted].to_list())

    if histogram_x == 'method RER':
        sns.histplot(y_annotated_RER_sorted, binwidth = 0.1, kde=True, stat='count', label='Method (Count)', color="green")
    else:
        sns.histplot(y_annotated_RER_sorted, binwidth = 0.1, kde=True, stat='count', label='Annotated (Count)', color="blue")
        sns.histplot(y_predicted_RER_sorted, binwidth = 0.1, kde=True, stat='count', label='Predicted (Count)', color="red")

    plt.xlabel(histogram_x.title())
    plt.ylabel("Count")
    plt.legend()
    if fold > 0:
        plt.savefig(dir + f"/fold_{fold}/" + metric + f"/histogram_{histogram_x.replace(' ', '_')}.pdf")
    else:
        plt.savefig(dir + "/all_folds/" + metric + f"/histogram_{histogram_x.replace(' ', '_')}.pdf")
    plt.close()

def draw_histogram_subplots(dataframe, categories, y_annotated = 'annot_RER', y_predicted = 'pred_RER', dir = "./plots", metric = 'area', fold = 1, group = 'species', save = True):
    if not categories:
        print(f"Aviso: Nenhuma categoria encontrada para o grupo '{group}' na métrica '{metric}'. Pulando plot.")
        return
    figsize = (15,6)

    if fold > 0:
        os.makedirs(dir + f"/fold_{fold}/" + metric, exist_ok=True)
    else:
        os.makedirs(dir + "/all_folds/" + metric, exist_ok=True)
        figsize = (15,10)

    ncols = 3
    nrows = math.ceil(len(categories)/ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    histogram_x = 'estimation'
    if 'RER' in y_annotated or 'RER' in y_predicted:
        if 'method' in y_annotated or 'method' in y_predicted:
            histogram_x = 'method RER'
        else:
            histogram_x = 'RER'

    if fold > 0:
        fig.suptitle(f"Histogram {histogram_x} per " + group + f" fold {fold}")
    else:
        fig.suptitle(f"Histogram {histogram_x} per " + group + " all folds")

    axes = axes.flat

    for i, category in enumerate(categories):
        filtered_df = dataframe[(dataframe[group] == category) & (dataframe['metric'] == metric) & (dataframe['real']>0)]
        if fold > 0:
            filtered_df = filtered_df[filtered_df['fold'] == fold]
        if filtered_df.empty:
            continue
        y_annotated_RER_sorted = sorted(filtered_df[y_annotated].to_list())
        y_predicted_RER_sorted = sorted(filtered_df[y_predicted].to_list())
        axes[i].set_title(category)
        if histogram_x == 'method RER':
            sns.histplot(y_annotated_RER_sorted, ax=axes[i], kde=True, stat='count', label='Method (Count)', color="green")
        else:
            sns.histplot(y_annotated_RER_sorted, ax=axes[i], kde=True, stat='count', label='Annotated (Count)', color="blue")
            sns.histplot(y_predicted_RER_sorted, ax=axes[i], kde=True, stat='count', label='Predicted (Count)', color="red")

        # plt.xlim(0, 200)
        axes[i].set_xlabel(histogram_x.title())
        axes[i].set_ylabel("Count")
        axes[i].legend()

    plt.tight_layout()
    if fold > 0:
        plt.savefig(dir + f"/fold_{fold}/" + metric + f"/histogram_{histogram_x.replace(' ', '_')}_" + group + ".pdf")
    else:
        plt.savefig(dir + "/all_folds/" + metric + f"/histogram_{histogram_x.replace(' ', '_')}_" + group + ".pdf")
    plt.close()

def draw_box_plot(dataframe, x='real', y_annotated = 'annot', y_predicted = 'pred', dir = "./plots", metric = 'area', fold = 1, save = True):
    filtered_df = dataframe[(dataframe['metric'] == metric) & (dataframe[x] > 0)]
    if fold > 0:
        filtered_df = filtered_df[filtered_df['fold'] == fold]
    if filtered_df.empty:
        return

    if fold > 0:
        os.makedirs(dir + f"/fold_{fold}/" + metric, exist_ok=True)
    else:
        os.makedirs(dir + "/all_folds/" + metric, exist_ok=True)

    plt.figure(figsize=(8,6))
    if fold > 0:
        plt.title(metric.title() + f" estimation box plot fold {fold}")
    else:
        plt.title(metric.title() + " estimation box plot all folds")

    filtered_df.boxplot(column=[x, y_annotated, y_predicted])

    plt.ylabel(metric.title() + " estimated")
    if fold > 0:
        plt.savefig(dir + f"/fold_{fold}/" + metric + "/boxplot.pdf")
    else:
        plt.savefig(dir + "/all_folds/" + metric + "/boxplot.pdf")
    plt.close()

    plt.figure(figsize=(8,6))
    if fold > 0:
        plt.title(metric.title() + f" estimation box plot fold {fold} (no outliers)")
    else:
        plt.title(metric.title() + " estimation box plot all folds (no outliers)")
    filtered_df.boxplot(column=[x, y_annotated, y_predicted], showfliers=False)

    plt.ylabel(metric.title() + " estimated")
    if fold > 0:
        plt.savefig(dir + f"/fold_{fold}/" + metric + "/boxplot_no_outlier.pdf")
    else:
        plt.savefig(dir + "/all_folds/" + metric + "/boxplot_no_outlier.pdf")
    plt.close()

def draw_box_plot_subplots(dataframe, categories, x='real', y_annotated = 'annot', y_predicted = 'pred', dir = "./plots", metric = 'area', fold = 1, group = 'species', save = True):
    figsize = (15,6)

    if fold > 0:
        os.makedirs(dir + f"/fold_{fold}/" + metric, exist_ok=True)
    else:
        os.makedirs(dir + "/all_folds/" + metric, exist_ok=True)
        figsize = (15,15)

    ncols = 3
    nrows = math.ceil(len(categories)/ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    if fold > 0:
        fig.suptitle("Box plot per " + group + f" fold {fold}")
    else:
        fig.suptitle("Box plot per " + group + f" all folds")

    axes = axes.flat

    for i, category in enumerate(categories):
        filtered_df = dataframe[(dataframe[group] == category) & (dataframe['metric'] == metric) & (dataframe['real'] > 0)]
        if fold > 0:
            filtered_df = filtered_df[filtered_df['fold'] == fold]
        if filtered_df.empty:
            continue

        filtered_df.boxplot(column=[x, y_annotated, y_predicted], ax=axes[i])

        axes[i].set_title(category)
        axes[i].set_ylabel(metric.title() + " estimated")

    plt.tight_layout()
    if fold > 0:
        plt.savefig(dir + f"/fold_{fold}/" + metric + "/boxplot_" + group + ".pdf")
    else:
        plt.savefig(dir + "/all_folds/" + metric + "/boxplot_" + group + ".pdf")
    plt.close()

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    if fold > 0:
        fig.suptitle("Box plot per " + group + f" fold {fold}")
    else:
        fig.suptitle("Box plot per " + group + f" all folds")

    axes = axes.flat

    for i, category in enumerate(categories):
        filtered_df = dataframe[
            (dataframe[group] == category) & (dataframe['metric'] == metric) & (dataframe['real'] > 0)]

        filtered_df.boxplot(column=[x, y_annotated, y_predicted], showfliers=False, ax=axes[i])

        axes[i].set_title(category)
        axes[i].set_ylabel(metric.title() + " estimated")

    plt.tight_layout()
    if fold > 0:
        plt.savefig(dir + f"/fold_{fold}/" + metric + "/boxplot_" + group +"_no_outlier.pdf")
    else:
        plt.savefig(dir + "/all_folds/" + metric + "/boxplot_" + group +"_no_outlier.pdf")
    plt.close()

# def save_statistics(x, y_annotated, y_predicted, dir = "./plots", metric = "area", fold = 0, category = "all"):
#     metric_index = {"area": 0, "perimeter": 1, "length": 2}
#     index = metric_index[metric]

#     ((m_a, b_a), (m_p, b_p)) = draw_linear_regression(x, y_annotated, y_predicted, dir=dir, metric=metric, fold=fold)

#     os.makedirs(dir + f"/fold_{fold+1}/" + metric, exist_ok=True)

#     mean, median, variance, standard_deviantion = compute_statistics(y_annotated[category][0])

#     print(f"Average (Annotation): mean = {mean}, median = {median}, sd = {standard_deviantion}")

#     r_squared = r2_score(x[category][0], y_annotated[category][0])
#     print(f"R squared (Anottation): {r_squared}")

#     with open(dir + "/statistics.csv", "a", newline='') as csvfile:
#         spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
#         spamwriter.writerow([fold+1, metric.title(), "Annotation", m_a, b_a, r_squared, mean, median, variance, standard_deviantion])

#     mean, median, variance, standard_deviantion = compute_statistics(y_predicted[category][0])

#     print(f"Average (Prediction): mean = {mean}, median = {median}, sd = {standard_deviantion}")

#     r_squared = r2_score(x[category][0], y_predicted[category][0])
#     print(f"R squared (Prediction): {r_squared}")

#     with open(dir + "/statistics.csv", "a", newline='') as csvfile:
#         spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
#         spamwriter.writerow([fold+1, category, metric.title(), "Prediction", m_p, b_p, r_squared, mean, median, variance, standard_deviantion])


with open(f"./plots/statistics_fix.csv", "w", newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    # spamwriter.writerow(["Fold", "Metric", "Type of mask", "m", "b", "R2", "Mean", "Median", "Variance", "Standard deviantion"])
    spamwriter.writerow(["Fold", "Category", "Metric", "Type of mask", "m", "b", "R2", "Mean", "Median", "Variance", "Standard deviantion", "RER Mean", "RER Median", "RER Standard deviation"])



df = pd.read_csv("dados_pra_grafico.csv")

width_df = df[df["metric"] == "width"]

print("Min:", width_df["annot_RER"].min())
print("Max:", width_df["annot_RER"].max())
print("Faixa:", width_df["annot_RER"].max() - width_df["annot_RER"].min())

print((width_df["annot_RER"] == 0).sum())
print(len(width_df))


for fold in sorted(df["fold"].unique()):
    if parser.parse_args().per_fold:
        os.makedirs(f"./plots/fold_{fold+1}", exist_ok=True)
        print(f"-------------Fold {fold+1}------------")
        metrics = ['area', 'perimeter', 'length', 'width']
        groups = ['species', 'dist', 'pattern_size']

        for metric in metrics:
            draw_scatter_plot(df, metric=metric, fold = fold+1, gen_simple=True)
            draw_histogram(df, metric=metric, fold = fold+1)
            draw_histogram(df, y_annotated='annot', y_predicted='pred', metric=metric, fold = fold+1)
            draw_histogram(df, y_annotated='method_RER', y_predicted='method_RER', metric=metric, fold = fold+1)
            draw_box_plot(df, metric=metric, fold = fold+1)
            for group in groups:
                fold_group = df.loc[df['real'] > 0][group].unique().tolist()
                draw_multiple_scatter_plots(df, fold_group, metric=metric, group=group, fold = fold+1, gen_simple=True)
                draw_histogram_subplots(df, fold_group, metric=metric, group=group, fold = fold+1)
                draw_histogram_subplots(df, fold_group, y_annotated='annot', y_predicted='pred', metric=metric, group=group, fold = fold+1)
                draw_histogram_subplots(df, fold_group, y_annotated='method_RER', y_predicted='method_RER', metric=metric, group=group, fold = fold+1)
                draw_box_plot_subplots(df, fold_group, metric=metric, group=group, fold = fold+1)

# General
print("-----------General----------")

metrics = ['area', 'perimeter', 'length', 'width']
groups = ['species', 'dist', 'pattern_size']

for metric in metrics:
    draw_scatter_plot(df, metric=metric, fold = 0, gen_simple=True)
    draw_histogram(df, metric=metric, fold = 0)
    draw_histogram(df, y_annotated='annot', y_predicted='pred', metric=metric, fold = 0)
    draw_histogram(df, y_annotated='method_RER', y_predicted='method_RER', metric=metric, fold = 0)
    draw_box_plot(df, metric=metric, fold = 0)
    for group in groups:
        fold_group = df.loc[df['real'] > 0][group].unique().tolist()
        draw_multiple_scatter_plots(df, fold_group, metric=metric, group=group, fold = 0, gen_simple=True)
        draw_histogram_subplots(df, fold_group, metric=metric, group=group, fold = 0)
        draw_histogram_subplots(df, fold_group, y_annotated='annot', y_predicted='pred', metric=metric, group=group, fold = 0)
        draw_histogram_subplots(df, fold_group, y_annotated='method_RER', y_predicted='method_RER', metric=metric, group=group, fold = 0)
        draw_box_plot_subplots(df, fold_group, metric=metric, group=group, fold = 0)