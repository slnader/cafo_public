# ==============================================================================
# Figure 2: ROC and PR Curves
# ==============================================================================

#Required modules (condatensorflow environment)
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
from sklearn import metrics as skmetrics
from sklearn.utils.class_weight import compute_sample_weight

def main():

    # ==========================================================================
    #0. Read in image accuracy data (prepared in build_image_level_data.py)
    # ==========================================================================

    #Data directory
    data_dir = '../data/csv'

    #Table directory
    fig_dir = '../figures'

    #Read in image accuracy data
    accuracy_df = pd.read_csv(os.path.join(data_dir,
    "test_image_accuracy_data.csv"))

    #Final poultry model
    poultry_model_version = 'final_poultry_model'

    #Final swine model
    swine_model_version = 'final_swine_model'

    #Set seed
    random.seed(94063)
    # ==========================================================================
    #1. Plot curves
    # ==========================================================================

    print("Creating AUC plots...")

    #Set up plot dimensions
    fig, axarr = plt.subplots(nrows=2, ncols=2)
    fig.set_size_inches(7,5)
    lw = 2
    fig.tight_layout()

    #Name models
    model_names = ['Poultry', 'Swine']
    model_versions = [poultry_model_version, swine_model_version]

    #Cafo proportion thresholds
    cafo_proportions = [[0,1], [1]]
    proportion_desc = ['All', 'Non-Occluded']
    linestyles = ['-', '--', '-.', ':']
    line_colors = ['#feb24c','#f03b20']

    #Calculate AUC
    for i in range(len(model_names)):

        for j in range(len(cafo_proportions)):

            #Description of image sample
            this_desc = proportion_desc[j]

            #Select data for curve
            roc_data =  accuracy_df.loc[(
            (accuracy_df.model_version==model_versions[i]) &
            ((accuracy_df.cafo_threshold.isin(cafo_proportions[j])) |
            (accuracy_df.image_class_cat=='notcafo'))),]

            #ROC curve
            fpr, tpr, thresholds = skmetrics.roc_curve(
            roc_data[['image_class_binary']], roc_data[['score']])

            #Plot curves
            axarr[i][0].plot(fpr, tpr, linestyle = linestyles[j],
            label = "%s"%(this_desc),
                         color = line_colors[j])
            axarr[i][0].legend(loc="lower right",
            title = "Test Sample").get_frame().set_linewidth(0.0)
            axarr[i][0].plot([0,1], [0,1], clip_on=True, scalex=False,
            scaley=False, color = '#bdbdbd')

            #PR curve
            precision,recall,thresholds = skmetrics.precision_recall_curve(
            roc_data[['image_class_binary']], roc_data[['score']])

            #Plot curve
            axarr[i][1].plot(recall, precision, linestyle = linestyles[j],
            label = "%s"%(this_desc), color = line_colors[j])

            #Junk classifier comparison
            random_scores = [random.random() for _ in range(len(roc_data))]
            rand_precision,rand_recall,thresholds = skmetrics.precision_recall_curve(
            roc_data[['image_class_binary']], random_scores)
            axarr[i][1].plot(rand_recall, rand_precision,
            color = '#bdbdbd', linestyle = linestyles[j])

    #Change axis and set titles
    axarr[0][0].set_title('ROC Curve')
    axarr[0][1].set_title('PR Curve')
    axarr.flat[0].set_xticklabels([0]+[1-x*1.0/100 for x in range(0,120,20)])
    axarr.flat[2].set_xticklabels([0]+[1-x*1.0/100 for x in range(0,120,20)])

    axarr.flat[0].set(ylabel='Poultry \n Sensitivity')
    axarr.flat[1].set(ylabel='Precision')
    axarr.flat[2].set(xlabel='Specificity', ylabel=' Swine \n Sensitivity')
    axarr.flat[3].set(xlabel='Sensitivity (Recall)', ylabel='Precision')

    #Label panels
    plt.figtext(0.07, 0.93, 'a', fontweight = 'bold')
    plt.figtext(0.07, 0.44, 'b', fontweight = 'bold')
    plt.figtext(0.56, 0.93, 'c', fontweight = 'bold')
    plt.figtext(0.56, 0.44, 'd', fontweight = 'bold')

    #Save figure
    fig.savefig(os.path.join(fig_dir, '2_Figure_fgroc_ROC_PR_Curves.pdf'),
    bbox_inches='tight')

    print("Done.")

#Add arguments
if __name__ == '__main__':
    main()
