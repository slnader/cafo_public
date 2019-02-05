# ==============================================================================
# Table 1: Image Accuracy Statistics
# ==============================================================================

#Required modules (condatensorflow environment)
import os
import psycopg2
import pandas as pd
import numpy as np
import getpass
import random
from sklearn import metrics as skmetrics
from sklearn.utils.class_weight import compute_sample_weight

def main():

    # ==============================================================================
    #0. Read in image accuracy data (prepared in build_image_level_data.py)
    # ==============================================================================
  
    #Data directory
    data_dir = '../data/csv'

    #Table directory
    tab_dir = '../tables'

    #Read in image accuracy data
    accuracy_df = pd.read_csv(os.path.join(data_dir, "test_image_accuracy_data.csv"))

    #Final poultry model
    poultry_model_version = 'final_poultry_model'

    #Final swine model
    swine_model_version = 'final_swine_model'

    # ==============================================================================
    #1. Calculate accuracy metrics for each model
    # ==============================================================================

    print("Creating image accuracy table...")

    all_results = []

    #For the poultry and swine models separately
    for this_model in [poultry_model_version, swine_model_version]:

        #With and without excluding corner images
        for this_threshold in [[0,1], [1]]:
            
            #Define model classes
            if 'poultry' in this_model:
                model_classes = ['poultry cafo', 'notcafo']
            else:
                model_classes = ['swine cafo', 'notcafo']

            #Define sample type
            if np.min(this_threshold)==0:
                sample_type = 'All images'
            else:
                sample_type = 'Non-Occluded'

            #Identify model test data
            model_data_set = accuracy_df.loc[((accuracy_df.model_version==this_model) &
                                              ((accuracy_df.cafo_threshold.isin(this_threshold)) | 
                                               (accuracy_df.image_class_cat=='notcafo'))
                                             ),]
            
            #Area under ROC curve
            this_auc = skmetrics.roc_auc_score(model_data_set[['image_class_binary']], 
                          model_data_set[['score']])
            
            
            #Precision and recall
            average_precision = skmetrics.average_precision_score(model_data_set[['image_class_binary']], 
                              model_data_set[['score']])
            
            #Save results
            result_tuple = (this_model, 
                            sample_type,
                            np.round(this_auc,3), 
                            np.round(average_precision,3), 
                            len(model_data_set.loc[(model_data_set.image_class_cat==model_classes[0]),]),
                            len(model_data_set.loc[(model_data_set.image_class_cat==model_classes[1]),])
                           )
            all_results.append(result_tuple)

    #Convert results to tabular
    tab_results = pd.DataFrame(all_results, columns = ['model', 'sample', 'auc', 'avg_precision', 'n_1', 'n_0'])

    #Convert to friendly names
    tab_results.loc[(tab_results.model==poultry_model_version), 'model'] = 'Poultry'
    tab_results.loc[(tab_results.model==swine_model_version), 'model'] = 'Swine'
 
    #Write to csv
    tab_results.to_csv(os.path.join(tab_dir, "1_Table_tbimacc_image_accuracy_statistics.csv"), index = False)


    print("Done.")

#Add arguments
if __name__ == '__main__':
    main() 

