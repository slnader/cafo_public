# ==============================================================================
#Facility consolidation algorithm
#This script combines CAFO objects into facilities, discarding false positives 
#found by recentering and rescoring
# ==============================================================================
import os
import pandas as pd
import numpy as np
import csv
from sklearn.metrics.pairwise import euclidean_distances
import utm
from datetime import datetime
pd.options.mode.chained_assignment = None 

def main():

    # ==============================================================================
    # Setup - path names
    # ==============================================================================

    #Directory for results
    data_dir = '../data'

    #Results csv
    results_csv = os.path.join(data_dir, 'csv', 'cam_results.csv')  

    #Rescored images
    score_csv = os.path.join(data_dir, 'csv', 'rescored_images.csv')

    #Read in result files
    square_footage_df = pd.read_csv(results_csv, header = None, 
        names = ['image_id', 'cluster', 'center', 'easting', 'northing', 'zone', 
        'square_footage','activation_bounds','pixel_center_x','pixel_center_y'])

    #Rescored centered images
    new_score_df = pd.read_csv(score_csv, header = None,
        names = ['image_cluster_id', 'image_id', 'poultry_score', 'swine_score'])

    #Score threshold for classifying images as CAFOs
    score_threshold = 0.5

    # ==============================================================================
    # Prep data
    # ==============================================================================

    #Add new poultry classification
    new_score_df['new_score_class'] = 'notcafo'
    new_score_df.loc[(((new_score_df.poultry_score >= score_threshold) | (new_score_df.swine_score >= score_threshold)) & (new_score_df.poultry_score > new_score_df.swine_score)), 'new_score_class'] = 'poultry cafo'
    new_score_df.loc[(((new_score_df.poultry_score >= score_threshold) | (new_score_df.swine_score >= score_threshold)) & (new_score_df.poultry_score <= new_score_df.swine_score)), 'new_score_class'] = 'swine cafo'
    
    #Add image cluster id
    square_footage_df['image_cluster_id'] = [str(square_footage_df['image_id'][i]) +'_'+ str(square_footage_df['cluster'][i]) + '.jpeg' for i in range(len(square_footage_df))]

    #Convert strings to arrays
    square_footage_df.activation_bounds = [x.replace("[", "").replace("]", "").replace(" ", "") for x in square_footage_df.activation_bounds]
    square_footage_df.activation_bounds = [x.split(",") for x in square_footage_df.activation_bounds.values]

    #Convert strings to integers within arrays
    new_list = []
    for arr in square_footage_df.activation_bounds:
        new_arr = [int(x) for x in arr]
        new_list.append(new_arr)
    square_footage_df.activation_bounds = new_list

    #Add county name
    square_footage_df["county_name"] = [x.split('_')[1] for x in square_footage_df.image_id]

    #Merge to new classification
    square_footage_df = square_footage_df.merge(new_score_df[['image_cluster_id', 'new_score_class', 'poultry_score']],
                                           on = 'image_cluster_id', how = 'left')

    #Add boundary image flag (does the object touch the edge of the image)
    square_footage_df['min_bounds'] = [np.min(x) for x in square_footage_df.activation_bounds.values]
    square_footage_df['max_bounds'] = [np.max(x) for x in square_footage_df.activation_bounds.values]
    square_footage_df['boundary_image'] = 0
    square_footage_df.loc[((square_footage_df.min_bounds==0) | (square_footage_df.max_bounds>=298)), 'boundary_image'] = 1

    #Print how many objects to process
    print("%d centroids to process."%(len(square_footage_df[1:])))

    #Remove edge cases in UTM zone 16 (1 boundary image)
    square_footage_df = square_footage_df.loc[(square_footage_df.zone!=16),]

    # ==============================================================================
    # Run facility creation algorithm
    # ==============================================================================

    #Initialize starting values
    all_results_df = pd.DataFrame()
    distance_threshold = 250
    square_footage_df['image_group'] = ''

    #Consolidate within each UTM zone
    for this_zone in list(set(square_footage_df.zone.values)):

        #Get data for zone
        this_zone_data = square_footage_df.loc[(square_footage_df.zone==this_zone), ]
        min_distances = [0]
        j = 0

        while np.min(min_distances) <= distance_threshold:

            print("Round %d for utm zone %d: minimum distance between centroids is %f."%(j, this_zone, np.min(min_distances)))
            #Reset index and calculate distances
            this_zone_data = this_zone_data.reset_index(drop=True)
            this_zone_distances = euclidean_distances(this_zone_data[['easting', 'northing']])
            
            #Get min distance 
            distances_no_diagonal = [np.delete(this_zone_distances[i],i) for i in range(len(this_zone_distances))]
            min_distances = [np.min(x) for x in distances_no_diagonal]

            #Initialize results
            result_df = pd.DataFrame(columns = list(square_footage_df)+ ['is_poultry'])
            reviewed_indices = []

            for i in range(len(this_zone_distances)):

                if i in reviewed_indices:
                    continue

                distance_vec = this_zone_distances[i]

                if min_distances[i] <= distance_threshold:
                    #Getting matching pair
                    pair_match = np.argwhere(distance_vec == min_distances[i])
                    pair_match = np.delete(pair_match, np.argwhere(pair_match==i))

                    #Sort by square footage
                    cluster_indices = list(pair_match)+[i]
                    pair_data = this_zone_data.iloc[(this_zone_data.index.isin(cluster_indices)),]
                    pair_data = pair_data.sort_values(['poultry_score', 'boundary_image', 'square_footage'], ascending = [False, True, False])
                    
                    #Get image ids in cluster
                    cluster_images = pair_data.image_cluster_id.values
                    
                    #Determine if either image remained poultry
                    is_poultry = np.max(pair_data.new_score_class=='poultry cafo')
                   
                    #Calculate average centroid
                    if is_poultry:
                        average_easting = np.mean(pair_data.loc[(pair_data.new_score_class=='poultry cafo'), 'easting'])
                        average_northing = np.mean(pair_data.loc[(pair_data.new_score_class=='poultry cafo'), 'northing'])      
                    else:
                        average_easting = np.mean(pair_data.easting)
                        average_northing = np.mean(pair_data.northing)

                    #Keep largest area
                    pair_data = pair_data.iloc[0]

                    #Replace centroid with average centroid
                    pair_data['easting'] = average_easting
                    pair_data['northing'] = average_northing
                
                    #Label poultry
                    pair_data['is_poultry'] = int(is_poultry)
                    
                    #Append to dataframe
                    result_df = result_df.append(pair_data)
                    
                    #Update reviewed indices
                    reviewed_indices = reviewed_indices + cluster_indices
                    
                    #Update square footage df
                    square_footage_df.loc[(square_footage_df.image_cluster_id.isin(cluster_images)), 'image_group'] = pair_data['image_id']

                else:

                    #If no pair, add object to list
                    pair_data = this_zone_data.iloc[i]
                    is_poultry = np.max(pair_data.new_score_class=='poultry cafo')
                    pair_data['is_poultry'] = int(is_poultry)
                    result_df = result_df.append(pair_data)
                    reviewed_indices.append(i)
                    
                    #Update square footage df
                    square_footage_df.loc[(square_footage_df.image_cluster_id.isin([pair_data['image_cluster_id']])), 'image_group'] = pair_data['image_id']

            #Add zone data to results            
            this_zone_data  = result_df

            #Increment counter
            j = j+1
            
        #Add to results
        all_results_df = all_results_df.append(result_df)


    #Add latitude and longitude
    all_results_df = all_results_df.reset_index(drop=True)
    all_results_df['latitude'] = None
    all_results_df['longitude'] = None

    #Convert utm coords to lat lon coords
    for i in range(len(all_results_df)):
        geo_coords = utm.to_latlon(all_results_df.easting.values[i], 
                                   all_results_df.northing.values[i], 
                                   all_results_df.zone.values[i], 'S')
        all_results_df.loc[i, 'latitude'] = geo_coords[0]
        all_results_df.loc[i, 'longitude'] = geo_coords[1]

    #Create facility id
    all_results_df['facility_id'] = all_results_df['image_id'] + '_' + all_results_df['cluster'].map(str) + all_results_df['center'].map(str)

    #Drop non-poultry objects 
    all_results_df = all_results_df.loc[((all_results_df.is_poultry==1) &
                                    (all_results_df.poultry_score>=score_threshold)),]

    #Keep required columns
    all_results_df= all_results_df[['facility_id', 'latitude', 'longitude', 'square_footage']]

    #Write out predicted positives to csv 
    all_results_df.to_csv(os.path.join(data_dir, 'csv', 'consolidated_facilities.csv'),
                     index = False)

    print("Done.")

if __name__ == '__main__':
    main() 

