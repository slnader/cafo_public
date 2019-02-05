# ==============================================================================
# Figure 3: CAM Algorithm Illustration
# ==============================================================================

import os
import pandas as pd
import numpy as np
import cv2
from PIL import Image as PILImage
from PIL import ImageDraw
from PIL import ImageOps
from PIL import ImageFont
from PIL import ImageEnhance
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
pd.options.mode.chained_assignment = None 

# ==============================================================================
#0. CAM Functions
# ==============================================================================

def generate_cam(conv, weights, class_idx, size_upsample = (299,299)):
    """
    Function to generate class activation map for inception model 

    Keyword Arguments:
    conv = last convolutional layer of cnn (1x8x8x2048 in inception)
    weights = matrix of weights for each class (nx2048)
    class_idx = index of class to generate cam for
    size_upsample = tuple of final image size in pixels
    
    Returns:
    class activation map pixel array (299x299)
    """
    #Get shape of conv layer
    _,h,w,f = conv.shape
    
    #Reshape
    feature_input = conv.reshape(h*w,f).transpose()
    
    #Multiply matrix with weights for class
    cam = weights[class_idx].dot(feature_input)
    
    #Sample the result up to 299x299 image size
    #size_upsample = (299,299)
    cam = cam.reshape(h, w)
    min_cam = np.min(cam)
    max_cam = np.max(cam)
    #min_cam = -10 #lowest value for image with only trees
    #max_cam = 25 #highest value for image with cafo
    cam = cam - min_cam
    cam_img = cam / max_cam
    cam_img = np.uint8(255 * cam_img)
    final_img = cv2.resize(cam_img, size_upsample)

    return final_img

def generate_poly(cam_array, threshold):
    """
    Function to obtain bounding box coordinates based on CAM

    Keyword Arguments:
    cam_array = 299x299 matrix of cam values
    threshold = value between 0 and 1 to define bounding box as percentage of max cam value
    
    Returns:
    top left and bottom right coordinates of bounding box
    """
    #Get max cam
    max_cam = np.max(cam_array)
    
    #Define value threshold
    bb_threshold = threshold*max_cam
    
    #Identify pixel locations that meet threshold
    point_array = []
    
    for i in range(cam_array.shape[0]):
        for j in range(cam_array.shape[1]):
            if cam_array[i][j] >= bb_threshold:
                point_array.append((j,i))
                
    return point_array

def get_pixel_clusters(poly, distance_threshold=150, start_clusters=5):
    """
    Function to obtain ordered array of polygon outline for visualization purposes

    Keyword Arguments:
    poly = unordered array of pixel location tuples (x,y) 
    distance_threshold = distance in meters that we require clusters to be separated by
    start_clusters = integer number of clusters to initially create with k-means
    
    Returns:
    ordered array of polygon outline for each cluster
    """
        
    #Run k-means with five centers (estimated max possible for image size)
    kmeans = KMeans(n_clusters=start_clusters, random_state=94063).fit(poly)

    #Get centers
    kmeans_centers = kmeans.cluster_centers_

    #Get distances between all centers
    center_distances = euclidean_distances(kmeans_centers)

    #Prune clusters based on 100m distance threshold
    possible_clusters = range(len(center_distances))
    clusters_removed = []
    clusters_to_keep = []

    for c in possible_clusters:
        separated_clusters = np.argwhere((center_distances[c]> 0) & (center_distances[c] < distance_threshold))
        clusters_to_remove = [x[0] for x in separated_clusters]
        already_removed = [x in clusters_removed for x in clusters_to_remove]

        if c in clusters_removed:
            clusters_removed = clusters_removed + clusters_to_remove
        elif len(already_removed) > 0:
            if np.max(already_removed)==True:
                clusters_removed = clusters_removed + clusters_to_remove
            else:
                clusters_to_keep.append(c)
                clusters_removed = clusters_removed + clusters_to_remove
        else:
            clusters_to_keep.append(c)
            clusters_removed = clusters_removed + clusters_to_remove
            
    #Rerun kmeans with appropriate number of clusters
    kmeans = KMeans(n_clusters=len(clusters_to_keep), random_state=94063).fit(poly)

    return kmeans.cluster_centers_, kmeans.labels_

def generate_poly_outline(poly, cluster_labels):
    """
    Function to obtain ordered array of polygon outline for visualization purposes

    Keyword Arguments:
    poly = unordered array of pixel location tuples (x,y) 
    cluster_labels = labels for which cluster pixel belongs to
    
    Returns:
    ordered array of polygon outline for each cluster
    """

    #Append clusters to points
    point_clusters = zip(cluster_labels, poly)

    #Draw polygons separately for each cluster
    ordered_array_list = []
    area_list = []
    
    for this_cluster in list(set(cluster_labels)):

        #Points in this cluster
        poly_points = [x[1] for x in point_clusters if x[0]==this_cluster]
        
        #Count for area
        poly_area = len(poly_points)
        
        #All unique x and y dims
        j_dim = list(set([x[1] for x in poly_points]))
        i_dim = list(set([x[0] for x in poly_points]))

        #Initialize ordered array
        ordered_array = []

        #Get maximum y dim for each x dim
        for i in i_dim:
            pixels_dim = [x for x in poly_points if x[0]==i]
            max_j = np.max([x[1] for x in pixels_dim])
            if i == np.min(i_dim):
                ordered_array.append((i, np.min([x[1] for x in pixels_dim])))
            ordered_array.append((i, max_j))

        #Get minimum y dim for each x dim in reverse order
        for i in sorted(i_dim, reverse=True):
            pixels_dim = [x for x in poly_points if x[0]==i]
            min_j = np.min([x[1] for x in pixels_dim])
            ordered_array.append((i, min_j))

        ordered_array_list.append(ordered_array)
        area_list.append(poly_area)

    return ordered_array_list, area_list

def run_cam_loop(image_path,  sess, start_clusters = 5):
    """
    Function to calculate pixel activations for images in a for loop
    
    Keyword Arguments:
    image_path = full path to image location
    sess = tensorflow Session
    start_clusters = integer number of clusters to initially create with k-means
    
    Returns:
    pixel activations, k means cluster centroids, k means labels 
    """
    
    #Read in image
    image_data = gfile.FastGFile(image_path, 'rb').read()

    #Get last layer of network
    feature_conv = sess.run(
      'import/mixed_10/join:0',
      {'import/DecodeJpeg/contents:0': image_data})

    #Get final weights
    weight_softmax = sess.graph.get_tensor_by_name('import/final_training_ops/weights/final_weights:0').eval()
    weight_softmax = weight_softmax.transpose()

    #Get cam
    final_result = generate_cam(feature_conv, weight_softmax, 1)

    #Get bbox
    activated_pixels = generate_poly(final_result, 0.5)
    pixel_clusters, pixel_labels = get_pixel_clusters(activated_pixels, 150, start_clusters)
    
    return activated_pixels, pixel_clusters, pixel_labels, final_result


def main():

    # ==============================================================================
    #1. Setup, file paths
    # ==============================================================================

    #Final poultry model
    poultry_model_version = 'final_poultry_model'

    #Final swine model
    swine_model_version = 'final_swine_model'

    #Directory for data
    data_dir = '../data'

    #Image directory
    image_dir = os.path.join(data_dir, 'images', 'fig3')

    #Figure directory
    fig_dir = '../figures'

    #Model directory
    model_dir = '../models'

    # Paths to saved models 
    poultry_graph_path = os.path.join(model_dir, poultry_model_version+'.pb')
    swine_graph_path = os.path.join(model_dir, swine_model_version+'.pb')

    #Location of CAM results
    cam_results_csv = os.path.join(data_dir, 'csv', 'cam_results.csv')
    score_csv = os.path.join(data_dir, 'csv', 'rescored_images.csv')

    #PIL fonts
    bold_font = ImageFont.truetype(os.path.join(data_dir, 'fonts/DejaVuSans-Bold.ttf'), 16)
    font = ImageFont.truetype(os.path.join(data_dir, 'fonts/DejaVuSans.ttf'), 16)

    #Image of interest
    image_id = 'north-carolina_duplin_11801_287_6_1_0_18_-944_13554'

    # ==============================================================================
    #2. Load model graphs
    # ==============================================================================

    print("Loading model graphs...")

    #Reset graph params
    tf.reset_default_graph()
    poultry_graph = tf.Graph()
    swine_graph = tf.Graph()

    # Load in poultry model
    with poultry_graph.as_default():
        with tf.gfile.FastGFile(poultry_graph_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def)

    #Load in swine model
    with swine_graph.as_default():
        with tf.gfile.FastGFile(swine_graph_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def)    


    # ==============================================================================
    #3. Prepare CAM results for plotting
    # ==============================================================================

    print("Loading CAM results...")
    
    #CAM results
    square_footage_df = pd.read_csv(cam_results_csv, header = None, 
        names = ['image_id', 'cluster', 'center', 'easting', 'northing', 'zone', 'square_footage','activation_bounds',
                'pixel_cluster_x', 'pixel_cluster_y'])

    #Add image cluster id
    square_footage_df['image_cluster_id'] = [str(square_footage_df['image_id'][i]) +'_'+ str(square_footage_df['cluster'][i]) + '.jpeg' for i in range(len(square_footage_df))]

    #Re-scored images
    new_score_df = pd.read_csv(score_csv, header = None, names = ['image_cluster_id', 'image_id', 'poultry_score', 'swine_score'])

    #Merge together
    cam_merge = square_footage_df.merge(new_score_df, on = 'image_cluster_id')

    # ==============================================================================
    #4. Score original image
    # ==============================================================================

    print("Scoring original image...")

    #Get path to original image
    img_path = os.path.join(image_dir, image_id + '.jpeg')

    #Read in image
    image_data = gfile.FastGFile(img_path, 'rb').read()

    with tf.Session(graph=poultry_graph) as sess:
        #Get final weights
        softmax_tensor = sess.graph.get_tensor_by_name('import/final_result:0')

        #Get model prediction for image
        poultry_predictions = sess.run(softmax_tensor, \
                     {'import/DecodeJpeg/contents:0': image_data,
                     'import/final_training_ops/dropout/Placeholder:0':1.0})

        poultry_score = poultry_predictions[0][1]

    with tf.Session(graph=swine_graph) as sess:
        #Get final weights
        softmax_tensor = sess.graph.get_tensor_by_name('import/final_result:0')

        #Get model prediction for image
        swine_predictions = sess.run(softmax_tensor, \
                     {'import/DecodeJpeg/contents:0': image_data,
                     'import/final_training_ops/dropout/Placeholder:0':1.0})

        swine_score = swine_predictions[0][1]
        
    #Identify new scores on recentered image
    new_poultry_score = cam_merge.loc[(cam_merge.image_id_x==image_id),'poultry_score'].values[0]
    new_swine_score = cam_merge.loc[(cam_merge.image_id_x==image_id),'swine_score'].values[0]


    # ==============================================================================
    #5. Run poultry and swine CAMs on original image
    # ==============================================================================

    print("Re-running poultry and swine CAMs on image...")

    #Run Poultry CAM on original image
    with tf.Session(graph=poultry_graph) as sess:

        #Get cam results
        activated_pixels, pixel_clusters, pixel_labels, final_result = run_cam_loop(img_path, sess)

    ##Combine with original image
    img = cv2.imread(img_path)
    heatmap = cv2.applyColorMap(final_result, cv2.COLORMAP_JET)
    result_img = heatmap * 0.3 + img * 0.5

    #Path to results
    outpath = os.path.join(image_dir, image_id + '_poultry.jpeg')

    #Write and read in 
    cv2.imwrite(outpath, result_img)

    #Path to recentered image
    img_path = os.path.join(image_dir, image_id + '_0.jpeg')

    #Run Swine CAM on recentered image
    with tf.Session(graph=swine_graph) as sess:

        #Get cam results
        activated_pixels, pixel_clusters, pixel_labels, final_result = run_cam_loop(img_path, sess)

    ##Combine with original image
    img = cv2.imread(img_path)
    heatmap = cv2.applyColorMap(final_result, cv2.COLORMAP_JET)
    result_img = heatmap * 0.3 + img * 0.5

    #Path to results
    outpath = os.path.join(image_dir, image_id + '_swine.jpeg')

    #Write and read in 
    cv2.imwrite(outpath, result_img)

    # ==============================================================================
    #5. Create full image
    # ==============================================================================

    print("Compiling figure...")

    #Initialize image
    new_im = PILImage.new('RGB', (299*2, 299*2), color = (255,255,255))
       
    #Original image
    org_im = PILImage.open(os.path.join(image_dir, image_id + '.jpeg')).convert('RGB')
    draw_image = ImageDraw.Draw(org_im)
    draw_image.text((0,0), 'a', 'white', font=bold_font)
    new_im.paste(org_im, (0,0))

    #Poultry activation, original image
    org_act = PILImage.open(os.path.join(image_dir, image_id + '_poultry.jpeg')).convert('RGB')
    draw_image = ImageDraw.Draw(org_act)
    draw_image.text((0,0), 'b', 'white', font=bold_font)
    draw_image.text((120,10), 'Poultry score: ' + '{0:1.2f}'.format(poultry_score), 'white', font=bold_font)
    draw_image.text((130,30), 'Swine score: ' + '{0:1.2f}'.format(swine_score), 'white', font=bold_font)
    new_im.paste(org_act, (299,0))

    #Centered image
    cent_im = PILImage.open(os.path.join(image_dir, image_id + '_0.jpeg')).convert('RGB')
    draw_image = ImageDraw.Draw(cent_im)
    draw_image.text((0,0), 'c', 'white', font=bold_font)
    new_im.paste(cent_im, (0,299))

    #Swine activation centered image
    cent_act = PILImage.open(os.path.join(image_dir, image_id + '_swine.jpeg')).convert('RGB')
    draw_image = ImageDraw.Draw(cent_act)
    draw_image.text((0,0), 'd', 'white',font=bold_font)
    draw_image.text((120,10), 'Poultry score: ' + '{0:1.2f}'.format(new_poultry_score), 'white', font=bold_font)
    draw_image.text((130,30), 'Swine score: ' + '{0:1.2f}'.format(new_swine_score), 'white', font=bold_font)
    new_im.paste(cent_act, (299,299))

    #Save image
    new_im.save(os.path.join(fig_dir, '3_Figure_fgcam_CAM_Algorithm_Illustration.png'), dpi=(300,300))

    print("Done.")

#Add arguments
if __name__ == '__main__':
    main() 


