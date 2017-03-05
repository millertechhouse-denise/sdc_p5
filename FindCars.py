import helper_functions as hp
import training as train
from sklearn.externals import joblib
import matplotlib.image as mpimg
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
import glob

svc = None
X_scaler = None


    
# Define a single function that can extract features using hog sub-sampling and make predictions
def process_image(img):
    ystart = 400
    ystop = 650
    scale = 1.5
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    spatial_size = 32
    hist_bins = 32
    
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = hp.convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1 
    nfeat_per_block = orient*cell_per_block**2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1 
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = hp.get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = hp.get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = hp.get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    bboxes = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = hp.bin_spatial(subimg, size=(spatial_size,spatial_size))
            hist_features = hp.color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                bboxes.append(((xbox_left, ytop_draw+ystart), (xbox_left+win_draw,ytop_draw+win_draw+ystart)))
                #cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
    
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    
    # Add heat to each box in box list
    heat = hp.add_heat(heat,bboxes)
    
    # Apply threshold to help remove false positives
    heat = hp.apply_threshold(heat,1)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    out_img = hp.draw_labeled_bboxes(np.copy(draw_img), labels)
                
    return out_img
                            
        
    
if __name__ == '__main__':
    
    global svc
    global X_scaler
    svc = joblib.load('saved_svc.pickle')
    X_scaler = joblib.load('saved_scalar.pickle')
    
    image = mpimg.imread('./test_images/test1.jpg')
    
    
    #process_image(image, clf)
    image_files = glob.glob('./test_images/*.jpg')
    
    i = 0
    for fname in image_files:
        #
        image = mpimg.imread(fname)
        new_image = process_image(image)
        plt.imshow(new_image)
        plt.show()
    
    #iterate through each window
    #predict if car found
    #draw square for each car found
    #create heatmap
    #draw sqare outside heat map
    #print('My SVC predicts: ', clf.predict(X_test[0:n_predict]))
    

    white_output = 'output_images/test_video.mp4'
    clip1 = VideoFileClip("project_video.mp4")
   # clip2 = VideoFileClip("challenge_video.mp4")
    white_clip = clip1.fl_image(process_image)
    white_clip.write_videofile(white_output, audio=False)
