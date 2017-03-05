import helper_functions as hp
import training as train
from sklearn.externals import joblib
import matplotlib.image as mpimg
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import cv2

def process_image(image, clf):
    
    spatial = 50
    histbin = 24
    
    scaler = 1
    
    color_space = 'HLS'
    
    #convert to HLS
    ctrans_tosearch = convert_color(img_tosearch, color_space)
    
    #convert entire image to HOG
    feature_image[:,:,hog_channel]
    orient, 
    pix_per_cell
    cell_per_block
    vis=False
    feature_vec=True

    feature_array = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), 
        cells_per_block=(cell_per_block, cell_per_block), visualization=vis, feature_vector=False)

                        
    X_scaler = joblib.load('saved_scalar.pickle')
    
  
    hist_bins = 24
    
    y_start_stop = [400, 720] # Min and max in y to search in slide_window()
    spatial_size = (50, 50) # Spatial binning dimensions
    orient = 12  # HOG orientations
    pix_per_cell = 12 # HOG pixels per cell
    cell_per_block = 6 # HOG cells per block
    hog_channel = 1 # Can be 0, 1, 2, or "ALL"
    spatial_feat = True # Spatial features on or off
    hist_feat = True # Histogram features on or off
    hog_feat = True # HOG features on or off
                            
    windows = hp.slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop, 
                    xy_window=(200, 200), xy_overlap=(0.5, 0.5))
                    
    print('len windows in image' + str(len(windows)))

    hot_windows = hp.search_windows(image, windows, clf, X_scaler, color_space=color_space)                    

    window_img = hp.draw_boxes(image, hot_windows, color=(0, 0, 255), thick=6)     
    print(len(hot_windows))
    print('done')               

    print(window_img.shape)
    plt.imshow(window_img)
    plt.show()
    
# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    
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
            print('subimg size ' + str(subimg.shape))
          
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
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                
    return draw_img
                            
        
    
if __name__ == '__main__':
    
    
    clf = joblib.load('saved_svc.pickle')
    
    image = mpimg.imread('./test_images/test1.jpg')
    
    ystart = 400
    ystop = 650
    scale = 1.5
    X_scaler = joblib.load('saved_scalar.pickle')
    print(X_scaler)
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    spatial_size = 32
    hist_bins = 32
    
    #process_image(image, clf)
    new_image = find_cars(image, ystart, ystop, scale, clf, X_scaler, orient, 
        pix_per_cell, cell_per_block, spatial_size, hist_bins)
    plt.imshow(new_image)
    plt.show()
    
    #iterate through each window
    #predict if car found
    #draw square for each car found
    #create heatmap
    #draw sqare outside heat map
    #print('My SVC predicts: ', clf.predict(X_test[0:n_predict]))
    

   # white_output = 'output_images/project_video_processed.mp4'
   # clip1 = VideoFileClip("project_video.mp4")
   # clip2 = VideoFileClip("challenge_video.mp4")
   # white_clip = clip1.fl_image(process_video)
   # white_clip.write_videofile(white_output, audio=False)
