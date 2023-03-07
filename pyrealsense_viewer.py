## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

#Set window positions
cv2.namedWindow("Assignment3")

cv2.moveWindow("Assignment3", 0,0)

window = np.zeros([480,480,3],dtype=np.uint8) 


#Intilaize Tracker
cap = cv2.VideoCapture(1)
#ok, frame = cap.read()

tracker = cv2.TrackerKCF_create() #maybe in loop

# Define an initial bounding box
boundingBox = (287, 23, 86, 320)


# Start streaming
pipeline.start(config)
frames = pipeline.wait_for_frames()
depth_frame = frames.get_depth_frame()
track_image = np.asanyarray(depth_frame.get_data())


ok = tracker.init(track_image, boundingBox)


try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            break

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())


        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))

        track_image = depth_colormap
        

        #Creating black window dimensions
        # black = np.zeros([images.shape[0],images.shape[1],3],dtype=np.uint8) 
        black = np.zeros([300,images.shape[1],3],dtype=np.uint8) 
        # attach lower black window
        window = np.vstack((images, black))

        #tracker
        #if track_image.
        boundingBox = tracker.update(track_image)

        if ok:
             # Tracking success
            p1 = (int(boundingBox[0]), int(boundingBox[1]))
            p2 = (int(boundingBox[0] + boundingBox[2]), int(boundingBox[1] + boundingBox[3]))
            cv2.rectangle(track_image, p1, p2, (255,0,0), 2, 1)
        else :
            # Tracking failure
            cv2.putText(track_image, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)


        
        
        # Show images
        cv2.imshow('Assignment3', window)
        cv2.imshow('Tracking', track_image)


        #exit
        k = cv2.waitKey(1)
        if k == 27:
           break

finally:

    # Stop streaming
    pipeline.stop()