import os
import cv2

def process_videos_into_frames(file_name, dir_name):
 
    print(dir_name, file_name)
    dir_to_save_name = dir_name + file_name.split(".")[0] + "/"
    print("Creating directory " + dir_to_save_name)
    os.makedirs(dir_to_save_name)

    video_file_name = dir_name + file_name
    vidcap = cv2.VideoCapture(video_file_name)
    success, image = vidcap.read()
    count = 0
 
    while success:
        file_name = "frame%d.jpg" % count
        full_file_name = dir_to_save_name + file_name
        cv2.imwrite(full_file_name, image)     # save frame as JPEG file      
        success,image = vidcap.read()
        count += 1

def process_folder_of_videos_into_frames(dir_name):

    if not dir_name.endswith("/"):
        dir_name += "/"
    
    number_of_files = len(os.listdir(dir_name))
    for i, file_name in enumerate(os.listdir(dir_name)):
        if file_name.endswith(".mp4"):
            print("[" + str(i) + "/" + str(number_of_files) + "], currently processing file: " + file_name)
            process_videos_into_frames(file_name, dir_name)

#process_folder_of_videos_into_frames("../MELD.Raw/dev_splits_complete")
#process_folder_of_videos_into_frames("../MELD.Raw/output_repeated_splits_test")
#process_folder_of_videos_into_frames("../MELD.Raw/train_splits")