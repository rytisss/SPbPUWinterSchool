import os
import glob
import numpy as np
import cv2


def gather_image_from_dir(input_dir):
    image_extensions = ['*.bmp', '*.jpg', '*.png']
    image_list = []
    for image_extension in image_extensions:
        image_list.extend(glob.glob(input_dir + image_extension))
    image_list.sort()
    return image_list


def get_file_name(path):
    file_name_with_ext = path.rsplit('\\', 1)[1]
    file_name, file_extension = os.path.splitext(file_name_with_ext)
    return file_name


def insertROI2BlackImage(imageROI, destination_width, destination_height):
    # make blank image
    destination_image = np.zeros([destination_height, destination_width, 3], dtype=np.uint8)
    # center 'imageROI' to destination image center
    center_x = destination_width // 2
    center_y = destination_height // 2
    # resize 'imageROI' to fit into destination image, also keep the aspect ratio
    roi_height, roi_width = imageROI.shape[:2]
    x_aspect_ratio = destination_width / roi_width
    y_aspect_ratio = destination_height / roi_height
    resize_ratio = x_aspect_ratio if x_aspect_ratio < y_aspect_ratio else y_aspect_ratio
    resized_width = int(resize_ratio * float(roi_width))
    resized_height = int(resize_ratio * float(roi_height))
    # prevention from too big width
    if resized_width > destination_width:
        resized_width = destination_width
    if resized_height > destination_height:
        resized_height = destination_height
    delta_w = destination_width - resized_width
    delta_h = destination_height - resized_height
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    resized_roi = cv2.resize(imageROI, (resized_width, resized_height))
    # calculate offset for roi in image (roi is centered)
    #resized_roi_height, resized_roi_width = resized_roi.shape[:2]
    #x_offset = int((destination_width - resized_roi_width) / 2)
    #y_offset = int((destination_height - resized_roi_height) / 2)
    #destination_image[y_offset:y_offset+resized_roi_height, x_offset:x_offset+resized_roi_width] = resized_roi
    value = [0,0,0]
    destination_image = cv2.copyMakeBorder(resized_roi, top, bottom, left, right, cv2.BORDER_REPLICATE, None, value)
    return destination_image


def preprocess():
    input_dir = r'C:\Users\Rytis\Desktop\SPbPUWinterSchool\final_project\original data/'
    output_dir = r'C:\Users\Rytis\Desktop\SPbPUWinterSchool\final_project\prepared_data_inside_boarders/'
    desired_size = (224,224)

    # gather all subdirectories in input_dir
    subdirs = glob.glob(input_dir + '*/')
    for subdir in subdirs:
        class_folder_name = os.path.basename(os.path.normpath(subdir))
        # make class directory in output
        class_output_dir = os.path.join(output_dir, class_folder_name, '')
        if not os.path.exists(class_output_dir):
            os.makedirs(class_output_dir)
        # gather images
        image_paths = gather_image_from_dir(subdir)
        for image_path in image_paths:
            image = cv2.imread(image_path)
            if image is None:
                print('Can\'t open ' + image_path)
                continue

            # get image name
            image_name = get_file_name(image_path)
            # resize to center
            #processed_image = cv2.resize(image, desired_size)#insertROI2BlackImage(image, desired_size[0], desired_size[1])
            precessed_image = insertROI2BlackImage(image, desired_size[0], desired_size[1])
            cv2.imwrite(class_output_dir + image_name + '.jpg', precessed_image)

            #cv2.imshow('image', image)
            #cv2.imshow('processed', processed_image)
            #cv2.waitKey(1)


if __name__ == "__main__":
    preprocess()
