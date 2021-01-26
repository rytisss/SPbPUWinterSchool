from final_project.model import create_MobileNetV3Small_model, preprocess_image
from final_project.preprocess import insertROI2BlackImage
import glob
import cv2
import tensorflow as tf


def gather_image_from_dir(input_dir):
    image_extensions = ['*.bmp', '*.jpg', '*.png']
    image_list = []
    for image_extension in image_extensions:
        image_list.extend(glob.glob(input_dir + image_extension))
    image_list.sort()
    return image_list


def main():
    tf.keras.backend.clear_session()
    test_image_dir = r'C:\Users\Rytis\Desktop\SPbPUWinterSchool\final_project\original data/1/'
    trained_weights = r'C:\Users\Rytis\Desktop\SPbPUWinterSchool\final_project\weights_output_inside_boarders/Doggo_or_catto-010-0.0012.hdf5'
    model = create_MobileNetV3Small_model(weigths_path=trained_weights)
    input_size = (224, 224)
    # test directory contains folder with different classes
    # collect those subdirectory with different numbers
    image_paths = gather_image_from_dir(test_image_dir)
    for image_path in image_paths:
        image = cv2.imread(image_path)
        image = insertROI2BlackImage(image, input_size[0], input_size[0])
        image_rb_invert = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rb_invert = preprocess_image(image_rb_invert)
        prediction = model.predict(image_rb_invert)
        cv2.imshow('image', image)
        print(prediction)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
