from final_project.model import create_MobileNetV3Small_model, preprocess_image
from final_project.preprocess import resize_and_extrapolate_image_boarders, gather_image_from_dir
import cv2
import tensorflow as tf


def cat_or_dog(prediction):
    animal = ''
    if prediction[0][0] >= 0.5:
        animal = 'dog'
    else:
        animal = 'cat'
    return animal


def main():
    tf.keras.backend.clear_session()
    test_image_dir = 'test_images/'
    trained_weights = 'weights/Doggo_or_catto-007-0.0087.hdf5'
    model = create_MobileNetV3Small_model(weigths_path=trained_weights)
    input_size = (224, 224)
    # test directory contains folder with different classes
    # collect those subdirectory with different numbers
    image_paths = gather_image_from_dir(test_image_dir)
    for image_path in image_paths:
        image = cv2.imread(image_path)
        image = resize_and_extrapolate_image_boarders(image, input_size[0], input_size[0])
        image_rb_invert = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rb_invert = preprocess_image(image_rb_invert)
        prediction = model.predict(image_rb_invert)
        cv2.imshow('image', image)
        animal = cat_or_dog(prediction)
        print(animal + ' | model output: ' + str(prediction[0][0]))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
