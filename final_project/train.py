import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from .model import create_MobileNetV3Small_model


def scheduler(epoch):
    step = epoch // 3
    init_lr = 0.001
    lr = init_lr / 2 ** step
    print('Epoch: ' + str(epoch) + ', learning rate = ' + str(lr))
    return lr


def train():
    tf.keras.backend.clear_session()

    batch_size = 16
    epoch_number = 10
    image_size = (224, 224)

    model = create_MobileNetV3Small_model()

    image_dir = r'C:\Users\Rytis\Desktop\SPbPUWinterSchool\final_project\prepared_data_inside_boarders'

    image_generator = ImageDataGenerator(rescale=1./255., validation_split=0.2)

    train_generator = image_generator.flow_from_directory(batch_size=batch_size,
                                                        directory=image_dir,
                                                        shuffle=True,
                                                        target_size=image_size,
                                                        subset='training',
                                                        class_mode='binary')
    test_output = r'C:\Users\Rytis\Desktop\SPbPUWinterSchool\final_project\test_output/'
    test_generator = image_generator.flow_from_directory(batch_size=batch_size,
                                                             directory=image_dir,
                                                             shuffle=True,
                                                             target_size=image_size,
                                                             subset='validation',
                                                             class_mode='binary',
                                                            save_to_dir=test_output,
                                                            save_prefix='N',
                                                            save_format='jpeg')

    output_dir = r'C:\Users\Rytis\Desktop\SPbPUWinterSchool\final_project\weights_output_inside_boarders/'
    outputPath = output_dir + "Doggo_or_catto-{epoch:03d}-{loss:.4f}.hdf5"

    # Learning rate scheduler
    learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
    model_checkpoint = ModelCheckpoint(outputPath,
                                       monitor='loss',
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=False)

    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=test_generator,
        validation_steps=test_generator.samples // batch_size,
        epochs=epoch_number,
        callbacks=[model_checkpoint, learning_rate_scheduler],
        shuffle=True
    )

    tf.keras.backend.clear_session()


if __name__ == "__main__":
    train()
    # test()
