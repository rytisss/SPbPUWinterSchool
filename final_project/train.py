import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator


def create_efficientNetB0_model(input_size=(224, 224, 3), weigths_path=None):
    model = tf.keras.Sequential([
        tf.keras.applications.EfficientNetB0(include_top=False,
                                             weights='imagenet',
                                             input_shape=input_size),
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dense(32, activation='relu'),
        Dense(8, activation='relu'),
        Dense(2, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=1e-3),
                  metrics=['binary_accuracy'])

    if weigths_path != None:
        model.load_weights(weigths_path)
    return model


def scheduler(epoch):
    step = epoch // 3
    init_lr = 0.001
    lr = init_lr / 2 ** step
    print('Epoch: ' + str(epoch) + ', learning rate = ' + str(lr))
    return lr


def train():
    tf.keras.backend.clear_session()

    batch_size = 16
    epoch_number = 5
    image_size = (224, 224)

    model = create_efficientNetB0_model()

    image_dir = r'C:\Users\Rytis\Desktop\SPbPUWinterSchool\final_project\prepared_data/'

    image_generator = ImageDataGenerator(rescale=1./255., validation_split=0.2)

    train_generator = image_generator.flow_from_directory(batch_size=batch_size,
                                                        directory=image_dir,
                                                        shuffle=True,
                                                        target_size=image_size,
                                                        subset="training",
                                                        class_mode='binary')

    test_generator = image_generator.flow_from_directory(batch_size=batch_size,
                                                             directory=image_dir,
                                                             shuffle=True,
                                                             target_size=image_size,
                                                             subset="validation",
                                                             class_mode='binary')

    output_dir = r'C:\Users\Rytis\Desktop\SPbPUWinterSchool\final_project\weights_output/'
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
