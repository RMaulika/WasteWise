# Week2/src/dataset_loader.py
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_generators(train_dir, val_dir, test_dir, image_size=(224,224), batch_size=32):
    train_aug = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    val_test = ImageDataGenerator(rescale=1./255)

    train_gen = train_aug.flow_from_directory(
        train_dir, target_size=image_size, batch_size=batch_size, class_mode='categorical', shuffle=True)
    val_gen = val_test.flow_from_directory(
        val_dir, target_size=image_size, batch_size=batch_size, class_mode='categorical', shuffle=False)
    test_gen = val_test.flow_from_directory(
        test_dir, target_size=image_size, batch_size=batch_size, class_mode='categorical', shuffle=False)

    return train_gen, val_gen, test_gen
