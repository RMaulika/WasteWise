# Week2/src/model_finetune.py
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, Model, optimizers

def build_finetune_model(num_classes, input_shape=(224,224,3), unfreeze_last_n=50, lr=1e-4):
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    # Freeze all then unfreeze last N layers
    for layer in base.layers:
        layer.trainable = False
    if unfreeze_last_n > 0:
        for layer in base.layers[-unfreeze_last_n:]:
            layer.trainable = True

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base.input, outputs=outputs)
    model.compile(optimizer=optimizers.Adam(lr), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

