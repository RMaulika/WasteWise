# Week2/src/train_model.py
import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

def train(model, train_gen, val_gen, epochs=25, output_dir="Week2/outputs"):
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, "best_model_week2.h5")
    cp = ModelCheckpoint(ckpt_path, save_best_only=True, monitor="val_accuracy", mode="max", verbose=1)
    es = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1)
    rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

    history = model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=[cp, es, rlrop])

    # plots
    plt.figure()
    plt.plot(history.history.get('accuracy', []), label='train_acc')
    plt.plot(history.history.get('val_accuracy', []), label='val_acc')
    plt.legend(); plt.title('Accuracy'); plt.savefig(os.path.join(output_dir,"accuracy_plot_week2.png"))

    plt.figure()
    plt.plot(history.history.get('loss', []), label='train_loss')
    plt.plot(history.history.get('val_loss', []), label='val_loss')
    plt.legend(); plt.title('Loss'); plt.savefig(os.path.join(output_dir,"loss_plot_week2.png"))

    return history, ckpt_path
