import random
import numpy as np
import tensorflow as tf
import pandas as pd
import tensorflow_addons as tfa
from Siamese_model import image_embedder, get_siamese_network
from functions import split_dataset, Generate_dataset, create_triplets


random.seed(5)
np.random.seed(5)
tf.random.set_seed(5)
Path = "replace this with link to dataset"

physical_devices = tf.config.list_physical_devices('GPU')
for gpu_instance in physical_devices:
    tf.config.experimental.set_memory_growth(gpu_instance, True)

train_list, test_list = split_dataset(directory=Path, split=0.9)
train_triplet = create_triplets(Path, train_list, max_files=55)
test_triplet = create_triplets(Path, test_list, max_files=5)
train_dataset = Generate_dataset(Path=Path,list=train_triplet)
train_dataset = train_dataset.shuffle(buffer_size=1024)
train_dataset = train_dataset.batch(2048, drop_remainder=False)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)





tf.debugging.set_log_device_placement(True)
gpus = tf.config.list_logical_devices('GPU')
strategy = tf.distribute.MirroredStrategy(gpus)
with strategy.scope():
    emb_mod, model= get_siamese_network([224 ,224, 3])

    checkpoint_path = r"/home/khlifi/Documents/model_semihard_triplet_loss/all/allweights_1024b_preprocessing/max_55.keras"
    model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),loss=tfa.losses.TripletSemiHardLoss(margin=0.3))
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                              verbose=1)
    history = model.fit(train_dataset, epochs=15, callbacks=[cp_callback])
    hist_df = pd.DataFrame(history.history)
    hist_json_file = '/home/khlifi/Documents/more_data_preprocessing_on/all/history_2048_50.json'
    with open(hist_json_file, mode='w') as f:
        hist_df.to_json(f)
    hist_csv_file = '/home/khlifi/Documents/more_data_preprocessing_on/all/history_2048_50.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)



model.save('/home/khlifi/Documents/model_semihard_triplet_loss/final/2048_batch_pre_on_max_50.keras')




























