#!/usr/bin/env python
# coding: utf-8
import argparse
import glob
import resource
import time
import datetime
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sn
import cv2
import tifffile as tif
import tensorflow as tf
import tensorflow_addons as tfa
import horovod.tensorflow as hvd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import *
from tensorflow.keras.initializers import *
from tensorflow.keras.callbacks import *
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import matplotlib.pyplot as plt


tf.keras.backend.set_floatx('float32')

os.chdir("") # home dir

parser = argparse.ArgumentParser(description='Training params')
parser.add_argument('--trial', type=int, default=6, metavar='N',
                    help='number of trials for training ')
parser.add_argument('--level', type=str, default='10X', metavar='NX',
                    help='input wsi level  (default: 10X)')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=1200, metavar='N',
                    help='number of epochs to train (default: 120)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--seed', type=int, default=4, metavar='N',
                    help='dataset randomization seed  for training (default: 4)')
parser.add_argument('--cache', type=bool, default=True, metavar='N',
                    help='use cache to accelerate training (default: True)')
args = parser.parse_args()

hvd.init()
gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()],'GPU')
    


# Constants
MODEL = "mobile_net_v3_large"
INPUT_NUMS = 64  # tiles per case
TILE_SIZE = 512
IMG_SHAPE = (INPUT_NUMS, TILE_SIZE, TILE_SIZE, 3)
CV_K = 0  # k-th fold
GRAY = False
AUGMENT = True
CYCLE_LEN = -1
BLOCK_LEN = 1
Ds_Repeat = True
AUTOTUNE = tf.data.experimental.AUTOTUNE
CACHE_DIR = "Caches/"
CSV_DIR = ""
GRAPH_DIR = ""
MODEL_DIR = ""
TENSORBOARD_DIR = ""
centers = []
centers_num = ""
name = ""
DATA_DIR = ""

# Arguments
trial = args.trial
level = args.level
BATCH_SIZE = args.batch_size // hvd.size()
EPOCHS = args.epochs
SEED = args.seed
shift_mil = False
output_path = f"{trial}_{level}_output.txt"
tfrec_path = f"{DATA_DIR}/{level}"
transfer = True
eps = K.epsilon()

# Load data
train_valids_X = np.load(f"{DATA_DIR}/train_valid_x_cv_5_seed_{SEED}.npy", allow_pickle=True)
X_trains = train_valids_X[CV_K, 0]
X_valids = train_valids_X[CV_K, 1]

# Label mapping
classes_to_index = {'nonprogress': 0, 'progress': 1}
train_labels = [classes_to_index[Path(tile).parent.name] for tile in X_trains]

# Compute class weights
class_weights = class_weight.compute_class_weight("balanced", np.unique(train_labels), train_labels)
class_weights = dict(enumerate(class_weights))

# Write to output file
with open(output_path, "w", encoding="utf-8") as f:
    f.write("")

# Parameters
PARA = pd.DataFrame({
    "name": name,
    "model": MODEL,
    "level": level,
    "tile_size": TILE_SIZE,
    "trial": trial,
    "Batch_size": BATCH_SIZE,
    "ColorNorm": True,
    "InputNum": 64,
    "Augmentation": AUGMENT,
    "ds_repeat": Ds_Repeat,
    "cycle_len": CYCLE_LEN,
    "block_len": BLOCK_LEN,
    "optimizer": 'Adam',
    "initial_lr": "0.001",
    "base_model_trainable": False,
    "fc_layers": '2',
    "weight_decay": '0.05',
    "ss": None,
    "pooling": "NoisyAnd",
    "dropout": None,
    "seed": SEED
}, index=[0])

# Title
TITLE = f"{str(MODEL).upper()}_NewModel_{level}_trial_{trial}_seed_{SEED}_tile_size_{TILE_SIZE}_"
SUB_DIR = TITLE + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")




GRAPH_DIR = 'graph_dir'
CSV_DIR = 'csv_dir'
MODEL_DIR = 'model_dir'
SUB_DIR = 'sub_dir'
INPUT_NUMS = 10
COLORS = ['b']
TITLE = 'title'

graph_dir = Path(GRAPH_DIR) / SUB_DIR
csv_dir = Path(CSV_DIR) / SUB_DIR
model_dir = Path(MODEL_DIR) / SUB_DIR

def ensure_dir_exists(directory):
    if not directory.exists():
        os.makedirs(directory)

def parse_fn_MultiInput(example):
    example_fmt = {
        "img_data": tf.io.FixedLenFeature([], tf.string, default_value=""),
        "label": tf.io.FixedLenFeature([], tf.int64)
    }
    parsed = tf.io.parse_example(example, features=example_fmt)
    image = tf.io.decode_raw(parsed["img_data"], tf.uint8)
    image = tf.reshape(image, (INPUT_NUMS, 512, 512, 3))
    image = tf.cast(image, tf.float32)
    label = tf.cast(parsed["label"], tf.int32)
    label = tf.one_hot(label, 2)
    return image, label

def augment_MultiInput(image, label):  
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.rot90(image, k=tf.random.uniform(shape=[], maxval=4, dtype=tf.int32))
    return image, label

def augment_MultiInput_eval(image, label):  
    return image, label

def parse_fn_MultiInput_valid(example):
    return parse_fn_MultiInput(example)

def plot_metrics(history, index=None):
    metrics =  ['loss', 'accuracy']
    plt.figure(figsize=(10, 15))
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(3, 2, n+1)
        plt.plot(history.epoch,  history.history[metric], color=COLORS[0], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric], color=COLORS[0], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        plt.ylim([min(plt.ylim()), plt.ylim()[1] if metric == 'loss' else 1])
        plt.tight_layout()
        plt.legend()
    plt.savefig(graph_dir / f"{TITLE}_train_validation_loss_auc_pr_{index}.png")

def save_model_summary(model=None, path=graph_dir):
    name = Path(graph_dir) / f"{TITLE}_model_summary.txt"
    with open(name, 'a') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

def Save_history(history=None, index=1, path=csv_dir):
    save_path = Path(csv_dir) / f"train_history_{index}.csv"
    df = pd.DataFrame(history.history)
    df.to_csv(save_path, index=None)

def Save_model(history=None, index=1):
    df = history.history
    train_acc = np.max(df["accuracy"])
    num = np.argmax(df["accuracy"])
    val_acc = df["val_accuracy"][num]
    model.save(f"{model_dir}/{TITLE}_acc-{train_acc:.3f}_val-acc-{val_acc:.3f}_fine_tune_{index}.h5")

if __name__ == "__main__":
    ensure_dir_exists(graph_dir)
    ensure_dir_exists(csv_dir)
    ensure_dir_exists(model_dir)

def Base_Model(MODEL=None, weights=None):
    IMG_SHAPE = (512, 512, 3)
    base_model = None

    model_dict = {
        "inception_resnet": tf.keras.applications.InceptionResNetV2,
        "mobile_net": tf.keras.applications.MobileNetV2,
        "dense_net_121": tf.keras.applications.DenseNet121,
        "resnet_101_v2": tf.keras.applications.ResNet101V2,
        "resnet_50_v2": tf.keras.applications.ResNet50V2,
        "resnet_152_v2": tf.keras.applications.ResNet152V2,
        "inception_v3": tf.keras.applications.InceptionV3,
        "xception": tf.keras.applications.Xception,
        "dense_net_169": tf.keras.applications.DenseNet169,
        "dense_net_201": tf.keras.applications.DenseNet201,
        "efficient_net_b7": tf.keras.applications.EfficientNetB7,
        "efficient_net_b6": tf.keras.applications.EfficientNetB6,
        "efficient_net_b5": tf.keras.applications.EfficientNetB5,
        "efficient_net_b4": tf.keras.applications.EfficientNetB4,
        "efficient_net_b0": tf.keras.applications.EfficientNetB0,
        "mobile_net_v3_large": tf.keras.applications.MobileNetV3Large,
        "mobile_net_v3_small": tf.keras.applications.MobileNetV3Small
    }

    if MODEL in model_dict:
        base_model = model_dict[MODEL](input_shape=IMG_SHAPE, include_top=False, weights=weights)

    base_model_num = len(base_model.layers)
    print(base_model_num)
    PARA["base_model"] = base_model_num

    return base_model


def safediv(a, b):
    # Calculating safe division
    safe_x = tf.where(tf.not_equal(b, 0.), b, tf.ones_like(b))
    resx = tf.where(tf.not_equal(b, 0.), tf.math.divide(x=a, y=safe_x), tf.zeros_like(safe_x))
    return resx
class NoisyAnd(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(NoisyAnd, self).__init__(**kwargs)

    def build(self, input_shape):
        self.a = tf.constant(10, dtype=tf.float32)
        b_init = tf.random_normal_initializer()
        self.b = tf.Variable(initial_value=b_init(shape=(1, input_shape[3]), dtype=tf.float32), trainable=True)
        super(NoisyAnd, self).build(input_shape)

    def call(self, x):
        mean = tf.reduce_mean(x, axis=[1, 2])
        mean = tf.cast(mean, dtype=tf.float32)
        res_a = (tf.sigmoid(tf.multiply(self.a, (mean - self.b))) - tf.sigmoid(-tf.multiply(self.a, self.b)))
        res_b = (tf.sigmoid(tf.multiply(self.a, (1 - self.b))) - tf.sigmoid(-tf.multiply(self.a, self.b)))
        res = safediv(res_a, res_b)
        return res

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim

    def get_config(self):
        config = super(NoisyAnd, self).get_config()
        config.update({
            "output_dim": self.output_dim
        })
        return config


PARA = {
    "base_model_trainable": True,
    "fc_layers": 'dropout(0.5)',
    "weight_decay": '0.15',
    "optimizer": 'AdamW',
    "initial_lr": '0.001',
    "class_weight": False
}


def get_files(subset):
    files = glob.glob(f"{tfrec_path}/*.tfrecords")
    return [i for j in subset for i in files if Path(j).name in i]


train_dataset = tf.data.Dataset.from_tensor_slices(tf.random.shuffle(get_files(X_trains), seed=SEED))
SHUFFLE_SIZE = BATCH_SIZE * hvd.size()
train_ds = train_dataset.interleave(map_func=tf.data.TFRecordDataset, cycle_length=CYCLE_LEN, block_length=BLOCK_LEN,
                                    num_parallel_calls=AUTOTUNE).map(parse_fn_MultiInput,
                                                                      num_parallel_calls=AUTOTUNE).map(
    augment_MultiInput, num_parallel_calls=AUTOTUNE).repeat().batch(BATCH_SIZE, drop_remainder=True).prefetch(AUTOTUNE)

X_valids = [str(i) for i in list(X_valids)]
all_files = glob.glob(f"*/{level}/*.tfrecords")
X_valids_stem = [Path(i).stem for i in X_valids]
X_valids_tfr = [i for i in all_files if Path(i).stem in X_valids_stem]

X_valids_cases = pd.read_csv(f"{DATA_DIR}/validation_all_tiles_{level}.csv")
X_valids_tiles = np.load(f"{DATA_DIR}/validation_all_tiles_{level}.npy", allow_pickle=True)
valid_tiles = X_valids_tiles[X_valids_cases.apply(lambda x: x["cases"] in X_valids, axis=1)]
valids_tiles = valid_tiles[np.random.permutation(len(valid_tiles))]

valid_labels = [classes_to_index[Path(tiles[0]).parent.parent.parent.name] for tiles in valid_tiles]
valid_labels = tf.one_hot(valid_labels, 2)

valid_index = list(range(len(valid_tiles)))
val_dataset = tf.data.Dataset.from_tensor_slices(valid_index)
val_dataset = val_dataset.shard(hvd.size(), hvd.rank())
valid_selected_index = list(val_dataset.as_numpy_iterator())
X_valids_tfr_selected = np.asarray(X_valids_tfr)[valid_selected_index]

lr = args.lr
scaled_lr = lr * hvd.size()
verbose = 1 if hvd.rank() == 0 else 0


class GCAdamW(tfa.optimizers.AdamW):
    def get_gradients(self, loss, params):
        grads = []
        gradients = super().get_gradients()
        for grad in gradients:
            grad_len = len(grad.shape)
            if grad_len > 1:
                axis = list(range(grad_len - 1))
                grad -= tf.reduce_mean(grad, axis=axis, keep_dims=True)
            grads.append(grad)

        return grads
class JointLoss(tf.keras.metrics.Metric):
    def __init__(self, name='jointloss', **kwargs):
        super(JointLoss, self).__init__(name=name, **kwargs)
        self.jointloss = self.add_weight(name='jl', initializer='zeros')
        self.count = self.add_weight(name="count", initializer='zeros')

    def update_state(self, loss_value, sample_weight=None):
        self.jointloss.assign_add(loss_value)
        self.count.assign_add(tf.constant(1, dtype=tf.float32))

    def result(self):
        return tf.math.divide(self.jointloss, self.count)

    def reset_states(self):
        self.jointloss.assign(0)
        self.count.assign(0)


def TwoBinaryCE(y_true, outputs):
    y_true = tf.cast(y_true, dtype=tf.float32)
    bce = keras.backend.binary_crossentropy(y_true, outputs, from_logits=False)
    bce = tf.reduce_mean(tf.reduce_sum(bce, axis=1))
    return bce


def Predict(tfr_cases):
    labels_all = []
    dense_outputs = np.empty((len(tfr_cases), 2))
    for i, tfr in enumerate(tfr_cases):
        valid_ds = (tf.data.TFRecordDataset(tfr, num_parallel_reads=AUTOTUNE)
                    .map(parse_fn_MultiInput_valid, AUTOTUNE)
                    .map(augment_MultiInput_eval, AUTOTUNE)
                    .batch(4)
                    .prefetch(4))
        outs_all = []
        for imgs, label in valid_ds:
            outs = representation(imgs)
            outs_all.append(outs)

        outs_all_merge = K.concatenate(outs_all, axis=0)  # N x H xW x C
        shapes = tf.shape(outs_all_merge)
        outs_all_merge = tf.reshape(outs_all_merge, [1, shapes[0], -1, shapes[3]])
        pool_output = pool(outs_all_merge)
        dense_output = dense(pool_output)
        dense_outputs[i, :] = dense_output
        labels_all.append(label)
    labels_merge = [tf.reshape(tf.gather(i, 0), [1, 2]) for i in labels_all]
    labels_merge = tf.concat(labels_merge, axis=0)
    return dense_outputs, labels_merge


if transfer:
    base_model = Base_Model(MODEL=MODEL, weights="imagenet")
    base_model.trainable = False
else:
    base_model = Base_Model(MODEL=MODEL, weights=None)
    base_model.trainable = True

representation_middle = Sequential([
    base_model,
    AveragePooling2D((16, 16)),  # change pooling size
    Conv2D(2, (1, 1), kernel_initializer="glorot_uniform", activation="linear"),
], name="representation")

pool_middle = Sequential([
    NoisyAnd(2, name="pooling")
])

dense_middle = Sequential([
    Dense(2, activation="softmax", name="cls", kernel_initializer="glorot_uniform")
])

Whole = Sequential([
    representation_middle,
    pool_middle,
    dense_middle
])

representation = Whole.layers[0]
pool = Whole.layers[1]
dense = Whole.layers[2]

if hvd.rank() == 0:
    with open(output_path, 'a') as f:
        Whole.summary(print_fn=lambda x: f.write(x + '\n'))

loss_fn = keras.losses.CategoricalCrossentropy()
lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
    scaled_lr, decay_steps=600)
optimizer = GCAdamW(learning_rate=lr_decayed_fn)

train_loss_tracker = JointLoss()
train_acc_metric = keras.metrics.CategoricalAccuracy(name='accuracy')
train_auc_metric = keras.metrics.AUC(name='auc', from_logits=True)
train_prc_metric = keras.metrics.AUC(name='prc', curve='PR', from_logits=True)

valid_loss_tracker = JointLoss()
valid_acc_metric = keras.metrics.CategoricalAccuracy(name='accuracy')
valid_auc_metric = keras.metrics.AUC(name='auc', from_logits=True)
valid_prc_metric = keras.metrics.AUC(name='prc', curve='PR', from_logits=True)

@tf.function
def train_step(images, labels, first_batch):
    k = 8  # number of tiles to join the backpropagation
    ix = tf.math.top_k(tf.random.uniform(shape=[INPUT_NUMS]), k, sorted=True).indices

    with tf.GradientTape() as tape:
        for num in range(images.shape[1]):
            img = images[:, num]
            if not tf.reduce_any(tf.equal(ix, num)):
                with tape.stop_recording():
                    out = representation(img)
            else:
                out = representation(img)

            outs_all = out if num == 0 else tf.concat([outs_all, out], 1)

        pool_output = pool(outs_all)
        dense_output = dense(pool_output)

        loss = loss_fn(labels, dense_output) + TwoBinaryCE(labels, pool_output)

    tape = hvd.DistributedGradientTape(tape)
    grads = tape.gradient(loss, Whole.trainable_weights)

    for grad in grads:
        tf.debugging.check_numerics(grad, message="checking grads")

    optimizer.apply_gradients(zip(grads, Whole.trainable_weights))

    if first_batch:
        hvd.broadcast_variables(Whole.variables, root_rank=0)
        hvd.broadcast_variables(optimizer.variables(), root_rank=0)

    train_loss_tracker.update_state(loss)
    train_acc_metric.update_state(labels, dense_output)
    train_auc_metric.update_state(labels[:, 1], dense_output[:, 1])
    train_prc_metric.update_state(labels[:, 1], dense_output[:, 1])

    return loss


def lr_warmup(initial_lr=None, warmup_epochs=None, current_epoch=None):
    return initial_lr * (current_epoch + 1) / warmup_epochs


train_steps = len(X_trains) * INPUT_NUMS // args.batch_size
valid_steps = len(X_valids) * INPUT_NUMS

print("train_steps", train_steps)

HISTORY = []

logical_gpus = tf.config.list_logical_devices('GPU')
def train_main(resume_epoch=None, trainable_layers=None, epochs=EPOCHS, learning_rate=scaled_lr, Patience=20, Lr_patience=10, index=0):
    """Construct models"""
    if transfer:
        if index == 0:
            base_model.trainable = False
        elif index != 0 and trainable_layers[index-1] == 0:
            base_model.trainable = True
        else:
            for layer in base_model.layers[trainable_layers[index-1]:]:
                if not isinstance(layer, BatchNormalization):
                    layer.trainable = True
    else:
        base_model.trainable = True

    """ initialize metrics """
    metrics = ["loss", "accuracy", "auc", "prc", "val_loss", "val_accuracy", "val_auc", "val_prc"]
    history = {metric: [] for metric in metrics}

    """initialize optimizer & loss func"""
    optimizer.learning_rate = learning_rate

    # finetune
    patience = Patience
    lr_patience = Lr_patience
    chunk_size = 50
    wait = 0
    lr_wait = 0  # lr drop at plateau
    best = 0
    total_steps = 0  # exponential lr decay
    warmup_epochs = hvd.size() // 4

    for epoch in range(epochs):
        if epoch < warmup_epochs:
            optimizer.learning_rate = lr_warmup(learning_rate, warmup_epochs, epoch)
        print("\nStart epoch", resume_epoch)
        start_time = time.time()

        for step, (images, labels) in enumerate(train_ds.take(train_steps)):
            loss = train_step(images, labels, step == 0)
            total_steps += 1
            lr_now = optimizer.learning_rate.numpy()
            loss = hvd.allreduce(loss)
            loss = loss.numpy()

            if step % 100 == 0 and hvd.rank() == 0:
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(f"Start epoch:{resume_epoch}\n")
                    f.write(f"loss at step {step}: loss: {loss},learning rate:{lr_now}\n")
                print(f"loss at step {step}: loss: {loss},learning rate:{lr_now}\n")

        train_loss = train_loss_tracker.result()
        train_acc = train_acc_metric.result()
        train_auc = train_auc_metric.result()
        train_prc = train_prc_metric.result()

        if hvd.rank() == 0:
            history["loss"].append(train_loss.numpy())
            history["accuracy"].append(train_acc.numpy())
            history["auc"].append(train_auc.numpy())
            history["prc"].append(train_prc.numpy())

        train_loss_tracker.reset_states()
        train_acc_metric.reset_states()
        train_auc_metric.reset_states()
        train_prc_metric.reset_states()

        if hvd.rank() == 0:
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(f"Epoch:{resume_epoch},Training Loss:{train_loss.numpy():.4f}, Acc: {train_acc.numpy():.4f}, AUC:{train_auc.numpy():.3f}, PRC:{train_prc.numpy():.3f}\n")
            print(f"Epoch:{resume_epoch},Training Loss:{train_loss.numpy():.4f}, Acc: {train_acc.numpy():.4f}, AUC:{train_auc.numpy():.3f}, PRC:{train_prc.numpy():.3f}\n")

        dense_outputs, valid_labels_selected = Predict(X_valids_tfr_selected)
        dense_outputs = tf.cast(dense_outputs, dtype=tf.float32)
        valid_loss = loss_fn(valid_labels_selected, dense_outputs)
        valid_loss_tracker.update_state(valid_loss)
        valid_acc_metric.update_state(valid_labels_selected, dense_outputs)
        valid_auc_metric.update_state(valid_labels_selected[:, 1], dense_outputs[:, 1])
        valid_prc_metric.update_state(valid_labels_selected[:, 1], dense_outputs[:, 1])

        valid_loss = hvd.allreduce(valid_loss_tracker.result())
        valid_acc = hvd.allreduce(valid_acc_metric.result())
        valid_auc = hvd.allreduce(valid_auc_metric.result())
        valid_prc = hvd.allreduce(valid_prc_metric.result())

        if hvd.rank() == 0:
            history["val_loss"].append(valid_loss.numpy())
            history["val_accuracy"].append(valid_acc.numpy())
            history["val_auc"].append(valid_auc.numpy())
            history["val_prc"].append(valid_prc.numpy())

        valid_loss_tracker.reset_states()
        valid_acc_metric.reset_states()
        valid_auc_metric.reset_states()
        valid_prc_metric.reset_states()

        if hvd.rank() == 0:
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(f"Epoch:{resume_epoch},Validation Loss:{valid_loss:.4f}, val_acc: {valid_acc:.4f}, val_AUC:{valid_auc:.3f}, val_PRC:{valid_prc:.3f}\n")
                f.write(f"Time taken: {(time.time() - start_time):.2f}s\n")
            print(f"Epoch:{resume_epoch},Validation Loss:{valid_loss:.4f}, val_acc: {valid_acc:.4f}, val_AUC:{valid_auc:.3f}, val_PRC:{valid_prc:.3f}\n")
            print("Time taken: %.2fs" % (time.time() - start_time))

        wait += 1
        lr_wait += 1

        if resume_epoch > 10 and resume_epoch % 5 == 0 and hvd.rank() == 0:
            Model_name = f"{MODEL}_{level}_trial_{trial}_{resume_epoch:03d}_acc-{train_acc:.3f}_val-acc-{valid_acc:.3f}_index_{index}.h5"
            Whole.save(f"{model_dir}/{Model_name}")

        if valid_acc > best:
            best = valid_acc
            best_weight_representation = representation.get_weights()
            best_weight_dense = dense.get_weights()
            wait = 0
            lr_wait = 0

            if hvd.rank() == 0:
                Model_name = f"{MODEL}_{level}_trial_{trial}_{resume_epoch:03d}_acc-{train_acc:.3f}_val-acc-{valid_acc:.3f}_index_{index}.h5"
                Whole.save(f"{model_dir}/{Model_name}")

        if lr_wait >= lr_patience:
            optimizer.learning_rate = optimizer.learning_rate.numpy() * tf.sqrt(0.1)

        if wait >= patience:
            break

        resume_epoch += 1

    if hvd.rank() == 0:
        pd.DataFrame(history).to_csv(f"{csv_dir}/history_index_{index}.csv")
        for i in history.keys():
            pd.DataFrame(history[i]).to_csv(f"{csv_dir}/history_index_{i}_{index}.csv")

        HISTORY.append(history)

    return best_weight_representation, best_weight_dense, resume_epoch
