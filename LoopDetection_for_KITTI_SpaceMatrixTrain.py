#! -*- coding: utf-8 -*-
#   version = 3.0
#   date = 2020-08-01
# __author__ = 'zyz'
import numpy as np
from keras.layers import *

from keras import backend as K
from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
import tensorflow as tf
import os, pickle, cv2
import matplotlib.pyplot as plt
from gen_kitti import gen_pk, Nrow, Ncol, load_model, save_model
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score
import scipy.io as sio
from gen import WarmupExponentialDecay
from vgg16 import VGG16
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
config = tf.ConfigProto()
config.allow_soft_placement=True
config.gpu_options.per_process_gpu_memory_fraction=0.95
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

dataset_name_train = ['06', '04', '08']
dataset_name_test = ['05']

nrow = 8
ncol = 26

n_pacth = 1 + 4 + 16
# n_pacth = (nrow // 2) * (ncol // 2)

def transpose(x):
    return K.permute_dimensions(x, (0, 2, 1))

def feature_extract(path='./dataset/KITTI/', dataset_name='CityCentre', ext='image_2',
                    image_shape=(Nrow, Ncol, 3),
                    weights='./weights/vgg16-places365_weights_tf_dim_ordering_tf_kernels.h5'):
    image_files = glob.glob(f'{path}/{dataset_name}/{ext}/*.png')
    base_model = VGG16(include_top=False, input_shape=image_shape, classes=365,
                       weights=weights)
    model = Model(base_model.input, base_model.get_layer('block4_pool').output)
    for file in image_files:
        img = cv2.imread(file)
        img = cv2.resize(img, (image_shape[1], image_shape[0]))
        features = model.predict(np.expand_dims(img, axis=0))
        p_file = file.split('/')[-1].split('.')[0]
        with open(f'{path}/{dataset_name}/pickles4/{ext}/{p_file}.pkl', 'wb') as f:
            pickle.dump(features, f)
    return


def expand_dim(tensor, axis=1):
    return K.expand_dims(tensor, axis=axis)

def mask_gen(x):
    return K.mean(x, axis=-1, keepdims=True)

def get_Inception_model(input_size):
    z = Input(input_size)
    y = z
    y1 = MaxPooling2D(pool_size=(2, 7), padding='same', strides=(2, 7))(y)
    y1 = Reshape((-1, 512))(y1)
    y2 = MaxPooling2D(pool_size=(4, 13), padding='same', strides=(4, 13))(y)
    y2 = Reshape((-1, 512))(y2)
    y3 = MaxPooling2D(pool_size=(8, 26), padding='same', strides=(8, 26))(y)
    y3 = Reshape((-1, 512))(y3)
    y = concatenate([y1, y2, y3], axis=1)
    # y = BatchNormalization()(y)
    # y = LeakyReLU(0.2)(y)
    model = Model(z, y)
    model.summary()
    return model

def norm_layer(x):
    x_norm = K.sqrt(K.sum(K.square(x), axis=-1, keepdims=True)) + K.epsilon()
    return x / x_norm

def sim_layer(x):
    x_norm = x[0] / (K.sqrt(K.sum(K.square(x[0]), axis=-1, keepdims=True)) + K.epsilon())
    y_norm = x[1] / (K.sqrt(K.sum(K.square(x[1]), axis=-1, keepdims=True) + K.epsilon()))
    sim = K.sum(x_norm * y_norm, axis=-1, keepdims=True)
    return sim

def cos_sim_layer(x):
    x0 = K.expand_dims(x[0], axis=2)
    x1 = K.expand_dims(x[1], axis=1)
    x0_norm = K.sqrt(K.sum(K.square(x0), axis=-1)) + K.epsilon()
    x1_norm = K.sqrt(K.sum(K.square(x1), axis=-1)) + K.epsilon()
    sim_score = K.sum(x0 * x1, axis=-1) / x0_norm / x1_norm
    # sim_score_max = K.max(sim_score, axis=-1, keepdims=True)
    sim_score_mean = K.mean(sim_score, axis=-1, keepdims=True)
    return sim_score - sim_score_mean

def get_sim_model():
    z11 = Input((n_pacth, 512))
    z21 = Input((n_pacth, 512))

    sim_score_space = Lambda(cos_sim_layer)([z11, z21])
    # sim_score_space = BatchNormalization()(sim_score_space)
    sim_score_space = Activation('relu')(sim_score_space)
    sim_score_space = Flatten()(sim_score_space)
    sim_score_space = Dense(256, activation=None, use_bias=False, name='space_weight1')(sim_score_space)
    sim_score_space = BatchNormalization()(sim_score_space)
    sim_score_space = Activation('relu')(sim_score_space)
    sim_score_space = Dense(1, activation=None, use_bias=False, name='space_weight')(sim_score_space)
    sim_score_space = BatchNormalization()(sim_score_space)
    sim_score_space = Activation('sigmoid')(sim_score_space)
    # sim_score_space = Dropout(0.2)(sim_score_space)
    # sim_score_total = Lambda(sim_layer)([z11, z21])
    # sim_score = concatenate([sim_score_total, sim_score_space])
    # sim_score = Dense(1, activation='sigmoid', use_bias=False,
    #                   kernel_regularizer=regularizers.l2(0.0001), name='sim')(sim_score)
    return Model([z11, z21], sim_score_space)

def get_model(input_size):
    x1 = Input(shape=input_size)
    x2 = Input(shape=input_size)
    base_model = get_Inception_model(input_size=input_size)
    z1 = base_model(x1)
    z2 = base_model(x2)
    sim_model = get_sim_model()
    sim_score = sim_model([z1, z2])
    model = Model(inputs=[x1, x2], outputs=sim_score)
    return base_model, sim_model, model

def contrastiveloss(y_true, y_pred, alpha=0.5):
    margin_ = 0.8
    _margin = 0.0
    dist = 1 - y_pred
    sqaure_pred = K.maximum(dist - _margin, 0)
    margin_sqaure = K.maximum(margin_ - dist, 0)
    return K.mean(alpha * y_true * sqaure_pred + (1 - alpha) * (1 - y_true) * margin_sqaure)

def binary_focal_loss(gamma=2, alpha=0.25):
    """
    Binary form of focal loss.
    适用于二分类问题的focal loss

    focal_loss(p_t) = -alpha_t * (1 - p_t)**gamma * log(p_t)
        where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        y_true shape need be (None,1)
        y_pred need be compute after sigmoid
        """
        # alpha = 1 - 0.5 * K.stop_gradient(smooth_F1(y_true, y_pred))
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)

        p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred) + K.epsilon()
        focal_loss = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)
        contrastive_loss = contrastiveloss(y_true, y_pred, 0.5)
        # F1_score = 1 - smooth_F1(y_true, y_pred)
        return contrastive_loss + K.mean(focal_loss)
    return binary_focal_loss_fixed


def precision(y_true, y_pred):
    # Calculates the precision
    y = K.round(y_true)
    true_positives = K.sum(K.round(K.clip(y * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    # Calculates the recall
    y = K.round(y_true)
    true_positives = K.sum(K.round(K.clip(y * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def smooth_F1(y_true, y_pred):
    return 2 * K.sum(y_true * y_pred) / K.sum((y_true + y_pred))

def train(feature_shape=(15, 20, 512), batch_size=4, l_r=1e-3, weight=None):
    epoch = 200
    base_model, sim_model, model = get_model(feature_shape)
    if weight is not None:
        if os.path.exists(weight):
            model.load_weights(weight)

    learning_rate = l_r
    optimizer = SGD(learning_rate, decay=1e-6, momentum=0.9, clipnorm=1., nesterov=True)

    lr_schedule = lambda epoch: l_r * (0.99 ** epoch)
    learning_rate = np.array([lr_schedule(i) for i in range(epoch)])
    changelr = LearningRateScheduler(lambda epoch: float(learning_rate[epoch]))
    tensorboard = TensorBoard(log_dir='./logs', write_graph=True)
    warm_up = WarmupExponentialDecay(lr_base=l_r, decay=0.0002, warmup_epochs=10)
    model.compile(loss=binary_focal_loss(gamma=2, alpha=0.5),
                  optimizer=optimizer, metrics=['acc', smooth_F1, recall, precision])
    checkpoint = ModelCheckpoint(
        filepath='./logs/sim-{epoch:02d}-{loss:.4f}-{smooth_F1:.4f}-{precision:.4f}-{recall:.4f}-{val_loss:.4f}-{val_smooth_F1:.4f}-{val_precision:.4f}-{val_recall:.4f}.h5',
        monitor='val_loss',
        save_best_only=False,
        save_weights_only=True)
    # early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=2)

    print('-----------Start training-----------')
    n_train = 100
    n_test = 100

    train_loader = gen_pk(dataset_name_train, batch_size)
    test_loader = gen_pk(dataset_name_test,  batch_size)
    History = model.fit_generator(train_loader, validation_data=test_loader,
                                  steps_per_epoch=n_train,
                                  epochs=epoch,
                                  shuffle=True, verbose=1,
                                  validation_steps=n_test,
                                  callbacks=[checkpoint, warm_up, tensorboard])
    model.save_weights('./logs/full_model.h5')
    base_model.save_weights('./logs/base_model.h5')
    sim_model.save_weights(f'./logs/sim_model.h5')
    # plt.figure()
    # plt.plot(History.history['loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    #
    # plt.plot(History.history['val_loss'])
    # plt.ylabel('val loss')
    # plt.xlabel('epoch')
    # plt.show()
    return

def sim_total(z1, z2):
    n = z1.shape[0]
    z1 = np.reshape(z1, [n, -1])
    z2 = np.reshape(z2, [n, -1])
    z_1 = z1 / (np.sqrt(np.sum(np.square(z1), axis=-1, keepdims=True)) + 1e-6)
    z_2 = z2 / (np.sqrt(np.sum(np.square(z2), axis=-1, keepdims=True)) + 1e-6)
    sim_score_total = np.sum(z_1 * z_2, axis=-1, keepdims=True)
    return sim_score_total

def ConfidenceMatrix(path='./dataset/TUM/',
                     dataset_name='00',
                     feature_shape = (Nrow, Ncol, 3), n=4):
    gt = sio.loadmat('%s/%s/KITTI%s_GT_6_26.mat' % (path, dataset_name, dataset_name))['truth']

    base_model, sim_model, model = get_model(feature_shape)
    model.load_weights('./logs/model.h5')
    W1 = sim_model.get_layer('space_weight1').get_weights()[0]
    W0 = sim_model.get_layer('space_weight').get_weights()[0]
    sio.savemat('W0.mat', {'W0':W0})
    sio.savemat('W1.mat',  {'W1':W1})
    W = np.reshape(np.dot(W1, W0), [21, 21])
    sio.savemat('W.mat', {'W':W})
    # plt.figure()
    # plt.imshow(W)
    # plt.show()
    # weightfile = glob.glob(f'./logs/weight/*.h5')
    # weights = []
    # for w in weightfile:
    #     model.load_weights(w)
    #     weights.append(model.get_weights())
    # new_weight = []
    # for weights_list_tuple in zip(*weights):
    #     new_weight.append([np.array(weights_).mean(axis=0) for weights_ in zip(*weights_list_tuple)])
    # model.set_weights(new_weight)
    # model.save_weights('./logs/model.h5')
    # print(sim_model.get_layer('sim').get_weights())

    # if isinstance(weight, list):
    # 	if os.path.exists(weight[0]):
    # 		base_model.load_weights(weight[0])
    # 	else:
    # 		return np.zeros(gt.shape, dtype=np.float32)
    # 	if os.path.exists(weight[1]):
    # 		sim_model.load_weights(weight[1])
    # 	else:
    # 		return np.zeros(gt.shape, dtype=np.float32)
    #
    # elif isinstance(weight, str):
    # 	if weight is not None:
    # 		if os.path.exists(weight):
    # 			model.load_weights(weight)
    # 		else:
    # 			return np.zeros(gt.shape, dtype=np.float32)
    # 	else:
    # 		return np.zeros(gt.shape, dtype=np.float32)

    X = []
    for i in range(0, gt.shape[0], n):
        d = load_model('%s/%s/pickles/image_2/%06i.pkl' % (path, dataset_name, i))
        d = base_model.predict(d)
        X.append(d)

    # with open(f'./logs/base_{dataset_name}.pkl', 'wb') as f:
    #     pickle.dump(X, f)

    Confidence_Matrix_space = np.zeros((len(X), len(X)), dtype=np.float)
    Confidence_Matrix_total_space = np.zeros((len(X), len(X)), dtype=np.float)
    mininterval = 0

    for i in range(len(X)):
        if i % 100 == 0:
            print(i)
        x = np.repeat(X[i], i - mininterval + 1, axis=0)
        y = np.concatenate(X[:i - mininterval + 1])
        result_space = sim_model.predict([x, y])[:, 0]
        Confidence_Matrix_space[i, :i - mininterval + 1] = result_space
        result_total = sim_total(x, y)[:, 0]
        result = (result_space + result_total) * 0.5
        Confidence_Matrix_total_space[i, :i - mininterval + 1] = result

    with open(f'./logs/CM_{dataset_name}_total_space.pkl', 'wb') as f:
        pickle.dump(Confidence_Matrix_total_space, f)

    return [Confidence_Matrix_space, Confidence_Matrix_total_space]

def evluation(path='./dataset/KITTI/', dataset_name='00',
              feature_shape = (Nrow, Ncol, 3),
              weight=None,
              plot=True, n=4):
    gt = sio.loadmat('%s/%s/KITTI%s_GT_6_26.mat' % (path, dataset_name, dataset_name))['truth']
    CM = ConfidenceMatrix(path=path, dataset_name=dataset_name, feature_shape=feature_shape, n=n)
    gt = gt[::n, ::n]
    gt = gt.reshape([-1, ])
    for i in range(len(CM)):
        CM_v = CM[i].reshape([-1, ])
        precisions, recalls, thresholds = precision_recall_curve(gt, CM_v, sample_weight=None)
        ap = average_precision_score(gt, CM_v)
        print(f'Average Precision: {ap}')
        plt.figure()
        plt.plot(recalls, precisions)
        plt.fill_between(recalls, precisions, step='post', alpha=0.2, color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall curve: AP={0:0.3f}'.format(ap))
        plt.box(True)
        plt.grid(b=True, linestyle='--')
        plt.savefig(f'./logs/precision-recall_curve_{dataset_name}_{i}_{ap}.png', format='png', dpi=150)
        plt.show()

        if plot:
            f, ax = plt.subplots()
            cax = ax.imshow(CM[i], cmap='coolwarm', interpolation='nearest', vmin=0., vmax=1.)
            cbar = f.colorbar(cax, ticks=[0, 0.5, 1])
            cbar.ax.set_yticklabels(['0', '0.5', '1'])
            plt.savefig(f'./logs/Confidence_Matrix_{dataset_name}_{i}_{ap}.png', format='png', dpi=150)
            plt.show()
    return

import glob

if __name__=='__main__':
    feature_shape = (nrow, ncol, 512)
    weightList = []
    # train(feature_shape=feature_shape, batch_size=16, l_r=1e-1, weight='./logs/model.h5')
    for dataset_name in ['05', '02', '00']:
        evluation(dataset_name=dataset_name,
                  feature_shape=feature_shape,
                  weight=weightList,
                  plot=True, n=1)
