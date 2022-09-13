#!/usr/bin/env python

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # no gpu
#os.environ["CUDA_VISIBLE_DEVICES"] = "0" #2080ti

import tensorflow as tf
import roslib
#roslib.load_manifest('my_package')
import sys
import rospy
import cv2
import numpy as np
import math
import threading
import time
import csv

from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from std_msgs.msg import String, Float32, Float32MultiArray, Int8
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from datetime import datetime
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2DTranspose, ConvLSTM2D, BatchNormalization
from tensorflow.keras.layers import TimeDistributed, Conv2D, LayerNormalization, Conv3D
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import backend as K
from os import listdir
from os.path import isfile, join, isdir
import matplotlib.pyplot as plt

def main(args):  
    rospy.init_node('lstm_AE', anonymous=True)
    q = build_q()
    ic = image_converter(q) #pass tf q
    evaluate(q)
    cv2.destroyAllWindows()
    q.close()



def sampling_model(distribution_params):
    mean, log_var = distribution_params
    epsilon = K.random_normal(shape=K.shape(mean), mean=0., stddev=1.)
    return mean + K.exp(log_var / 2) * epsilon

def sampling(input_1,input_2):
    #input1 = layers.Lambda(sampling_model, name='encoder_output')([mean, log_var])
    mean = keras.Input(shape=input_1, name='input_layer1')
    log_var = keras.Input(shape=input_2, name='input_layer2')
    out = layers.Lambda(sampling_model, name='encoder_output')([mean, log_var])
    enc_2 = tf.keras.Model([mean,log_var], out,  name="Encoder_2")
    return enc_2

def encoder(input_encoder):    
    inputs = keras.Input(shape=input_encoder, name='input_layer')    
    # Block-1
    x = layers.Conv2D(32, kernel_size=3, strides= 2, padding='same', name='conv_1')(inputs)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.LeakyReLU(name='lrelu_1')(x)
    # Block-2
    x = layers.Conv2D(64, kernel_size=3, strides= 2, padding='same', name='conv_2')(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.LeakyReLU(name='lrelu_2')(x)
    # Block-3
    x = layers.Conv2D(64, 3, 2, padding='same', name='conv_3')(x)
    x = layers.BatchNormalization(name='bn_3')(x)
    x = layers.LeakyReLU(name='lrelu_3')(x)
    # Block-4
    x = layers.Conv2D(64, 3, 2, padding='same', name='conv_4')(x)
    x = layers.BatchNormalization(name='bn_4')(x)
    x = layers.LeakyReLU(name='lrelu_4')(x)
    # Block-5
    x = layers.Conv2D(64, 3, 2, padding='same', name='conv_5')(x)
    x = layers.BatchNormalization(name='bn_5')(x)
    x = layers.LeakyReLU(name='lrelu_5')(x)
    # Final Block
    flatten = layers.Flatten()(x)
    mean = layers.Dense(200, name='mean')(flatten)
    log_var = layers.Dense(200, name='log_var')(flatten)
    model = tf.keras.Model(inputs, (mean, log_var), name="Encoder")
    return model

def decoder(input_decoder):   
    inputs = keras.Input(shape=input_decoder, name='input_layer')
    x = layers.Dense(4096, name='dense_1')(inputs)
    x = layers.Reshape((8,8,64), name='Reshape')(x)
    # Block-1
    x = layers.Conv2DTranspose(64, 3, strides= 2, padding='same',name='conv_transpose_1')(x)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.LeakyReLU(name='lrelu_1')(x)
    # Block-2
    x = layers.Conv2DTranspose(64, 3, strides= 2, padding='same', name='conv_transpose_2')(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.LeakyReLU(name='lrelu_2')(x)
    # Block-3
    x = layers.Conv2DTranspose(64, 3, 2, padding='same', name='conv_transpose_3')(x)
    x = layers.BatchNormalization(name='bn_3')(x)
    x = layers.LeakyReLU(name='lrelu_3')(x)
    # Block-4
    x = layers.Conv2DTranspose(32, 3, 2, padding='same', name='conv_transpose_4')(x)
    x = layers.BatchNormalization(name='bn_4')(x)
    x = layers.LeakyReLU(name='lrelu_4')(x)
    # Block-5
    outputs = layers.Conv2DTranspose(3, 3, 2,padding='same', activation='sigmoid', name='conv_transpose_5')(x)
    model = tf.keras.Model(inputs, outputs, name="Decoder")
    return model

def mse_loss(y_true, y_pred):
    r_loss = K.mean(K.square(y_true - y_pred), axis = [1,2,3])
    return 1000 * r_loss

def kl_loss(mean, log_var):
    the_kl_loss =  -0.5 * K.sum(1 + log_var - K.square(mean) - K.exp(log_var), axis = 1)
    return the_kl_loss

def vae_loss(y_true, y_pred, mean, var):
    r_loss = mse_loss(y_true, y_pred)
    the_kl_loss = kl_loss(mean, var)
    return  r_loss + the_kl_loss

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images, enc, dec, final, optimizer):

    with tf.GradientTape() as encoder, tf.GradientTape() as decoder:
      
        mean, log_var = enc(images, training=True)
        latent = final([mean, log_var])
        generated_images = dec(latent, training=True)
        loss = vae_loss(images, generated_images, mean, log_var)

    gradients_of_enc = encoder.gradient(loss, enc.trainable_variables)
    gradients_of_dec = decoder.gradient(loss, dec.trainable_variables)    
    
    optimizer.apply_gradients(zip(gradients_of_enc, enc.trainable_variables))
    optimizer.apply_gradients(zip(gradients_of_dec, dec.trainable_variables))
    return loss



def train(dataset, epochs, enc, dec, final, optimizer):
    for epoch in range(epochs):
        start = time.time()
        i = 0
        loss_ = []
        for image_batch in dataset:
            #print('tf.shape(image_batch[0])',tf.shape(image_batch[0]))
            #print('type(image_batch[0])',type(image_batch[0]))
            #exit()
            i += 1
            loss = train_step(tf.reshape(image_batch[0],[-1, 256, 256, 1] ), enc, dec, final, optimizer)
            #loss = train_step(image_batch[0], enc, dec, final, optimizer )

        seed = image_batch[:25]
        # Save the model every 15 epochs
        #if (epoch + 1) % 15 == 0:
        #checkpoint.save(file_prefix = checkpoint_prefix)
        #enc.save_weights('tf_vae/cartoon/training_weights/enc_'+ str(epoch)+'.h5')
        #dec.save_weights('tf_vae/cartoon/training_weights/dec_'+ str(epoch)+'.h5')
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start),
               '; Epoch Loss',np.mean(loss))
    #checkpoint.save(file_prefix = checkpoint_prefix)
    enc.save_weights('tf_vae/turtle/training_weights/enc_'+ str(epoch)+'.h5')
    dec.save_weights('tf_vae/turtle/training_weights/dec_'+ str(epoch)+'.h5')
    # Generate after the final epoch
    generate_and_save_images([enc,final,dec], epochs, tf.reshape(seed[0],[-1, 256, 256, 1]) )

def generate_and_save_images(model, epoch, test_input):

  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
    #mean, var = enc(test_input, training=False)
    mean, var = model[0](test_input, training=False)
    #latent = final([mean, var])
    latent = model[1]([mean, var])
    #predictions = dec(latent, training=False)
    predictions = model[2](latent, training=False)
    #print(predictions.shape)
    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(5, 5, i+1)
        pred = predictions[i, :, :, :] * 255
        pred = np.array(pred)  
        pred = pred.astype(np.uint8)
        #cv2.imwrite('tf_ae/images/image'+ str(i)+'.png',pred)
        
        plt.imshow(pred)
        plt.axis('off')

    plt.savefig('tf_vae/turtle/images/image_at_epoch_{:d}.png'.format(epoch))
    plt.show()


def setup_training(num_epochs):
    img_height, img_width = 256, 256
    batch_size = 128

    os.makedirs('tf_vae/celeb/training_weights', exist_ok=True)
    os.makedirs('tf_vae/celeb/images', exist_ok=True)

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        #'../cartoonset100k',
        '/home/jack/data/celebaOther/train/',
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode=None)

    normalization_layer = layers.experimental.preprocessing.Rescaling(scale= 1./255)
    normalized_ds = train_ds.map(lambda x: normalization_layer(x))
    image_batch = next(iter(normalized_ds))
    first_image = image_batch[0]
    print(np.min(first_image), np.max(first_image))

    input_encoder = (256, 256)
    input_decoder = (200,)

    input_1 = (200,)
    input_2 = (200,)

    enc = encoder(input_encoder)
    dec = decoder(input_decoder)

    optimizer = tf.keras.optimizers.Adam(lr = 0.0005)
    final = sampling(input_1,input_2)
    
    train(normalized_ds, num_epochs, enc, dec, final, optimizer)

#setup_training(150)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def plot_prediction(predictions):
    pred = predictions[0, :, :, :] * 255
    pred = np.array(pred)  
    pred = pred.astype(np.uint8)
    #cv2.imwrite('tf_ae/images/image'+ str(i)+'.png',pred)
    plt.imshow(pred)
    plt.axis('off')
    plt.show()
    #for i in range(3):
    #    print(pred[:,:,i])

    pred = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)
    #pred = rgb2gray(pred)
    print("shape(pred))",pred.shape)
    plt.imshow(pred)
    plt.show()
    #print(pred)
    
def exec_main_loop(model, model_inf, q, pubS):
    saveDot3 = False #
    saveDot5 = False
    saveDot7 = False # True
    saveDot9 = False # True
    saveBest = False #True
    toLearn = False #True
    
    best_ssim = 0
    
    dot25 = True
    window = 10
    initial_train = 10 #300 #5000 #100 #2000
    
    #inference_thread = threading.Thread(target=inference_thread_f, args=(model, model_inf,))
    #inference_thread.start()

    vid_gen = lambda: generator_from_queue(q,Config.BATCH_SIZE, initial_train)
    vid_dataset = tf.data.Dataset.from_generator(
        vid_gen,
        (tf.float32, (tf.float32,tf.float32) ) )#,
    
    #X = y = q.dequeue_many( 3 * 10 )
    print('dequed')
    
    print('take one',type(vid_dataset.take(1)))
    print('type',type(vid_dataset))
    
    #exit()
    ###################### VAE ###################### VAE
    train_VAE = False
    input_encoder = ( 256, 256, 1 )
    input_decoder = (200,)
    test_num_epochs = 500
    input_1 = (200,)
    input_2 = (200,)

    enc = encoder(input_encoder)
    dec = decoder(input_decoder)
    #enc.summary()
    #dec.summary()
    #exit()
    optimizer = tf.keras.optimizers.Adam(lr = 0.0005)
    final = sampling(input_1,input_2)
    if train_VAE:
        train(vid_dataset, test_num_epochs, enc, dec, final, optimizer)
    else:
        enc.load_weights('tf_vae/turtle/training_weights/enc_499.h5')
        dec.load_weights('tf_vae/turtle/training_weights/dec_499.h5')
        print('loaded weights')
    #exit()

    ###################### end VAE ###################### end VAE

    
    timestr = time.strftime("%Y%m%d-%H%M%S")    
    reconFile = open("csvICRA/SSIM-1dot5e-4_VAE"+ timestr+ ".csv","w")
    reconFile.write("Epoch, SSIM, Average\n")
    #logdir = "~/logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    #tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    
    num_train = 1   #this is our train interations going forward 
    vid_gen = lambda: generator_from_queue(q,Config.BATCH_SIZE, num_train)
    vid_dataset = tf.data.Dataset.from_generator(
        vid_gen,
        (tf.float32, (tf.float32,tf.float32) ) )#,

    ssim_lst= [.380, .380, .380, .380, .380, .380, .380, .380,.380]
    
    old_time = current_ms()
    fit_frequency = 10
    reset_frequency = 10000 # so the rl can learn reward from images
    # Main loop
    #model_reload = input("Reload previous model dot3, dot5, dot7 or dot9 or best? 3, 5, 7, 8, 9, b : ")
    
    window = 1
    for i in range(sys.maxsize**10): # billions of loops ############### MAIN LOOP

        now_set = q.dequeue_many(Config.BATCH_SIZE * window)
        #print(current_ms() - old_time, 'ms per')
        old_time = current_ms()
        #now_set = np.reshape(now_set,(-1,window,256,256,1)) #vae
        now_set = np.reshape(now_set,(window,256,256,1)) #vae
        now_image = np.reshape(now_set,(256,256,1)) #vae

        #gen_frames = model.predict( now_set, batch_size=1 ) #vae
        mean, var = enc(now_set, training=False) #wants (-1, 256,256,1)
        latent = final([mean, var])
        predictions = dec(latent, training=False)
        #plot_prediction(predictions)
        
        pred = np.reshape(predictions,(256,256,3)) #
        #pred = predictions[0, :, :, :] * 255
        pred = np.array(pred)  
        #pred = pred.astype(np.uint8)
        pred = rgb2gray(pred) #cv2.cvtColor(pred, cv2.COLOR_GRAY2RGB)

        now_image = np.reshape(now_image,(256,256)) #

        struct_similiar = np.array([ssim( pred, now_image, data_range=1) ] )

        if struct_similiar[0] > best_ssim and saveBest:
            best_ssim = struct_similiar[0]   
            model.save_weights('models/bestModel')  # save good model
            #saveBest = False
            print(datetime.now().strftime("%m-%d_%H:%M"),'save model to best')
            #exit('exiting with', ssim_save)
 
        pubS.publish(struct_similiar.astype(dtype=np.float32))
      
        ssim_lst.append(struct_similiar[0])
        if i % 10 == 0:
            mov_avg = sum(ssim_lst[-50:])/50
            line_str = "#" * int(struct_similiar[0]*100)
            fileWrite = (str(i) + ',' + str(struct_similiar[0])  + ',' + str(round(mov_avg, 2)) + '\n' )
            print('i:',i, ',  ssim:',round(struct_similiar[0],4), ',  mov_avg:',round(mov_avg, 2) )
            x = reconFile.write(fileWrite)

    exit()
    model.save(Config.MODEL_PATH)
    return model

  
def evaluate(q):
    #pub = rospy.Publisher('latent', numpy_msg(Floats), queue_size=1 )#Float32MultiArray, queue_size=2)
    pubS = rospy.Publisher('ssim', Float32, queue_size=1)    
    #rospy.init_node('SSIM_latent_space')

    #model = get_model(True)
    model, model_inf = get_func_model(True)
    print("got models")
    #print("\n ################# model.summary")
    #print(model.summary())
    #print("\n ################# model_inf.summary")
    #print(model_inf.summary())
    #exit()
    exec_main_loop(model, model_inf, q, pubS)    



def current_ms():
    return round(time.time() * 1000)

class image_converter(object):
  def __init__(self,q):
    self.bridge = CvBridge()
    self.q = q
    self.image_sub = rospy.Subscriber("/camera/image",Image,self.callback)

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)
    #print("type",type(cv_image[0,0,0]))
    q_image = cv2.resize(cv_image, (256, 256))
    q_image = cv2.cvtColor(q_image, cv2.COLOR_BGR2GRAY)
    q_image = np.array(q_image, dtype=np.float32) / 256.0
    self.q.enqueue( np.reshape( q_image, ( 256, 256, 1 ) ) )
    #print('size',self.q.size())
    #cv2.imshow("Image window", cv_image)
    #cv2.waitKey(3)

def build_q():
    q_size = 100 #20
    shape=(256,256,1)
    q = tf.queue.FIFOQueue(q_size, [tf.float32], shapes=shape)
    return q

  
class Config:
    BATCH_SIZE = 1 # 4 was original
    EPOCHS = 1 # change back to 3 (Jack)
    MODEL_PATH = "/home/jack/src/video-anomaly-detection-master/notebooks/lstmautoencoder/model.hdf5"
  


    
def encoder_model(window=10, height=256,width=256):
    """
    Parameters
    ----------
    reload_model : bool
        Load saved model or retrain it
    """
    normalizer_1 = LayerNormalization()
    normalizer_2 = LayerNormalization()
    normalizer_3 = LayerNormalization()
    normalizer_4 = LayerNormalization()
    
    model_input = keras.Input(shape=(window, width, height, 1))
    conv_2d_layer_1 = Conv2D(128, (11, 11), strides=4, padding="same")
    time_D_layer_1 =  TimeDistributed(conv_2d_layer_1)(model_input)
    normalize_layer_1 = normalizer_1(time_D_layer_1)    
    conv_2d_layer_2 = Conv2D(64, (5, 5), strides=2, padding="same")
    time_D_layer_2 =  TimeDistributed(conv_2d_layer_2)(normalize_layer_1) 
    normalize_layer_2 = normalizer_2(time_D_layer_2)
    # # # # #
    lstm_layer_1 = ConvLSTM2D(64, (3, 3), padding="same", return_sequences=True)(normalize_layer_2)
    lstm_norm_1 = normalizer_3(lstm_layer_1)
    lstm_layer_2 = ConvLSTM2D(32, (3, 3), padding="same", return_sequences=True)(lstm_norm_1)
    
    return model_input, normalizer_4(lstm_layer_2)

def decoder_model(encoder_model):
    normalizer_5 = LayerNormalization()
    normalizer_6 = LayerNormalization()
    normalizer_7 = LayerNormalization()
    lstm_layer_3 = ConvLSTM2D(64, (3, 3), padding="same", return_sequences=True)(encoder_model)
    lstm_norm_3 = normalizer_5(lstm_layer_3)
    # # # # #
    conv_2d_layer_D1 = Conv2DTranspose(64, (5, 5), strides=2, padding="same")
    time_D_layer_D1 =  TimeDistributed(conv_2d_layer_D1)(lstm_norm_3)
    normalize_layer_D1 = normalizer_6(time_D_layer_D1)
    conv_2d_layer_D2 = Conv2DTranspose(128, (11, 11), strides=4, padding="same")
    time_D_layer_D2 =  TimeDistributed(conv_2d_layer_D2)(normalize_layer_D1)
    normalize_layer_D2 = normalizer_7(time_D_layer_D2)
    conv_2d_layer_D3 = Conv2D(1, (11, 11), activation="sigmoid", padding="same")
    model_output = TimeDistributed( conv_2d_layer_D3 )( normalize_layer_D2 )

    return model_output

def generator_from_queue(q, batch_size, gLoop):
    window = 10
    for i in range(gLoop):
      X = y = q.dequeue_many( batch_size * window )
      X = np.reshape(X, (-1,10,256,256))
      y = (np.reshape(y, (-1,10,256,256)), np.zeros((1, 10, 32, 32, 32), dtype=np.float32) )
      yield (X, y)

def generator_from_queue_VAE(q, batch_size, gLoop):
    window = 10
    for i in range(gLoop):
      X = y = q.dequeue_many( batch_size * window )
      X = np.reshape(X, (-1,10,256,256))
      y = (np.reshape(y, (-1,10,256,256)), np.zeros((1, 10, 32, 32, 32), dtype=np.float32) )
      yield X


def generator_from_queue_test(q, batch_size, gLoop):
    window = 10
    for i in range(gLoop):
      X = y = q.dequeue_many( batch_size * window )
      X = np.reshape(X, (-1,10,256,256))
      print('new type', type(X))
      g1 = tf.random.Generator.from_seed(1)
      y_train = [g1.normal(shape=[1,10,256,256,1]).astype(np.float32), g1.normal(shape=[1,10,32,32, 32]).astype(np.float32)]
      yield (X, y_train)




def get_func_model(reload_model=True):  # this is predict next 10
    """
    Parameters
    ----------
    reload_model : bool
        Load saved model or retrain it
    """

    model_inputs, encode_only = encoder_model()
    model_all_layers = decoder_model(encode_only)
    #with tf.device('/gpu:0'):    
    model = keras.Model( inputs=[model_inputs],
                         outputs=[model_all_layers],
                         name="FullConvLSTM_AE") # (None, 10, 32, 32, 32)
    #model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=1e-4, decay=1e-5, epsilon=1e-6), metrics=["mae"])
    model.compile( loss=['mse','mse'], optimizer=tf.keras.optimizers.Adam(lr=1.5e-4, decay=1e-5, epsilon=1e-6),
                   metrics=["mae"], loss_weights=[1.0, 0.0])
        
    #with tf.device('/gpu:1'):
    model_inf = keras.Model( inputs=[model_inputs],
                             outputs=[model_all_layers],#outputs=[encode_only],
                             name="encoder_only") # (None, 10, 32, 32, 32)
    model_inf.compile( loss=['mse','mse'], optimizer=tf.keras.optimizers.Adam(lr=1.5e-4, decay=1e-5, epsilon=1e-6),
                       metrics=["mae"], loss_weights=[1.0, 0.0])
    print('models compliled')
    return model, model_inf


      
if __name__ == '__main__':
    main(sys.argv)
