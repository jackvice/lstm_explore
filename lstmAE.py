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
#from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2DTranspose, ConvLSTM2D, BatchNormalization
from tensorflow.keras.layers import TimeDistributed, Conv2D, LayerNormalization, Conv3D
from tensorflow.keras.models import Sequential, load_model
from os import listdir
from os.path import isfile, join, isdir




def main(args):  
    rospy.init_node('lstm_AE', anonymous=True)
    q = build_q()
    ic = image_converter(q) #pass tf q
    evaluate(q)
    cv2.destroyAllWindows()
    q.close()




class inference_obj(object):
    def __init__(self, model, model_inf): #copy weights from model
        self.pub_ssim = rospy.Publisher('ssim', Float32, queue_size=1)
        #self.pub_latent = rospy.Publisher('latent', numpy_msg(Floats))#, queue_size=65536 )#Float32MultiArray, queue_size=2)
        self.pub_ae_image = rospy.Publisher('ae_image', numpy_msg(Floats))#, queue_size=65536 )#Float32MultiArray, queue_size=2)
        self.debug = rospy.Publisher('debug', String , queue_size=1 )#Float32MultiArray, queue_size=2)
        self.bridge = CvBridge()
        self.fifo_set = np.zeros( (1,15,256,256,1), dtype=float ) #initialize fifo buffer of 20 frames
        #self.call_count = 20 # start at 20 so we get a weight update first    
        self.model = model # to copy weights
        self.model_inf = model_inf
        self.i = 0 #
        ssim_lst= [.380, .380, .380, .380, .380, .380, .380, .380,.380]
        self.image_subscribe = rospy.Subscriber("/camera/image", Image, self.callback)# ,queue_size = 20
        #self.old_time = current_ms()
        
        
    def callback(self, data):
        old_time = current_ms()
        if self.i % 90 == 0: #about every three seconds
            self.model_inf.set_weights(self.model.get_weights())
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        cv_image = cv2.resize(cv_image, (256, 256))
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        cv_image = np.reshape( ( np.array( cv_image, dtype=np.float32) / 256.0),
                                (1, 1, 256, 256, 1 ) ) # Scale 0-1 ?
        self.fifo_set = np.concatenate((cv_image, self.fifo_set[:,:-1,:,:,:]), axis=1) # push new image to front

        predicted_frames = self.model_inf.predict( self.fifo_set[ : ,:10, : , : , : ], batch_size = 1 )
        
        #self.pub_ae_image.publish( predicted_frames[ 0 , 0].flatten() ) # 1st frame from prediction
        
        #self.debug.publish(str(current_ms() - old_time) + ' callback total ms, i: ' + str(self.i) )
        self.debug.publish('predicted_frames_type '+str(type(predicted_frames) ))
        self.old_time = current_ms()
        
        #sin_val = 3.5 # where to put the sine curve to deminish reward after learned         
        struct_similiar = np.array([ssim( np.reshape(self.fifo_set[0,14], (256, 256) ), #use the 5th
                                          np.reshape(predicted_frames[0,4], (256,256) ),
                                          data_range=1 )])
        #struct_similiar = np.array([0.9])
        self.pub_ssim.publish(struct_similiar.astype(dtype=np.float32))
        
        #if self.i % 3000 == 0:
        if self.i % 300 == 0:
            trueImg = self.fifo_set[0,14] * 256
            trueImg = trueImg.astype(int)
            cv2.imwrite("outImages/inferenceTrue"+ str(self.i) +".png", trueImg )
            rImg = predicted_frames[0,4] * 256
            rImg = rImg.astype(int)
            cv2.imwrite("outImages/inferenceRecon"+ str(self.i) +".png", rImg )
        self.i += 1


def inference_thread_f(model, model_inf):
    ic = inference_obj(model, model_inf)
    try:
        rospy.spin()
    except:
        print("fail")

    
def exec_main_loop(model, model_inf, q, pubS):

    saveDot3 = False #
    saveDot5 = False
    saveDot7 = False # True
    saveDot9 = False # True
    saveBest = False #True
    toLearn = False

    """
    saveDot3 = True # False #
    saveDot5 = True
    saveDot7 = True # True
    saveDot9 = True
    saveBest = True
    toLearn = True  
    """    
    
    best_ssim = 0
    
    dot25 = True
    window = 10
    initial_train = 10 #300 #5000 #100 #2000
    
    inference_thread = threading.Thread(target=inference_thread_f, args=(model, model_inf,))
    inference_thread.start()

    vid_gen = lambda: generator_from_queue(q,Config.BATCH_SIZE, initial_train)
    vid_dataset = tf.data.Dataset.from_generator(
        vid_gen,
        (tf.float32, (tf.float32,tf.float32) ) )#,
    
    X = y = q.dequeue_many( 3 * 10 )
    print('dequed')

    model.fit(vid_dataset,
                batch_size=Config.BATCH_SIZE,
                epochs=Config.EPOCHS,
                shuffle=False)
    print('done Fit')
    timestr = time.strftime("%Y%m%d-%H%M%S")    
    reconFile = open("csvICRA/SSIM-1dot5e-4"+ timestr+ ".csv","w")
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
    model_reload = input("Reload previous model dot3, dot5, dot7 or dot9 or best? 3, 5, 7, 8, 9, b : ")
    if model_reload == '3':
        print('loading previous .3 model')
        model.load_weights('models/dot3Model')
    elif model_reload == '5':
        print('loading previous .5 model')
        model.load_weights('models/dot5Model')
    elif model_reload == '7':
        print('loading previous .7 model')
        model.load_weights('models/dot7Model')
    elif model_reload == '9':
        print('loading previous .9 model')
        model.load_weights('models/dot7Model')
    elif model_reload == '8':
        print('loading previous .85 model')
        model.load_weights('models/dot85Model')
    elif model_reload == 'b':
        print('loading previous best model')
        model.load_weights('models/bestModel')
    else:
        print('loading no model')
        #model.save_weights('naiveModel')
    for i in range(sys.maxsize**10): # billions of loops ############### MAIN LOOP
      if (i % fit_frequency == 0) and toLearn:
            model.fit(vid_dataset, 
                      batch_size=Config.BATCH_SIZE,
                      epochs=Config.EPOCHS, shuffle=False, verbose=0)
      now_set = q.dequeue_many(Config.BATCH_SIZE * window)
      #print(current_ms() - old_time, 'ms per')
      old_time = current_ms()
      now_set = np.reshape(now_set,(-1,window,256,256,1))
      #print('now shape',now_set.shape)
      #exit()

      gen_frames = model.predict( now_set, batch_size=1 )

      if False: # i % 100 == 0:
        #print('now.shape',tf.shape(now_set))
        #print('gen_frames.shape',tf.shape(gen_frames))
        tImg = now_set[0,4] * 256
        tImg = tImg.astype(int)
        rImg = gen_frames[0,4] * 256
        rImg = rImg.astype(int)

        cv2.imwrite("outImages/generated.png", rImg )
        cv2.imwrite("outImages/actual.png", tImg )
        tImg = np.reshape(tImg,(256,256))
        rImg = np.reshape(rImg,(256,256))
        reconFile.flush()
        #exit()
      struct_similiar = np.array([ssim( np.reshape(now_set[0,9], (256, 256) ), #use the 10th frame.
                              np.reshape(gen_frames[0,9], (256,256) ),
                              data_range=1 )])
      #if struct_similiar[0] > 0.25 and dot25 == True:
      #    model.save_weights('ssimDot25Model')
      #    dot25=False
      
      if struct_similiar[0] > 0.3 and saveDot3: 
          model.save_weights('models/dot3Model')  # save good model
          saveDot3 = False
          print(datetime.now().strftime("%m-%d_%H:%M"),'save model to dot3')
          #exit('exiting with', ssim_save)
      if struct_similiar[0] > 0.5 and saveDot5: 
          model.save_weights('models/dot5Model')  # save good model
          saveDot5 = False
          print(datetime.now().strftime("%m-%d_%H:%M"),'save model to dot5')
          #exit('exiting with', ssim_save)
      if struct_similiar[0] > 0.7 and saveDot7:   
          model.save_weights('models/dot7Model')  # save good model
          saveDot7 = False
          print(datetime.now().strftime("%m-%d_%H:%M"),'save model to dot7')
          #exit('exiting with', ssim_save)
      if struct_similiar[0] > 0.9 and saveDot9:   
          model.save_weights('models/dot9Model')  # save good model
          saveDot9 = False
          print(datetime.now().strftime("%m-%d_%H:%M"),'save model to dot9')
          #exit('exiting with', ssim_save)
      if struct_similiar[0] > best_ssim and saveBest:
          best_ssim = struct_similiar[0]   
          model.save_weights('models/bestModel')  # save good model
          #saveBest = False
          print(datetime.now().strftime("%m-%d_%H:%M"),'save model to best')
          #exit('exiting with', ssim_save)
     #sin_val = 3.5 # where to put the sine curve to deminish reward after learned 
 
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
    pubS = rospy.Publisher('ssim_old', numpy_msg(Floats), queue_size=1)    
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

    test = get_single_test()
    print(test.shape)
    sz = test.shape[0] - 10 + 1
    sequences = np.zeros((sz, 10, 256, 256, 1))
    # apply the sliding window technique to get the sequences
    for i in range(0, sz):
        clip = np.zeros((10, 256, 256, 1))
        for j in range(0, 10):
            clip[j] = test[i + j, :, :, :]
        sequences[i] = clip

    print("got data")
    # get the reconstruction cost of all the sequences
    reconstructed_sequences = model.predict(sequences,batch_size=1)
    
    sequences_reconstruction_cost = np.array([np.linalg.norm(np.subtract(sequences[i],
                                                                         reconstructed_sequences[i]))
                                              for i in range(0,sz)])
    sa = ( ( sequences_reconstruction_cost - np.min(sequences_reconstruction_cost)) /
           np.max(sequences_reconstruction_cost) )
    sr = 1.0 - sa

    # plot the regularity scores
    #plt.plot(sr)
    #plt.ylabel('regularity score Sr(t)')
    #plt.xlabel('frame t')
    #plt.show()

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
