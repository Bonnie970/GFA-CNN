import os, pprint, time, h5py
import numpy as np
import tensorflow as tf
import tensorlayer as tl
pp = pprint.PrettyPrinter()
from utils_realAugm import *
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
from nets2 import vgg2
import tensorflow.contrib.slim as slim

flags = tf.app.flags
flags.DEFINE_integer("id_classes", 15, "The number of subject categories")
flags.DEFINE_integer("anti_classes", 2, "The number of spoofing categories")
flags.DEFINE_float("lam", 0.3, "weights of the pcLoss")
flags.DEFINE_integer("output_size", 224, "the height of images")
flags.DEFINE_integer("batch_size", 32, "The number of batch images [64]")
#flags.DEFINE_string("antiType", "crop_backgrd_mfsdPhoto", "the types of for antispoofing- raw/rpyPhotoAdv/rpyPhotoCon ...")
flags.DEFINE_string("tstPath", "../FDA_codes/Replay-Attack/rpy_2_mfsdStyle/rpy_2_mfsdStyle_test.txt", "testing data path")
flags.DEFINE_string("lblIndx", "./id_labels/mfsd_ids.txt", "lables index path")
flags.DEFINE_string("pre_model", "../models/mfsd2rpy/tpc_mfsdPhoto_realAugm3-3/3_model.ckpt", "pretrained weights") # The model is trained on dataset MFSD
flags.DEFINE_boolean("is_resize", True, "True for training, False for testing [False]")
FLAGS = flags.FLAGS

fbl = open(FLAGS.lblIndx)
files = fbl.readlines()
lblDic = {}
for xx in files:
    if(xx.split(' ')[2] == 'train'):
        ldc = {xx.split(' ')[0]:xx.split(' ')[3]}
        lblDic.update(ldc)
 
def obtain_testAccuracy(sess,acc_id,acc_anti,probs_anti, preLbl_anti, input_x_id, input_y_id, input_x_anti, input_y_anti, tstPath):
    f = open(tstPath)
    data_files = f.readlines()
    f.close()
    data_files = data_files[0::1]
    shuffle(data_files)
    batch_idxs = len(data_files) // FLAGS.batch_size

    test_accID = 0
    test_accAnti = 0
    test_cost = 0
    grdLbls = []
    preLbls = []
    probas = []
    for idx in range(0, batch_idxs):
        batch_files = data_files[idx*FLAGS.batch_size:(idx+1)*FLAGS.batch_size]
        # batch_files_anti = [v.replace('crop_backgrd', FLAGS.antiType) for v in batch_files]

            
        batch_id = [get_tst_image(batch_file, is_resize=FLAGS.is_resize, resize_w=FLAGS.output_size, is_grayscale = 0) for batch_file in batch_files]
        batch_labels_id = [get_idxxx(batch_file, FLAGS.id_classes) for batch_file in batch_files]
        batch_images_id = np.array(batch_id).astype(np.float32)
            
        batch_anti = [get_tst_image(batch_file, is_resize=FLAGS.is_resize, resize_w=FLAGS.output_size, is_grayscale = 0) for batch_file in batch_files]
        batch_labels_anti = [get_tst_antiLabel(batch_file, FLAGS.anti_classes) for batch_file in batch_files]
        batch_images_anti = np.array(batch_anti).astype(np.float32)
        
        tAccID, tAccAnti, pb, pl, gl = sess.run([acc_id, acc_anti, probs_anti, preLbl_anti,input_y_anti],feed_dict={input_x_id:batch_images_id, 
                                                   input_y_id:batch_labels_id, input_x_anti:batch_images_anti, input_y_anti:batch_labels_anti})
        pl = np.int64(pl)
        test_accID = test_accID + tAccID
        test_accID_avg = test_accID/(idx+1)
        test_accAnti = test_accAnti + tAccAnti
        test_accAnti_avg = test_accAnti/(idx+1)
        preLbls = np.append(preLbls, pl)
        grdLbls = np.append(grdLbls, gl)
        probas = np.append(probas, pb)

        if idx % 1 == 0:
            fp = open('../res/mfsd_2_rpy/print.txt', 'a+w')
            print('**Step %d, tAccID = %.4f, tAccAnti = %.4f%% **'%(idx,test_accID_avg,test_accAnti_avg))
            print(gl)
            print(pl)
            print>> fp, ('**Step %d, tAccID = %.4f, tAccAnti = %.4f%% **'%(idx,test_accID_avg,test_accAnti_avg))
            print>> fp, (gl)
            print>> fp, (pl)
            fp.close()
    return test_accAnti_avg, grdLbls, preLbls, probas

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    ##========================= DEFINE MODEL ===========================##
    input_x_id = tf.placeholder(tf.float32, [None, FLAGS.output_size, FLAGS.output_size, 3])
    input_x_anti = tf.placeholder(tf.float32, [None, FLAGS.output_size, FLAGS.output_size, 3])
    input_y_id = tf.placeholder(tf.int64, shape=[None, ], name='y_id_grdth')
    input_y_anti = tf.placeholder(tf.int64, shape=[None, ], name='y_anti_grdth')
    
    with slim.arg_scope(vgg2.vgg_arg_scope()):
        net_simaese_id, end_points1_id = vgg2.vgg_siamese(input_x_id) 
    with slim.arg_scope(vgg2.vgg_arg_scope()):
        net_simaese_anti, end_points1_anti = vgg2.vgg_siamese(input_x_anti) 
    net_id, end_points_id = vgg2.vgg_id(net_simaese_id, num_classes=FLAGS.id_classes, is_training=False) 
    net_anti, end_points_anti = vgg2.vgg_anti(net_simaese_anti, num_classes=FLAGS.anti_classes, is_training=False) 
    
    y_id = tf.reshape(net_id, [-1, FLAGS.id_classes])
    y_anti = tf.reshape(net_anti, [-1, FLAGS.anti_classes])
    probs_anti = tf.nn.softmax(y_anti) 

    correct_prediction_id = tf.equal(tf.cast(tf.argmax(y_id, 1), tf.float32), tf.cast(input_y_id, tf.float32))
    acc_id = tf.reduce_mean(tf.cast(correct_prediction_id, tf.float32))
    preLbl_id = tf.cast(tf.argmax(y_id, 1), tf.float32)
    
    correct_prediction_anti = tf.equal(tf.cast(tf.argmax(y_anti, 1), tf.float32), tf.cast(input_y_anti, tf.float32))
    acc_anti = tf.reduce_mean(tf.cast(correct_prediction_anti, tf.float32))
    preLbl_anti = tf.cast(tf.argmax(y_anti, 1), tf.float32)

    variables = slim.get_model_variables()

    ##========================= Test MODELS ================================##
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)

    saver_restore = tf.train.Saver(variables)
    saver_restore.restore(sess, FLAGS.pre_model)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord) 
                
#    [test_accAnti, grdLbls, preLbls, probas] = obtain_testAccuracy(sess,acc_id,acc_anti,probs_anti, preLbl_anti,input_x_id, 
#                                                                input_y_id, input_x_anti, input_y_anti, FLAGS.tstPath)
    [test_accAnti, grdLbls, preLbls, probas] = obtain_testAccuracy(sess,acc_id,acc_anti, probs_anti, preLbl_anti, input_x_id, 
                                                                input_y_id, input_x_anti, input_y_anti, FLAGS.tstPath)


if __name__ == '__main__':
    tf.app.run()


