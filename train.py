import bilsm_crf_model
from keras import backend as K
from tensorflow.python import debug as tfdbg
import tensorflow as tf

from keras import callbacks
import os
EPOCHS = 10
batch_size=16
USE_MULTI_INPUT = False
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# log.info("use gpu id:" + args.gpu_id)
debug = False

def gen_callbacks():
    tb = callbacks.TensorBoard('tensorboard-logs/',
                           batch_size=batch_size, histogram_freq=int(1))
    return [tb]

def training():
    if debug:
#         sess = K.get_session()
#         sess = tfdbg.TensorBoardDebugWrapperSession(sess, "0.0.0.0:9009")
#         K.set_session(sess)
        K.set_session(
            tfdbg.TensorBoardDebugWrapperSession(tf.Session(), "chaowei-SYS-7048GR-TR:9010"))
    
    
    
    
    if USE_MULTI_INPUT:
        model, (train_x, train_y), (test_x, test_y) = bilsm_crf_model.create_model2()
        # train model
        model.fit([train_x, train_x], train_y,batch_size=batch_size,epochs=EPOCHS, validation_data=[[test_x, test_x], test_y],callbacks=gen_callbacks())
        model.save('model/crf_multi.h5')
    else:
        model, (train_x, train_y), (test_x, test_y) = bilsm_crf_model.create_model()
        # train model
        model.fit(train_x, train_y,batch_size=batch_size,epochs=EPOCHS, validation_data=[test_x, test_y],callbacks=gen_callbacks())
        model.save('model/crf.h5')
        
if __name__ == "__main__":
    training()
