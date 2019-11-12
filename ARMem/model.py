import tensorflow as tf
from tensorflow.contrib import layers
import os

# AR_memory
class Model(object):
    def __init__(self, config, input_x=None, memories=None, targets=None):
        self.config = config
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.regularizer = layers.l2_regularizer(self.config.l2_lambda)
        self.sess = None
        self.saver = None
        self.input_x = input_x
        self.memories = memories
        self.targets = targets
        self._build_model()
        
    def _build_model(self):
        self.add_placeholder()
        # get auto-regression
        # input_ar : [b, nf]
        with tf.variable_scope("inputs"):
            input_ar, ar_loss = self.auto_regressive(self.input_x, self.config.ar_lambda, self.config.x_len, scope="inputs_")
            
        with tf.variable_scope("memories"):    
            # memory : [b, (n+1)*m, nf] -> [b*m, n+1, nf]
            memories = tf.concat(tf.split(self.memories, self.config.msteps, axis=1), axis=0)
            # memories = tf.reshape(self.memories, shape=[-1, self.config.nsteps+1, self.config.nfeatures])
            # memory_ar : [b*m, nf]
            memory_ar, ar_loss_ = self.auto_regressive(memories, self.config.ar_lambda, self.config.x_len+1, scope="memories")
        
        # context: [b, nf]
        context = self.attention(input_ar, memory_ar)
        linear_inputs = tf.concat([input_ar, context], axis=1) # [b, 2nf] 
        self.predictions = tf.layers.dense(linear_inputs, self.config.nfeatures, activation=tf.nn.tanh,
                                           kernel_initializer=layers.xavier_initializer(), use_bias=False)
        #self.predictions = input_ar
        self.loss = tf.losses.mean_squared_error(labels=self.targets, predictions=self.predictions)
        #self.loss = tf.reduce_mean(tf.abs(self.targets - self.predictions))
        
        # metric
        error = tf.reduce_sum((self.targets - self.predictions)**2) ** 0.5
        denom = tf.reduce_sum((self.targets - tf.reduce_mean(self.targets))**2) ** 0.5
        self.rse = error / denom
        self.mape = tf.reduce_mean(tf.abs((self.targets - self.predictions)/self.targets))
        self.smape = tf.reduce_mean(2*tf.abs(self.targets-self.predictions)/(tf.abs(self.targets)+tf.abs(self.predictions)))
        self.mae = tf.reduce_mean(tf.abs(self.targets - self.predictions))
        '''
        if self.config.l2_lambda > 0:
            reg_vars = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            reg_term = layers.apply_regularization(self.regularizer, reg_vars)
            self.loss += reg_term
        '''    
        # self.loss = self.loss + ar_loss + ar_loss_
        
        self.add_train_op()
        self.initialize_session()

    def add_placeholder(self):
        if self.input_x is None:
            self.input_x = tf.placeholder(shape=[None, self.config.nsteps, self.config.nfeatures],dtype=tf.float32, name="x")
        if self.targets is None:
            self.targets = tf.placeholder(shape=[None, self.config.nfeatures], dtype=tf.float32, name="targets")
        if self.memories is None:
            self.memories = tf.placeholder(shape=[None, (self.config.nsteps+1) * self.config.msteps, self.config.nfeatures], dtype=tf.float32,
                                       name="memories")
        # self.targets = tf.placeholder(shape=[None], dtype=tf.int32, name="targets")
        self.dropout = tf.placeholder(dtype=tf.float32, name="dropout")
        
    def auto_regressive(self, inputs, ar_lambda, x_len, scope, reuse=False):
        # y_t,d = sum_i(w_i * y_i,d) + b_d
        w = tf.get_variable(shape=[x_len, self.config.nfeatures],
                            initializer=layers.xavier_initializer(),
                            name="w")
        bias = tf.get_variable(shape=[self.config.nfeatures],
                    initializer=tf.zeros_initializer(),
                    name="bias")     
        w_ = tf.expand_dims(w, axis=0)
        # weighted: [b, nf]
        weighted = tf.reduce_sum(inputs * w_, axis=1) + bias
        
        ar_loss = ar_lambda * tf.reduce_sum(tf.square(w))
        return weighted, ar_loss
    
    def attention(self, inputs, memories):
        # use MLP to compute attention score
        # given input, attend memories(m1, m2, m3, mnstep)
        
        # inputs : [b, nf] -> [b, 1, nf]
        query = tf.expand_dims(inputs, axis=1)
        # [b, 1, attention_size]
        query = tf.layers.dense(query, self.config.attention_size,
                               activation=None,use_bias=False,
                               kernel_initializer=layers.xavier_initializer(),
                               kernel_regularizer=self.regularizer)
        
        # memories : [b*m, nf] -> [b, m, nf]  
        key = tf.reshape(memories, shape=[-1, self.config.msteps, self.config.nfeatures])
        # [b, m, attention_size]
        key = tf.layers.dense(key, self.config.attention_size,
                              activation=None,use_bias=False,
                               kernel_initializer=layers.xavier_initializer(),
                               kernel_regularizer=self.regularizer)
        
        bias = tf.get_variable(shape=[self.config.attention_size],
                              initializer=tf.zeros_initializer(),
                              name="attention_bias")
        
        # projection : [b, m, attention_size]
        projection = tf.nn.tanh(query + key + bias)
        # sim_matrix : [b, m, 1]
        sim_matrix = tf.layers.dense(projection, 1, activation=None)
        sim_matrix = tf.nn.softmax(sim_matrix, 1)
        # context : [b, 1, nf] -> [b, nf]
        context = tf.matmul(tf.transpose(sim_matrix, [0,2,1]), key)
        context = tf.squeeze(context, axis=1)

        return context
    
 
    def add_train_op(self):
        opt = tf.train.AdamOptimizer(self.config.lr)
        vars = tf.trainable_variables()
        grads = tf.gradients(self.loss, vars)
        clipped_grads,_ = tf.clip_by_global_norm(grads,self.config.clip)
        self.train_op = opt.apply_gradients(zip(grads,vars), global_step=self.global_step)
        
    def initialize_session(self):
        """Defines self.sess and initialize the variables"""
        if not self.config.allow_gpu:
            config = tf.ConfigProto(device_count={'GPU': 0})
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"            
        else:   
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            config.gpu_options.per_process_gpu_memory_fraction = 0.9
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        
    def save_session(self, model_name):
        """Saves session = weights"""
        self.saver.save(self.sess, model_name)
        
    def restore_session(self, dir_model):
        """Reload weights into session
        Args:
            sess:tf.Session()
            dir_model: dir with weights
        """
        self.saver.restore(self.sess, tf.train.latest_checkpoint(dir_model))
    
    def train(self, input_x, mem, targets):
        feed_dict = {
            self.input_x: input_x,
            self.memories: mem,
            self.targets: targets,
            self.dropout: self.config.dropout
        }
    
        output_feed = [self.train_op, self.loss, self.rse, self.smape, self.mae, self.global_step]
        _, loss, rse, smape, mae, step = self.sess.run(output_feed, feed_dict)
      
        return loss, rse, smape, mae, step
        
        
    def eval(self, input_x, mem, targets):
        feed_dict = {
            self.input_x: input_x,
            self.memories: mem,
            self.targets: targets,
            self.dropout: 1.0
        }
        output_feed = [self.predictions, self.loss, self.rse, self.smape, self.mae]
        pred, loss, rse, smape, mae = self.sess.run(output_feed, feed_dict)

        return pred, loss, rse, smape, mae

    
if __name__ == "__main__":
    from config import Config
    config = Config()
    model = Model(config)
    print("done")