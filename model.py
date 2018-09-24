import numpy as np
import tensorflow as tf
import my_attention_decoder_fn
import my_loss
import my_seq2seq

from tensorflow.python.ops.nn import dynamic_rnn
#from tensorflow.contrib.rnn.python.ops.core_rnn_cell import GRUCell, LSTMCell, MultiRNNCell
from tensorflow.python.ops.rnn_cell_impl import GRUCell, LSTMCell, MultiRNNCell
#from tensorflow.contrib.seq2seq.python.ops import attention_decoder_fn

#from tensorflow.contrib.seq2seq.python.ops.seq2seq import dynamic_rnn_decoder
#from tensorflow.contrib.seq2seq.python.ops.loss import sequence_loss
from tensorflow.contrib.lookup.lookup_ops import HashTable, KeyValueTensorInitializer
from tensorflow.contrib.layers.python.layers import layers
from output_projection import output_projection_layer
from tensorflow.python.ops import variable_scope

from utils import sample_gaussian
from utils import gaussian_kld

PAD_ID = 0
UNK_ID = 1
GO_ID = 2
EOS_ID = 3
_START_VOCAB = ['_PAD', '_UNK', '_GO', '_EOS']

class Seq2SeqModel(object):
    def __init__(self,
            num_symbols,
            num_embed_units,
            num_units,
            num_layers,
            is_train,
            vocab=None,
            topic_pos=None,
            non_topic_pos = None,
            func_pos = None,
            embed=None,
            #learning_rate=0.005,
            learning_rate=0.1,
            learning_rate_decay_factor=0.9995,
            max_gradient_norm=5.0,
            num_samples=512,
            max_length=30,
            latent_size=128,
            use_lstm=False,
            num_classes=3,
            full_kl_step=80000,
            num_keywords = 4,
            num_topic_symbols = 4000,
            num_non_topic_symbols = 40000):
        
        self.posts = tf.placeholder(tf.string, shape=(None, None))  # batch*len
        self.posts_length = tf.placeholder(tf.int32, shape=(None))  # batch
        self.responses = tf.placeholder(tf.string, shape=(None, None))  # batch*len
        self.responses_length = tf.placeholder(tf.int32, shape=(None))  # batch
        self.keywords = tf.placeholder(tf.string, shape=(None, None))  # batch*num_keywords
        self.keywords_length = tf.placeholder(tf.int32, shape=(None))  # batch
        self.labels = tf.placeholder(tf.float32, shape=(None, num_classes))
        self.use_prior = tf.placeholder(tf.bool)
        #self.is_anneal= tf.placeholder(tf.bool)
        self.global_t = tf.placeholder(tf.int32)
        #self.topic_one_hot = tf.reshape(tf.one_hot(topic_pos, num_symbols, 1.0, 0.0), [-1, num_symbols]) # num_topic_symbols * num_symbols
        #self.non_topic_one_hot = tf.reshape(tf.one_hot(non_topic_pos, num_symbols, 1.0, 0.0), [-1, num_symbols]) # num_non_topic_symbols * num_symbols
        self.topic_mask = tf.reduce_sum(tf.one_hot(topic_pos, num_symbols, 1.0, 0.0), axis = 0)
        self.func_mask = tf.reduce_sum(tf.one_hot(func_pos, num_symbols, 1.0, 0.0), axis = 0)
        self.ordinary_mask = tf.cast(tf.ones_like(self.topic_mask), tf.float32) - self.topic_mask - self.func_mask
        #self.no_topic_mask = tf.constant(1.0, shape = [])
        # build the vocab table (string to index)
        if is_train:
            self.symbols = tf.Variable(vocab, trainable=False, name="symbols")
        else:
            self.symbols = tf.Variable(np.array(['.']*num_symbols), name="symbols")
        self.symbol2index = HashTable(KeyValueTensorInitializer(self.symbols, 
            tf.Variable(np.array([i for i in range(num_symbols)], dtype=np.int32), False)), 
            default_value=UNK_ID, name="symbol2index")

        self.posts_input = self.symbol2index.lookup(self.posts)   # batch*len
        self.responses_target = self.symbol2index.lookup(self.responses)   #batch*len
        self.keywords_input = self.symbol2index.lookup(self.keywords) # batch*num_keywords
        
        batch_size, decoder_len = tf.shape(self.responses)[0], tf.shape(self.responses)[1]
        self.responses_input = tf.concat([tf.ones([batch_size, 1], dtype=tf.int32)*GO_ID,
            tf.split(self.responses_target, [decoder_len-1, 1], 1)[0]], 1)   # batch*len
        self.decoder_mask = tf.reshape(tf.cumsum(tf.one_hot(self.responses_length-1, 
            decoder_len), reverse=True, axis=1), [-1, decoder_len])        
        
        # build the embedding table (index to vector)
        if embed is None:
            # initialize the embedding randomly
            self.embed = tf.get_variable('embed', [num_symbols, num_embed_units], tf.float32)
        else:
            # initialize the embedding by pre-trained word vectors
            self.embed = tf.get_variable('embed', dtype=tf.float32, initializer=embed)

        self.pattern_embed = tf.get_variable('pattern_embed', [num_classes, num_embed_units], tf.float32)
        
        self.encoder_input = tf.nn.embedding_lookup(self.embed, self.posts_input) #batch*len*unit
        #self.encoder_input = tf.nn.embedding_lookup(self.embed, self.keywords_input)
        self.decoder_input = tf.nn.embedding_lookup(self.embed, self.responses_input)
        #self.attn_input = tf.nn.embedding_lookup(self.embed, self.keywords_input) #batch*num_keywords*unit
        #self.attn_input = tf.contrib.layers.fully_connected(self.attn_input, num_embed_units, activation_fn=tf.tanh)
        #self.attn_input = tf.contrib.layers.fully_connected(self.attn_input, num_units, activation_fn=None) #batch*num_keywords*num_units

        if use_lstm:
            #cell = MultiRNNCell([LSTMCell(num_units)] * num_layers)
            cell_fw = LSTMCell(num_units)
            cell_bw = LSTMCell(num_units)
            cell_dec = LSTMCell(2*num_units)
        else:
            #cell = MultiRNNCell([GRUCell(num_units)] * num_layers)
            cell_fw = GRUCell(num_units)
            cell_bw = GRUCell(num_units)
            cell_dec = GRUCell(2*num_units)

        # rnn encoder
        with variable_scope.variable_scope("encoder"):
            encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.encoder_input, 
                self.posts_length, dtype=tf.float32)

            post_sum_state = tf.concat(encoder_state, 1) #batch*2num_units
            encoder_output = tf.concat(encoder_output, 2) #batch*len*2num_units


        with variable_scope.variable_scope("encoder", reuse = True):
            decoder_state, decoder_last_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.decoder_input, 
                self.responses_length, dtype=tf.float32)
            response_sum_state = tf.concat(decoder_last_state, 1)

        with variable_scope.variable_scope("recog_net"):
            recog_input = tf.concat([post_sum_state, response_sum_state], 1)
            recog_mulogvar = tf.contrib.layers.fully_connected(recog_input, latent_size * 2, activation_fn=None, scope="muvar")
            recog_mu, recog_logvar = tf.split(recog_mulogvar, 2, axis=1)

        with variable_scope.variable_scope("prior_net"):
            prior_fc1 = tf.contrib.layers.fully_connected(post_sum_state, latent_size * 2, activation_fn=tf.tanh, scope="fc1")
            prior_mulogvar = tf.contrib.layers.fully_connected(prior_fc1, latent_size * 2, activation_fn=None, scope="muvar")
            prior_mu, prior_logvar = tf.split(prior_mulogvar, 2, axis=1)

            # use sampled Z or posterior Z
            #if self.use_prior:
            #    latent_sample = sample_gaussian(prior_mu, prior_logvar)
            #else:
            #    latent_sample = sample_gaussian(recog_mu, recog_logvar)

            latent_sample = tf.cond(self.use_prior,
                                lambda: sample_gaussian(prior_mu, prior_logvar),
                                lambda: sample_gaussian(recog_mu, recog_logvar)) #batch*latent_size

        with variable_scope.variable_scope("discriminator"):
            #dis_input = tf.concat([post_sum_state, latent_sample], 1)
            dis_input = latent_sample
            pattern_fc1 = tf.contrib.layers.fully_connected(dis_input, latent_size, activation_fn=tf.tanh, scope="pattern_fc1")
            self.pattern_logits = tf.contrib.layers.fully_connected(pattern_fc1, num_classes, activation_fn=None, scope="pattern_logits") #batch*num_classes

        temp_pred = tf.argmax(self.pattern_logits, 1)
        temp_onehot = tf.one_hot(temp_pred, num_classes, 1.0, 0.0)
        #self.label_embedding = tf.cond(self.use_prior,
        #                               lambda: tf.matmul(temp_onehot, self.pattern_embed),
        #                               lambda: tf.matmul(self.labels, self.pattern_embed)) #batch*num_embed
        self.label_embedding = tf.matmul(self.labels, self.pattern_embed) #batch*num_embed

        #label_embedding = tf.matmul(self.labels, self.pattern_embed)

        #with variable_scope.variable_scope("decoder"): 
            # get decoding loop function
            #decoder_fn_train = my_attention_decoder_fn.attention_decoder_fn_train(encoder_state, 
            #        attention_keys, attention_values, attention_score_fn, attention_construct_fn, label_embedding)
        #extra_info = tf.concat([self.label_embedding, latent_sample], 1)
            # get output projection function
        output_fn, my_sequence_loss = output_projection_layer(2*num_units, num_symbols, self.embed, latent_size, self.keywords_input, num_embed_units, num_keywords, decoder_len, num_topic_symbols, num_non_topic_symbols, self.topic_mask, self.ordinary_mask, self.func_mask)

            # get attention function
        attention_keys, attention_values, attention_score_fn, attention_construct_fn = my_attention_decoder_fn.prepare_attention(encoder_output, 'luong', 2*num_units)
            
        #if is_train:
        with variable_scope.variable_scope("dec_start"):
            temp_start = tf.concat([post_sum_state, self.label_embedding, latent_sample], 1)
            dec_fc1 = tf.contrib.layers.fully_connected(temp_start, 2*num_units, activation_fn=tf.tanh, scope="dec_start_fc1")
            dec_fc2 = tf.contrib.layers.fully_connected(dec_fc1, 2*num_units, activation_fn=None, scope="dec_start_fc2")
        #else:

        if is_train:
            # rnn decoder
            #with variable_scope.variable_scope("decoder"):

            # calculate the loss of decoder
            #self.decoder_loss = sampled_sequence_loss(self.decoder_output, 
            #        self.responses_target, self.decoder_mask)
            #st_yt = tf.concat([self.decoder_output, self.decoder_input], 2) #batch*len*(2num_units+embed)
            #with variable_scope.variable_scope("loss"):
            extra_info = tf.concat([self.label_embedding, latent_sample], 1)
            decoder_fn_train = my_attention_decoder_fn.attention_decoder_fn_train(dec_fc2, 
                attention_keys, attention_values, attention_score_fn, attention_construct_fn, extra_info)
            self.decoder_output, _, _ = my_seq2seq.dynamic_rnn_decoder(cell_dec, decoder_fn_train, 
                self.decoder_input, self.responses_length, scope = "decoder")

            self.decoder_loss = my_loss.sequence_loss(logits = self.decoder_output, 
                targets = self.responses_target, weights = self.decoder_mask, 
                extra_information = latent_sample, label_embedding = self.label_embedding, softmax_loss_function = my_sequence_loss)            
            #self.klloss = tf.reduce_mean(gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar))
            temp_klloss = tf.reduce_mean(gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar))
            #kl_weight = tf.cond(self.is_anneal,
            #                    lambda: tf.minimum(tf.to_float(self.global_t)/full_kl_step, 1.0),
            #                    lambda: tf.constant(1.0))
            self.kl_weight = tf.minimum(tf.to_float(self.global_t)/full_kl_step, 1.0)
            self.klloss = self.kl_weight * temp_klloss
            #self.klloss = self.kl_weight * temp_klloss
            temp_labels = tf.argmax(self.labels, 1)
            self.disloss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.pattern_logits, labels=temp_labels))
            self.loss = self.decoder_loss + self.klloss + self.disloss  # need to anneal the kl_weight

            self.pred_class = temp_pred # [batch, 1]
            self.latent_z = latent_sample # [batch, latent_size]
            
                # building graph finished and get all parameters
            self.params = tf.trainable_variables()
        
            # initialize the training process
            self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
            self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
            self.global_step = tf.Variable(0, trainable=False)
            
            # calculate the gradient of parameters
            #opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            opt = tf.train.MomentumOptimizer(self.learning_rate, 0.9)
            #opt = tf.train.AdamOptimizer(self.learning_rate)
            gradients = tf.gradients(self.loss, self.params)
            clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients, 
                    max_gradient_norm)
            self.update = opt.apply_gradients(zip(clipped_gradients, self.params), 
                    global_step=self.global_step)

        else:
            # rnn decoder
            #with variable_scope.variable_scope("decoder", reuse = True):
            #temp_latent = sample_gaussian(recog_mu, recog_logvar)
            #with variable_scope.variable_scope("decoder"):

            # generating the response
            #self.generation_index = tf.argmax(self.decoder_distribution, 2)
            #with variable_scope.variable_scope("loss"):
            #    self.temp = output_fn(post_sum_state, latent_sample, is_reuse = False)
            decoder_fn_inference = my_attention_decoder_fn.attention_decoder_fn_inference(output_fn, 
                dec_fc2, attention_keys, attention_values, attention_score_fn, 
                attention_construct_fn, self.embed, GO_ID, EOS_ID, max_length, num_symbols, latent_sample, self.label_embedding) 
            #self.decoder_distribution, _, _, self.decoder_type = my_seq2seq.dynamic_rnn_decoder(cell_dec, decoder_fn_inference, scope = "decoder")
            self.decoder_distribution, _, _ = my_seq2seq.dynamic_rnn_decoder(cell_dec, decoder_fn_inference, scope="decoder")
            self.generation_index = tf.argmax(tf.split(self.decoder_distribution,
                [2, num_symbols-2], 2)[1], 2) + 2 # for removing UNK
            self.generation = tf.nn.embedding_lookup(self.symbols, self.generation_index) 

            self.pred_class = temp_pred # [batch, 1]
            self.latent_z = latent_sample # [batch, latent_size]
            
            self.params = tf.trainable_variables()
        
        self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2, 
                max_to_keep=3, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)

    def print_parameters(self):
        for item in self.params:
            print('%s: %s' % (item.name, item.get_shape()))
    
    def step_decoder(self, session, data, forward_only=False, global_t=None):
        input_feed = {self.posts: data['posts'],
                self.posts_length: data['posts_length'],
                self.responses: data['responses'],
                self.responses_length: data['responses_length'],
                self.keywords: data['keywords'],
                self.keywords_length: data['keywords_length'],
                self.labels: data['labels'],
                self.use_prior: False}
        if global_t is not None:
            input_feed[self.global_t] = global_t
        if forward_only:  #On the dev set
            output_feed = [self.loss, self.klloss, self.decoder_loss, self.disloss, self.latent_z, self.pred_class]
        else:
            output_feed = [self.loss, self.klloss, self.decoder_loss, self.disloss, self.gradient_norm, self.update]
        return session.run(output_feed, input_feed)
    
    def inference(self, session, data, label_no):
        if label_no == 0:
            temp_labels = np.tile(np.array([1, 0, 0]),(len(data['posts']),1))
        else:
            if label_no == 1:
                temp_labels = np.tile(np.array([0, 1, 0]), (len(data['posts']), 1))
            else:
                temp_labels = np.tile(np.array([0, 0, 1]), (len(data['posts']), 1))
        #temp_labels = np.zeros([len(data['posts']), 3])
        input_feed = {self.posts: data['posts'], self.posts_length: data['posts_length'], 
                      self.responses: data['posts'], self.responses_length: data['posts_length'],
                      self.keywords: data['keywords'], self.keywords_length: data['keywords_length'],
                      self.labels: temp_labels, self.use_prior: True}
        #output_feed = [self.generation, self.pred_class, self.decoder_type, self.latent_z]
        output_feed = [self.generation, self.pred_class, self.latent_z]
        return session.run(output_feed, input_feed)

