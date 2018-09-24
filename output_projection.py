import tensorflow as tf
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.ops import variable_scope

def output_projection_layer(num_units, num_symbols, embedding, latent_size, keywords_input, num_embed_units, num_keywords, max_time, num_topic_symbols, num_non_topic_symbols, topic_mask, ordinary_mask, func_mask, name="output_projection"):
    def output_fn(outputs, latent_z, label_embedding):
        #return layers.linear(outputs, num_symbols, scope=name)
        units_embed = tf.shape(outputs)[1] #batch*2num_units
        embed_dim = num_embed_units #num_symbols*num_embed
        latent_dim = tf.shape(latent_z)[1] #batch*latent_size
        label_dim = tf.shape(label_embedding)[1] #batch*label_dim
        #with variable_scope.variable_scope('decoder/%s' % name):
        #with variable_scope.variable_scope('decoder_loss', reuse = is_train):
        with variable_scope.variable_scope(name):
            local_d = tf.reshape(outputs, [-1, num_units])
            local_l = tf.reshape(tf.concat([outputs, latent_z], 1), [-1, num_units + latent_size])
            local_d2 = tf.reshape(tf.concat([outputs, latent_z, label_embedding], 1), [-1, num_units + latent_size + num_embed_units])

            l_fc1 = tf.contrib.layers.fully_connected(local_l, num_units + latent_size, activation_fn=tf.tanh, scope = 'l_fc1')
            l_fc2 = tf.contrib.layers.fully_connected(l_fc1, 3, activation_fn=None, scope = 'l_fc2')
            p_dis = tf.nn.softmax(l_fc2)
            p_dis_1, p_dis_2, p_dis_3 = tf.split(p_dis, 3, axis = 1)
            p_dis_1 = tf.reshape(tf.tile(p_dis_1, [1, num_symbols]), [-1, num_symbols])
            p_dis_2 = tf.reshape(tf.tile(p_dis_2, [1, num_symbols]), [-1, num_symbols]) #[batch, num_symbols]
            p_dis_3 = tf.reshape(tf.tile(p_dis_3, [1, num_symbols]), [-1, num_symbols])
            type_index = p_dis # batch*3
            
            #temp_keywords = tf.reshape(tf.tile(keywords_input, [1, max_time]), [-1, num_keywords])

            #temp_keywords = tf.reshape(keywords_input, [-1, num_keywords])
            #temp_one_hot = tf.reshape(tf.one_hot(temp_keywords, num_symbols, 1.0, 0.0), [-1, num_keywords, num_symbols])
            #sum_one_hot = tf.reduce_sum(temp_one_hot, 1)


            #temp_embed = tf.nn.embedding_lookup(embedding, temp_keywords)
            #temp_embed = tf.reshape(tf.tile(tf.reshape(temp_embed, [-1, num_keywords*num_embed_units]), [1, max_time]), [-1, num_keywords*num_embed_units])
            #temp_embed = tf.reshape(temp_embed, [-1, num_keywords*num_embed_units])
            #temp_sim = tf.concat([local_d, temp_embed], 1)
            #temp_sim = local_d
            #w_fc1 = tf.contrib.layers.fully_connected(temp_sim, num_units, activation_fn=tf.tanh, scope = 'w_fc1')
            w_fc2 = tf.contrib.layers.fully_connected(local_d, num_symbols, activation_fn=None, scope = 'w_fc2')
            #p_w = tf.nn.softmax(w_fc2) # (batch * len, num_topic_symbols)
            p_w = tf.exp(w_fc2)
            p_w = p_w * tf.tile(tf.reshape(topic_mask, [1, num_symbols]), [tf.shape(local_d)[0], 1])
            temp_normal = tf.tile(tf.reduce_sum(p_w, 1, keep_dims=True), [1, num_symbols])
            y_prob_d = tf.div(p_w, temp_normal) # topic words

            #topic_one_hot = tf.reshape(tf.one_hot(topic_pos, num_symbols, 1.0, 0.0), [-1, num_symbols])
            #p_w = tf.matmul(p_w, topic_one_hot) # (batch * len, num_symbols)
            #p_w = tf.exp(w_fc2)
            #p_w = p_w * sum_one_hot
            #temp_normal = tf.tile(tf.reduce_sum(p_w, 1, keep_dims = True), [1, num_symbols])
            #y_prob_d = tf.div(p_w, temp_normal)
            #y_prob_d = p_w * sum_one_hot
            #y_prob_d = p_w
            #total_word = tf.reshape(tf.one_hot(temp_keywords, num_symbols, 1.0, 0.0), [-1, num_keywords, num_symbols])
            #temp_p = tf.reshape(tf.tile(tf.reshape(p_w, [-1, num_keywords, 1]), [1, 1, num_symbols]), [-1, num_keywords, num_symbols])
            #y_prob_d = tf.reduce_sum(total_word*temp_p, 1) # Distribution over topic words

            #d1_fc1 = tf.contrib.layers.fully_connected(local_d, num_units, activation_fn=tf.tanh, scope = 'd1_fc1')
            d1_fc2 = tf.contrib.layers.fully_connected(local_d, num_symbols, activation_fn=None, scope = 'd1_fc2') # (batch * len, num_symbols)
            temp_d1 = tf.exp(d1_fc2)
            temp_d1 = temp_d1 * tf.tile(tf.reshape(ordinary_mask, [1, num_symbols]), [tf.shape(local_d)[0], 1])
            temp_normal = tf.tile(tf.reduce_sum(temp_d1, 1, keep_dims=True), [1, num_symbols])
            y_prob_d1 = tf.div(temp_d1, temp_normal) # ordinary words
            #y_prob_d1 = tf.nn.softmax(d1_fc2) # Distribution about ordinary word
            #y_prob_d1 = tf.matmul(y_prob_d1, non_topic_one_hot) # (batch * len, num_symbols)

            #d2_fc1 = tf.contrib.layers.fully_connected(local_d2, num_units + latent_size + num_embed_units, activation_fn=tf.tanh, scope = 'd2_fc1')
            d2_fc2 = tf.contrib.layers.fully_connected(local_d2, num_symbols, activation_fn=None, scope = 'd2_fc2')
            temp_d2 = tf.exp(d2_fc2)
            temp_d2 = temp_d2 * tf.tile(tf.reshape(func_mask, [1, num_symbols]), [tf.shape(local_d)[0], 1])
            temp_normal = tf.tile(tf.reduce_sum(temp_d2, 1, keep_dims=True), [1, num_symbols])
            #y_prob_d1 = tf.div(temp_d1, temp_normal)
            y_prob_d2 = tf.div(temp_d2, temp_normal) # function-related word
            #y_prob_d2 = tf.nn.softmax(d2_fc2) # Distribution about attribute-related word
            #y_prob_d2 = tf.matmul(y_prob_d2, non_topic_one_hot)

            y_prob = p_dis_1 * y_prob_d + p_dis_2 * y_prob_d1 + p_dis_3 * y_prob_d2
        return y_prob, type_index

    '''
    def sampled_sequence_loss(outputs, targets, masks):
        with variable_scope.variable_scope('decoder/%s' % name):
            weights = tf.transpose(tf.get_variable("weights", [num_units, num_symbols]))
            bias = tf.get_variable("biases", [num_symbols])
            
            local_labels = tf.reshape(targets, [-1, 1])
            local_outputs = tf.reshape(outputs, [-1, num_units])
            local_masks = tf.reshape(masks, [-1])
            
            local_loss = tf.nn.sampled_softmax_loss(weights, bias, local_labels,
                    local_outputs, num_samples, num_symbols)
            local_loss = local_loss * local_masks
            
            loss = tf.reduce_sum(local_loss)
            total_size = tf.reduce_sum(local_masks)
            total_size += 1e-12 # to avoid division by 0 for all-0 weights
            
            return loss / total_size
    '''
    def my_sequence_loss(outputs, targets, latent_z, label_embedding, max_time):
        units_embed = tf.shape(outputs)[1]
        embed_dim = num_embed_units
        latent_dim = tf.shape(latent_z)[1]
        label_dim = tf.shape(label_embedding)[1]
        #with variable_scope.variable_scope('decoder/%s' % name):
        #with variable_scope.variable_scope('decoder_loss'):
        with variable_scope.variable_scope("decoder/%s" % name):
            #weights = tf.transpose(tf.get_variable("weights", [num_units, num_symbols]))
            #bias = tf.get_variable("biases", [num_symbols])
            #w11 = tf.get_variable("w11", [units_embed, embed_dim])
            #b11 = tf.get_variable("b11", [num_symbols])
            #w12 = tf.get_variable("w12", [latent_dim, embed_dim])
            #b12 = tf.get_variable("b12", [num_symbols])

            #local_labels = tf.reshape(targets, [-1, 1])
            local_labels = tf.reshape(targets, [-1])
            #local_d = tf.reshape(outputs, [-1, units_embed])

            local_d = tf.reshape(outputs, [-1, num_units])
            local_l = tf.reshape(tf.concat([outputs, latent_z], 1), [-1, num_units + latent_size])
            local_d2 = tf.reshape(tf.concat([outputs, latent_z, label_embedding], 1), [-1, num_units + latent_size + num_embed_units])

            l_fc1 = tf.contrib.layers.fully_connected(local_l, num_units + latent_size, activation_fn=tf.tanh, scope = 'l_fc1')
            l_fc2 = tf.contrib.layers.fully_connected(l_fc1, 3, activation_fn=None, scope = 'l_fc2')
            p_dis = tf.nn.softmax(l_fc2)
            p_dis_1, p_dis_2, p_dis_3 = tf.split(p_dis, 3, axis = 1)
            p_dis_1 = tf.reshape(tf.tile(p_dis_1, [1, num_symbols]), [-1, num_symbols])
            p_dis_2 = tf.reshape(tf.tile(p_dis_2, [1, num_symbols]), [-1, num_symbols]) #[batch, num_symbols]
            p_dis_3 = tf.reshape(tf.tile(p_dis_3, [1, num_symbols]), [-1, num_symbols])
            
            #temp_keywords = tf.reshape(tf.tile(keywords_input, [1, max_time]), [-1, num_keywords])
            #temp_keywords = tf.reshape(keywords_input, [-1, num_keywords])
            #temp_one_hot = tf.reshape(tf.one_hot(temp_keywords, num_symbols, 1.0, 0.0), [-1, num_keywords, num_symbols])
            #sum_one_hot = tf.reduce_sum(temp_one_hot, 1)
            #temp_embed = tf.reshape(temp_embed, [-1, num_keywords*num_embed_units])
            #temp_embed = tf.reshape(tf.tile(tf.reshape(temp_embed, [-1, num_keywords*num_embed_units]), [1, max_time]), [-1, num_keywords*num_embed_units])
            #temp_sim = tf.concat([local_d, temp_embed], 1)
            #temp_sim = local_d
            #w_fc1 = tf.contrib.layers.fully_connected(temp_sim, num_units, activation_fn=tf.tanh, scope = 'w_fc1')
            #w_fc2 = tf.contrib.layers.fully_connected(w_fc1, num_keywords, activation_fn=None, scope = 'w_fc2')


            w_fc2 = tf.contrib.layers.fully_connected(local_d, num_symbols, activation_fn=None, scope = 'w_fc2')
            #p_w = tf.nn.softmax(w_fc2) # (batch * len, num_topic_symbols)
            p_w = tf.exp(w_fc2)
            p_w = p_w * tf.tile(tf.reshape(topic_mask, [1, num_symbols]), [tf.shape(local_d)[0], 1])
            temp_normal = tf.tile(tf.reduce_sum(p_w, 1, keep_dims=True), [1, num_symbols])
            y_prob_d = tf.div(p_w, temp_normal)

            #w_fc2 = tf.contrib.layers.fully_connected(local_d, num_topic_symbols, activation_fn=None, scope='w_fc2')
            #p_w = tf.nn.softmax(w_fc2)
            #topic_one_hot = tf.reshape(tf.one_hot(topic_pos, num_symbols, 1.0, 0.0), [-1, num_symbols])
            #p_w = tf.matmul(p_w, topic_one_hot)
            #y_prob_d = p_w * sum_one_hot
            #y_prob_d = p_w
            #p_w = tf.exp(w_fc2)
            #p_w = p_w * sum_one_hot
            #temp_normal = tf.tile(tf.reduce_sum(p_w, 1, keep_dims = True), [1, num_symbols])
            #y_prob_d = tf.div(p_w, temp_normal)
            #p_w = tf.nn.softmax(w_fc2)
            #total_word = tf.reshape(tf.one_hot(temp_keywords, num_symbols, 1.0, 0.0), [-1, num_keywords, num_symbols])
            #temp_p = tf.reshape(tf.tile(tf.reshape(p_w, [-1, num_keywords, 1]), [1, 1, num_symbols]), [-1, num_keywords, num_symbols])
            #y_prob_d = tf.reduce_sum(total_word*temp_p, 1) # Distribution over topic words

            #d1_fc1 = tf.contrib.layers.fully_connected(local_d, num_units, activation_fn=tf.tanh, scope = 'd1_fc1')
            d1_fc2 = tf.contrib.layers.fully_connected(local_d, num_symbols, activation_fn=None, scope = 'd1_fc2') # (batch * len, num_symbols)
            temp_d1 = tf.exp(d1_fc2)
            temp_d1 = temp_d1 * tf.tile(tf.reshape(ordinary_mask, [1, num_symbols]), [tf.shape(local_d)[0], 1])
            temp_normal = tf.tile(tf.reduce_sum(temp_d1, 1, keep_dims=True), [1, num_symbols])
            y_prob_d1 = tf.div(temp_d1, temp_normal)

            #d1_fc2 = tf.contrib.layers.fully_connected(local_d, num_non_topic_symbols, activation_fn=None, scope = 'd1_fc2')
            #y_prob_d1 = tf.nn.softmax(d1_fc2) # Distribution about ordinary word
            #y_prob_d1 = tf.matmul(y_prob_d1, non_topic_one_hot)
            d2_fc2 = tf.contrib.layers.fully_connected(local_d2, num_symbols, activation_fn=None, scope = 'd2_fc2')
            temp_d2 = tf.exp(d2_fc2)
            temp_d2 = temp_d2 * tf.tile(tf.reshape(func_mask, [1, num_symbols]), [tf.shape(local_d)[0], 1])
            temp_normal = tf.tile(tf.reduce_sum(temp_d2, 1, keep_dims=True), [1, num_symbols])
            #y_prob_d1 = tf.div(temp_d1, temp_normal)
            y_prob_d2 = tf.div(temp_d2, temp_normal)
            #d2_fc1 = tf.contrib.layers.fully_connected(local_d2, num_units + latent_size + num_embed_units, activation_fn=tf.tanh, scope = 'd2_fc1')
            #d2_fc2 = tf.contrib.layers.fully_connected(local_d2, num_non_topic_symbols, activation_fn=None, scope = 'd2_fc2')
            #y_prob_d2 = tf.nn.softmax(d2_fc2) # Distribution about attribute-related word
            #y_prob_d2 = tf.matmul(y_prob_d2, non_topic_one_hot)

            y_prob = p_dis_1 * y_prob_d + p_dis_2 * y_prob_d1 + p_dis_3 * y_prob_d2

            labels_onehot = tf.one_hot(local_labels, num_symbols) #[batch*max_time, num_symbols]
            labels_onehot = tf.clip_by_value(labels_onehot, 0.0, 1.0)
            y_prob = tf.clip_by_value(y_prob, 1e-18, 1.0)
            #cross_entropy = tf.reshape(tf.reduce_sum(labels_onehot * tf.log(labels_onehot / y_prob), 1), [-1, 1])
            cross_entropy = tf.reshape(tf.reduce_sum(-labels_onehot * tf.log(y_prob), 1), [-1, 1])
            #local_masks = tf.reshape(masks, [-1])
            
            #local_loss = tf.nn.sampled_softmax_loss(weights, bias, local_labels,
            #        local_outputs, num_samples, num_symbols)
            #local_loss = local_loss * local_masks
            
            #loss = tf.reduce_sum(local_loss)
            #total_size = tf.reduce_sum(local_masks)
            #total_size += 1e-12 # to avoid division by 0 for all-0 weights
            
            return cross_entropy   
    
    return output_fn, my_sequence_loss
    
