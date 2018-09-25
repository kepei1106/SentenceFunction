import tensorflow as tf
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.ops import variable_scope

def output_projection_layer(num_units, num_symbols, latent_size, num_embed_units, topic_mask, ordinary_mask, func_mask, name="output_projection"):
    def output_fn(outputs, latent_z, label_embedding):
        with variable_scope.variable_scope(name):
            local_d = tf.reshape(outputs, [-1, num_units])
            local_l = tf.reshape(tf.concat([outputs, latent_z], 1), [-1, num_units + latent_size])
            local_d2 = tf.reshape(tf.concat([outputs, latent_z, label_embedding], 1), [-1, num_units + latent_size + num_embed_units])

            # type controller
            l_fc1 = tf.contrib.layers.fully_connected(local_l, num_units + latent_size, activation_fn=tf.tanh, scope = 'l_fc1')
            l_fc2 = tf.contrib.layers.fully_connected(l_fc1, 3, activation_fn=None, scope = 'l_fc2')
            p_dis = tf.nn.softmax(l_fc2)
            p_dis_1, p_dis_2, p_dis_3 = tf.split(p_dis, 3, axis = 1)
            p_dis_1 = tf.reshape(tf.tile(p_dis_1, [1, num_symbols]), [-1, num_symbols])
            p_dis_2 = tf.reshape(tf.tile(p_dis_2, [1, num_symbols]), [-1, num_symbols])
            p_dis_3 = tf.reshape(tf.tile(p_dis_3, [1, num_symbols]), [-1, num_symbols])
            type_index = p_dis

            # topic words
            w_fc2 = tf.contrib.layers.fully_connected(local_d, num_symbols, activation_fn=None, scope = 'w_fc2')
            p_w = tf.exp(w_fc2)
            p_w = p_w * tf.tile(tf.reshape(topic_mask, [1, num_symbols]), [tf.shape(local_d)[0], 1])
            temp_normal = tf.tile(tf.reduce_sum(p_w, 1, keep_dims=True), [1, num_symbols])
            y_prob_d = tf.div(p_w, temp_normal)

            # ordinary words
            d1_fc2 = tf.contrib.layers.fully_connected(local_d, num_symbols, activation_fn=None, scope = 'd1_fc2')
            temp_d1 = tf.exp(d1_fc2)
            temp_d1 = temp_d1 * tf.tile(tf.reshape(ordinary_mask, [1, num_symbols]), [tf.shape(local_d)[0], 1])
            temp_normal = tf.tile(tf.reduce_sum(temp_d1, 1, keep_dims=True), [1, num_symbols])
            y_prob_d1 = tf.div(temp_d1, temp_normal)

            # function-related words
            d2_fc2 = tf.contrib.layers.fully_connected(local_d2, num_symbols, activation_fn=None, scope = 'd2_fc2')
            temp_d2 = tf.exp(d2_fc2)
            temp_d2 = temp_d2 * tf.tile(tf.reshape(func_mask, [1, num_symbols]), [tf.shape(local_d)[0], 1])
            temp_normal = tf.tile(tf.reduce_sum(temp_d2, 1, keep_dims=True), [1, num_symbols])
            y_prob_d2 = tf.div(temp_d2, temp_normal)

            y_prob = p_dis_1 * y_prob_d + p_dis_2 * y_prob_d1 + p_dis_3 * y_prob_d2
        return y_prob, type_index

    def my_sequence_loss(outputs, targets, latent_z, label_embedding, max_time):
        with variable_scope.variable_scope("decoder/%s" % name):
            local_labels = tf.reshape(targets, [-1])
            local_d = tf.reshape(outputs, [-1, num_units])
            local_l = tf.reshape(tf.concat([outputs, latent_z], 1), [-1, num_units + latent_size])
            local_d2 = tf.reshape(tf.concat([outputs, latent_z, label_embedding], 1), [-1, num_units + latent_size + num_embed_units])

            # type controller
            l_fc1 = tf.contrib.layers.fully_connected(local_l, num_units + latent_size, activation_fn=tf.tanh, scope = 'l_fc1')
            l_fc2 = tf.contrib.layers.fully_connected(l_fc1, 3, activation_fn=None, scope = 'l_fc2')
            p_dis = tf.nn.softmax(l_fc2)
            p_dis_1, p_dis_2, p_dis_3 = tf.split(p_dis, 3, axis = 1)
            p_dis_1 = tf.reshape(tf.tile(p_dis_1, [1, num_symbols]), [-1, num_symbols])
            p_dis_2 = tf.reshape(tf.tile(p_dis_2, [1, num_symbols]), [-1, num_symbols])
            p_dis_3 = tf.reshape(tf.tile(p_dis_3, [1, num_symbols]), [-1, num_symbols])

            # topic words
            w_fc2 = tf.contrib.layers.fully_connected(local_d, num_symbols, activation_fn=None, scope = 'w_fc2')
            p_w = tf.exp(w_fc2)
            p_w = p_w * tf.tile(tf.reshape(topic_mask, [1, num_symbols]), [tf.shape(local_d)[0], 1])
            temp_normal = tf.tile(tf.reduce_sum(p_w, 1, keep_dims=True), [1, num_symbols])
            y_prob_d = tf.div(p_w, temp_normal)

            # ordinary words
            d1_fc2 = tf.contrib.layers.fully_connected(local_d, num_symbols, activation_fn=None, scope = 'd1_fc2')
            temp_d1 = tf.exp(d1_fc2)
            temp_d1 = temp_d1 * tf.tile(tf.reshape(ordinary_mask, [1, num_symbols]), [tf.shape(local_d)[0], 1])
            temp_normal = tf.tile(tf.reduce_sum(temp_d1, 1, keep_dims=True), [1, num_symbols])
            y_prob_d1 = tf.div(temp_d1, temp_normal)

            # function-related words
            d2_fc2 = tf.contrib.layers.fully_connected(local_d2, num_symbols, activation_fn=None, scope = 'd2_fc2')
            temp_d2 = tf.exp(d2_fc2)
            temp_d2 = temp_d2 * tf.tile(tf.reshape(func_mask, [1, num_symbols]), [tf.shape(local_d)[0], 1])
            temp_normal = tf.tile(tf.reduce_sum(temp_d2, 1, keep_dims=True), [1, num_symbols])
            y_prob_d2 = tf.div(temp_d2, temp_normal)

            y_prob = p_dis_1 * y_prob_d + p_dis_2 * y_prob_d1 + p_dis_3 * y_prob_d2

            # cross entropy
            labels_onehot = tf.one_hot(local_labels, num_symbols)
            labels_onehot = tf.clip_by_value(labels_onehot, 0.0, 1.0)
            y_prob = tf.clip_by_value(y_prob, 1e-18, 1.0)
            cross_entropy = tf.reshape(tf.reduce_sum(-labels_onehot * tf.log(y_prob), 1), [-1, 1])
            
            return cross_entropy   
    
    return output_fn, my_sequence_loss
