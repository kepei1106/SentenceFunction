import numpy as np
import tensorflow as tf
import sys
import time
import random
import pickle as pkl
import math
random.seed(time.time())

from model import Seq2SeqModel, _START_VOCAB
try:
    from wordseg_python import Global
except:
    Global = None

tf.app.flags.DEFINE_boolean("is_train", True, "Set to False to inference.")
tf.app.flags.DEFINE_integer("symbols", 40000, "vocabulary size.")
#tf.app.flags.DEFINE_integer("topic_symbols", 8000, "topic vocabulary size.")
tf.app.flags.DEFINE_integer("topic_symbols", 10000, "topic vocabulary size.")
tf.app.flags.DEFINE_integer("full_kl_step", 80000, "Total steps to finish annealing")
tf.app.flags.DEFINE_integer("embed_units", 100, "Size of word embedding.")
tf.app.flags.DEFINE_integer("units", 256, "Size of each model layer.")
tf.app.flags.DEFINE_integer("layers", 2, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("batch_size", 128, "Batch size to use during training.")
#tf.app.flags.DEFINE_string("data_dir", "./data", "Data directory")
tf.app.flags.DEFINE_string("data_dir", "/home/kepei/seq2seq_rec_nostop/data", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./train", "Training directory.")
tf.app.flags.DEFINE_integer("per_checkpoint", 1000, "How many steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("inference_version", 0, "The version for inferencing.")
tf.app.flags.DEFINE_boolean("log_parameters", True, "Set to True to show the parameters")
tf.app.flags.DEFINE_string("inference_path", "", "Set filename of inference, default isscreen")
tf.app.flags.DEFINE_string("num_keywords", 2, "Number of keywords extracted from responses")

FLAGS = tf.app.flags.FLAGS

def load_data(path, fname):
    with open('%s/%s.post' % (path, fname)) as f:
        post = [line.strip().split() for line in f.readlines()]
    with open('%s/%s.response' % (path, fname)) as f:
        response = [line.strip().split() for line in f.readlines()]
    with open('%s/%s.keyword' % (path, fname)) as f:
        keyword = [line.strip().split() for line in f.readlines()]
    with open('%s/%s.label' % (path, fname)) as f:
        label = [line.strip().split('\t') for line in f.readlines()]    
    data = []
    for p, r, k, l in zip(post, response, keyword, label):
        data.append({'post': p, 'response': r, 'keyword': k, 'label':l})
    return data

def build_vocab(path, data, stop_list, func_list):
    print("Creating vocabulary...")
    vocab = {}
    vocab_topic = {}
    for i, pair in enumerate(data):
        if i % 100000 == 0:
            print("    processing line %d" % i)
        for token in pair['post']+pair['response']:
            if token in vocab:
                vocab[token] += 1
            else:
                vocab[token] = 1
        for token in pair['keyword']: # remove stopwords from topic_vocab
            if token not in stop_list:
                if token in vocab_topic:
                    vocab_topic[token] += 1
                else:
                    vocab_topic[token] = 1
    vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
    vocab_topic_list = sorted(vocab_topic, key = vocab_topic.get, reverse = True)
    print 'length of vocabulary list:', len(vocab_list)
    print 'length of vocabulary topic list:', len(vocab_topic_list)
    #print ''

    if len(vocab_list) > FLAGS.symbols:
        vocab_list = vocab_list[:FLAGS.symbols] # remove topic words with low frequency
    vocab_topic_list_new = []
    for word in vocab_topic_list:
        if word in vocab_list:
            vocab_topic_list_new.append(word) # common topic words

    print 'length of new vocabulary list new:', len(vocab_list)
    print 'length of new vocabulary topic list new:', len(vocab_topic_list_new)
    #print 'vocab_topic_list=', len(vocab_topic_list)
    f_topic = open('vocab_topic_list.txt','w')
    for i in range(len(vocab_topic_list_new)):
        f_topic.writelines('%s\n' % vocab_topic_list_new[i])
    f_topic.close()
    #if len(vocab_topic_list) > FLAGS.topic_symbols:
        #vocab_topic_list = vocab_topic_list[:FLAGS.topic_symbols]
    if len(vocab_topic_list_new) > FLAGS.topic_symbols:
        vocab_topic_list_new = vocab_topic_list_new[:FLAGS.topic_symbols]
    topic_pos_list = []
    non_topic_pos_list = []
    topic_cnt = 0
    for ele in vocab_topic_list_new:
        if ele in vocab_list and ele not in func_list:
            topic_cnt += 1
            topic_pos_list.append(vocab_list.index(ele))
        #else:
            #topic_pos_list.append(1)
    non_topic_cnt = 0
    for word_idx in range(len(vocab_list)):
        if word_idx not in topic_pos_list:
            non_topic_pos_list.append(word_idx)
            non_topic_cnt += 1
    print 'topic_cnt = ', topic_cnt
    print 'non_topic_cnt = ', non_topic_cnt
    f_pos = open('topic_pos_list.txt','w')
    for i in range(len(topic_pos_list)):
        f_pos.writelines('%s\n' % topic_pos_list[i])
    f_pos.close()

    func_pos_list = []
    for ele in func_list.items():
        if ele[0] in vocab_list:
            func_pos_list.append(vocab_list.index(ele[0]))

    print("Loading word vectors...")
    vectors = {}
    with open('%s/vector.txt' % path) as f:
        for i, line in enumerate(f):
            if i % 100000 == 0:
                print("    processing line %d" % i)
            s = line.strip()
            word = s[:s.find(' ')]
            vector = s[s.find(' ')+1:]
            vectors[word] = vector
    
    embed = []
    for word in vocab_list:
        if word in vectors:
            vector = map(float, vectors[word].split())
        else:
            vector = np.zeros((FLAGS.embed_units), dtype=np.float32)
        embed.append(vector)
    embed = np.array(embed, dtype=np.float32)
            
    return vocab_list, embed, vocab_topic_list_new, topic_pos_list, non_topic_pos_list, func_pos_list

def gen_batched_data(data):
    encoder_len = max([len(item['post']) for item in data])+1
    decoder_len = max([len(item['response']) for item in data])+1
    
    posts, responses, posts_length, responses_length, keywords, keywords_length, labels = [], [], [], [], [], [], []
    def padding(sent, l):
        return sent + ['_EOS'] + ['_PAD'] * (l-len(sent)-1)
    def padding_key(sent, l):
        return sent + ['_PAD'] * (l-len(sent))
        
    for item in data:
        posts.append(padding(item['post'], encoder_len))
        responses.append(padding(item['response'], decoder_len))
        posts_length.append(len(item['post'])+1)
        responses_length.append(len(item['response'])+1)
        keywords.append(padding_key(item['keyword'], FLAGS.num_keywords))
        keywords_length.append(len(item['keyword']))
        labels.append(item['label'])

    batched_data = {'posts': np.array(posts),
            'responses': np.array(responses),
            'posts_length': posts_length, 
            'responses_length': responses_length,
            'keywords': np.array(keywords),
            'keywords_length': keywords_length,
            'labels':np.array(labels)}
    return batched_data



def train(model, sess, data_train, global_t):
    #selected_data = [random.choice(data_train) for i in range(FLAGS.batch_size)]
    batched_data = gen_batched_data(data_train)
    outputs = model.step_decoder(sess, batched_data, global_t = global_t)
    return outputs

def evaluate(model, sess, data_dev):
    loss = np.zeros((1, ))
    kl_loss, dec_loss, dis_loss = np.zeros((1, )), np.zeros((1, )), np.zeros((1, ))
    st, ed, times = 0, FLAGS.batch_size, 0
    latent_z, pred_class = [], []
    #print 'start evaluation'
    while st < len(data_dev):
        selected_data = data_dev[st:ed]
        batched_data = gen_batched_data(selected_data)
        outputs = model.step_decoder(sess, batched_data, forward_only=True, global_t = FLAGS.full_kl_step)
        kl_loss += outputs[1]
        dec_loss += outputs[2]
        dis_loss += outputs[3]
        loss += outputs[0]
        latent_z.append(outputs[4])
        pred_class.append(outputs[5])
        #loss += kl_loss + dec_loss + dis_loss
        st, ed = ed, ed+FLAGS.batch_size
        times += 1
    #print 'end evaluation'
    loss /= times
    kl_loss /= times
    dec_loss /= times
    dis_loss /= times
    show = lambda a: '[%s]' % (' '.join(['%.2f' % x for x in a]))
    print('perplexity on dev set: %s  kl_loss: %s  dec_loss: %s  dis_loss: %s ' % (show(np.exp(dec_loss)), show(kl_loss), show(dec_loss), show(dis_loss)))
    return latent_z, pred_class

def inference(model, sess, posts, test_key, label_no):
    length = [len(p)+1 for p in posts]
    length_key = [len(k) for k in test_key]
    def padding(sent, l):
        return sent + ['_EOS'] + ['_PAD'] * (l-len(sent)-1)
    def padding_key(sent, l):
        return sent + ['_PAD'] * (l-len(sent))

    batched_posts = [padding(p, max(length)) for p in posts]
    batched_keywords = [padding_key(k, FLAGS.num_keywords) for k in test_key]
    batched_data = {'posts': np.array(batched_posts), 
            'posts_length': np.array(length, dtype=np.int32),
            'keywords': np.array(batched_keywords),
            'keywords_length': np.array(length_key, dtype=np.int32)}

    results_inf = model.inference(sess, batched_data, label_no)
    #responses, pred, word_type, latent_z = results_inf[0], results_inf[1], results_inf[2], results_inf[3]
    responses, pred, latent_z = results_inf[0], results_inf[1], results_inf[2]
    #We should also get labels
    results = []
    results_type = []
    results_z = []
    results_pred = []
    res_cnt = 0
    for response in responses:
        result = []
        result_type = []
        result_z = []
        result_pred = []
        token_cnt = 0
        for token in response:
            if token != '_EOS':
                result.append(token)
                #result_type.append(np.argmax(np.array(word_type[res_cnt][token_cnt])))
                #result_type.append(word_type[res_cnt][token_cnt][2])
                token_cnt += 1
            else:
                break
        for num in latent_z[res_cnt]:
            result_z.append(num)
        result_pred.append(pred[res_cnt])
        res_cnt += 1
        results.append(result)
        #results_type.append(result_type)
        results_z.append(result_z)
        results_pred.append(result_pred)
    #return results, results_pred, results_type, results_z
    return results, results_pred, results_z

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    data_train = load_data(FLAGS.data_dir, 'weibo_pair_train_pattern')
    data_dev = load_data(FLAGS.data_dir, 'weibo_pair_dev_pattern')
    # data_test = load_data(FLAGS.data_dir, 'weibo_pair_test')
    stop_list = {}
    stop_file = open('stopword_utf8.txt', 'r')
    line_stop = stop_file.readline()
    while line_stop:
        temp = line_stop.strip()
        if temp not in stop_list:
            stop_list[temp] = 1
        else:
            stop_list[temp] += 1
        line_stop = stop_file.readline()
    stop_file.close()
    print 'stop_list=', len(stop_list)

    func_list = {}
    func_file = open('functionword-utf8.txt', 'r')
    line_func = func_file.readline()
    while line_func:
        temp = line_func.strip()
        if temp not in func_list:
            func_list[temp] = 1
        else:
            func_list[temp] += 1
        line_func = func_file.readline()
    func_file.close()
    print 'func_list=', len(func_list)

    vocab, embed, vocab_topic, topic_pos, non_topic_pos, func_pos = build_vocab(FLAGS.data_dir, data_train, stop_list, func_list)
    print 'topic_vocab=', len(vocab_topic)
    print 'func_vocab=', len(func_pos)
    file_func = open('function_pos.txt','w')
    for i in range(len(func_pos)):
        file_func.write(str(func_pos[i])+'\n')
    file_func.close()
    if FLAGS.is_train:
        '''
        data_train = load_data(FLAGS.data_dir, 'weibo_pair_train_pattern')
        data_dev = load_data(FLAGS.data_dir, 'weibo_pair_dev_pattern')
        #data_test = load_data(FLAGS.data_dir, 'weibo_pair_test')
        stop_list = {}
        stop_file = open('stopword_utf8.txt','r')
        line_stop = stop_file.readline()
        while line_stop:
            temp = line_stop.strip()
            if temp not in stop_list:
                stop_list[temp] = 1
            else:
                stop_list[temp] += 1
            line_stop = stop_file.readline()
        stop_file.close()
        print 'stop_list=', len(stop_list)

        vocab, embed, vocab_topic, topic_pos = build_vocab(FLAGS.data_dir, data_train, stop_list)
        print 'topic_vocab=', len(vocab_topic)
        '''
        model = Seq2SeqModel(
                FLAGS.symbols, 
                FLAGS.embed_units,
                FLAGS.units, 
                FLAGS.layers,
                is_train=True,
                vocab=vocab,
                topic_pos=topic_pos,
                non_topic_pos = non_topic_pos,
                func_pos = func_pos,
                embed=embed,
                full_kl_step=FLAGS.full_kl_step,
                num_keywords = FLAGS.num_keywords,
                num_topic_symbols = FLAGS.topic_symbols,
                num_non_topic_symbols = FLAGS.symbols - FLAGS.topic_symbols)
        if FLAGS.log_parameters:
            model.print_parameters()
        
        if tf.train.get_checkpoint_state(FLAGS.train_dir):
            print("Reading model parameters from %s" % FLAGS.train_dir)
            model.saver.restore(sess, tf.train.latest_checkpoint(FLAGS.train_dir))
            model.symbol2index.init.run()
        else:
            print("Created model with fresh parameters.")
            tf.global_variables_initializer().run()
            model.symbol2index.init.run()
        
        temp_total_losses, total_loss_step, kl_loss_step, decoder_loss_step, dis_loss_step, time_step = np.zeros((1, )), np.zeros((1, )), np.zeros((1, )), np.zeros((1, )), np.zeros((1, )), .0
        previous_losses = [1e18]*6

        num_batch = len(data_train) / FLAGS.batch_size
        random.shuffle(data_train)
        pre_train = [data_train[i:i+FLAGS.batch_size] for i in range(0, len(data_train), FLAGS.batch_size)]
        if len(data_train) % FLAGS.batch_size != 0:
            pre_train.pop() 
        random.shuffle(pre_train)
        ptr = 0
        global_t = 0
        while True:
            if model.global_step.eval() % FLAGS.per_checkpoint == 0:
                show = lambda a: '[%s]' % (' '.join(['%.2f' % x for x in a]))
                #print type(total_loss_step)
                #print type(kl_loss_step)
                #print type(decoder_loss_step)
                #print type(dis_loss_step)
                print("global step %d learning rate %.4f step-time %.2f perplexity %s kl_loss %s dec_loss %s dis_loss %s"
                        % (model.global_step.eval(), model.learning_rate.eval(), 
                            time_step, show(np.exp(decoder_loss_step)), show(kl_loss_step), show(decoder_loss_step), show(dis_loss_step)))
                model.saver.save(sess, '%s/checkpoint' % FLAGS.train_dir, global_step=model.global_step)
                #latent_z, pred_class = evaluate(model, sess, data_dev)
                evaluate(model, sess, data_dev)
                '''
                f_q = open('z_q.out', 'w')
                f_s = open('z_s.out', 'w')
                f_i = open('z_i.out', 'w')
                for i in range(len(latent_z)):
                    if pred_class[i][0] == 0:
                        for j in latent_z[i]:
                            f_q.write('%f ' % (j))
                        f_q.write('\n')
                    else:
                        if pred_class[i][0] == 1:
                            for j in latent_z[i]:
                                f_s.write('%f ' % (j))
                            f_s.write('\n')
                        else:
                            if pred_class[i][0] == 2:
                                for j in latent_z[i]:
                                    f_i.write('%f ' % (j))
                                f_i.write('\n')
                f_q.close()
                f_s.close()
                f_i.close()
                break
                '''
                #evaluate(model, sess, data_test)
                #temp_total_losses = decoder_loss_step + kl_loss_step + dis_loss_step
                if np.sum(temp_total_losses) > max(previous_losses):
                    sess.run(model.learning_rate_decay_op)
                previous_losses = previous_losses[1:]+[np.sum(temp_total_losses)]
                #total_loss_step, time_step = np.zeros((1, )), .0
                temp_total_losses, total_loss_step, kl_loss_step, decoder_loss_step, dis_loss_step, time_step = np.zeros((1, )), np.zeros((1, )), np.zeros((1, )), np.zeros((1, )), np.zeros((1, )), .0

            global_t = model.global_step.eval()
            start_time = time.time()
            temp_loss = train(model, sess, pre_train[ptr], global_t)
            total_loss_step += temp_loss[0] / FLAGS.per_checkpoint
            kl_loss_step += temp_loss[1] / FLAGS.per_checkpoint
            decoder_loss_step += temp_loss[2] / FLAGS.per_checkpoint
            dis_loss_step += temp_loss[3] / FLAGS.per_checkpoint
            if global_t>=1:
                temp_total_losses += decoder_loss_step + kl_loss_step*FLAGS.full_kl_step/global_t + dis_loss_step
            time_step += (time.time() - start_time) / FLAGS.per_checkpoint
            #global_t += 1
            ptr += 1
            if ptr == num_batch:
                random.shuffle(pre_train)
                ptr = 0
            
    else:
        pkl_file = open('/home/kepei/seq2seq_rec_nostop/PMI.pkl', 'rb')
        PMI = pkl.load(pkl_file)
        model = Seq2SeqModel(
                FLAGS.symbols, 
                FLAGS.embed_units, 
                FLAGS.units, 
                FLAGS.layers, 
                is_train=False,
                topic_pos = topic_pos,
                func_pos = func_pos,
                vocab=None,
                num_keywords = FLAGS.num_keywords,
                num_topic_symbols = FLAGS.topic_symbols)
        if FLAGS.inference_version == 0:
            model_path = tf.train.latest_checkpoint(FLAGS.train_dir)
        else:
            model_path = '%s/checkpoint-%08d' % (FLAGS.train_dir, FLAGS.inference_version)
        print('restore from %s' % model_path)
        model.saver.restore(sess, model_path)
        model.symbol2index.init.run()

        def split(sent):
            if Global == None:
                return sent.split()

            sent = sent.decode('utf-8', 'ignore').encode('gbk', 'ignore')
            tuples = [(word.decode("gbk").encode("utf-8"), pos) 
                    for word, pos in Global.GetTokenPos(sent)]
            return [each[0] for each in tuples]
        
        if FLAGS.inference_path == '':
            while True:
                sys.stdout.write('post: ')
                sys.stdout.flush()
                post = split(sys.stdin.readline())
                res_pred = inference(model, sess, [post])
                response, pred = res_pred[0], res_pred[1]
                print('response: %s' % ''.join(response))
                sys.stdout.flush()
        else:
            posts = []
            posts_ori = []
            test_key = []
            with open(FLAGS.inference_path) as f:
                for line in f:
                    #sent = line.strip().split('\t')[0]
                    sent = line.strip()
                    posts_ori.append(sent)
                    cur_post = split(sent)
                    #cur_post = sent.split(' ')
                    posts.append(cur_post)
                    temp_pmi = {}
                    for i in cur_post:
                        if i in PMI:
                            for j,k in PMI[i].items():
                                if j in temp_pmi:
                                    temp_pmi[j] += math.log(k)
                                else:
                                    temp_pmi[j] = math.log(k)
                    temp_pmi = sorted(temp_pmi.items(), key = lambda x:x[1], reverse = True)
                    kw_candidate = []
                    for i in range(min(len(temp_pmi), FLAGS.num_keywords)):
                        kw_candidate.append(temp_pmi[i][0])
                    test_key.append(kw_candidate)
            with open('test_keyword.out', 'w') as f1:
                for key in test_key:
                    #f1.writelines(str(key[0])+' '+str(key[1])+' '+str(key[2])+' '+str(key[3])+'\r\n')
                    f1.writelines(str(key[0]) + ' ' + str(key[1])  + '\r\n')


            responses, preds, res_type, res_z = [[], [], []], [[], [], []], [[], [], []], [[],[],[]]
            st, ed = 0, FLAGS.batch_size
            while st < len(posts):
                for i in range(3):
                    temp = inference(model, sess, posts[st: ed], test_key[st: ed], i)
                    responses[i] += temp[0]
                    preds[i] += temp[1]
                    #res_type[i] += temp[2]
                    #res_z[i] += temp[3]
                    res_z[i] += temp[2]
                #print len(temp[0])
                #print len(temp[1])
                #for i in range(len(temp[1])):
                #    preds.append(temp[1][i])
                #preds += temp[1]
                st, ed = ed, ed+FLAGS.batch_size
            #f_q = open('z_q.out', 'w')
            #f_s = open('z_s.out', 'w')
            #f_i = open('z_i.out', 'w')
            with open(FLAGS.inference_path+'.out', 'w') as f:
                #for r, l in zip(responses, preds):
                for i in range(len(posts)):
                    #f.writelines('%s\t%s\n' % (''.join(p), ''.join(r)))
                    #f.writelines('%s\n' % (''.join(responses[0][i])))
                    for k in range(3):
                        f.writelines('%s\n' % (''.join(responses[k][i])))
                    f.writelines('\n')
                        #for j in responses[k][i]:
                            #f.write('%s ' % (j))
                       # f.write('\t')
                    '''
                        for j in res_z[k][i]:
                            f.write('%f ' % (j))
                        f.write('\n')
                        #f.writelines('%s\n' % (''.join(responses[1][i])))
                        #f.writelines('%s\n' % (''.join(responses[2][i])))
                        if preds[k][i][0] == k:
                            if k == 0:
                                for j in res_z[k][i]:
                                    f_q.write('%f ' % (j))
                                f_q.write('\n')
                            else:
                                if k == 1:
                                    for j in res_z[k][i]:
                                        f_s.write('%f ' % (j))
                                    f_s.write('\n')
                                else:
                                    for j in res_z[k][i]:
                                        f_i.write('%f ' % (j))
                                    f_i.write('\n')
                        
                        else:
                            if preds[1][i][0] == 1:
                                for j in res_z[1][i]:
                                    f_s.write('%f ' % (j))
                                f_s.write('\n')
                            else:
                                for j in res_z[2][i]:
                                    f_i.write('%f ' % (j))
                                f_i.write('\n')
                  '''
            '''
            st, ed = 0,1
            f_out = open(FLAGS.inference_path+'.out', 'w')
            while st < len(posts):
                sample_cnt = 0
                temp_response = [[], [], []]
                right_cnt = 0
                f_res = [0, 0, 0]
                while sample_cnt <= 50:
                    temp = inference(model, sess, posts[st: ed], test_key[st: ed])
                    if f_res[temp[1][0]] == 0:
                        temp_response[temp[1][0]].append(temp[0][0])
                        f_res[temp[1][0]] = 1
                        right_cnt += 1
                    sample_cnt += 1
                #responses += temp[0]
                #print len(temp[0])
                #print len(temp[1])
                #for i in range(len(temp[1])):
                    #preds.append(temp[1][i])
                #preds += temp[1]
                f_out.writelines('%s\n' % (''.join(posts[st])))
                for i in range(3):
                    if f_res[i] != 0:
                        f_out.writelines('%s\t%s\n' % (''.join(temp_response[i][0]), str(i)))
                    else:
                        f_out.writelines('%s\n' % ('None'))
                st, ed = ed, ed+1
            '''

            #with open(FLAGS.inference_path+'.out', 'w') as f:
                #for r, l in zip(responses, preds):
                    #f.writelines('%s\t%s\n' % (''.join(p), ''.join(r)))
                    #f.writelines('%s\t%s\n' % (''.join(r), str(l)))


