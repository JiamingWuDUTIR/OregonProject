import numpy as np
import gensim


class TextProcessor:
    def __init__(self, sent_lst):
        # sent list 中每个 item 是一个句子str，用空格分好了词
        self.prefix = ['<pad>', '<s>', '<\s>', '<unk>']  # 特殊符号占位
        self.sent_lst = sent_lst
        self.word_freq_dct = {}
        self.word2id = {}
        self.id2word = {}
        self.vocab = None
        self.weights = None

    def build_word_freq_dct(self):
        # 统计词频，获得词频字典
        for sent in self.sent_lst:
            for word in sent.split(' '):
                word = word.strip()
                if len(word) != 0:
                    self.word_freq_dct[word] = self.word_freq_dct.get(word, 0) + 1
        print('Original %d words in vocabulary.' % len(self.word_freq_dct))

    def build_word2id(self, vocab_size=None):
        # 按词频排序，做截断，建立 word: id 字典
        if not vocab_size:
            vocab_size = len(self.word_freq_dct)
        sorted_freq_lst = sorted(self.word_freq_dct.items(), key=lambda x: x[1], reverse=True)
        original_len = len(sorted_freq_lst)
        original_times = sum([i[1] for i in sorted_freq_lst])

        if vocab_size < original_len:  # 如果词表过多，做一个截断
            sorted_freq_lst = sorted_freq_lst[:vocab_size]
        truncated_times = sum(i[1] for i in sorted_freq_lst)

        words_remain = [i[0] for i in sorted_freq_lst]
        self.vocab = self.prefix[:] + words_remain  # 词表加入了4个预设定占位符

        self.word2id = dict([(token, i+2) for i, token in enumerate(self.vocab)])  # 1留给未登录词，故从2开始
        self.id2word = dict(zip(self.word2id.values(), self.word2id.keys()))

        print('After truncated low frequent word:')
        truncated_ratio = truncated_times / original_times
        print('words num: %d/%d; words freq: %.3f' % (len(self.vocab)-len(self.prefix), original_len, truncated_ratio))

    def build_weights(self, keyed_vectors=None):
        # 构造词向量matrix
        # 预训练w2v embedding 导入方法：
        # from gensim.models import KeyedVectors
        # key_words = KeyedVectors.load_word2vec_format(w2v_file)
        if type(keyed_vectors) is gensim.models.keyedvectors.Word2VecKeyedVectors:
            keyed_vectors = keyed_vectors.wv  # 如果有训练好的w2v类，直接作为词向量
            dim = keyed_vectors.vector_size

            weight_id0 = np.zeros(shape=(1, dim), dtype=np.float32)  # mask 0 的weights 全0
            weight_id1 = np.random.uniform(low=-0.1, high=0.1, size=(1, dim))  # 未登入词，id为1的weight
            weights = [np.random.uniform(low=-0.1, high=0.1, size=(dim,))  # 特殊符和正常的单词的weights
                       if word not in keyed_vectors
                       else keyed_vectors[word] for word in self.vocab]
            weights = np.array(weights)
            weights_all = np.concatenate([weight_id0, weight_id1, weights])
            weights_all = np.array(weights_all, dtype=np.float32)

            exist_cnt = len([word for word in self.vocab if word in keyed_vectors])
            print('Words exit in w2v file: %d/%d, rate: %f%%' % (exist_cnt, len(self.vocab), 100 * exist_cnt / len(self.vocab)))
            print('Shape of weight matrix:', weights_all.shape)
            self.weights = weights_all
            return weights_all
        else:
            print('Build weights error, not a pre-train weights file!')

    def view_sent_length_freq(self):
        # 统计所有句子的长度，按频率排序
        sent_length_lst = [len(sent.split(' ')) for sent in self.sent_lst]
        sent_length_dct = {}
        for length in sent_length_lst:
            sent_length_dct[length] = sent_length_dct.get(length, 0) + 1
        sort_length_lst = sorted(sent_length_dct.items(), key=lambda x: x[0])
        print('length of sentence: length : freq')
        for i, j in sort_length_lst:
            print(i, j)

    def get_truncate_id_list(self, sent_lst, truncated_len):
        # 把所有句子变成一个id的array，做截断和补齐
        # 返回id矩阵 和 记录了句子真实长度的矩阵
        sent_lst_id_array = np.zeros(shape=[len(sent_lst), truncated_len], dtype=np.int32)
        sent_len_lst = []  # 记录句子真实长度
        for i, sent in enumerate(sent_lst):
            sent_id_lst = []
            for word in sent.split(' '):
                word = word.strip()
                if len(word) != 0:
                    if word in self.word2id:
                        sent_id_lst.append(self.word2id[word])
                    else:
                        sent_id_lst.append(1)  # 未登录词 id=1

            if len(sent_id_lst) < truncated_len:
                sent_len_lst.append(len(sent_id_lst))
                sent_id_lst.extend([0]*(truncated_len-len(sent_id_lst)))
            else:
                sent_id_lst = sent_id_lst[:truncated_len]
                sent_len_lst.append(truncated_len)
            sent_lst_id_array[i, :] = np.array(sent_id_lst, dtype=np.int32)
        sent_len_array = np.array(sent_len_lst, dtype=np.int32)
        return sent_lst_id_array, sent_len_array


