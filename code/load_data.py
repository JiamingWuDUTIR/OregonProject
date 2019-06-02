import pickle
import pandas as pd
from gensim.models import KeyedVectors
from text_preprocess import TextProcessor
from gensim.models import KeyedVectors


def pickle_load(path):
    with open(path, 'rb') as pickle_open:
        obj = pickle.load(pickle_open)
    print('Load pickle data from', path)
    return obj


def pickle_dump(obj, path):
    with open(path, 'wb') as pickle_open:
        pickle.dump(obj, pickle_open)
    print('Dump pickle data to', path)


def load_imdb_data(vocab_size = 40000, max_seq_len = 500):
    data_pickle_path = '../data/imdb_data_3col.pkl'
    df = pickle_load(data_pickle_path)
    
    train_df = df[df['train_tag'] == 'train']
    test_df = df[df['train_tag'] == 'test']
    
    train_sent_lst = train_df['sentence']
    # 构建词表
    data_processor = TextProcessor(train_sent_lst)
    data_processor.build_word_freq_dct()
    data_processor.build_word2id(vocab_size)

    # 构建词向量矩阵
    wv_path = '../data/2-W2V.50d.txt'
    key_words = KeyedVectors.load_word2vec_format(wv_path)
    data_processor.build_weights(key_words)
    

    train_seqs, train_lens = data_processor.get_truncate_id_list(train_df['sentence'], max_seq_len)
    test_seqs, test_lens = data_processor.get_truncate_id_list(test_df['sentence'], max_seq_len)

    train_data = {'data': train_seqs, 'data_len': train_lens, 'label': train_df['label']}
    test_data = {'data': test_seqs, 'data_len': test_lens, 'label': test_df['label']}

    print('Train data shape:', train_data['data'].shape, 'label length:', len(train_data['label']))
    print('Test data shape:', test_data['data'].shape, 'label length:', len(test_data['label']))
    
    return train_data, test_data, data_processor.weights
    
    