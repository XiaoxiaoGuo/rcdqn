import numpy as np
import pickle


def save_glove_numpy():
    embs = []
    word2id = {'<PAD>': 0,
               '<OOV>': 1}

    embs.append(np.asarray([0.0] * 100, 'float32'))
    embs.append(np.asarray([0.0] * 100, 'float32'))

    emb_sum = 0
    count = 0
    with open('glove.6B.100d.txt', 'r') as f:
        for i, line in enumerate(f):
            array = line.strip().split(' ')
            word = array[0]
            word2id[word] = len(word2id)
            e = np.asarray(array[1:], 'float32')
            emb_sum += e
            count += 1
            embs.append(e)
    emb_sum /= count
    embs[word2id['<OOV>']] = emb_sum

    # special token <ACT>
    word2id['<ACT>'] = len(word2id)
    embs.append(-emb_sum)

    save_data = {'word2id': word2id, 'embs': embs}
    with open('dict.pt', 'wb') as f:
        pickle.dump(save_data, f)


def get_dict_emb(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data['word2id'], np.asfarray(data['embs'])


if __name__ == "__main__":
    save_glove_numpy()
