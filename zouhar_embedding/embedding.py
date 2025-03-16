from zouhar_embedding.MetricLearner import RNNMetricLearner, RNNVocab
import torch
import math
import tqdm
class ZouharEmbedder():

    model = RNNMetricLearner(
        dimension=300,
        feature_size=593
    )

    def __init__(self):
        self.vocab = RNNVocab()
        self.vocab.get_multi_data()
        self.vocab.get_vocab_all()
        self.vocab.get_vocab_size()
        self.load_model_state()

    def load_model_state(self):
        self.model.load_state_dict(torch.load("/csse/research/NativeLanguageID/mthesis-phonological/zouhar-embedding-project/models/rnn_metric_learning_token_ort_all.pt"))

    # def embed_word(self, word):
    #     x = {}
    #     x["token_ort"] = word
    #     x["token_ipa"] = None
    #     x_list = []
    #     x_list.append(x)
    #     print(x_list)
    #     word = self.vocab.onehot_encode(x_list)
    #     print(word)
    #     return list(self.model.forward(word).detach().cpu().numpy())
    
    def embed_list(self, words):
        words = [[word, None] for word in words]
        a = ["token_ort", "token_ipa"]
        x = [dict(zip(a,word)) for word in words]
        word = self.vocab.onehot_encode(x)
        BATCH_SIZE = 256
        data_out = []
        for i in tqdm.tqdm(range(math.ceil(len(word) / BATCH_SIZE))):
            batch = [f for f, _ in word[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]]
            data_out += list(
                self.model.forward(batch).detach().cpu().numpy()
            )
        return data_out

    def vec(self, word):
        if type(word) != type('str'):
            raise TypeError('this function takes a single word in the form of a str')
        tmp = []
        tmp.append(word)
        words = tmp
        words = [[word, None] for word in words]
        a = ["token_ort", "token_ipa"]
        x = [dict(zip(a,word)) for word in words]
        word = self.vocab.onehot_encode(x)
        BATCH_SIZE = 1
        data_out = []
        for i in range(math.ceil(len(word) / BATCH_SIZE)):
            batch = [f for f, _ in word[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]]
            data_out += list(
                self.model.forward(batch).detach().cpu().numpy()
            )
        data_out = data_out.pop() #This is because we want only the vector and not the batch_list of length 1
        return data_out
