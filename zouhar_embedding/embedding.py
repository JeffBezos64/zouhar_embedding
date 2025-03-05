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
    
    def embed_word(self, word):
        x = {}
        x["token_ort"] = word
        x["token_ipa"] = None
        x_list = []
        x_list.append(x)
        print(x_list)
        word = self.vocab.onehot_encode(x_list)
        print(word)
        BATCH_SIZE = 32
        data_out = []
        for i in tqdm.tqdm(range(math.ceil(len(word) / BATCH_SIZE))):
            batch = [f for f, _ in word[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]]
            data_out += list(
                self.model.forward(batch).detach().cpu().numpy()
            )
        return data_out

    def embed_list(self, word_list):
        pass 