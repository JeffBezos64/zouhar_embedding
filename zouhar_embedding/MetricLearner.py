import torch
from sklearn.metrics.pairwise import cosine_distances
import random
import tqdm
import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import datasets
import torch
import torch.nn.functional as F 

class RNNMetricLearner(torch.nn.Module):
    

    def __init__(
        self,
        target_metric="l2",
        feature_size=24,
        dimension=300,
        safe_eval=False,
    ):
        super().__init__()

        self.model = torch.nn.LSTM(
            input_size=feature_size,
            hidden_size=dimension//2,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True,
        )
        self.batch_size_eval = 2048
        self.batch_size_train = 128

        # TODO: contrastive learning
        self.loss = torch.nn.MSELoss()
        self.panphon_distance = Dict()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)

        if target_metric == "l2":
            self.dist_embd = torch.nn.PairwiseDistance(p=2)
        elif target_metric == "cos":
            self.dist_embd = torch.nn.CosineSimilarity()
        else:
            raise Exception(f"Unknown metric {target_metric}")

        self.evaluator = Evaluator(safe_eval=safe_eval)

        # move the model to GPU
        self.to(device)

    def forward(self, ws):
        # TODO: here we use -1 for padding because 0.0 is already
        # used somewhere. This may not matter much but good to be aware of.
        ws = torch.nn.utils.rnn.pad_sequence(
            [torch.Tensor(x) for x in ws],
            batch_first=True, padding_value=-1.0,
        ).to(device)
        output, (h_n, c_n) = self.model(ws)

        # take last vector for all elements in the batch
        output = output[:, -1, :]

        return output

class RNNVocab():
    def __init__(self):
        self.data = None
        self.vocab = None
        self.vocab_size = None
        self.UNK_SYMBOL = "😕"

    def token_onehot(self, word):
        indices = [self.vocab[c] for c in word if c in self.vocab]
        return F.one_hot(torch.tensor(indices), num_classes=self.vocab_size).float()

    def onehot_encode(self, sample=None):
        """takes a dict of words and returns their one hot encoding for the model"""
        return [(self.token_onehot(x["token_ort"]), x["token_ipa"]) for x in sample]

    
    def get_vocab_size(self):
        self.vocab_size = len(self.vocab)

    
    def get_vocab_all(self):
        UNK_SYMBOL = "😕"
        vocab_raw = [c for word in self.data for c in word["token_ort"]]
        characters = {UNK_SYMBOL} | set(vocab_raw)
        vocab = {char: idx for idx, char in enumerate(characters)}
        self.vocab = vocab

    
    def get_multi_data(self, purpose_key="main"):
        data = datasets.load_dataset("json", data_files="vocab/train.jsonl",split="train")
        self.data = data

    def __str__(self):
        try:
            return f'self.data length is {len(self.data)}, self.vocab length is {len(self.vocab)}, self.vocab_size is {self.vocab_size}, self.UNK_SYMBOL is {self.UNK_SYMBOL}'
        except TypeError:
            return f'Not fully instantiated'
