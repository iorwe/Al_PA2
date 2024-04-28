from torch.utils.data import Dataset
from gensim.models import KeyedVectors
import os
import torch
import tqdm

class TextDataset(Dataset):
    def __init__(self, file_path, file_name,sentence_length):
        self.file_path = os.path.join(file_path, file_name)
        self.sentence_length = sentence_length
        self.data = []
        word_vectors = KeyedVectors.load_word2vec_format(os.path.join(file_path, 'wiki_word2vec_50.bin'), binary=True)

        with open(self.file_path, 'r') as file:
            data = file.readlines()
            for line in tqdm.tqdm(data,leave=False,desc='Loading data'):
                sentence = line.split()
                label = int(sentence[0])
                words = sentence[1:]
                vectors=[torch.tensor(word_vectors[word]) for word in words if word in word_vectors][:sentence_length]
                if len(vectors) < sentence_length:
                    vectors += torch.zeros(sentence_length - len(vectors), 50)
                self.data.append((torch.stack(vectors), label))

    def __getitem__(self, idx):
        vectors, label = self.data[idx]
        return vectors, label
    
    def __len__(self):
        return len(self.data)
    