from tensorflow.keras.utils import Sequence
import math
from tensorflow.keras.preprocessing.sequence import pad_sequences

class DynamicPadding(Sequence):
    '''
    implements batch sampling with dynamic padding. Credits to

    https://towardsdatascience.com/divide-hugging-face-transformers-training-time-by-2-or-more-21bf7129db9q-21bf7129db9e
    '''

    def __init__(self, tokenized_seqs, targets, batch_size):
        self.y = tokenized_seqs
        self.z = targets
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.y) / self.batch_size)
        
    def __getitem__(self, idx):
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
              self.batch_size]
        batch_z = self.z[idx * self.batch_size:(idx + 1) *
              self.batch_size]
        # padding w.r.t. the longest sequence in the batch
        batch_y = pad_sequences(batch_y, padding='post')
        return batch_y, batch_z