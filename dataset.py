# import some packages you need here
import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F


class Shakespeare(Dataset):
    """ Shakespeare dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        input_file: txt file

    Note:
        1) Load input file and construct character dictionary {index:character}.
        You need this dictionary to generate characters.
        2) Make list of character indices using the dictionary
        3) Split the data into chunks of sequence length 30. 
        You should create targets appropriately.
    """

    def __init__(self, input_file):

        # Load input file 
        with open(input_file, 'rb') as f :
            
            sentences = []
            
            for sentence in f: 
                sentence = sentence.strip() # strip
                sentence = sentence.lower() # lower
                sentence = sentence.decode('ascii', 'ignore')  
                
                if len(sentence) > 0 :
                    sentences.append(sentence)
                
        # construct character dictionary {index:character}            
        self.total_data = ' '.join(sentences)     
        char_vocab = sorted(list(set(self.total_data)))
        self.char_to_index = dict((char, index) for index, char in enumerate(char_vocab))  
        self.index_to_char = {value : key for key, value in self.char_to_index.items()}
        
        
        
        # Make list of character indices using the dictionary
        # Split the data into chunks of sequence length 30.         
                
        seq_length = 30

        self.X = []
        self.y = []
            
        for index in range(len(self.total_data) - seq_length):
          
            X_sample = self.total_data[index: index + seq_length]
            X_encoded = [self.char_to_index[c] for c in X_sample]
            self.X.append(X_encoded)

            # shift
            y_sample = self.total_data[index + 1 : index + seq_length + 1]
            y_encoded = [self.char_to_index[c] for c in y_sample]
            self.y.append(y_encoded)
     
        # to Torch.Tensor & one_hot encoding
        self.X = torch.LongTensor(self.X)    
        self.X = F.one_hot(self.X).float()
        self.y = torch.LongTensor(self.y)    
        

    def __len__(self):

        return len(self.X)
    

    def __getitem__(self, idx):

        input  = self.X[idx]
        target = self.y[idx]

        return input, target


if __name__ == '__main__':
    
    dataset = Shakespeare('./shakespeare_train.txt')
    
    print(f"self.X {dataset.X.shape}")
    print(f"self.X[5] {dataset.X[5]}")
    print(f"self.y {dataset.y.shape}")
    print(f"self.y[5] {dataset.y[5]}")
