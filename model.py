import torch.nn as nn
import torch

class CharRNN(nn.Module):

                                                     
    def __init__(self, dic_size, hidden_dim, n_layers, drop_prob):
    
        super(CharRNN,self).__init__()
        
        self.input_dim  = dic_size        # 36
        self.hidden_dim = hidden_dim      # 36 (dic_size = hidden_dim)
        self.n_layers = n_layers          # 2-
        self.drop_prob = drop_prob        # 0.3
        
        self.rnn = nn.RNN(self.input_dim, self.hidden_dim, self.n_layers, dropout=self.drop_prob, batch_first=True)
        self.fc  = nn.Linear(self.hidden_dim, self.input_dim)
        

    def forward(self, input, hidden):

        output, hidden = self.rnn(input, hidden)
        output= self.fc(output)

        return output, hidden

    # https://stackoverflow.com/questions/55350811/in-language-modeling-why-do-i-have-to-init-hidden-weights-before-every-new-epoc
    # https://discuss.pytorch.org/t/initialization-of-first-hidden-state-in-lstm-and-truncated-bptt/58384
    # https://pytorch.org/docs/stable/generated/torch.nn.RNN.html : Defaults to zeros if not provided.
    def init_hidden(self, batch_size):

        # h0
        return torch.zeros(self.n_layers, batch_size, self.hidden_dim) 


class CharLSTM(nn.Module):

    def __init__(self, dic_size, hidden_dim, n_layers, drop_prob):
        
        super(CharLSTM,self).__init__()
        
        self.input_dim  = dic_size        # 36
        self.hidden_dim = hidden_dim      # 36 (dic_size = hidden_dim)
        self.n_layers = n_layers          # 2-
        self.drop_prob = drop_prob        # 0.3    

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.n_layers, dropout=self.drop_prob, batch_first=True)
        self.fc   = nn.Linear(self.hidden_dim, self.input_dim)


    def forward(self, input, hidden):

        output, hidden = self.lstm(input, hidden)
        output= self.fc(output)
        
        return output, hidden

    def init_hidden(self, batch_size):

        # h0, c0
        return torch.zeros(self.n_layers, batch_size, self.hidden_dim), torch.zeros(self.n_layers, batch_size, self.hidden_dim)
    
    

if __name__ == '__main__':
    
    model = CharLSTM(36, 36, 3, 0.3)
    x = torch.randn(8, 30, 36)
    hidden = model.init_hidden(8)
    print(model(x, hidden)[0].shape)
    
