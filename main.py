import dataset
import model as M
from model import CharRNN, CharLSTM
from dataset import Shakespeare

# import some packages you need here
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import torch.nn as nn
import torch

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
# from tqdm.auto import tqdm


def train(model, trn_loader, device, criterion, optimizer):
    """ Train function

    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim

    Returns:
        trn_loss: average loss value
    """

    model.train()  # train

    total_loss = 0    

    for batch in trn_loader:
            
        input, target = map(lambda x: x.to(device), batch)
                
        # for lstm
        if isinstance(model.init_hidden(len(input)), tuple):
            h0 = model.init_hidden(len(input))[0].to(device)
            c0 = model.init_hidden(len(input))[1].to(device)
            hidden = (h0, c0)
        
        else : hidden = model.init_hidden(len(input)).to(device)
        
        predict, _ = model(input, hidden)
        
        # Make 2D tensor : https://discuss.pytorch.org/t/rnn-for-many-to-many-classification-task/15457/3
        # >>> X : Batch * seq_length(30), dic_size(36), y : Batch * seq_length(30)
        loss = criterion(predict.reshape(-1, predict.shape[-1]), target.reshape(-1))

        # backward
        optimizer.zero_grad()  
        loss.backward()  
        optimizer.step()  
        
        # loss : sum
        total_loss += loss.item()    

    # get average loss (total/batch)
    trn_loss = total_loss / len(trn_loader)

    return trn_loss

def validate(model, val_loader, device, criterion):
    """ Validate function

    Args:
        model: network
        val_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        val_loss: average loss value
    """

    model.eval()    # eval

    total_loss = 0

    for batch in val_loader:    
        
        input, target = map(lambda x: x.to(device), batch)
        
        # for lstm
        if isinstance(model.init_hidden(len(input)), tuple):
            h0 = model.init_hidden(len(input))[0].to(device)
            c0 = model.init_hidden(len(input))[1].to(device)
            hidden = (h0, c0)
        
        else : hidden = model.init_hidden(len(input)).to(device)  # batch_size : len(input)
        
        predict, _ = model(input, hidden)
        
        loss = criterion(predict.reshape(-1, predict.shape[-1]), target.reshape(-1))

        # loss : sum
        total_loss += loss.item()    

    # get average loss (total/batch)
    val_loss = total_loss / len(val_loader)

    return val_loss


def main():
    """ Main function

        Here, you should instantiate
        1) DataLoaders for training and validation. 
        2) Try SubsetRandomSampler to create these DataLoaders.   # SubsetRandomSampler
           # https://velog.io/@olxtar/PyTorch-%EC%82%AC%EC%9A%A9%EB%B2%95
        3) model
        4) optimizer
        5) cost function: use torch.nn.CrossEntropyLoss

    """
    
    # train_config
    validation_split = 0.2
    batch_size = 256       
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lr = 0.001
    epochs = 30
    #model_name = 'CharRNN'
    model_name = 'CharLSTM'
    print(f'device : {device}')   
    print(f'model name : {model_name}')   
       
    # DataLoaders for training and validation.         
    dataset = Shakespeare('./shakespeare_train.txt')
    dataset_size = len(dataset)
    print(f'len - char_to_index : {len(dataset.char_to_index)}')
    
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))  
    
    # split  
    train_indices, val_indices = indices[split:], indices[:split]   
    print(f'len - train_indices : {len(train_indices)} / len - val_indices : {len(val_indices)}')
     
    # SubsetRandomSampler to create these DataLoaders.  
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    
    trn_loader = DataLoader(dataset, batch_size=batch_size,sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size,sampler=valid_sampler) 
    
    # model
    model = getattr(M, model_name)(dic_size=len(dataset.char_to_index) ,hidden_dim=64, n_layers=3, drop_prob=0.3).to(device)
    print(model)
    
    # optimizer
    # cost function: use torch.nn.CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)    
    

    trn_loss_list, val_loss_list = [], []

    for epoch in range(epochs) :
        
        trn_loss = train(model, trn_loader, device, criterion, optimizer)
        trn_loss_list.append(trn_loss)
        
        val_loss = validate(model, val_loader, device, criterion)
        val_loss_list.append(val_loss)
        
        print(f"Epoch {epoch:2d}; trn_loss: {trn_loss:.3f} / val_loss: {val_loss:.3f}")  
    
    # model_save
    torch.save(model, f"model_{model_name}.pt")
    
    # visualize    
    plt.figure(figsize=(10,4))
    sns.lineplot(trn_loss_list, marker='o', color='green', label='train')
    sns.lineplot(val_loss_list, marker='o', color='red', label='validate')
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))
    plt.ylim(1.5, 2.5)
    plt.legend()
    plt.title(model_name)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.savefig(f'train_{model_name}.png')
    plt.show()

    

if __name__ == '__main__':
    
    main()