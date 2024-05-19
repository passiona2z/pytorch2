# import some packages you need here
import dataset
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import Shakespeare

def generate(model, seed_characters, temperature=1, *args):
    """ Generate characters

    Args:
        model: trained model
        seed_characters: seed characters
        temperature: T  
        # https://towardsdatascience.com/how-to-sample-from-language-models-682bceb97277
        # https://seongqjini.com/pytorch-softmax%EC%86%8C%ED%94%84%ED%8A%B8%EB%A7%A5%EC%8A%A4%EC%97%90-temperature-scaling-%EC%82%AC%EC%9A%A9%EC%BD%94%EB%93%9C-%EA%B5%AC%ED%98%84/
        args: other arguments if needed

    Returns:
        samples: generated characters
    """
    
    model.eval()
    batch_size = 1
    char_to_index = args[0]
    index_to_char = args[1]

    char_li = [char_to_index[character] for character in seed_characters]       # Get index
    hidden = model.init_hidden(batch_size)
    length = 100
    
    # init
    X = np.zeros((1, length+1, len(char_to_index)))  # 1,31,30
    X = torch.FloatTensor(X)
    
    # one-hot
    for i, idx in enumerate(char_li) :
        X[:, i, idx] = 1   
    
    result = []
    result += list(seed_characters)
    
    for i in range(length - len(char_li)):
        
        i += len(char_li)
        
        output, hidden = model(X[:,:i+1,:], hidden)
      
        # pred_ind = torch.argmax(output, dim=-1)[0][-1].item()
        # https://stackoverflow.com/questions/42593231/is-numpy-random-choice-with-replacement-equivalent-to-multinomial-sampling-for-a
        # softmax : out_distribution > sampling(torch.multinomial)  
        out_distribution = F.softmax(output/temperature, dim=-1)[:,-1,:]
        pred_ind = torch.multinomial(out_distribution, num_samples=1).item()
        
        # next word : one-hot
        X[:,i+1, pred_ind] = 1
        # next word : append 
        result.append(index_to_char[pred_ind])
        
    return "".join(result)
        
    

if __name__ == '__main__':

    dataset = Shakespeare('./shakespeare_train.txt')    
    model = torch.load('./model_CharLSTM.pt').cpu()

    temperature = 5   # 0.1 - 10
     
    seed_characters = 'you'
    
    result = generate(model, seed_characters, temperature, dataset.char_to_index, dataset.index_to_char)      

    print(result)