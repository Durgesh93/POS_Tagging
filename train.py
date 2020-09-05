import torch.nn as nn
from  TreeBankDataSet import CustomDataset
from torch.autograd import Variable
import torch
from torch.utils.data import random_split,DataLoader
from tqdm import tqdm
from pos_tagger import GRUModel
import matplotlib.pyplot as plt



def plot_acc(acc_arr):
     plt.figure()
     plt.title('Accuracy/Epoch')
     plt.plot(acc_arr)
     plt.xlabel('Epoch Number')
     plt.ylabel('Accuracy value')
     plt.show()

def train(EMBEDDING_DIM=400,hidden_dim=400,learning_rate=0.1,num_epochs=10,batch_size=40,CELL_TYPE='s',opt='Adam',dname='brown',loss='nll',spl=0.8):
    print(dname)
    print('\n###########   Starting Data Preprocessing       #################### \n')    
    
    cdataset = CustomDataset(dname=dname)  
    vocab_size = cdataset.get_vocab_size()
    target_size = cdataset.get_target_size()
    output_dim = target_size
    train_size = int(spl * len(cdataset))
    test_size = len(cdataset) - train_size
    train_dataset, test_dataset = random_split(cdataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size,shuffle=True,drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size,shuffle=True,drop_last=True)
    model = GRUModel(EMBEDDING_DIM, hidden_dim, output_dim,vocab_size,batch_size,CELL_TYPE=CELL_TYPE)
    
    if torch.cuda.is_available():
        model.cuda()
    else:
        print('Cuda not available')
        return 
     
    criterion=None
    if loss == 'nll':
        criterion = nn.NLLLoss()
    else:
        print('Invalid loss function')
        return
 
       
    optimizer=None
    
    if opt=='Adam':    
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif opt=='SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    else:
        print('Invalid optimizer')
        return
 
    
    if CELL_TYPE not in ['s','m1','m2','m3']:
        print('Invalid cell type')
        return
    
    print('\n###########   Starting Training       #################### \n')
    acc_arr=[]      
    for epoch in range(num_epochs):
        training_loss = []
        for batch_no,(sample, labels) in  tqdm(enumerate(train_loader),total=len(train_loader),desc='Training Epoch {}'.format(epoch+1)):
            if torch.cuda.is_available():
                sample = Variable(sample.cuda())
                labels = Variable(labels.cuda())
            else:
                sample = Variable(sample)
                labels = Variable(labels)  
            optimizer.zero_grad()
            outputs = model(sample)
            outputs=outputs.view(-1,target_size)
            labels=labels.view(-1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            training_loss.append(loss.item())
        
            
        accuracy=0
        correct = 0
        total = 0
        
        for batch_no,(test_sample, test_labels) in tqdm(enumerate(test_loader),total=len(test_loader),desc='Testing Epoch {}'.format(epoch+1)):
            
            if torch.cuda.is_available():
                test_sample = Variable(test_sample.cuda())
                test_labels = Variable(test_labels.cuda())
            else:
                sample = Variable(test_sample)
                labels = Variable(test_labels)
            
            test_outputs = model(test_sample)
            _, predicted = torch.max(test_outputs.data, 2)
            predicted = predicted.view(-1)
            test_labels = test_labels.view(-1)
            
            total += len(test_labels)
            correct += (predicted == test_labels).sum()
            accuracy = 100 * correct / total
            acc_arr.append(accuracy)
            
    plot_acc(acc_arr)
           
            
       