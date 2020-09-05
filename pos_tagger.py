import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
from gru import GRUCell,GRUCellM1,GRUCellM2,GRUCellM3


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,vocab_size,batch_size,CELL_TYPE='s'):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.word_embeddings = nn.Embedding(vocab_size, input_dim)

        if CELL_TYPE=='s':
            self.gru_cell = GRUCell(input_dim, hidden_dim,batch_size)
        elif CELL_TYPE=='m1':
            self.gru_cell = GRUCellM1(input_dim, hidden_dim,batch_size)
        elif CELL_TYPE=='m2':
            self.gru_cell = GRUCellM2(input_dim, hidden_dim,batch_size)
        elif CELL_TYPE=='m3':
            self.gru_cell = GRUCellM3(input_dim, hidden_dim,batch_size)
        self.fc = nn.Linear(hidden_dim, output_dim)
     
    def forward(self, x):
        x=self.word_embeddings(x)
        batch_size = self.batch_size
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(batch_size,self.hidden_dim).cuda())
            
        else:
            h0 = Variable(torch.zeros(batch_size,self.hidden_dim))
            
        outs = torch.zeros(batch_size,x.size(1),self.hidden_dim).cuda()
        
        hn=h0
        
        for seq in range(x.size(1)):
            hn = self.gru_cell(x[:,seq,:], hn) 
            outs[:,seq,:]=hn
            
        out = self.fc(outs)
        out = F.log_softmax(out,dim=2)
        return out
    