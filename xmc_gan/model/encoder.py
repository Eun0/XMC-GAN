import torch 
import torch.nn as nn

from torch.nn.uitls.rnn import pack_padded_sequence, pad_packed_sequence


class RNN_ENCODER(nn.Module):
    def __init__(self,cfg):
        super(RNN_ENCODER,self).__init__()
        self.n_steps = cfg.TEXT.MAX_LENGTH
        self.ntoken = cfg.TEXT.VOCA_SIZE
        self.ninput = 300
        self.drop_prob = 0.5
        self.nlayers = 1
        self.bidirectional = True 
        self.rnn_type = cfg.TEXT.RNN_TYPE
        if self.bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        self.nhidden = cfg.TEXT.EMBEDDING_DIM // self.num_directions

        self._define_modules()
        self._init_weights()

    def _define_modules(self):
        self.encoder = nn.Embedding(self.ntoken,self.ninput)
        self.drop = nn.Dropout(self.drop_prob)
        if self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(self.ninput,self.nhidden,self.nlayers,batch_first=True,
                                dropout=self.drop_prob,
                                bidirectional=self.bidirectional)
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(self.ninput,self.nhidden,self.nlayers,batch_first=True,
                                dropout=self.drop_prob,
                                bidirectional = self.bidirectional)
        else:
            raise NotImplementedError()

        print('Use rnn encoder')

    def _init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange,initrange)
    
    def _init_hidden(self,batch_size):
        weight = next(self.parameters()).data 
        if self.rnn_type =='LSTM':
            return(nn.Parameter(weight.new(self.nlayers*self.num_directions,batch_size,self.nhidden).zero_()),
                    nn.Parameter(weight.new(self.nlayers*self.num_directions,batch_size,self.nhidden).zero_()))
        else:
            return nn.Parameter(weight.new(self.nlayers * self.num_directions,batch_size,self.nhidden).zero_())

    def forward(self,caps,cap_lens, **kwargs):

        caps = caps.cuda()
        cap_lens = cap_lens.cuda()

        sorted_cap_lens, sorted_idx = cap_lens.sort(descending=True)
        sorted_caps = caps[sorted_idx]
        sorted_cap_lens = sorted_cap_lens.tolist()

        batch_size = sorted_caps.size(0)
        hiddens = self._init_hidden(batch_size)

        sorted_embs = self.drop(self.encoder(sorted_caps))

        sorted_embs = pack_padded_sequence(sorted_embs, sorted_cap_lens, batch_first = True)
        sorted_outputs, sorted_hiddens = self.rnn(sorted_embs, hiddens)

        #sorted_outputs = pad_packed_sequence(sorted_outputs, batch_first= True, total_length = self.n_steps)[0]
        sorted_outputs = pad_packed_sequence(sorted_outputs, batch_first= True)[0]

        sorted_words_embs = sorted_outputs.transpose(1,2)

        if self.rnn_type == 'LSTM':
            sorted_sent_embs = sorted_hiddens[0].transpose(0,1).contiguous()
        else:
            sorted_sent_embs = sorted_hiddens.transpose(0,1).contiguous()
        
        sorted_sent_embs = sorted_sent_embs.view(-1,self.nhidden * self.num_directions)

        mask = (caps == 0) 

        words_embs = sorted_words_embs[sorted_idx.argsort()]
        sent_embs = sorted_sent_embs[sorted_idx.argsort()]

        return words_embs, sent_embs, mask