import os
import numpy as np
import sys
import argparse
import pickle
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

# from HRFL_utils import loadTrainValidTest, workout2index, dataIteratorSupervised, generator_for_autotrain

use_cuda = torch.cuda.is_available()
cuda0 = torch.device('cuda:0')
cuda1 = torch.device('cuda:1')


class LSTMTarget(nn.Module):
    def __init__(self, country='France', userId=None, inputAtts=['distance','altitude','time_elapsed'], targetAtts=['heart_rate'], includeTemporal=True, hidden_dim=64, context_final_dim=32, lr=0.005, max_local_epochs=200, weight_decay=.001, use_cuda=True):
        super(LSTMTarget, self).__init__()
        self.local_epochs = max_local_epochs
        self.lr = lr
        self.dropout_rate = .2
        self.wd = weight_decay

        self.inputAtts = inputAtts
        self.targetAtts = 'tar_' + targetAtts[0]
        self.num_steps = 500
        self.includeTemporal = includeTemporal
        self.trimmed_workout_len = self.num_steps

        self.input_dim = len(self.inputAtts)
        self.output_dim = 1
        self.hidden_dim = hidden_dim
        self.context_final_dim = context_final_dim

        # build the context embedding for workout profile forecasting
        total_input_dim = self.input_dim
        if self.includeTemporal:
            self.context1_dim = self.input_dim + 1
            self.context2_dim = 1

            self.context_layer_1 = nn.LSTM(input_size = self.context1_dim, hidden_size = self.hidden_dim, batch_first=True)
            self.context_layer_2 = nn.LSTM(input_size = self.context2_dim, hidden_size = self.hidden_dim, batch_first=True)
            self.dropout_context = nn.Dropout(self.dropout_rate)
            # then apply nn.Linear on the concat between context1 and context2 inputs
            self.project = nn.Linear(self.hidden_dim * 2, self.context_final_dim)

            if use_cuda:
                self.context_layer_1 = self.context_layer_1.to(cuda0)
                self.context_layer_2 = self.context_layer_2.to(cuda0)
                self.project = self.project.to(cuda0)

            total_input_dim += self.context_final_dim

        self.lstm_stacked = nn.LSTM(input_size=total_input_dim,
            hidden_size=self.hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=.2)
        self.linear = nn.Linear(self.hidden_dim, self.output_dim)

        if use_cuda:
            self.lstm_stacked = self.lstm_stacked.to(cuda0)
            self.linear = self.linear.to(cuda0)

        # print("Parameters of context_layer_2")
        # print(self.context_layer_2._all_weights)    # names of all weights
        # print(self.context_layer_2.weight_ih_l0.shape)
        # print(self.context_layer_2.bias_ih_l0.shape)
        # print(self.context_layer_2.weight_hh_l0.shape)
        # print(self.context_layer_2.bias_hh_l0.shape)


        self.optimizer = torch.optim.RMSprop(
            [
                {'params': [p for n, p in self.named_parameters()]}
            ], lr=self.lr, weight_decay=self.wd
        )


    def forward(self, all_inputs):
        
        h_t = self.init_hidden(all_inputs, n_layers=2).float() #[stack,batch,hid]
        c_t = self.init_hidden(all_inputs, n_layers=2).float()

        if use_cuda:
            h_t = h_t.to(cuda0)
            c_t = c_t.to(cuda0)

        result, (h_t, c_t) = self.lstm_stacked(all_inputs, (h_t, c_t))
        result = F.selu(self.linear(result))
        return result


    def inner_train(self, trainBatch):
        # with torch.autograd.set_detect_anomaly(True):
        batches = list(self.trainBatch)
        num_batches = len(batches)
        loss_func = nn.MSELoss(reduction='mean')

        for epoch in range(self.local_epochs):
            for b in range(num_batches):
                inputs, target = self.embed_inputs(batches[b])
                inputs = inputs.float().to(cuda0)
                out = self.forward(inputs)
                self.optimizer.zero_grad()
                self.loss = loss_func(out, target.to(cuda0))
                self.loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 0.1)
                self.optimizer.step()
            if (epoch+1) % 10 == 0 or epoch==0:
                print('-------Epoch: %d, loss %1.5f' % (epoch+1, self.loss.item()))


    def init_hidden(self, x, n_layers=1):
        return Variable(torch.zeros(n_layers, x.size(0), self.hidden_dim)) # dimension 0: batch

    def embed_inputs(self, batch):
        inputs = Variable(torch.tensor(batch[0]['input'])).to(cuda0)   # size = [52, 500, 3]
        context_input_1 = batch[0]['context_input_1']
        context_input_2 = batch[0]['context_input_2']
        context_input_1 = Variable(torch.from_numpy(context_input_1).float())
        context_input_2 = Variable(torch.from_numpy(context_input_2).float())
        if use_cuda:
            context_input_1 = context_input_1.to(cuda0)
            context_input_2 = context_input_2.to(cuda0)

        context_input_1 = self.dropout_context(context_input_1)
        context_input_2 = self.dropout_context(context_input_2)

        hidden_1 = self.init_hidden(inputs).to(cuda0)  # [1, 52, 64]
        cell_1 = self.init_hidden(inputs).to(cuda0)
        hidden_2 = self.init_hidden(inputs).to(cuda0)
        cell_2 = self.init_hidden(inputs).to(cuda0)

        outputs_1, (_, _) = self.context_layer_1(context_input_1, (hidden_1, cell_1))
        outputs_2, (_, _) = self.context_layer_2(context_input_2, (hidden_2, cell_2))

        all_outputs = torch.cat([outputs_1, outputs_2], dim=-1)
        all_outputs = self.project(all_outputs)

        all_inputs = torch.cat([inputs, all_outputs], dim=-1)
        outputs = torch.tensor(batch[1]).float()
        return all_inputs, outputs


if __name__ == "__main__":
    LSTM_model = LSTMTarget(country='France', userId=11737814, inputAtts=['distance','altitude','time_elapsed'], targetAtts=['heart_rate'], includeTemporal=True, hidden_dim=64, context_final_dim=32, lr=0.005, max_local_epochs=201, weight_decay=.001, use_cuda=True)

    # LSTM_model.inner_train()
