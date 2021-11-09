from collections import OrderedDict
import torch
from torch import nn
from torch.nn.utils import spectral_norm

use_cuda = torch.cuda.is_available()
cuda0 = torch.device('cuda:0')
cuda1 = torch.device('cuda:1')

class LSTMHyper(nn.Module):
    def __init__(self, input_dim=3, output_dim=1, hidden_dim = 64, embed_dim=-1, context_final_dim=32, num_nodes=10, n_hidden=3, lr=0.005, wd=.001, spec_norm=False, use_cuda=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.context_final_dim = context_final_dim

        if embed_dim == -1:
            embed_dim = int(1 + num_nodes / 4)

        self.embeddings = nn.Embedding(num_embeddings=num_nodes, embedding_dim=embed_dim)

        layers = [
            spectral_norm(nn.Linear(embed_dim, hidden_dim)) if spec_norm else nn.Linear(embed_dim, hidden_dim),
        ]
        for _ in range(n_hidden):
            layers.append(nn.ReLU(inplace=True))
            layers.append(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim),
            )

        self.mlp = nn.Sequential(*layers)


        #########################################
        # Dimension of weights - target network # 
        #########################################
        ##### context encoder LSTM
        # ------ context 1:
        # weight_ih_l0 = [256, 4]        bias_ih_l0 = [256]
        # weight_hh_l0 = [256, 64]       bias_hh_l0 = [256]
        # ------ context 2:
        # weight_ih_l0 = [256, 1]        bias_ih_l0 = [256]
        # weight_hh_l0 = [256, 64]       bias_hh_l0 = [256]

        ##### main LSTM
        # weight_ih_l0 = Size([256, 131])     bias_ih_l0 = Size([256])
        # weight_hh_l0 = Size([256, 64])      bias_hh_l0 = Size([256])
        # weight_ih_l1 = Size([256, 64])      bias_ih_l1 = Size([256])
        # weight_hh_l1 = Size([256, 64])      bias_hh_l1 = Size([256])

        self.con1_weight_ih = nn.Linear(hidden_dim, 4*hidden_dim * 4)
        self.con1_bias_ih = nn.Linear(hidden_dim, 4*hidden_dim)
        self.con1_weight_hh = nn.Linear(hidden_dim, 4*hidden_dim * hidden_dim)
        self.con1_bias_hh = nn.Linear(hidden_dim, 4*hidden_dim)

        self.con2_weight_ih = nn.Linear(hidden_dim, 4*hidden_dim * 1)
        self.con2_bias_ih = nn.Linear(hidden_dim, 4*hidden_dim)
        self.con2_weight_hh = nn.Linear(hidden_dim, 4*hidden_dim * hidden_dim)
        self.con2_bias_hh = nn.Linear(hidden_dim, 4*hidden_dim)

        self.project_weight = nn.Linear(hidden_dim, 2*hidden_dim * context_final_dim)
        self.project_bias = nn.Linear(hidden_dim, context_final_dim)

        self.lstm1_weight_ih = nn.Linear(hidden_dim, 4*hidden_dim * (context_final_dim+3))
        self.lstm1_bias_ih = nn.Linear(hidden_dim, 4*hidden_dim)
        self.lstm1_weight_hh = nn.Linear(hidden_dim, 4*hidden_dim * hidden_dim)
        self.lstm1_bias_hh = nn.Linear(hidden_dim, 4*hidden_dim)

        self.lstm2_weight_ih = nn.Linear(hidden_dim, 4*hidden_dim * hidden_dim)
        self.lstm2_bias_ih = nn.Linear(hidden_dim, 4*hidden_dim)
        self.lstm2_weight_hh = nn.Linear(hidden_dim, 4*hidden_dim * hidden_dim)
        self.lstm2_bias_hh = nn.Linear(hidden_dim, 4*hidden_dim)

        self.linear_weight = nn.Linear(hidden_dim, hidden_dim * 1)
        self.linear_bias = nn.Linear(hidden_dim, 1)

        if use_cuda:
            self.embeddings = self.embeddings.to(cuda0)
            self.mlp = self.mlp.to(cuda0)
            self.con1_weight_ih = self.con1_weight_ih.to(cuda0)
            self.con1_bias_ih = self.con1_bias_ih.to(cuda0)
            self.con1_weight_hh = self.con1_weight_hh.to(cuda0)
            self.con1_bias_hh = self.con1_bias_hh.to(cuda0)
            self.con2_weight_ih = self.con2_weight_ih.to(cuda0)
            self.con2_bias_ih = self.con2_bias_ih.to(cuda0)
            self.con2_weight_hh = self.con2_weight_hh.to(cuda0)
            self.con2_bias_hh = self.con2_bias_hh.to(cuda0)
            self.project_weight = self.project_weight.to(cuda0)
            self.project_bias = self.project_bias.to(cuda0)
            self.lstm1_weight_ih = self.lstm1_weight_ih.to(cuda0)
            self.lstm1_bias_ih = self.lstm1_bias_ih.to(cuda0)
            self.lstm1_weight_hh = self.lstm1_weight_hh.to(cuda0)
            self.lstm1_bias_hh = self.lstm1_bias_hh.to(cuda0)
            self.lstm2_weight_ih = self.lstm2_weight_ih.to(cuda0)
            self.lstm2_bias_ih = self.lstm2_bias_ih.to(cuda0)
            self.lstm2_weight_hh = self.lstm2_weight_hh.to(cuda0)
            self.lstm2_bias_hh = self.lstm2_bias_hh.to(cuda0)
            self.linear_weight = self.linear_weight.to(cuda0)
            self.linear_bias = self.linear_bias.to(cuda0)

        if spec_norm:
            self.con1_weight_ih = spectral_norm(self.con1_weight_ih)
            self.con1_bias_ih = spectral_norm(self.con1_bias_ih)
            self.con1_weight_hh = spectral_norm(self.con1_weight_hh)
            self.con1_bias_hh = spectral_norm(self.con1_bias_hh)

            self.con2_weight_ih = spectral_norm(self.con2_weight_ih)
            self.con2_bias_ih = spectral_norm(self.con2_bias_ih)
            self.con2_weight_hh = spectral_norm(self.con2_weight_hh)
            self.con2_bias_hh = spectral_norm(self.con2_bias_hh)

            self.project_weight = spectral_norm(self.project_weight)
            self.project_bias = spectral_norm(self.project_bias)

            self.lstm1_weight_ih = spectral_norm(self.lstm1_weight_ih)
            self.lstm1_bias_ih = spectral_norm(self.lstm1_bias_ih)
            self.lstm1_weight_hh = spectral_norm(self.lstm1_weight_hh)
            self.lstm1_bias_hh = spectral_norm(self.lstm1_bias_hh)

            self.lstm2_weight_ih = spectral_norm(self.lstm2_weight_ih)
            self.lstm2_bias_ih = spectral_norm(self.lstm2_bias_ih)
            self.lstm2_weight_hh = spectral_norm(self.lstm2_weight_hh)
            self.lstm2_bias_hh = spectral_norm(self.lstm2_bias_hh)

            self.linear_weight = spectral_norm(self.linear_weight)
            self.linear_bias = spectral_norm(self.linear_bias)

        self.optimizer = torch.optim.RMSprop([
                {'params': [p for n, p in self.named_parameters()]}
            ], lr=lr, weight_decay=wd
        )
        self.loss_func = torch.nn.MSELoss()

    def forward(self, idx):
        embedded_inputs = self.embeddings(idx)
        features = self.mlp(embedded_inputs)
        
        target_weights = OrderedDict({
            "context_layer_1.weight_ih_l0": self.con1_weight_ih(features).view(4*self.hidden_dim, 4),
            "context_layer_1.weight_hh_l0": self.con1_weight_hh(features).view(4*self.hidden_dim, self.hidden_dim),
            "context_layer_1.bias_ih_l0": self.con1_bias_ih(features).view(-1),
            "context_layer_1.bias_hh_l0": self.con1_bias_hh(features).view(-1),
            "context_layer_2.weight_ih_l0": self.con2_weight_ih(features).view(4*self.hidden_dim, 1),
            "context_layer_2.weight_hh_l0": self.con2_weight_hh(features).view(4*self.hidden_dim, self.hidden_dim),
            "context_layer_2.bias_ih_l0": self.con2_bias_ih(features).view(-1),
            "context_layer_2.bias_hh_l0": self.con2_bias_hh(features).view(-1),
            "project.weight": self.project_weight(features).view(self.context_final_dim, 2*self.hidden_dim),
            "project.bias": self.project_bias(features).view(-1),
            "lstm_stacked.weight_ih_l0": self.lstm1_weight_ih(features).view(4*self.hidden_dim, self.context_final_dim+3),
            "lstm_stacked.weight_hh_l0": self.lstm1_weight_hh(features).view(4*self.hidden_dim, self.hidden_dim),
            "lstm_stacked.bias_ih_l0": self.lstm1_bias_ih(features).view(-1),
            "lstm_stacked.bias_hh_l0": self.lstm1_bias_hh(features).view(-1),
            "lstm_stacked.weight_ih_l1": self.lstm2_weight_ih(features).view(4*self.hidden_dim, self.hidden_dim),
            "lstm_stacked.weight_hh_l1": self.lstm2_weight_hh(features).view(4*self.hidden_dim, self.hidden_dim),
            "lstm_stacked.bias_ih_l1": self.lstm2_bias_ih(features).view(-1),
            "lstm_stacked.bias_hh_l1": self.lstm2_bias_hh(features).view(-1),
            "linear.weight": self.linear_weight(features).view(1, self.hidden_dim),
            "linear.bias": self.linear_bias(features).view(-1),
        })
        return target_weights


if __name__ == "__main__":
    HyperNet = LSTMHyper(
        input_dim=3,
        output_dim=1,
        hidden_dim = 64,
        embed_dim=5,
        context_final_dim=32,
        num_nodes=10,
        n_hidden=3,
        spec_norm=True)

    for name, p in HyperNet.named_parameters():
        if p.requires_grad_:
            print(name, p.data.shape)