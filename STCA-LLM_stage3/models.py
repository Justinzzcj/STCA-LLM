import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from attention import Attention, selfAttention
from einops import rearrange
from gat import GraphAttentionLayer
import numpy as np


device = "cuda:0" if torch.cuda.is_available() else "cpu"

class GPT4TS(nn.Module):
    def __init__(self, device="cuda:0", gpt_layers=6):
        super(GPT4TS, self).__init__()
        self.gpt2 = GPT2Model.from_pretrained(
            r"D:\pythonproject\STCA-LLM\gptweights\gpt2_weights_stage2", output_attentions=True, output_hidden_states=True
        )
        self.gpt2.h = self.gpt2.h[:gpt_layers]
        self.U = 2

        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
                if "mlp" in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

    def forward(self, x):
        return self.gpt2(inputs_embeds=x).last_hidden_state


class stcllm(nn.Module):
    def __init__(self, configs, device, adj):
        super(stcllm, self).__init__()
        self.device = device
        self.input_len = configs.seq_len
        self.input_dim = configs.input_size
        self.output_len = configs.output_size
        self.d_model = configs.d_model
        self.adj = adj
        self.dropout = configs.dropout
        self.num_channels = [16, 32, 20]
        self.adj = torch.Tensor(self.adj).to(self.device)
        self.gpt = GPT4TS(device=self.device, gpt_layers=6)
        self.gat = GraphAttentionLayer(
            in_features = self.d_model,
            out_features = self.d_model,
            n_heads = 32,
            is_concat = True,
            dropout = self.dropout,
        )
        self.attention = Attention(dim=self.d_model, dropout=self.dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.fc = nn.Linear(self.d_model, self.output_len)
        self.proj2 = nn.Linear(self.input_len, self.d_model)
        self.conv = nn.Conv1d(self.input_len, self.d_model, kernel_size=1)
        self.layer_norm1 = nn.LayerNorm(self.d_model)
        self.layer_norm2 = nn.LayerNorm(self.d_model)
        self.layer_norm3 = nn.LayerNorm(self.d_model)
        self.layer_norm4 = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(self.dropout)


    def forward(self, x):
        means = x.mean(dim=1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x = x / stdev
        x = x.permute(0, 2, 1)
       
        x_time = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        x_time = self.layer_norm1(x_time)
        
        
        x_space = self.proj2(x)
        x_space = self.gat(x_space, self.adj.unsqueeze(-1))
        x_space = self.layer_norm2(x_space)
        

        x_s_to_t = self.attention(x_time, x_space)
        x_t_to_s = self.attention(x_space, x_time)
        x_s_to_t = self.layer_norm3(x_s_to_t + x_space)
        x_t_to_s = self.layer_norm4(x_t_to_s + x_time)
        output = self.layer_norm(x_s_to_t+ x_t_to_s)   
        output = self.gpt(output)
        output = self.fc(output)
        output = rearrange(output, "b m l -> b l m")
        output = output * stdev + means
        output = output.permute(0, 2, 1)
        return output
