import torch
import math
import einops
from torch import nn
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
from mamba_ssm import Mamba
class MaTrRec1(SequentialRecommender):
    def __init__(self, config, dataset):
        super(MaTrRec1, self).__init__(config, dataset)

        self.hidden_size = config["hidden_size"]
        self.loss_type = config["loss_type"]
        self.n_layers = config["n_layers"]
        self.initializer_range = config["initializer_range"]
        self.d_state = config["d_state"]
        self.d_conv = config["d_conv"]
        self.expand = config["expand"]
        self.dropout_prob = config["dropout_prob"]
        self.n_heads = config["n_heads"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        ) 
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.dropout_prob)
        
        self.maeaa_layers = nn.ModuleList([
            MaEAALayer(
                hidden_size=self.hidden_size,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                dropout_prob=self.dropout_prob,
                attn_dropout_prob=self.attn_dropout_prob,
                hidden_dropout_prob=self.hidden_dropout_prob,
                n_layers=self.n_layers,
                hidden_act=self.hidden_act,
            ) for _ in range(self.n_layers)
        ])
        
        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len):
        item_emb = self.item_embedding(item_seq)
        item_emb = self.LayerNorm(item_emb)
        item_emb = self.dropout(item_emb)
        for i in range(self.n_layers):
            item_emb = self.maeaa_layers[i](item_emb)
        seq_output = self.gather_indexes(item_emb, item_seq_len - 1)
        return seq_output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(
            seq_output, test_items_emb.transpose(0, 1)
        )  # [B, n_items]
        return scores
class MaEAALayer(nn.Module):
    def __init__(self, hidden_size, d_state, d_conv, expand, dropout_prob, 
                 n_layers, hidden_dropout_prob, attn_dropout_prob, hidden_act):
        super().__init__()
        self.num_layers = n_layers
        self.mamba = MambaLayer(d_model=hidden_size, d_state=d_state, d_conv=d_conv,
                                expand=expand, dropout=dropout_prob)
        self.att = EAAttention(hidden_size=hidden_size,attn_dropout_prob=attn_dropout_prob)
        self.feed_forward = FeedForward(hidden_size=hidden_size,inner_size=4*hidden_size,
                                        hidden_dropout_prob=hidden_dropout_prob,
                                        hidden_act=hidden_act,layer_norm_eps=1e-12,)
    def forward(self, input_tensor):
        hidden_states = self.mamba(input_tensor)
        hidden_states = self.att(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        return hidden_states
    
    
class MambaLayer(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, dropout):
        super().__init__()
        self.mamba = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
    def forward(self, input_tensor):
        hidden_states = self.mamba(input_tensor)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    

        
class EAAttention(nn.Module):
    def __init__(self, hidden_size,attn_dropout_prob):
        super().__init__()
        self.to_query = nn.Linear(hidden_size, hidden_size)
        self.to_key = nn.Linear(hidden_size,  hidden_size)
        self.w_g = nn.Parameter(torch.randn(hidden_size, 1))
        self.scale_factor = hidden_size ** -0.5
        self.elu = nn.GELU()
        self.Proj = nn.Linear(hidden_size, hidden_size)
        self.LNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.fl = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(attn_dropout_prob)
    def forward(self, input_tensor):
        query = self.to_query(input_tensor)
        key = self.to_key(input_tensor)
        query = torch.nn.functional.normalize(query, dim=-1) #BxNxD
        key = torch.nn.functional.normalize(key, dim=-1) #BxNxD
        query = self.elu(query)
        query_weight = query @ self.w_g # BxNx1 (BxNxD @ Dx1)
        A = query_weight * self.scale_factor # BxNx1
        A = torch.nn.functional.normalize(A, dim=1) # BxNx1
        G = torch.sum(A * query, dim=1) # BxD
        G = einops.repeat(
            G, "b d -> b repeat d", repeat=key.shape[1]
        ) # BxNxD
        hidden_states = self.Proj(G* key)
        hidden_states = self.fl(hidden_states * self.LNorm(input_tensor))
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class FeedForward(nn.Module):

    def __init__(
        self, hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps
    ):
        super(FeedForward, self).__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.intermediate_act_fn = self.get_hidden_act(hidden_act)

        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def get_hidden_act(self, act):
        ACT2FN = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "swish": nn.SiLU(),
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        return ACT2FN[act]
    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states