import torch
import math
from torch import nn
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
from rotary_embedding_torch import RotaryEmbedding


class CLARoRec1(SequentialRecommender):
    def __init__(self, config, dataset):
        super(CLARoRec1, self).__init__(config, dataset)

        self.hidden_size = config["hidden_size"]
        self.loss_type = config["loss_type"]
        self.n_layers = config["n_layers"]
        self.initializer_range = config["initializer_range"]
        self.dropout_prob = config["dropout_prob"]
        self.n_heads = config["n_heads"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.seq_len = self.max_seq_length
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        ) 
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.csaro_layers = CSARoLayer(
                hidden_size=self.hidden_size,
                dropout_prob=self.dropout_prob,
                attn_dropout_prob=self.attn_dropout_prob,
                hidden_dropout_prob=self.hidden_dropout_prob,
                n_layers=self.n_layers,
                n_heads=self.n_heads,
                hidden_act=self.hidden_act,
                seq_len=self.seq_len
            )
        
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
        item_emb = self.csaro_layers(item_emb)
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
class CSARoLayer(nn.Module):
    def __init__(self, hidden_size, dropout_prob, n_layers, 
                 hidden_dropout_prob, attn_dropout_prob, n_heads, hidden_act,seq_len):
        super().__init__()
        self.n_layers = n_layers
        self.cnnlayer = Cnnlayer(hidden_size=hidden_size, dropout_prob=dropout_prob)
        self.att_layers = nn.ModuleList([
            MultiHeadAttention(
            n_heads=n_heads, hidden_size=hidden_size, hidden_dropout_prob=hidden_dropout_prob, 
                attn_dropout_prob=attn_dropout_prob, layer_norm_eps=1e-12)
            for _ in range(n_layers)
        ])
        self.feed_forward = FeedForward(hidden_size=hidden_size,inner_size=4*hidden_size,
                                        hidden_dropout_prob=hidden_dropout_prob,
                                        hidden_act=hidden_act, layer_norm_eps=1e-12,)
    def forward(self, input_tensor):
        hidden_states = self.cnnlayer(input_tensor)
        for att_layer in self.att_layers:
            hidden_states = att_layer(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        return hidden_states
        

    
class Cnnlayer(nn.Module):
    def __init__(self, hidden_size, dropout_prob): 
        super(Cnnlayer, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=hidden_size, out_channels=4*hidden_size, 
                                kernel_size=5, stride=1, padding=2, groups=hidden_size)
        self.gelu = nn.GELU()
        self.conv1d2 = nn.Conv1d(in_channels=4*hidden_size, out_channels=hidden_size, 
                                 kernel_size=5, stride=1, padding=2, groups=hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        
    def forward(self, input_tensor):
        hidden_states = input_tensor.permute(0, 2, 1)  # B x D x N
        hidden_states = self.conv1d(hidden_states)  # B x 4D x N
        hidden_states = self.gelu(hidden_states)
        hidden_states = self.conv1d2(hidden_states)  # B x D x N
        hidden_states = hidden_states.permute(0, 2, 1)  # B x N x D
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_heads,
        hidden_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        layer_norm_eps,
    ):
        super(MultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.rope = RotaryEmbedding(dim=self.all_head_size)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x



    def forward(self, input_tensor):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)
        
        mixed_query_layer = self.rope.rotate_queries_or_keys(mixed_query_layer)
        mixed_key_layer = self.rope.rotate_queries_or_keys(mixed_key_layer)
        
        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)
        value_layer = self.transpose_for_scores(mixed_value_layer).permute(0, 2, 1, 3)

        attention_scores = torch.matmul(query_layer, key_layer)

        attention_scores = attention_scores / self.sqrt_attention_head_size

        attention_probs = self.softmax(attention_scores)

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
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