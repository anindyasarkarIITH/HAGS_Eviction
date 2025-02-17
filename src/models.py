import torch
from torch import nn
import torch.nn.functional as F

from modules.transformer import TransformerEncoder


class MULTModel(nn.Module):
    def __init__(self, hyp_params):
        """
        Construct a MulT model.
        """
        super(MULTModel, self).__init__()
        self.orig_d_l, self.orig_d_a, self.orig_d_v = hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v
        self.d_l, self.d_a, self.d_v = 30, 30, 30
        self.num_parcel = 100
        self.vonly = hyp_params.vonly
        self.aonly = hyp_params.aonly
        self.lonly = hyp_params.lonly
        self.num_heads = hyp_params.num_heads
        self.layers = hyp_params.layers
        self.attn_dropout = hyp_params.attn_dropout
        self.attn_dropout_a = hyp_params.attn_dropout_a
        self.attn_dropout_v = hyp_params.attn_dropout_v
        self.relu_dropout = hyp_params.relu_dropout
        self.res_dropout = hyp_params.res_dropout
        self.out_dropout = hyp_params.out_dropout
        self.embed_dropout = hyp_params.embed_dropout
        self.attn_mask = hyp_params.attn_mask

        #combined_dim = self.d_l + self.d_a + self.d_v
        combined_dim = self.d_a + self.d_v
        
        output_dim =  hyp_params.output_dim        # This is actually not a hyperparameter :-)

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)

        # 2. Crossmodal Attentions
        
        if self.lonly:
            self.trans_ab_mem = self.get_network(self_type='l_mem', layers=3)
        if self.aonly:
            self.trans_a_with_l = self.get_network(self_type='al')
            self.trans_a_with_v = self.get_network(self_type='av')
        if self.vonly:
            self.trans_v_with_l = self.get_network(self_type='vl')
            self.trans_v_with_a = self.get_network(self_type='va')
        
        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)
       
        # Projection layers
        self.proj_search_info = nn.Linear(self.num_parcel, combined_dim)
        self.proj_search2 = nn.Linear(combined_dim, combined_dim)
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['f']:
            embed_dim, attn_dropout = 30, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            #embed_dim, attn_dropout = 2*self.d_l, self.attn_dropout
            embed_dim, attn_dropout = 2*30, self.attn_dropout
        elif self_type == 'a_mem':
            #embed_dim, attn_dropout = 2*self.d_a, self.attn_dropout
            embed_dim, attn_dropout = 30, self.attn_dropout
        elif self_type == 'v_mem':
            #embed_dim, attn_dropout = 2*self.d_v, self.attn_dropout
            embed_dim, attn_dropout = self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")
        
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)
            
    def forward(self, x_a, x_v, search_info):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        for i in range(1):
            #x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
            x_a = x_a.transpose(1, 2)
            x_v = x_v.transpose(1, 2)
       
            # Project the textual/visual/audio features
            #proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
            proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
            proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
            proj_x_a = proj_x_a.permute(2, 0, 1)
            proj_x_v = proj_x_v.permute(2, 0, 1)
            #proj_x_l = proj_x_l.permute(2, 0, 1)
            
            if self.aonly:
                # (V) --> A
                h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
                h_as = h_a_with_vs #torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
                last_h_a = h_as
              
            if self.vonly:
                # (A) --> V
                #h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
                h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
                h_vs = h_v_with_as #torch.cat([h_v_with_ls, h_v_with_as], dim=2)
                last_h_v = h_vs
                
            last_hs = torch.cat([last_h_a, last_h_v], dim=2)
            if self.lonly:
                h_hidd = self.trans_ab_mem(last_hs)
                if type(h_hidd) == tuple:
                    h_hidd = h_hidd[0]
                l_final = h_hidd[-1]
        
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(l_final)), p=self.out_dropout, training=self.training))
        last_hs_proj += l_final
        
        search_embed = self.proj_search2(F.relu(self.proj_search_info(search_info.clone())))
        search_aware_rep = torch.add(last_hs_proj, search_embed) #torch.add(last_hs_proj,search_embed)
        
        output = self.out_layer(search_aware_rep)
        #output = self.out_layer(last_hs_proj)
        return output, last_hs
    

class Model_search(torch.nn.Module):
    def __init__(self):
        super(Model_search, self).__init__()        
        self.pointwise_search_info = torch.nn.Conv2d(99, 1, 1, 1)
        
        # final MLP layer to transform combine representation to action space for searching
        self.linear_relu_stack_rl = torch.nn.Sequential(
            torch.nn.Linear(201, 150),    
            torch.nn.ReLU(),
            torch.nn.Linear(150, 100),   
        )

    def forward(self, search_info, query_left, grid_prob):
        
        combined_rep = torch.cat((grid_prob, search_info, query_left.unsqueeze(1)), 1)
        
        ## final fc layers
        logits_rl = self.linear_relu_stack_rl(combined_rep)
        return logits_rl  
    
