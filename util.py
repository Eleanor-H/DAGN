'''
Adapted from https://github.com/llamazing/numnet_plus
'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tools import allennlp as util
from igraph import Graph


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)


class ResidualGRU(nn.Module):
    def __init__(self, hidden_size, dropout=0.1, num_layers=2):
        super(ResidualGRU, self).__init__()
        self.enc_layer = nn.GRU(input_size=hidden_size, hidden_size=hidden_size // 2, num_layers=num_layers,
                                batch_first=True, dropout=dropout, bidirectional=True)
        self.enc_ln = nn.LayerNorm(hidden_size)

    def forward(self, input):
        output, _ = self.enc_layer(input)
        return self.enc_ln(output + input)



class FFNLayer(nn.Module):
    def __init__(self, input_dim, intermediate_dim, output_dim, dropout, layer_norm=True):
        super(FFNLayer, self).__init__()
        self.fc1 = nn.Linear(input_dim, intermediate_dim)
        if layer_norm:
            self.ln = nn.LayerNorm(intermediate_dim)
        else:
            self.ln = None
        self.dropout_func = nn.Dropout(dropout)
        self.fc2 = nn.Linear(intermediate_dim, output_dim)

    def forward(self, input):
        inter = self.fc1(self.dropout_func(input))
        inter_act = gelu(inter)
        if self.ln:
            inter_act = self.ln(inter_act)
        return self.fc2(inter_act)


class ArgumentGCN(nn.Module):

    def __init__(self, node_dim, extra_factor_dim=0, iteration_steps=1):
        super(ArgumentGCN, self).__init__()

        self.node_dim = node_dim
        self.iteration_steps = iteration_steps

        self._node_weight_fc = torch.nn.Linear(node_dim + extra_factor_dim, 1, bias=True)

        self._self_node_fc = torch.nn.Linear(node_dim, node_dim, bias=True)

        self._node_fc_argument = torch.nn.Linear(node_dim, node_dim,
                                                 bias=False)  
        self._node_fc_punctuation = torch.nn.Linear(node_dim, node_dim,
                                                    bias=False)  

    def forward(self,
                node,  
                node_mask,  
                argument_graph,  
                punctuation_graph,  
                extra_factor=None):
        ''' '''
        '''
        Current: 2 relation patterns.
            - argument edge. (most of them are causal relations)
            - punctuation edges. (including periods and commas)
        '''

        node_len = node.size(1)

        diagmat = torch.diagflat(torch.ones(node.size(1), dtype=torch.long,
                                            device=node.device))  
        diagmat = diagmat.unsqueeze(0).expand(node.size(0), -1, -1)  
        dd_graph = node_mask.unsqueeze(1) * node_mask.unsqueeze(-1) * (1 - diagmat)  

        graph_argument = dd_graph * argument_graph
        graph_punctuation = dd_graph * punctuation_graph

        node_neighbor_num = graph_argument.sum(-1) + graph_punctuation.sum(-1)
        node_neighbor_num_mask = (node_neighbor_num >= 1).long()
        node_neighbor_num = util.replace_masked_values(node_neighbor_num.float(), node_neighbor_num_mask, 1)

        all_weight = []
        for step in range(self.iteration_steps):

            ''' (1) Node Relatedness Measure '''
            if extra_factor is None:
                d_node_weight = torch.sigmoid(self._node_weight_fc(node)).squeeze(
                    -1)  
            else:
                d_node_weight = torch.sigmoid(self._node_weight_fc(torch.cat((node, extra_factor), dim=-1))).squeeze(
                    -1)  

            all_weight.append(d_node_weight)  

            self_node_info = self._self_node_fc(node)

            ''' (2) Message Propagation (each relation type) '''
            
            node_info_argument = self._node_fc_argument(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graph_argument,
                0)  
            node_info_argument = torch.matmul(node_weight, node_info_argument)

            
            node_info_punctuation = self._node_fc_punctuation(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graph_punctuation,
                0)  
            node_info_punctuation = torch.matmul(node_weight, node_info_punctuation)

            agg_node_info = (node_info_argument + node_info_punctuation) / node_neighbor_num.unsqueeze(-1)

            ''' (3) Node Representation Update '''
            node = F.relu(self_node_info + agg_node_info)

        all_weight = [weight.unsqueeze(1) for weight in all_weight]
        all_weight = torch.cat(all_weight, dim=1)

        return node, all_weight


class ArgumentGCN_wreverseedges_double(nn.Module):

    def __init__(self, node_dim, extra_factor_dim=0, iteration_steps=1):
        super(ArgumentGCN_wreverseedges_double, self).__init__()

        self.node_dim = node_dim
        self.iteration_steps = iteration_steps

        self._node_weight_fc = torch.nn.Linear(node_dim + extra_factor_dim, 1, bias=True)

        self._self_node_fc = torch.nn.Linear(node_dim, node_dim, bias=True)

        self._node_fc_argument = torch.nn.Linear(node_dim, node_dim,
                                                 bias=False)  
        self._node_fc_punctuation = torch.nn.Linear(node_dim, node_dim,
                                                    bias=False)  
        self._node_fc_argument_prime = torch.nn.Linear(node_dim, node_dim,
                                                       bias=False)  
        self._node_fc_punctuation_prime = torch.nn.Linear(node_dim, node_dim,
                                                          bias=False)  

        self._node_fc_argument_2 = torch.nn.Linear(node_dim, node_dim,
                                                   bias=False)  
        self._node_fc_punctuation_2 = torch.nn.Linear(node_dim, node_dim,
                                                      bias=False)  
        self._node_fc_argument_prime_2 = torch.nn.Linear(node_dim, node_dim,
                                                         bias=False)  
        self._node_fc_punctuation_prime_2 = torch.nn.Linear(node_dim, node_dim,
                                                            bias=False)  

    def forward(self,
                node,  
                node_mask,  
                argument_graph,  
                punctuation_graph,  
                extra_factor=None):
        ''' '''
        '''
        Current: 2 relation patterns & reversed directed edges.
            - argument edge. (most of them are causal relations)
            - punctuation edges. (including periods and commas)

        Date: 20/11/2020
        Change log: 
            * add double node_info.
        '''

        node_len = node.size(1)

        diagmat = torch.diagflat(torch.ones(node.size(1), dtype=torch.long,
                                            device=node.device))  
        diagmat = diagmat.unsqueeze(0).expand(node.size(0), -1, -1)  
        dd_graph = node_mask.unsqueeze(1) * node_mask.unsqueeze(-1) * (1 - diagmat)  

        graph_argument = dd_graph * argument_graph
        graph_punctuation = dd_graph * punctuation_graph

        ''' The reverse directed edges. '''
        graph_argument_re = dd_graph * graph_argument.permute(0, 2, 1)
        graph_punctuation_re = dd_graph * graph_punctuation.permute(0, 2, 1)

        node_neighbor_num = graph_argument.sum(-1) + graph_punctuation.sum(-1) + \
                            graph_argument_re.sum(-1) + graph_punctuation_re.sum(-1)
        node_neighbor_num_mask = (node_neighbor_num >= 1).long()
        node_neighbor_num = util.replace_masked_values(node_neighbor_num.float(), node_neighbor_num_mask, 1)

        all_weight = []
        for step in range(self.iteration_steps):

            ''' (1) Node Relatedness Measure '''
            if extra_factor is None:
                d_node_weight = torch.sigmoid(self._node_weight_fc(node)).squeeze(
                    -1)  
            else:
                d_node_weight = torch.sigmoid(self._node_weight_fc(torch.cat((node, extra_factor), dim=-1))).squeeze(
                    -1)  

            all_weight.append(d_node_weight)  

            self_node_info = self._self_node_fc(node)  

            ''' (2) Message Propagation (each relation type) '''
            
            node_info_argument = self._node_fc_argument(node)
            node_info_argument_2 = self._node_fc_argument_2(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graph_argument,
                0)  
            node_info_argument = torch.matmul(node_weight, node_info_argument)
            node_info_argument_2 = torch.matmul(node_weight, node_info_argument_2)

            
            node_info_argument_prime = self._node_fc_argument_prime(node)
            node_info_argument_prime_2 = self._node_fc_argument_prime_2(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graph_argument_re,
                0)  
            node_info_argument_prime = torch.matmul(node_weight, node_info_argument_prime)
            node_info_argument_prime_2 = torch.matmul(node_weight, node_info_argument_prime_2)

            
            node_info_punctuation = self._node_fc_punctuation(node)
            node_info_punctuation_2 = self._node_fc_punctuation_2(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graph_punctuation,
                0)  
            node_info_punctuation = torch.matmul(node_weight, node_info_punctuation)
            node_info_punctuation_2 = torch.matmul(node_weight, node_info_punctuation_2)

            
            node_info_punctuation_prime = self._node_fc_punctuation_prime(node)
            node_info_punctuation_prime_2 = self._node_fc_punctuation_prime_2(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graph_punctuation_re,
                0)  
            node_info_punctuation_prime = torch.matmul(node_weight, node_info_punctuation_prime)
            node_info_punctuation_prime_2 = torch.matmul(node_weight, node_info_punctuation_prime_2)

            agg_node_info = (node_info_argument + node_info_punctuation +
                             node_info_argument_prime + node_info_punctuation_prime +
                             node_info_argument_2 + node_info_punctuation_2 +
                             node_info_argument_prime_2 + node_info_punctuation_prime_2
                             ) / node_neighbor_num.unsqueeze(-1)

            ''' (3) Node Representation Update '''
            node = F.relu(self_node_info + agg_node_info)

        all_weight = [weight.unsqueeze(1) for weight in all_weight]
        all_weight = torch.cat(all_weight, dim=1)

        return node, all_weight


class ArgumentGCN_fromGTN_2_3(nn.Module):
    ''' mostly 5-hops and 5-channels. '''

    def __init__(self, node_dim, extra_factor_dim=0, iteration_steps=1):
        super(ArgumentGCN_fromGTN_2_3, self).__init__()

        self.node_dim = node_dim
        self.iteration_steps = iteration_steps

        self._node_weight_fc = torch.nn.Linear(node_dim + extra_factor_dim, 1, bias=True)

        self._self_node_fc = torch.nn.Linear(node_dim, node_dim, bias=True)

        self._node_fc_argument = torch.nn.Linear(node_dim, node_dim,
                                                 bias=False)  # relation-specific transform matrices.
        self._node_fc_punctuation = torch.nn.Linear(node_dim, node_dim,
                                                    bias=False)  # relation-specific transform matrices.
        self._node_fc_domain = torch.nn.Linear(node_dim, node_dim,
                                               bias=False)  # relation-specific transform matrices.

        self.node_fc_meta = torch.nn.ModuleList(
                                [torch.nn.ModuleList([torch.nn.Linear(node_dim, node_dim, bias=False) for c in range(5)])
                                    for h in range(5)])

    def forward(self,
                node, 
                graph_argument,
                graph_punctuation,
                graph_domain,
                graph_meta_paths,  # list[list[tensor]]
                node_neighbor_num,
                len_meta_paths,
                extra_factor=None,
                ):

        node_len = node.size(1)
        n_hops = len(graph_meta_paths)
        n_chans = len(graph_meta_paths[0])

        all_weight = []
        meta_path_node_weights = ()
        for step in range(self.iteration_steps):

            ''' (1) Node Relatedness Measure '''
            if extra_factor is None:
                d_node_weight = torch.sigmoid(self._node_weight_fc(node)).squeeze(
                    -1)  # eq (9). alpha_i. (bsz x n_choices, n_node)
            else:
                d_node_weight = torch.sigmoid(self._node_weight_fc(torch.cat((node, extra_factor), dim=-1))).squeeze(
                    -1)  # alpha_i.

            all_weight.append(d_node_weight)  # Here record the weights.

            self_node_info = self._self_node_fc(node)  # v_i.

            ''' (2) Message Propagation (each relation type) '''
            # type 1. argument edges.
            node_info_argument = self._node_fc_argument(node)  # (bsz x n_choices, n_nodes, embed_size)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),  # (bsz x n_choices, n_node, n_node)
                graph_argument,
                0)  # filtering weights not type 1
            node_info_argument = torch.matmul(node_weight, node_info_argument)

            # type 2. punctuation edges.
            node_info_punctuation = self._node_fc_punctuation(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graph_punctuation,
                0)  # filtering weights not type 2
            node_info_punctuation = torch.matmul(node_weight, node_info_punctuation)

            # type3. domain graph
            node_info_domain = self._node_fc_domain(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graph_domain,
                0)
            node_info_domain = torch.matmul(node_weight, node_info_domain)

            if graph_meta_paths is not None:
                # type meta-paths.
                feats_names = locals()
                step_meta_path_node_weights = ()
                for h in range(n_hops):
                    h_metapath_node_weights = ()
                    for c in range(n_chans):
                        feats_names["node_info_h{}_c{}".format(h, c)] = self.node_fc_meta[h][c](node)
                        node_weight = util.replace_masked_values(
                            d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                            graph_meta_paths[h][c],
                            0)
                        h_metapath_node_weights += (node_weight, )
                        feats_names["node_info_h{}_c{}".format(h, c)] = torch.matmul(node_weight,
                                                                    feats_names["node_info_h{}_c{}".format(h, c)])
                    step_meta_path_node_weights += (h_metapath_node_weights, )
                meta_path_node_weights += (step_meta_path_node_weights, )

            agg_node_info = node_info_argument + node_info_punctuation + node_info_domain
            if graph_meta_paths is not None:
                for h in range(n_hops):
                    for c in range(n_chans):
                        agg_node_info += feats_names["node_info_h{}_c{}".format(h, c)]
            agg_node_info /= node_neighbor_num.unsqueeze(-1)

            ''' (3) Node Representation Update '''
            node = F.relu(self_node_info + agg_node_info)

        all_weight = [weight.unsqueeze(1) for weight in all_weight]
        all_weight = torch.cat(all_weight, dim=1)

        return node, all_weight, meta_path_node_weights


class ArgumentGTN_normedH_multiple(nn.Module):
    ''' The output Hs are normed and turned into masks with a threshold. '''

    def __init__(self,
                 num_edges,
                 num_channels, node_dim, num_layers,
                 mp_thre=0.25
                 ):
        super(ArgumentGTN_normedH_multiple, self).__init__()

        self.num_channels = num_channels
        self.node_dim = node_dim

        self.num_layers = num_layers
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(GTLayer(num_edges+1, num_channels, first=True))  # +1 for identity matrix.
            else:
                layers.append(GTLayer(num_edges+1, num_channels, first=False))  # +1 for identity matrix.
        self.layers = nn.ModuleList(layers)
        self.linear1 = nn.Linear(self.node_dim * self.num_channels, self.node_dim)

        # meta-path threshold
        self.thre = mp_thre

    def gcn_conv(self, H, node_embeds, node_mask=None):
        '''

        :param H: (bsz, n_nodes, n_nodes). single channel.
        :param node_embeds: (bsz, n_nodes, embed_size)
        :param node_mask: (bsz, n_nodes)
        :return:
            - output: (bsz, n_nodes, embed_size)


        '''
        H = self.norm_in_batch(H)  # (bsz, n_nodes, n_nodes)
        output = torch.bmm(H.permute(0, 2, 1), node_embeds)  # (bsz, n_nodes, embed_size)
        output = output.float()

        if node_mask is not None:
            output = output * node_mask.unsqueeze(-1)

        return output

    def normalization(self, H):
        for i in range(self.num_channels):
            if i == 0:
                H_ = self.norm(H[i, :, :]).unsqueeze(0)
            else:
                H_ = torch.cat((H_, self.norm(H[i, :, :]).unsqueeze(0)), dim=0)
        return H_

    def normalization_in_batch(self, H):
        '''

        :param H:  (bsz, n_channels, n_nodes, n_nodes)
        :return:
            - H_: (bsz, n_channels, n_nodes, n_nodes)
        '''
        for i in range(self.num_channels):
            if i == 0:
                H_ = self.norm_in_batch(H[:, i, :, :]).unsqueeze(1)  # (bsz, 1, n_nodes, n_nodes)
            else:
                H_ = torch.cat((H_, self.norm_in_batch(H[:, i, :, :]).unsqueeze(1)), dim=1)  # (bsz, n_channels, n_nodes, n_nodes)
        return H_.to(H.device)

    def norm(self, H, add=False):
        H = H.t()
        if add == False:
            H = H * ((torch.eye(H.shape[0]) == 0).type(torch.FloatTensor).to(H.device))
        else:
            H = H * ((torch.eye(H.shape[0]) == 0).type(torch.FloatTensor).to(H.device)) + torch.eye(H.shape[0]).type(
                torch.FloatTensor).to(H.device)
        deg = torch.sum(H, dim=1)
        deg_inv = deg.pow(-1)
        deg_inv[deg_inv == float('inf')] = 0
        deg_inv = deg_inv * torch.eye(H.shape[0]).type(torch.FloatTensor).to(H.device)
        H = torch.mm(deg_inv, H)
        H = H.t()
        H = H.float()
        return H

    def norm_in_batch(self, H):
        '''

        :param H: (bsz, n_nodes, n_nodes)
        :return:
            - H_: (bsz, n_nodes, n_nodes)
        '''
        bsz = H.size(0)
        for i in range(bsz):
            if i == 0:
                H_ = self.norm(H[i, :, :]).unsqueeze(0)
            else:
                H_ = torch.cat((H_, self.norm(H[i, :, :]).unsqueeze(0)), dim=0)
        return H_

    def forward(self, node, node_mask, A):
        '''

        :param node:  torch.FloatTensor (bsz, n_nodes, embed_size)
        :param node_mask:  torch.LongTensor (bsz, n_nodes)
        :param adj_matrices:  list. [graph_argument, graph_punctuation, I]. Each: torch.FloatTensor (bsz, n_nodes, n_nodes) (dep)
        :param A: tensor. (bsz, 1, n_edges, n_nodes, nodes)
        :return:
        '''

        ''' eq (3) softly selects adjacency matrices (edge types)'''
        Ws = []
        Hs = []
        H_normed = []
        H_masks = []
        for i in range(self.num_layers):
            if i == 0:
                H, W = self.layers[i](A)  # H: (bsz, n_channels, n_nodes, n_nodes). The meta-path.
            else:
                H = self.normalization_in_batch(H)
                H, W = self.layers[i](A, H)  # W: (n_channels, n_edges)
            Ws.append(W[0]) 
            Hs.append(H)
            H_normed.append(self.normalization_in_batch(H))
            H_masks.append(torch.gt(self.normalization_in_batch(H), self.thre).long())

        ''' eq (5) GTNs '''
        for i in range(self.num_channels):
            if i == 0:
                node_ = F.relu(self.gcn_conv(H[:, i, :, :], node, node_mask=node_mask))  # (bsz, n_nodes, embed_size)
            else:
                node_tmp = F.relu(self.gcn_conv(H[:, i, :, :], node, node_mask=node_mask))  # (bsz, n_nodes, embed_size)
                node_ = torch.cat((node_, node_tmp), dim=2)  # (bsz, n_nodes, embed_size x n_channels)

        updated_node = self.linear1(node_)  # (bsz, n_nodes, embed_size)
        updated_node = updated_node.float()

        return updated_node, (Hs, H_normed, H_masks), Ws


class GTLayer(nn.Module):

    def __init__(self, in_channels, out_channels, first=True):
        super(GTLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first = first
        if self.first == True:
            self.conv1 = GTConv(in_channels, out_channels)
            self.conv2 = GTConv(in_channels, out_channels)
        else:
            self.conv1 = GTConv(in_channels, out_channels)

    def forward(self, A, H_=None):
        '''

        :param A:  (bsz, 1, n_edges=3, n_nodes, n_nodes)
        :param H_:  (bsz, n_channels, n_nodes, n_nodes)
        :return:
        '''
        if self.first == True:
            a = self.conv1(A)  # (bsz, n_channels, n_nodes, n_nodes)
            b = self.conv2(A)  # (bsz, n_channels, n_nodes, n_nodes)
            H = torch.matmul(a, b)
            H = H.float()
            W = [(F.softmax(self.conv1.weight, dim=1)).detach(), (F.softmax(self.conv2.weight, dim=1)).detach()]
        else:
            a = self.conv1(A)  # (bsz, n_channels, n_nodes, n_nodes)
            H_ = H_.to(A.device)
            H = torch.matmul(H_, a)
            H = H.float()
            H = H.to(A.device)
            W = [(F.softmax(self.conv1.weight, dim=1)).detach()]  # (num_channels, num_edges)
        return H, W


class GTConv(nn.Module):

    def __init__(self, in_channels, out_channels):  # in: num_edges, out: num_channels
        super(GTConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, 1, 1))  # 1 x 1 convolution.
        self.bias = None
        self.scale = nn.Parameter(torch.Tensor([0.1]), requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        self.weight.data.normal_(mean=0.0, std=0.1)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, A):
        '''

        :param A: (bsz, 1, n_edges=3, n_nodes, n_nodes)

        (bsz, 1, n_edges=3, n_nodes, n_nodes) * (bsz, n_channels, n_edges, 1, 1)
        -> (bsz, n_channels, n_edges, n_nodes, n_nodes)
        -> (bsz, n_channels, n_nodes, n_nodes)

        :return
            - A: (bsz, n_channels, n_nodes, n_nodes)
        '''

        tmp = self.weight.unsqueeze(0).expand(A.size(0), -1, -1, -1, -1)  # (bsz, n_channel, n_edges, 1, 1)
        A = torch.sum(A * F.softmax(tmp, dim=2), dim=2)
        return A



def compute_segments(split, verbose=False):
    """ compute adjacency matrix based on computed graph of segments.
    1. compute segments (consecutive series of zeros);
    2. traverse first half of segments and connect segments containing same domain id.

    Parameters
    ----------
    split : list
        split words of the sentence.
    verbose : bool, optional
        print intermediate result, by default False

    Returns
    -------
    tuple
        The first element is segments as 2D list.

        eg:
        - segments[0] = [1, 2, 3]
        - segments[1] = [4, 5]
        - segments[2] = [6, 7, 8, 9, 10]
        - segments[3] = [11]

        The second element is split_index as 1D list.

        eg: split_index = [1, 4, 6, 8, 9]
    """

    split_index = []
    segments = []
    cur_seg = -1
    in_seg = False
    for i, n in enumerate(split):
        # [segment start]: meet 0 for the first time, then start a new segment
        if n == 0 and not in_seg:
            cur_seg += 1
            segments.append([])
            segments[cur_seg] = [i]
            in_seg = True
        # [through segment]: meet 0 after 0, then remains the same segment
        elif n == 0 and in_seg:
            segments[cur_seg].append(i)
        # [segment end]: meet > 0 after 0, then the segment ends
        elif n > 0 and in_seg:
            split_index.append(i)
            in_seg = False
        # [split word again]: meet > 0 after > 0
        elif n > 0 and not in_seg:
            split_index.append(i)
        # [sentence end]: meet < 0
        elif n < 0:
            break
    return segments, split_index


def get_segment_id(index, segments):
    """get segment id of given word index

    Parameters
    ----------
    index : int
        index of the word in the sentence
    segments : list
        2D list of segments

    Returns
    -------
    int
        segment id. If -1, means not found.
    """
    cur_seg_id = -1
    for seg_id in range(len(segments)):
        if index in segments[seg_id]:
            cur_seg_id = seg_id
    return cur_seg_id


def compute_domain_adjacency_matrix(domain, split, verbose=False):
    """ compute adjacency matrix based on computed graph of segments.
    1. compute segments (consecutive series of zeros);
    2. traverse first half of segments and connect segments containing same domain id.

    Parameters
    ----------
    domain : list
        domain ids of the sentence.
    split : list
        split words of the sentence.
    verbose : bool, optional
        print intermediate result, by default False

    Returns
    -------
    list
        2D adjacency matrix.

        eg: the value means the count of connection

        [[0, 1, 2]
        ,[1, 0, 0]
        ,[2, 0, 0]]
    """

    segments, split_index = compute_segments(split, verbose)
    seg_amount = len(segments)  # matrix size
    if verbose: print(f'found {seg_amount} segments.')
    if verbose: print(segments)
    if verbose: print(split_index)

    # initialize graph
    g = Graph()
    g.add_vertices(seg_amount)

    for d_index, domain_id in enumerate(domain):
        # skip split word and non-domain word
        if d_index in split_index or domain_id == 0: continue

        # terminate when meet sentence end
        if domain_id < 0: break

        cur_seg_id = get_segment_id(d_index, segments)
        assert (cur_seg_id != -1)
        if verbose: print(f'# current domain id is {domain_id}, in segment {cur_seg_id}')

        if domain_id > 0:
            # search if following segments have the same domain id
            for next_seg_id in range(cur_seg_id + 1, seg_amount):
                if verbose: print(
                    f'-- looking for domain_id ({domain_id}) in segment {next_seg_id}: {segments[next_seg_id]}')
                for index_in_seg in segments[next_seg_id]:
                    # find the same id in the segment
                    if domain[index_in_seg] == domain_id:
                        if verbose: print(f'---- found {domain_id} in segment {next_seg_id} at {index_in_seg}')
                        g.add_edges([[cur_seg_id, next_seg_id]])
                        break

    return list(g.get_adjacency())







