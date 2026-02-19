# -*- coding: utf-8 -*-

'''
argument numnet Version 17.2

Adapted from: https://github.com/llamazing/numnet_plus
Date: 16/1/2021
Author: Yinya Huang

Change log:
    1. GNN -> jointly using (1) GNN, (2) GTN, (3) Transformer
    2.

Change log of V14.1
    1. GCN + GTN

Change log of V14.4
    1. GCN + GTN + highway

Change log of V15.0
    * The framework
        1. EDU embeddings -> BertEncoder -> updated EDU embeddings
        2. BertEncoder -> self-attention weights (n_layers, n_heads). size = (bsz, n_nodes, n_nodes)
        3. (argument_graph, punctuation_graph) -> GTN -> Hs (meta-paths. 1-hop, 2-hop, 3-hop)
        4. Hs -> masks (strategy 1, 2, 3) -> self-attention weights
            * strategy 1: copy top-layer weights
            * strategy 2: different layers
            * strategy 3: different heads

    * GTN -> meta-graphs
    * BertEncoder -> self-attention weights
    * GCN -> updating the nodes.

Change log of V15.1
    * remove Highway Network from V15.0

Change log of V15.2
    * use GCN node_weights, instead of Transformer self-attention weights.
        * which means GTN -> meta-paths -> (mask) -> GCN node_weights

Change log of V15.2.1
    * meta-paths 在 GCN 中用不同的参数

Change log of V15.2.2
    * 只用 GCN output node embeddings

Change log of V15.2.2.1
    * GTN -> (norm + threshold) -> GCN

Change log of V17.0:
    1. Multiple ways of masking the adjacency matrices.
    2. Feeding multiple groups of adjacency matrices into GTN and obtaining groups of meta-paths.
    3. Training time: using all groups of meta-paths.
    4. Test time: only using meta-paths(1) for prediction.

Change log of V17.2:
    * GCNs do not share weights.

Change log of V17.9:
    * evolve from V17.2
    * update the meta_paths in _adjmatrices_gtn_gcn_nodefeats()

28/10/2022
Change log of V17.2.9.5: (from V17.2.9.1)
    * domain graph
    * global graph representation fuse

'''


'''
TODOs:

[ ] priority definition
    1. relations 之间
    2. relation & punctuation 之间：relation > punctuation
    3. punctuation 之间：{"<s>", "</s>"} > {",", ".", ";", ":"}

[-] 相邻 relations / punctuations 选择优先级高者的 id 覆盖整个子序列。 => split_into_spans_7() -> split_into_spans_8(). 

[-] sequence[0] & sequence[-1] 的 split 不考虑

[ ] substitute domain_words to D_ids (V3.0)

[-] check get_gcn_info_vector()

'''


import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import numpy as np
import random
from typing import List, Dict, Any, Tuple
from itertools import groupby
from operator import itemgetter
import copy
from util import (FFNLayer, ResidualGRU, 
                  ArgumentGCN_fromGTN_2_3, ArgumentGTN_normedH_multiple, 
                  compute_segments, compute_domain_adjacency_matrix)
from tools import allennlp as util
from transformers import BertPreTrainedModel, RobertaModel




class HighwayLinear(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.0):
        super(HighwayLinear, self).__init__()

        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        self.reset_params()

    def reset_params(self):
        nn.init.kaiming_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        x = self.linear(x)
        return x


class ExtendedDAGNs(BertPreTrainedModel):
    '''
    Adapted from https://github.com/llamazing/numnet_plus.

    Inputs of forward(): see try_data_5.py - the outputs of arg_tokenizer()
        - input_ids: list[int]
        - attention_mask: list[int]
        - segment_ids: list[int]
        - argument_bpe_ids: list[int]. value={ -1: padding,
                                                0: non_arg_non_dom,
                                                1: (relation, head, tail)  关键词在句首
                                                2: (head, relation, tail)  关键词在句中，先因后果
                                                3: (tail, relation, head)  关键词在句中，先果后因
                                                }
        - domain_bpe_ids: list[int]. value={ -1: padding,
                                              0:non_arg_non_dom,
                                           D_id: domain word ids.}
        - punctuation_bpe_ids: list[int]. value={ -1: padding,
                                                   0: non_punctuation,
                                                   1: punctuation}


    '''

    def __init__(self,
                 config,
                 init_weights:bool,
                 max_rel_id,
                 hidden_size: int,
                 dropout_prob: float = 0.1,
                 merge_type: int = 8,
                 use_gcn: bool = False,
                 use_pool: bool = False,
                 gcn_steps: int=1,
                 n_metapath_channels: int=2,
                 n_metapath_layers: int=2
                 ) -> None:
        super().__init__(config)

        self.max_rel_id = max_rel_id
        self.merge_type = merge_type
        self.use_gcn = use_gcn
        self.use_pool = use_pool
        assert self.use_gcn or self.use_pool

        ''' from modeling_roberta '''
        self.roberta = RobertaModel(config)

        if self.use_pool:
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            self.classifier = nn.Linear(config.hidden_size, 1)

        # self._dropout = torch.nn.Dropout(p=dropout_prob)

        ''' global graph representation fuse 
                node attention pooling 
                '''
        self.multihead_attn = nn.MultiheadAttention(config.hidden_size, num_heads=1)

        ''' from numnet '''
        # if self.use_gcn:
        modeling_out_dim = hidden_size
        node_dim = modeling_out_dim

        self._gcn_input_proj = nn.Linear(node_dim * 2, node_dim)
        self._gcn_1 = ArgumentGCN_fromGTN_2_3(node_dim=node_dim, iteration_steps=gcn_steps)
        self._gcn_2 = ArgumentGCN_fromGTN_2_3(node_dim=node_dim, iteration_steps=gcn_steps)
        self._gcn_3 = ArgumentGCN_fromGTN_2_3(node_dim=node_dim, iteration_steps=gcn_steps)
        self._gcn_4 = ArgumentGCN_fromGTN_2_3(node_dim=node_dim, iteration_steps=gcn_steps)
        self._gcn_5 = ArgumentGCN_fromGTN_2_3(node_dim=node_dim, iteration_steps=gcn_steps)
        self._gtn = ArgumentGTN_normedH_multiple(num_channels=n_metapath_channels, num_edges=3,
                                                 node_dim=node_dim, num_layers=n_metapath_layers)  # num_layers: hops.

        self.merge_node_feats = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self._iteration_steps = gcn_steps
        self._gcn_prj_ln = nn.LayerNorm(node_dim)
        self._gcn_enc = ResidualGRU(hidden_size, dropout_prob, 2)
        ''' end of self.use_gcn '''

        # add bert proj
        self._proj_sequence_h = nn.Linear(hidden_size, 1, bias=False)
        self._proj_number = nn.Linear(hidden_size * 2, 1, bias=False)
        self._proj_sequence_g0 = FFNLayer(hidden_size, hidden_size, 1,
                                          dropout_prob)  # input_dim, intermediate_dim, output_dim, dropout.

        # span num extraction
        self._proj_span_num = FFNLayer(3 * hidden_size, hidden_size, 1, dropout_prob)

        self._proj_gcn_pool = FFNLayer(3 * hidden_size, hidden_size, 1, dropout_prob)
        self._proj_gcn_pool_4 = FFNLayer(4 * hidden_size, hidden_size, 1, dropout_prob)
        self._proj_gcn_pool_3 = FFNLayer(2 * hidden_size, hidden_size, 1, dropout_prob)

        if init_weights:
            self.init_weights()

    def highway_network(self, x1, x2):
        """
        Reference: https://github.com/galsang/BiDAF-pytorch/blob/8e16379e33af725daec3ec6c170dfae69e54d51b/model/model.py
        """
        """
        :param x1: (batch, seq_len, embed_size)
        :param x2: (batch, seq_len, embed_size)
        :return: (batch, seq_len, hidden_size * 2)
        """
        # (batch, seq_len, embed_size * 2)
        x = torch.cat([x1, x2], dim=-1)
        for i in range(2):
            h = getattr(self, 'highway_linear{}'.format(i))(x)
            g = getattr(self, 'highway_gate{}'.format(i))(x)
            x = g * h + (1 - g) * x
        # (batch, seq_len, hidden_size * 2)
        return x

    def list_depth(self, items: list) -> int:
        max_depth = 1 if isinstance(items, list) else 0
        if max_depth:
            for item in items:
                if isinstance(item, list):
                    max_depth = max(max_depth, self.list_depth(item) + 1)
        else:
            return max_depth
        return max_depth

    def split_into_spans_9(self, seq, seq_mask, split_bpe_ids):
        '''

            :param seq: (bsz, seq_length, embed_size)
            :param seq_mask: (bsz, seq_length)
            :param split_bpe_ids: (bsz, seq_length). value = {-1, 0, 1, 2, 3, 4}.
            :return:
                - encoded_spans: (bsz, n_nodes, embed_size)
                - span_masks: (bsz, n_nodes)
                - edges: (bsz, n_nodes - 1)
                - node_in_seq_indices: list of list of list(len of span).

        Change log:
            - edges: change from torch.tensor to list.
            - edges: truncate the head and tail edge_ids since they DO NOT contribute to the adjacency graphs.

        '''

        def _consecutive(seq: list, vals: np.array):
            groups_seq = []
            output_vals = copy.deepcopy(vals)
            for k, g in groupby(enumerate(seq), lambda x: x[0] - x[1]):
                groups_seq.append(list(map(itemgetter(1), g)))
            output_seq = []
            for i, ids in enumerate(groups_seq):
                output_seq.append(ids[0])
                if len(ids) > 1:
                    output_vals[ids[0]:ids[-1] + 1] = min(output_vals[ids[0]:ids[-1] + 1])
            return groups_seq, output_seq, output_vals

        embed_size = seq.size(-1)
        device = seq.device
        encoded_spans = []
        span_masks = []
        edges = []
        node_in_seq_indices = []
        for item_seq_mask, item_seq, item_split_ids in zip(seq_mask, seq, split_bpe_ids):
            item_seq_len = item_seq_mask.sum().item()
            item_seq = item_seq[:item_seq_len]
            item_split_ids = item_split_ids[:item_seq_len]
            item_split_ids = item_split_ids.cpu().numpy()
            split_ids_indices = np.where(item_split_ids > 0)[0].tolist()
            grouped_split_ids_indices, split_ids_indices, item_split_ids = _consecutive(
                split_ids_indices, item_split_ids)  # remove the consecutive.
            n_split_ids = len(split_ids_indices)

            item_spans, item_mask = [], []
            item_edges = []
            item_node_in_seq_indices = []
            item_edges.append(item_split_ids[split_ids_indices[0]])
            for i in range(n_split_ids):
                if i == n_split_ids - 1:
                    span = item_seq[split_ids_indices[i] + 1:]
                    if not len(span) == 0:
                        item_spans.append(span.sum(0))
                        item_mask.append(1)

                else:
                    span = item_seq[split_ids_indices[i] + 1:split_ids_indices[i + 1]]
                    if not len(span) == 0:
                        item_spans.append(span.sum(0))
                        item_mask.append(1)
                        item_edges.append(item_split_ids[split_ids_indices[i + 1]])
                        item_node_in_seq_indices.append([i for i in range(grouped_split_ids_indices[i][-1] + 1,
                                                                          grouped_split_ids_indices[i + 1][0])])

            encoded_spans.append(item_spans)
            span_masks.append(item_mask)
            edges.append(item_edges)
            node_in_seq_indices.append(item_node_in_seq_indices)

        max_nodes = max(map(len, span_masks))
        span_masks = [spans + [0] * (max_nodes - len(spans)) for spans in span_masks]
        span_masks = torch.from_numpy(np.array(span_masks))
        span_masks = span_masks.to(device).long()

        pad_embed = torch.zeros(embed_size, dtype=seq.dtype, device=seq.device)
        encoded_spans = [spans + [pad_embed] * (max_nodes - len(spans)) for spans in encoded_spans]
        encoded_spans = [torch.stack(lst, dim=0) for lst in encoded_spans]
        encoded_spans = torch.stack(encoded_spans, dim=0)
        encoded_spans = encoded_spans.to(device).float()

        # Truncate head and tail of each list in edges HERE.
        #     Because the head and tail edge DO NOT contribute to the argument graph and punctuation graph.
        truncated_edges = [item[1:-1] for item in edges]

        return encoded_spans, span_masks, truncated_edges, node_in_seq_indices

    def get_gcn_info_vector(self, indices, node, size, device):
        '''

        :param indices: list(len=bsz) of list(len=n_nodes) of list(len=varied).
        :param node: (bsz, n_nodes, embed_size)
        :param size: value=(bsz, seq_len, embed_size)
        :param device:
        :return:
        '''

        batch_size = size[0]
        # gcn_info_vec = torch.zeros(size=size, dtype=torch.float, device=device)  # (bsz x n_choices, n_nodes, embed_size)
        gcn_info_vec = torch.zeros(size=size, dtype=node.dtype, device=device)  # (bsz x n_choices, n_nodes, embed_size)

        for b in range(batch_size):
            # assert len(indices[b]) == len(node[b]), '{}, {}'.format(len(indices[b]), len(node[b]))  # TODO: check this.
            for ids, emb in zip(indices[b], node[b]):  # list[int], (embed_size)
                gcn_info_vec[b, ids] = emb

        return gcn_info_vec

    def get_adjacency_matrices_2(self, edges:List[List[int]], n_nodes:int, device:torch.device):
        '''
        Convert the edge_value_list into adjacency matrices.
            * argument graph adjacency matrix. Asymmetric (directed graph).
            * punctuation graph adjacency matrix. Symmetric (undirected graph).

            : argument
                - edges:list[list[str]]. len_out=(bsz x n_choices), len_in=n_edges. value={-1, 0, 1, 2, 3, 4, 5}.

            Note: relation patterns
                1 - (relation, head, tail)  关键词在句首
                2 - (head, relation, tail)  关键词在句中，先因后果
                3 - (tail, relation, head)  关键词在句中，先果后因
                4 - (head, relation, tail) & (tail, relation, head)  (1) argument words 中的一些关系
                5 - (head, relation, tail) & (tail, relation, head)  (2) punctuations

        '''

        batch_size = len(edges)
        # max_n_nodes = max([len(item_list) for item_list in edges])
        argument_graph = torch.zeros(
            (batch_size, n_nodes, n_nodes))  # NOTE: the diagonal should be assigned 0 since is acyclic graph.
        punct_graph = torch.zeros(
            (batch_size, n_nodes, n_nodes))  # NOTE: the diagonal should be assigned 0 since is acyclic graph.
        for b, sample_edges in enumerate(edges):
            for i, edge_value in enumerate(sample_edges):
                if edge_value == 1:  # (relation, head, tail)  关键词在句首
                    try:
                        argument_graph[b, i + 1, i + 2] = 1
                    except Exception:
                        pass
                elif edge_value == 2:  # (head, relation, tail)  关键词在句中，先因后果
                    argument_graph[b, i, i + 1] = 1
                elif edge_value == 3:  # (tail, relation, head)  关键词在句中，先果后因
                    argument_graph[b, i + 1, i] = 1
                elif edge_value == 4:  # (head, relation, tail) & (tail, relation, head) ON ARGUMENT GRAPH
                    argument_graph[b, i, i + 1] = 1
                    argument_graph[b, i + 1, i] = 1
                elif edge_value == 5:  # (head, relation, tail) & (tail, relation, head) ON PUNCTUATION GRAPH
                    try:
                        punct_graph[b, i, i + 1] = 1
                        punct_graph[b, i + 1, i] = 1
                    except Exception:
                        pass

        argument_graph = argument_graph.float()
        punct_graph = punct_graph.float()

        return argument_graph.to(device), punct_graph.to(device)

    def _mask_nodes(self, edu_mask, mask_strategy):
        '''

        :param edu_mask: span_mask. (bsz x n_choices, n_nodes)
        :param mask_strategy: first/mid/last/random 1/3 nodes.
        :return:
            - masked: (bsz x n_choices, n_nodes). 1=node, 0=not_node.
        '''

        n_nodes_each_sample = torch.sum(edu_mask, dim=1)  # (bsz x n_choices)
        mid_idx_of_nodes_each_sample = (torch.floor(1/2 * n_nodes_each_sample)).long().cpu().numpy().tolist()
        n_nodes_to_mask_each_sample = torch.floor(1/3 * n_nodes_each_sample).long().cpu().numpy().tolist()

        mask = torch.zeros(edu_mask.size())  # (bsz x n_choices, n_nodes)
        bsz = mask.size(0)
        if mask_strategy == "first":
            for b, n in zip(range(bsz), n_nodes_to_mask_each_sample):
                mask[b, :n] = 1
        elif mask_strategy == "last":
            for b, n in zip(range(bsz), n_nodes_to_mask_each_sample):
                mask[b, :-n] = 1
        elif mask_strategy == "mid":
            for b, n, m in zip(range(bsz), n_nodes_to_mask_each_sample, mid_idx_of_nodes_each_sample):
                if n % 2 == 1:
                    mask[b, m - n//2: m + n//2] = 1
                else:
                    mask[b, m - n//2 - 1: m + n//2] = 1
        elif mask_strategy == "random":
            for b, n, nn in zip(range(bsz), n_nodes_to_mask_each_sample, n_nodes_each_sample):
                ids = random.sample(range(nn), n)
                mask[b, ids] = 1

        return mask.to(edu_mask.device)

    def _get_masked_adj_matrices(self, adj_matrix, node_mask):
        '''

        :param adj_matrix:  (bsz, n_nodes, n_nodes)
        :param node_mask:  (bsz, n_nodes). 1=to_mask
        :return:
            - masked_adj_matrix: (bsz, n_nodes, n_nodes)
        '''
        node_mask = (1-node_mask).unsqueeze(1) * (1-node_mask).unsqueeze(1)

        masked_adj_matrix = adj_matrix * node_mask

        return masked_adj_matrix.long()

    def _adjmatrices_gtn_gcn_nodefeats(self, gtn, gcn,
                                       dd_graph, graph_argument, graph_punctuation,
                                       graph_domain,
                                       node_feats, node_masks, A):

        ''' GTN
        Outputs:
                - gtn_node_feats: (bsz x n_choices, n_nodes, embed_size)
                - updated_adj_graphs: list[tensor]. len(list)=_gcn.n_layers. 
                        tensor.size() = (bsz x n_choices, n_channels, n_nodes, n_nodes). 
                        The meta-paths (Hs). 
                - weights: list[tensor]. tensor.size() = (n_channels, n_edges)
                - graph_argument: (bsz x n_choices, n_nodes, n_nodes)
                - graph_punctuation: (bsz x n_choices, n_nodes, n_nodes)
                - graph_domain: (bsz x n_choices, n_nodes, n_nodes) 
        '''
        gtn_node_feats, updated_adj_graphs, weights = gtn(node=node_feats,
                                                          node_mask=node_masks,
                                                          A=A)

        assert isinstance(updated_adj_graphs, tuple)
        Hs_weights, Hs_normed, Hs_normed_masks = updated_adj_graphs

        n_hops = len(Hs_normed_masks)
        n_channels = Hs_normed_masks[0].size(1)

        meta_paths = ()
        for h in range(n_hops):
            h_meta_paths = ()
            for c in range(n_channels):
                h_meta_paths += (Hs_normed_masks[h][:, c, :, :], )
            meta_paths += (h_meta_paths,)
        assert len(meta_paths) <= 5
        assert max(map(len, meta_paths)) <= 5

        ''' GCN
        Outputs:
                - gcn_node_feats: (bsz x n_choices, n_nodes, embed_size)
                - node_weight: (bsz x n_choices, gcn_steps, n_node). (NODE weights.) 
        '''
        graph_meta_paths = [[dd_graph * item for item in h_meta_paths]
                            for h_meta_paths in meta_paths]

        node_neighbor_num = graph_argument.sum(-1) + graph_punctuation.sum(-1) + graph_domain.sum(-1)
        for h_meta_paths in graph_meta_paths:
            for meta_path in h_meta_paths:
                node_neighbor_num += meta_path.sum(-1)
        node_neighbor_num_mask = (node_neighbor_num >= 1).long()
        node_neighbor_num = util.replace_masked_values(node_neighbor_num.float(),
                                                       node_neighbor_num_mask, 1)

        gcn_node_feats, node_weight, _ = gcn(node=node_feats,
                                          graph_argument=graph_argument,
                                          graph_punctuation=graph_punctuation,
                                          graph_domain=graph_domain,
                                          graph_meta_paths=graph_meta_paths,
                                          node_neighbor_num=node_neighbor_num,
                                          len_meta_paths=len(meta_paths))

        return gcn_node_feats, node_weight

    def _downstream_prediction(self,
                               node_feats,
                               node_in_seq_indices,
                               sequence_output,
                               passage_mask,
                               question_mask
                               ):

        gcn_info_vec = self.get_gcn_info_vector(node_in_seq_indices, node_feats,
                                                size=sequence_output.size(),
                                                device=sequence_output.device)

        gcn_updated_sequence_output = self._gcn_enc(
            self._gcn_prj_ln(sequence_output + gcn_info_vec))  # (bsz x n_choices, seq_len, embed_size)

        # passage hidden and question hidden
        sequence_h2_weight = self._proj_sequence_h(gcn_updated_sequence_output).squeeze(
            -1)  # (bsz x num_choices, seq_len)
        passage_h2_weight = util.masked_softmax(sequence_h2_weight.float(),
                                                passage_mask.float())
        passage_h2 = util.weighted_sum(gcn_updated_sequence_output,
                                       passage_h2_weight)  # (bsz x num_choices, embed_size)
        question_h2_weight = util.masked_softmax(sequence_h2_weight.float(), question_mask.float())
        question_h2 = util.weighted_sum(gcn_updated_sequence_output,
                                        question_h2_weight)  # (bsz x num_choices, embed_size)

        gcn_output_feats = torch.cat([passage_h2,
                                      question_h2,
                                      gcn_updated_sequence_output[:, 0]],
                                     dim=1)  # (bsz x num_choices, embed_size x 3)
        gcn_logits = self._proj_span_num(gcn_output_feats)  # (bsz x num_choices, 1)

        return passage_h2, question_h2, gcn_logits

    def get_domain_matrices(self, domain_bpe_ids, split_ids, n_nodes):
        '''
        Inputs:
            - domain_bpe_ids: torch.tensor, (bsz x n_choices, max_seq_length)
            - split_ids: torch.tensor, (bsz x n_choices, max_seq_length)
            - n_nodes: int
        Output:
            - domain_graph: (bsz x n_choices, n_nodes, n_nodes)
        '''
        max_seg_amount = -1
        adj_matrices = []
        for _domain, _split in zip(domain_bpe_ids, split_ids):
            split = copy.deepcopy(_split)
            split = split.cpu().numpy()
            segments, split_index = compute_segments(split)
            max_seg_amount = max(len(segments), max_seg_amount)
            adj_matrix = compute_domain_adjacency_matrix(_domain, _split)  # value=0,1,2,.. the value means the count of connection.
            adj_matrices.append(adj_matrix)

        assert n_nodes >= max_seg_amount, "n_nodes={}, max_seq_amount={}".format(
            n_nodes, max_seg_amount
        )  # if not, the new code segments might be faulty.
        domain_graph = torch.zeros(len(adj_matrices), n_nodes, n_nodes)
        for i, adjm in enumerate(adj_matrices):

            # padding
            if len(adjm) > 0:
                assert len(adjm) == len(adjm[0])
                _pad = n_nodes - len(adjm)
                adjm = [item + [0] * _pad for item in adjm]
                adjm += [[0] * n_nodes] * _pad
                assert len(adjm) == len(adjm[0])

                domain_graph[i] = torch.tensor(adjm)

        domain_graph = domain_graph > 0

        return domain_graph.float().to(domain_bpe_ids.device)

    def forward(self,
                input_ids: torch.LongTensor,  # (bsz, num_choices, max_seq_length)
                attention_mask: torch.LongTensor,

                passage_mask: torch.LongTensor,  # (bsz, num_choices, max_seq_length)
                question_mask: torch.LongTensor,

                argument_bpe_ids: torch.LongTensor,  # (bsz, num_choices, max_seq_length)
                domain_bpe_ids: torch.LongTensor,
                punct_bpe_ids: torch.LongTensor,

                labels: torch.LongTensor,
                input_tokens: List[str],
                token_type_ids: torch.LongTensor = None,
                position_ids: torch.LongTensor = None,
                example_id: torch.LongTensor = None,
                ) -> Tuple:

        num_choices = input_ids.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None

        flat_passage_mask = passage_mask.view(-1, passage_mask.size(-1)) if passage_mask is not None else None
        flat_question_mask = question_mask.view(-1, question_mask.size(-1)) if question_mask is not None else None

        flat_argument_bpe_ids = argument_bpe_ids.view(-1, argument_bpe_ids.size(-1)) if argument_bpe_ids is not None else None
        flat_domain_bpe_ids = domain_bpe_ids.view(-1, domain_bpe_ids.size(-1)) if domain_bpe_ids is not None else None 
        flat_punct_bpe_ids = punct_bpe_ids.view(-1, punct_bpe_ids.size(-1)) if punct_bpe_ids is not None else None

        bert_outputs = self.roberta(flat_input_ids, attention_mask=flat_attention_mask,
                                    token_type_ids=None)
        sequence_output = bert_outputs[0]  # (bsz x num_choices, seq_length, embed_size)
        pooled_output = bert_outputs[1]  # <s> embedding. (bsz x num_choices, embed_size).

        if self.use_gcn:

            ''' The GCN branch. Suppose to go back to baseline once remove. '''
            new_punct_id = self.max_rel_id + 1
            new_punct_bpe_ids = new_punct_id * flat_punct_bpe_ids  # punct_id: 1 -> 4. for incorporating with argument_bpe_ids.
            _flat_all_bpe_ids = flat_argument_bpe_ids + new_punct_bpe_ids  # -1:padding, 0:non, 1-3: arg, 4:punct.
            overlapped_punct_argument_mask = (_flat_all_bpe_ids > new_punct_id).long()
            flat_all_bpe_ids = _flat_all_bpe_ids * (1 - overlapped_punct_argument_mask) + flat_argument_bpe_ids * overlapped_punct_argument_mask
            assert flat_argument_bpe_ids.max().item() <= new_punct_id
            assert flat_all_bpe_ids.max().item() <= new_punct_id

            ''' EDU embeddings & EDU masks & edges & node_indices
            Outputs:
                - encoded_spans: (bsz x n_choices, n_nodes, embed_size)
                - span_mask: (bsz x n_choices, n_nodes)
                - edges: list[list[int]]
                - node_in_seq_indices: list[list[list[int]]]
            '''
            encoded_spans, span_mask, edges, node_in_seq_indices = self.split_into_spans_9(sequence_output,
                                                                                           flat_attention_mask,
                                                                                           flat_all_bpe_ids)

            ''' adjacency matrices
            Outputs:
                - argument_graph: (bsz x n_choices, n_nodes, n_nodes)
                - punctuation_graph: (bsz x n_choices, n_nodes, n_nodes)
            '''
            argument_graph, punctuation_graph = self.get_adjacency_matrices_2(edges, n_nodes=encoded_spans.size(1), device=encoded_spans.device)  # checked.

            ''' Outputing the domain_words_edges adj matrices (named as domain_graph).
                                                            Inputs: 
                                                                - flat_domain_bpe_ids: (bsz x n_choices, max_seq_length)
                                                                - flat_all_bpe_ids: (bsz x n_choices, max_seq_length)
                                                            Output:
                                                                - domain_graph: (bsz x n_choices, n_nodes, n_nodes)
                                                            '''
            domain_graph = self.get_domain_matrices(domain_bpe_ids=flat_domain_bpe_ids, split_ids=flat_all_bpe_ids,
                                                    n_nodes=encoded_spans.size(1))

            ''' Preparation for GTN -> GCN. '''
            diagmat = torch.diagflat(torch.ones(encoded_spans.size(1), dtype=torch.long,
                                                device=encoded_spans.device))  # diagonal matrix. (n_node, n_node)
            diagmat = diagmat.unsqueeze(0).expand(encoded_spans.size(0), -1, -1)  # (bsz, n_node, n_node).
            dd_graph = span_mask.unsqueeze(1) * span_mask.unsqueeze(-1) * (1 - diagmat)  # the last item: acyclic graph.

            ''' (0) Initial adjacency matrices. '''
            graph_argument = dd_graph * argument_graph  # diag=0, pad=0, real_edges=1.
            graph_punctuation = dd_graph * punctuation_graph  # (bsz, n_node, n_node)
            graph_domain = dd_graph * domain_graph

            I = torch.eye(graph_argument.size(-1)).type(graph_argument.dtype)  # (n_nodes, n_nodes)
            I = I.unsqueeze(0).expand(graph_argument.size(0), -1, -1)  # (bsz, n_nodes, n_nodes)
            I = I.to(encoded_spans.device)
            A = torch.cat([I.unsqueeze(1),
                           graph_argument.unsqueeze(1),
                           graph_punctuation.unsqueeze(1),
                           graph_domain.unsqueeze(1),
                           ],
                          dim=1)  # (bsz, n_edges=3, n_nodes, n_nodes)
            A = A.to(encoded_spans.device)
            A = A.unsqueeze(1)  # (bsz, 1, n_edges=3, n_nodes, n_nodes)

            gcn_node_feats, node_weight = self._adjmatrices_gtn_gcn_nodefeats(
                self._gtn, self._gcn_1,
                dd_graph, graph_argument, graph_punctuation, graph_domain,
                encoded_spans, span_mask, A)

            ''' (1) masking the first 1/3 nodes. '''
            mask_edus_first = self._mask_nodes(span_mask, "first")  # (bsz x n_choices, n_nodes)
            graph_argument_mask_first = self._get_masked_adj_matrices(graph_argument, mask_edus_first)
            graph_punctuation_mask_first = self._get_masked_adj_matrices(graph_punctuation, mask_edus_first)
            graph_domain_mask_first = self._get_masked_adj_matrices(graph_domain, mask_edus_first)
            I_mask_first = self._get_masked_adj_matrices(I, mask_edus_first)
            A_first = torch.cat([I_mask_first.unsqueeze(1),
                                 graph_argument_mask_first.unsqueeze(1),
                                 graph_punctuation_mask_first.unsqueeze(1),
                                 graph_domain_mask_first.unsqueeze(1)
                                 ],
                                dim=1)
            A_first = A_first.unsqueeze(1).to(encoded_spans.device)
            gcn_node_feats_first, node_weight_first = self._adjmatrices_gtn_gcn_nodefeats(
                self._gtn, self._gcn_2,
                dd_graph, graph_argument_mask_first, graph_punctuation_mask_first,
                graph_domain_mask_first,
                encoded_spans, span_mask, A_first)

            ''' (2) masking the last 1/3 nodes. '''
            mask_edus_last = self._mask_nodes(span_mask, "last")
            graph_argument_mask_last = self._get_masked_adj_matrices(graph_argument, mask_edus_last)
            graph_punctuation_mask_last = self._get_masked_adj_matrices(graph_punctuation, mask_edus_last)
            graph_domain_mask_last = self._get_masked_adj_matrices(graph_domain, mask_edus_last)
            I_mask_last = self._get_masked_adj_matrices(I, mask_edus_last)
            A_last = torch.cat([I_mask_last.unsqueeze(1),
                                graph_argument_mask_last.unsqueeze(1),
                                graph_punctuation_mask_last.unsqueeze(1),
                                graph_domain_mask_last.unsqueeze(1)
                                ],
                               dim=1)
            A_last = A_last.unsqueeze(1).to(encoded_spans.device)
            gcn_node_feats_last, node_weight_last = self._adjmatrices_gtn_gcn_nodefeats(
                self._gtn, self._gcn_3,
                dd_graph, graph_argument_mask_last, graph_punctuation_mask_last, graph_domain_mask_last,
                encoded_spans, span_mask, A_last
            )

            ''' (3) masking the mid 1/3 nodes. '''
            mask_edus_mid = self._mask_nodes(span_mask, "mid")
            graph_argument_mask_mid = self._get_masked_adj_matrices(graph_argument, mask_edus_mid)
            graph_punctuation_mask_mid = self._get_masked_adj_matrices(graph_punctuation, mask_edus_mid)
            graph_domain_mask_mid = self._get_masked_adj_matrices(graph_domain, mask_edus_mid)
            I_mask_mid = self._get_masked_adj_matrices(I, mask_edus_mid)
            A_mid = torch.cat([I_mask_mid.unsqueeze(1),
                               graph_argument_mask_mid.unsqueeze(1),
                               graph_punctuation_mask_mid.unsqueeze(1),
                               graph_domain_mask_mid.unsqueeze(1)
                               ],
                              dim=1)
            A_mid = A_mid.unsqueeze(1).to(encoded_spans.device)
            gcn_node_feats_mid, node_weight_mid = self._adjmatrices_gtn_gcn_nodefeats(
                self._gtn, self._gcn_4,
                dd_graph, graph_argument_mask_mid, graph_punctuation_mask_mid, graph_domain_mask_mid,
                encoded_spans, span_mask, A_mid
            )

            ''' (4) masking the random 1/3 nodes. '''
            mask_edus_random = self._mask_nodes(span_mask, "random")
            graph_argument_mask_random = self._get_masked_adj_matrices(graph_argument, mask_edus_random)
            graph_punctuation_mask_random = self._get_masked_adj_matrices(graph_punctuation, mask_edus_random)
            graph_domain_mask_random = self._get_masked_adj_matrices(graph_domain, mask_edus_random)
            I_mask_random = self._get_masked_adj_matrices(I, mask_edus_random)
            A_random = torch.cat([I_mask_random.unsqueeze(1),
                                  graph_argument_mask_random.unsqueeze(1),
                                  graph_punctuation_mask_random.unsqueeze(1),
                                  graph_domain_mask_random.unsqueeze(1)
                                  ],
                                 dim=1)
            A_random = A_random.unsqueeze(1).to(encoded_spans.device )
            gcn_node_feats_random, node_weight_random = self._adjmatrices_gtn_gcn_nodefeats(
                self._gtn, self._gcn_5,
                dd_graph, graph_argument_mask_random, graph_punctuation_mask_random, graph_domain_mask_random,
                encoded_spans, span_mask, A_random
            )

            passage_h2, question_h2, gcn_logits = self._downstream_prediction(
                gcn_node_feats, node_in_seq_indices, sequence_output, flat_passage_mask, flat_question_mask)
            passage_h2_first, question_h2_first, gcn_mask_first_logits = self._downstream_prediction(
                gcn_node_feats_first, node_in_seq_indices, sequence_output, flat_passage_mask, flat_question_mask)
            passage_h2_last, question_h2_last, gcn_mask_last_logits = self._downstream_prediction(
                gcn_node_feats_last, node_in_seq_indices, sequence_output, flat_passage_mask, flat_question_mask)
            passage_h2_mid, question_h2_mid, gcn_mask_mid_logits = self._downstream_prediction(
                gcn_node_feats_mid, node_in_seq_indices, sequence_output, flat_passage_mask, flat_question_mask)
            passage_h2_random, question_h2_random, gcn_mask_random_logits = self._downstream_prediction(
                gcn_node_feats_random, node_in_seq_indices, sequence_output, flat_passage_mask, flat_question_mask)

        if self.use_pool:
            ''' The baseline branch. The output. '''
            pooled_output = self.dropout(pooled_output)  # (bsz x num_choices, embed_size)
            baseline_logits = self.classifier(pooled_output)  # (bsz x num_choices, 1)

        if self.use_gcn and self.use_pool:
            if self.merge_type == 8:
                ''' fuse: global graph embedding + pooled_output.'''

                pooled_output = self.dropout(pooled_output)

                node = gcn_node_feats
                _node = node.permute(1, 0, 2)
                global_node, _ = self.multihead_attn(_node, _node, _node)  # (n_node, b x n, embed_size)
                global_node = global_node.sum(dim=0)  # (b x n, embed_size)
                cls_graph_global = self._gcn_prj_ln(pooled_output + global_node)
                merged_feats = torch.cat([passage_h2, question_h2, cls_graph_global],
                                         dim=1)  # (bsz x num_choices, embed_size x 3)
                logits = self._proj_gcn_pool(merged_feats)  # (bsz x num_chocies, 1)

                _node_first = gcn_node_feats_first.permute(1, 0, 2)
                global_node_first, _ = self.multihead_attn(_node_first, _node_first, _node_first)
                global_node_first = global_node_first.sum(dim=0)
                cls_graph_global_first = self._gcn_prj_ln(pooled_output + global_node_first)
                merged_feats_first = torch.cat([passage_h2_first, question_h2_first, cls_graph_global_first], dim=1)
                mask_first_logits = self._proj_gcn_pool(merged_feats_first)

                _node_last = gcn_node_feats_last.permute(1, 0, 2)
                global_node_last, _ = self.multihead_attn(_node_last, _node_last, _node_last)
                global_node_last = global_node_last.sum(dim=0)
                cls_graph_global_last = self._gcn_prj_ln(pooled_output + global_node_last)
                merged_feats_last = torch.cat([passage_h2_last, question_h2_last, cls_graph_global_last], dim=1)
                mask_last_logits = self._proj_gcn_pool(merged_feats_last)

                _node_mid = gcn_node_feats_mid.permute(1, 0, 2)
                global_node_mid, _ = self.multihead_attn(_node_mid, _node_mid, _node_mid)
                global_node_mid = global_node_mid.sum(dim=0)
                cls_graph_global_mid = self._gcn_prj_ln(pooled_output + global_node_mid)
                merged_feats_mid = torch.cat([passage_h2_mid, question_h2_mid, cls_graph_global_mid], dim=1)
                mask_mid_logits = self._proj_gcn_pool(merged_feats_mid)

                _node_random = gcn_node_feats_random.permute(1, 0, 2)
                global_node_random, _ = self.multihead_attn(_node_random, _node_random, _node_random)
                global_node_random = global_node_random.sum(dim=0)
                cls_graph_global_random = self._gcn_prj_ln(pooled_output + global_node_random)
                merged_feats_random = torch.cat([passage_h2_random, question_h2_random, cls_graph_global_random], dim=1)
                mask_random_logits = self._proj_gcn_pool(merged_feats_random)


            else:
                raise Exception()
        elif self.use_gcn:
            logits = gcn_logits
        elif self.use_pool:
            logits = baseline_logits
        else:
            raise Exception

        reshaped_logits = logits.squeeze(-1).view(-1, num_choices)  # (bsz, num_choices)
        outputs = (reshaped_logits, )
        if self.use_gcn:
            reshaped_logits_first = mask_first_logits.squeeze(-1).view(-1, num_choices)  # (bsz, num_choices)
            reshaped_logits_last = mask_last_logits.squeeze(-1).view(-1, num_choices)  # (bsz, num_choices)
            reshaped_logits_mid = mask_mid_logits.squeeze(-1).view(-1, num_choices)  # (bsz, num_choices)
            reshaped_logits_random = mask_random_logits.squeeze(-1).view(-1, num_choices)  # (bsz, num_choices)
            outputs += (reshaped_logits_first, reshaped_logits_last, reshaped_logits_mid, reshaped_logits_random,)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss_1 = loss_fct(reshaped_logits, labels)
            loss = loss_1
            if self.use_gcn:
                loss_2 = loss_fct(reshaped_logits_first, labels)
                loss_3 = loss_fct(reshaped_logits_last, labels)
                loss_4 = loss_fct(reshaped_logits_mid, labels)
                loss_5 = loss_fct(reshaped_logits_random, labels)
                loss += loss_2 + loss_3 + loss_4 + loss_5

            outputs = (loss,) + outputs

        return outputs
