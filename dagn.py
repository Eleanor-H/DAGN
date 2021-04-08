# -*- coding: UTF-8 -*-
'''
DAGN Version 2.1.3.2

Adapted from: https://github.com/llamazing/numnet_plus
Date: 8/11/2020
Author: Yinya Huang
'''

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import numpy as np
from typing import List, Dict, Any, Tuple
from itertools import groupby
from operator import itemgetter
import copy
from util import FFNLayer, ResidualGRU, ArgumentGCN, ArgumentGCN_wreverseedges_double
from tools import allennlp as util
from transformers import BertPreTrainedModel, RobertaModel, BertModel


class DAGN(BertPreTrainedModel):
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
                 init_weights: bool,
                 max_rel_id,
                 hidden_size: int,
                 dropout_prob: float = 0.1,
                 merge_type: int = 1,
                 token_encoder_type: str = "roberta",
                 gnn_version: str = "GCN",
                 use_pool: bool = False,
                 use_gcn: bool = False,
                 gcn_steps: int=1) -> None:
        super().__init__(config)

        self.token_encoder_type = token_encoder_type
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

        ''' from numnet '''
        if self.use_gcn:
            modeling_out_dim = hidden_size
            node_dim = modeling_out_dim

            self._gcn_input_proj = nn.Linear(node_dim * 2, node_dim)
            if gnn_version == "GCN":
                self._gcn = ArgumentGCN(node_dim=node_dim, iteration_steps=gcn_steps)
            elif gnn_version == "GCN_reversededges_double":
                self._gcn = ArgumentGCN_wreverseedges_double(node_dim=node_dim, iteration_steps=gcn_steps)
            else:
                print("gnn_version == {}".format(gnn_version))
                raise Exception()
            self._iteration_steps = gcn_steps
            self._gcn_prj_ln = nn.LayerNorm(node_dim)
            self._gcn_enc = ResidualGRU(hidden_size, dropout_prob, 2)

            self._proj_sequence_h = nn.Linear(hidden_size, 1, bias=False)

            # span num extraction
            self._proj_span_num = FFNLayer(3 * hidden_size, hidden_size, 1, dropout_prob)

            self._proj_gcn_pool = FFNLayer(3 * hidden_size, hidden_size, 1, dropout_prob)
            self._proj_gcn_pool_4 = FFNLayer(4 * hidden_size, hidden_size, 1, dropout_prob)
            self._proj_gcn_pool_3 = FFNLayer(2 * hidden_size, hidden_size, 1, dropout_prob)

        if init_weights:
            self.init_weights()


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
                split_ids_indices, item_split_ids)
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

        :param indices: list(len=bsz) of list(len=n_notes) of list(len=varied).
        :param node: (bsz, n_nodes, embed_size)
        :param size: value=(bsz, seq_len, embed_size)
        :param device:
        :return:
        '''

        batch_size = size[0]
        gcn_info_vec = torch.zeros(size=size, dtype=torch.float, device=device)

        for b in range(batch_size):
            for ids, emb in zip(indices[b], node[b]):
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
        argument_graph = torch.zeros(
            (batch_size, n_nodes, n_nodes))  # NOTE: the diagonal should be assigned 0 since is acyclic graph.
        punct_graph = torch.zeros(
            (batch_size, n_nodes, n_nodes))  # NOTE: the diagonal should be assigned 0 since is acyclic graph.
        for b, sample_edges in enumerate(edges):
            for i, edge_value in enumerate(sample_edges):
                if edge_value == 1:  # (relation, head, tail)  关键词在句首. Note: not used in graph_version==4.0.
                    try:
                        argument_graph[b, i + 1, i + 2] = 1
                    except Exception:
                        pass
                elif edge_value == 2:  # (head, relation, tail)  关键词在句中，先因后果. Note: not used in graph_version==4.0.
                    argument_graph[b, i, i + 1] = 1
                elif edge_value == 3:  # (tail, relation, head)  关键词在句中，先果后因. Note: not used in graph_version==4.0.
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
        return argument_graph.to(device), punct_graph.to(device)


    def forward(self,
                input_ids: torch.LongTensor,
                attention_mask: torch.LongTensor,

                passage_mask: torch.LongTensor,
                question_mask: torch.LongTensor,

                argument_bpe_ids: torch.LongTensor,
                domain_bpe_ids: torch.LongTensor,
                punct_bpe_ids: torch.LongTensor,

                labels: torch.LongTensor,
                token_type_ids: torch.LongTensor = None,
                ) -> Tuple:

        num_choices = input_ids.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None

        flat_passage_mask = passage_mask.view(-1, passage_mask.size(-1)) if passage_mask is not None else None
        flat_question_mask = question_mask.view(-1, question_mask.size(-1)) if question_mask is not None else None

        flat_argument_bpe_ids = argument_bpe_ids.view(-1, argument_bpe_ids.size(-1)) if argument_bpe_ids is not None else None
        flat_domain_bpe_ids = domain_bpe_ids.view(-1, domain_bpe_ids.size(-1)) if domain_bpe_ids is not None else None  
        flat_punct_bpe_ids = punct_bpe_ids.view(-1, punct_bpe_ids.size(-1)) if punct_bpe_ids is not None else None

        bert_outputs = self.roberta(flat_input_ids, attention_mask=flat_attention_mask, token_type_ids=None)
        sequence_output = bert_outputs[0]
        pooled_output = bert_outputs[1]  


        if self.use_gcn:
            ''' The GCN branch. Suppose to go back to baseline once remove. '''
            new_punct_id = self.max_rel_id + 1
            new_punct_bpe_ids = new_punct_id * flat_punct_bpe_ids  # punct_id: 1 -> 4. for incorporating with argument_bpe_ids.
            _flat_all_bpe_ids = flat_argument_bpe_ids + new_punct_bpe_ids  # -1:padding, 0:non, 1-3: arg, 4:punct.
            overlapped_punct_argument_mask = (_flat_all_bpe_ids > new_punct_id).long()
            flat_all_bpe_ids = _flat_all_bpe_ids * (1 - overlapped_punct_argument_mask) + flat_argument_bpe_ids * overlapped_punct_argument_mask
            assert flat_argument_bpe_ids.max().item() <= new_punct_id

            # encoded_spans: (bsz x n_choices, n_nodes, embed_size)
            # span_mask: (bsz x n_choices, n_nodes)
            # edges: list[list[int]]
            # node_in_seq_indices: list[list[list[int]]]
            encoded_spans, span_mask, edges, node_in_seq_indices = self.split_into_spans_9(sequence_output,
                                                                                           flat_attention_mask,
                                                                                           flat_all_bpe_ids)

            argument_graph, punctuation_graph = self.get_adjacency_matrices_2(edges, n_nodes=encoded_spans.size(1), device=encoded_spans.device)

            node, node_weight = self._gcn(node=encoded_spans, node_mask=span_mask,
                                          argument_graph=argument_graph,
                                          punctuation_graph=punctuation_graph)  

            gcn_info_vec = self.get_gcn_info_vector(node_in_seq_indices, node,
                                                    size=sequence_output.size(), device=sequence_output.device)  

            gcn_updated_sequence_output = self._gcn_enc(self._gcn_prj_ln(sequence_output + gcn_info_vec))  

            # passage hidden and question hidden
            sequence_h2_weight = self._proj_sequence_h(gcn_updated_sequence_output).squeeze(-1)  
            passage_h2_weight = util.masked_softmax(sequence_h2_weight.float(), flat_passage_mask.float())  
            passage_h2 = util.weighted_sum(gcn_updated_sequence_output, passage_h2_weight)  
            question_h2_weight = util.masked_softmax(sequence_h2_weight.float(), flat_question_mask.float())
            question_h2 = util.weighted_sum(gcn_updated_sequence_output, question_h2_weight)  

            gcn_output_feats = torch.cat([passage_h2, question_h2, gcn_updated_sequence_output[:, 0]], dim=1)  
            gcn_logits = self._proj_span_num(gcn_output_feats)  


        if self.use_pool:
            ''' The baseline branch. The output. '''
            pooled_output = self.dropout(pooled_output)  
            baseline_logits = self.classifier(pooled_output)  


        if self.use_gcn and self.use_pool:
            ''' Merge gcn_logits & baseline_logits. TODO: different way of merging. '''

            if self.merge_type == 1:
                logits = gcn_logits + baseline_logits

            elif self.merge_type == 2:
                pooled_output = self.dropout(pooled_output)
                merged_feats = torch.cat([gcn_updated_sequence_output[:, 0], pooled_output], dim=1)  
                logits = self._proj_gcn_pool_3(merged_feats)  

            elif self.merge_type == 3:
                pooled_output = self.dropout(pooled_output)
                merged_feats = torch.cat([gcn_updated_sequence_output[:, 0], pooled_output,
                                          gcn_updated_sequence_output[:, 0], pooled_output], dim=1)  
                logits = self._proj_gcn_pool_4(merged_feats)  

            elif self.merge_type == 4:
                pooled_output = self.dropout(pooled_output)
                merged_feats = torch.cat([passage_h2, question_h2, pooled_output], dim=1)  
                logits = self._proj_gcn_pool(merged_feats)  

            elif self.merge_type == 5:
                pooled_output = self.dropout(pooled_output)
                merged_feats = torch.cat([passage_h2, question_h2, gcn_updated_sequence_output[:, 0], pooled_output],
                                         dim=1)  
                logits = self._proj_gcn_pool_4(merged_feats)  


        elif self.use_gcn:
            logits = gcn_logits
        elif self.use_pool:
            logits = baseline_logits
        else:
            raise Exception


        reshaped_logits = logits.squeeze(-1).view(-1, num_choices)  
        outputs = (reshaped_logits, )

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs
