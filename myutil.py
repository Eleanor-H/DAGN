import torch
import torch.nn as nn
import torch.nn.functional as F
from tools import allennlp as util


class ArgumentGCN(nn.Module):

    def __init__(self, node_dim, extra_factor_dim=0, iteration_steps=1):
        super(ArgumentGCN, self).__init__()

        self.node_dim = node_dim
        self.iteration_steps = iteration_steps

        self._node_weight_fc = torch.nn.Linear(node_dim + extra_factor_dim, 1, bias=True)

        self._self_node_fc = torch.nn.Linear(node_dim, node_dim, bias=True)

        self._node_fc_argument = torch.nn.Linear(node_dim, node_dim, bias=False)  # relation-specific transform matrices.
        self._node_fc_punctuation = torch.nn.Linear(node_dim, node_dim, bias=False)  # relation-specific transform matrices.


    def forward(self,
                node, # span features. 1 node type. Homogeneous graph.
                node_mask,  # 1=is_node, 0=not_node. size=(bsz, n_node)
                argument_graph, # directed graph (asymmetric adjacency matrix). size=(bsz, n_node, n_node)
                punctuation_graph,  # undirected edges (symmetric adjacency matrix). size=(bsz, n_node, n_node)
                extra_factor=None):
        ''' '''
        '''
        Current: 2 relation patterns.
            - argument edge. (most of them are causal relations)
            - punctuation edges. (including periods and commas)
        '''

        node_len = node.size(1)

        diagmat = torch.diagflat(torch.ones(node.size(1), dtype=torch.long,
                                            device=node.device))  # diagonal matrix. (n_node, n_node)
        diagmat = diagmat.unsqueeze(0).expand(node.size(0), -1, -1)  # (bsz, n_node, n_node).
        dd_graph = node_mask.unsqueeze(1) * node_mask.unsqueeze(-1) * (1 - diagmat)  # the last item: acyclic graph.

        graph_argument = dd_graph * argument_graph
        graph_punctuation = dd_graph * punctuation_graph

        node_neighbor_num = graph_argument.sum(-1) + graph_punctuation.sum(-1)
        node_neighbor_num_mask = (node_neighbor_num >= 1).long()
        node_neighbor_num = util.replace_masked_values(node_neighbor_num.float(), node_neighbor_num_mask, 1)

        all_weight = []
        for step in range(self.iteration_steps):


            ''' (1) Node Relatedness Measure '''
            if extra_factor is None:
                d_node_weight = torch.sigmoid(self._node_weight_fc(node)).squeeze(-1)  # Eleanor. eq (10). alpha_i. (bsz x n_choices, n_node)
            else:
                d_node_weight = torch.sigmoid(self._node_weight_fc(torch.cat((node, extra_factor), dim=-1))).squeeze(-1)  # alpha_i.

            all_weight.append(d_node_weight)  # Eleanor. Here record the weights.

            self_node_info = self._self_node_fc(node)  # v_i.

            ''' (2) Message Propagation (each relation type) '''
            # type 1. argument edges.
            node_info_argument = self._node_fc_argument(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
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

            agg_node_info = (node_info_argument + node_info_punctuation) / node_neighbor_num.unsqueeze(-1)

            ''' (3) Node Representation Update '''
            node = F.relu(self_node_info + agg_node_info)

        all_weight = [weight.unsqueeze(1) for weight in all_weight]
        all_weight = torch.cat(all_weight, dim=1)

        return node, all_weight


class ArgumentGCN_wreverseedges(nn.Module):

    def __init__(self, node_dim, extra_factor_dim=0, iteration_steps=1):
        super(ArgumentGCN_wreverseedges, self).__init__()

        self.node_dim = node_dim
        self.iteration_steps = iteration_steps

        self._node_weight_fc = torch.nn.Linear(node_dim + extra_factor_dim, 1, bias=True)

        self._self_node_fc = torch.nn.Linear(node_dim, node_dim, bias=True)

        self._node_fc_argument = torch.nn.Linear(node_dim, node_dim, bias=False)  # relation-specific transform matrices.
        self._node_fc_punctuation = torch.nn.Linear(node_dim, node_dim, bias=False)  # relation-specific transform matrices.
        self._node_fc_argument_prime = torch.nn.Linear(node_dim, node_dim,
                                                 bias=False)  # relation-specific transform matrices.
        self._node_fc_punctuation_prime = torch.nn.Linear(node_dim, node_dim,
                                                    bias=False)  # relation-specific transform matrices.


    def forward(self,
                node, # span features. 1 node type. Homogeneous graph.
                node_mask,  # 1=is_node, 0=not_node. size=(bsz, n_node)
                argument_graph, # directed graph (asymmetric adjacency matrix). size=(bsz, n_node, n_node)
                punctuation_graph,  # undirected edges (symmetric adjacency matrix). size=(bsz, n_node, n_node)
                extra_factor=None):
        ''' '''
        '''
        Current: 2 relation patterns & reversed directed edges.
            - argument edge. (most of them are causal relations)
            - punctuation edges. (including periods and commas)
        '''

        node_len = node.size(1)

        diagmat = torch.diagflat(torch.ones(node.size(1), dtype=torch.long,
                                            device=node.device))  # diagonal matrix. (n_node, n_node)
        diagmat = diagmat.unsqueeze(0).expand(node.size(0), -1, -1)  # (bsz, n_node, n_node).
        dd_graph = node_mask.unsqueeze(1) * node_mask.unsqueeze(-1) * (1 - diagmat)  # the last item: acyclic graph.

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
                d_node_weight = torch.sigmoid(self._node_weight_fc(node)).squeeze(-1)  # Eleanor. eq (10). alpha_i. (bsz x n_choices, n_node)
            else:
                d_node_weight = torch.sigmoid(self._node_weight_fc(torch.cat((node, extra_factor), dim=-1))).squeeze(-1)  # alpha_i.

            all_weight.append(d_node_weight)  # Eleanor. Here record the weights.

            self_node_info = self._self_node_fc(node)  # v_i.

            ''' (2) Message Propagation (each relation type) '''
            # type 1. argument edges.
            node_info_argument = self._node_fc_argument(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graph_argument,
                0)  # filtering weights not type 1
            node_info_argument = torch.matmul(node_weight, node_info_argument)

            # type 1'. reversed argument edges.
            node_info_argument_prime = self._node_fc_argument_prime(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graph_argument_re,
                0)  # filtering weights not type 1'.
            node_info_argument_prime = torch.matmul(node_weight, node_info_argument_prime)

            # type 2. punctuation edges.
            node_info_punctuation = self._node_fc_punctuation(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graph_punctuation,
                0)  # filtering weights not type 2
            node_info_punctuation = torch.matmul(node_weight, node_info_punctuation)

            # type 2'. reversed punctuation edges.
            node_info_punctuation_prime = self._node_fc_punctuation_prime(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graph_punctuation_re,
                0)  # filtering weights not type 2'
            node_info_punctuation_prime = torch.matmul(node_weight, node_info_punctuation_prime)


            agg_node_info = (node_info_argument + node_info_punctuation +
                             node_info_argument_prime + node_info_punctuation_prime) / node_neighbor_num.unsqueeze(-1)

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

        self._node_fc_argument = torch.nn.Linear(node_dim, node_dim, bias=False)  # relation-specific transform matrices.
        self._node_fc_punctuation = torch.nn.Linear(node_dim, node_dim, bias=False)  # relation-specific transform matrices.
        self._node_fc_argument_prime = torch.nn.Linear(node_dim, node_dim,
                                                 bias=False)  # relation-specific transform matrices.
        self._node_fc_punctuation_prime = torch.nn.Linear(node_dim, node_dim,
                                                    bias=False)  # relation-specific transform matrices.

        self._node_fc_argument_2 = torch.nn.Linear(node_dim, node_dim,
                                                 bias=False)  # relation-specific transform matrices.
        self._node_fc_punctuation_2 = torch.nn.Linear(node_dim, node_dim,
                                                    bias=False)  # relation-specific transform matrices.
        self._node_fc_argument_prime_2 = torch.nn.Linear(node_dim, node_dim,
                                                       bias=False)  # relation-specific transform matrices.
        self._node_fc_punctuation_prime_2 = torch.nn.Linear(node_dim, node_dim,
                                                          bias=False)  # relation-specific transform matrices.


    def forward(self,
                node, # span features. 1 node type. Homogeneous graph.
                node_mask,  # 1=is_node, 0=not_node. size=(bsz, n_node)
                argument_graph, # directed graph (asymmetric adjacency matrix). size=(bsz, n_node, n_node)
                punctuation_graph,  # undirected edges (symmetric adjacency matrix). size=(bsz, n_node, n_node)
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
                                            device=node.device))  # diagonal matrix. (n_node, n_node)
        diagmat = diagmat.unsqueeze(0).expand(node.size(0), -1, -1)  # (bsz, n_node, n_node).
        dd_graph = node_mask.unsqueeze(1) * node_mask.unsqueeze(-1) * (1 - diagmat)  # the last item: acyclic graph.

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
                d_node_weight = torch.sigmoid(self._node_weight_fc(node)).squeeze(-1)  # Eleanor. eq (10). alpha_i. (bsz x n_choices, n_node)
            else:
                d_node_weight = torch.sigmoid(self._node_weight_fc(torch.cat((node, extra_factor), dim=-1))).squeeze(-1)  # alpha_i.

            all_weight.append(d_node_weight)  # Eleanor. Here record the weights.

            self_node_info = self._self_node_fc(node)  # v_i.

            ''' (2) Message Propagation (each relation type) '''
            # type 1. argument edges.
            node_info_argument = self._node_fc_argument(node)
            node_info_argument_2 = self._node_fc_argument_2(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graph_argument,
                0)  # filtering weights not type 1
            node_info_argument = torch.matmul(node_weight, node_info_argument)
            node_info_argument_2 = torch.matmul(node_weight, node_info_argument_2)

            # type 1'. reversed argument edges.
            node_info_argument_prime = self._node_fc_argument_prime(node)
            node_info_argument_prime_2 = self._node_fc_argument_prime_2(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graph_argument_re,
                0)  # filtering weights not type 1'.
            node_info_argument_prime = torch.matmul(node_weight, node_info_argument_prime)
            node_info_argument_prime_2 = torch.matmul(node_weight, node_info_argument_prime_2)

            # type 2. punctuation edges.
            node_info_punctuation = self._node_fc_punctuation(node)
            node_info_punctuation_2 = self._node_fc_punctuation_2(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graph_punctuation,
                0)  # filtering weights not type 2
            node_info_punctuation = torch.matmul(node_weight, node_info_punctuation)
            node_info_punctuation_2 = torch.matmul(node_weight, node_info_punctuation_2)

            # type 2'. reversed punctuation edges.
            node_info_punctuation_prime = self._node_fc_punctuation_prime(node)
            node_info_punctuation_prime_2 = self._node_fc_punctuation_prime_2(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graph_punctuation_re,
                0)  # filtering weights not type 2'
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


class ArgumentGCN_sentence(nn.Module):
    '''
        A variant of class ArgumentGCN.
            - removing argument graphs. for sentence splitting.

        '''

    def __init__(self, node_dim, extra_factor_dim=0, iteration_steps=1):
        super(ArgumentGCN_sentence, self).__init__()

        self.node_dim = node_dim
        self.iteration_steps = iteration_steps

        self._node_weight_fc = torch.nn.Linear(node_dim + extra_factor_dim, 1, bias=True)

        self._self_node_fc = torch.nn.Linear(node_dim, node_dim, bias=True)

        self._node_fc_argument = torch.nn.Linear(node_dim, node_dim, bias=False)  # relation-specific transform matrices.
        self._node_fc_punctuation = torch.nn.Linear(node_dim, node_dim, bias=False)  # relation-specific transform matrices.


    def forward(self,
                node, # span features. 1 node type. Homogeneous graph.
                node_mask,  # 1=is_node, 0=not_node. size=(bsz, n_node)
                punctuation_graph,  # undirected edges (symmetric adjacency matrix). size=(bsz, n_node, n_node)
                extra_factor=None):
        ''' '''
        '''
        Current: 1 relation patterns.
            - punctuation edges. (including periods and commas)
        '''

        node_len = node.size(1)

        diagmat = torch.diagflat(torch.ones(node.size(1), dtype=torch.long,
                                            device=node.device))  # diagonal matrix. (n_node, n_node)
        diagmat = diagmat.unsqueeze(0).expand(node.size(0), -1, -1)  # (bsz, n_node, n_node).
        dd_graph = node_mask.unsqueeze(1) * node_mask.unsqueeze(-1) * (1 - diagmat)  # the last item: acyclic graph.

        # graph_argument = dd_graph * argument_graph
        graph_punctuation = dd_graph * punctuation_graph

        # node_neighbor_num = graph_argument.sum(-1) + graph_punctuation.sum(-1)
        node_neighbor_num = graph_punctuation.sum(-1)
        node_neighbor_num_mask = (node_neighbor_num >= 1).long()
        node_neighbor_num = util.replace_masked_values(node_neighbor_num.float(), node_neighbor_num_mask, 1)

        all_weight = []
        for step in range(self.iteration_steps):


            ''' (1) Node Relatedness Measure '''
            if extra_factor is None:
                d_node_weight = torch.sigmoid(self._node_weight_fc(node)).squeeze(-1)  # Eleanor. eq (10). alpha_i. (bsz x n_choices, n_node)
            else:
                d_node_weight = torch.sigmoid(self._node_weight_fc(torch.cat((node, extra_factor), dim=-1))).squeeze(-1)  # alpha_i.

            all_weight.append(d_node_weight)  # Eleanor. Here record the weights.

            self_node_info = self._self_node_fc(node)  # v_i.

            ''' (2) Message Propagation (each relation type) '''
            # type 1. argument edges.
            # node_info_argument = self._node_fc_argument(node)
            # node_weight = util.replace_masked_values(
            #     d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
            #     graph_argument,
            #     0)  # filtering weights not type 1
            # node_info_argument = torch.matmul(node_weight, node_info_argument)

            # type 2. punctuation edges.
            node_info_punctuation = self._node_fc_punctuation(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graph_punctuation,
                0)  # filtering weights not type 2
            node_info_punctuation = torch.matmul(node_weight, node_info_punctuation)

            # agg_node_info = (node_info_argument + node_info_punctuation) / node_neighbor_num.unsqueeze(-1)

            agg_node_info = node_info_punctuation / node_neighbor_num.unsqueeze(-1)

            ''' (3) Node Representation Update '''
            node = F.relu(self_node_info + agg_node_info)

        all_weight = [weight.unsqueeze(1) for weight in all_weight]
        all_weight = torch.cat(all_weight, dim=1)

        return node, all_weight


class ArgumentGCN_2(nn.Module):
    '''
    A variant of class ArgumentGCN.
        - The argument "argument_graph" of the forward() method is a list of adjacency matrices
            instead of a single adjacency matrix.
        - The calculation within the forward() method includes all adjacency matrices.

    '''

    def __init__(self, node_dim, extra_factor_dim=0, iteration_steps=1):
        super(ArgumentGCN_2, self).__init__()

        self.node_dim = node_dim
        self.iteration_steps = iteration_steps

        self._node_weight_fc = torch.nn.Linear(node_dim + extra_factor_dim, 1, bias=True)

        self._self_node_fc = torch.nn.Linear(node_dim, node_dim, bias=True)

        self._node_fc_argument_1 = torch.nn.Linear(node_dim, node_dim,
                                                   bias=False)  # relation-specific transform matrices.
        self._node_fc_argument_2 = torch.nn.Linear(node_dim, node_dim,
                                                 bias=False)  # relation-specific transform matrices.
        self._node_fc_argument_3 = torch.nn.Linear(node_dim, node_dim,
                                                 bias=False)  # relation-specific transform matrices.
        self._node_fc_argument_4 = torch.nn.Linear(node_dim, node_dim,
                                                 bias=False)  # relation-specific transform matrices.
        self._node_fc_punctuation = torch.nn.Linear(node_dim, node_dim, bias=False)  # relation-specific transform matrices.


    def forward(self,
                node, # span features. 1 node type. Homogeneous graph.
                node_mask,  # 1=is_node, 0=not_node. size=(bsz, n_node)
                argument_graphs:list, # list of directed graph (asymmetric adjacency matrix). size=(bsz, n_node, n_node)
                punctuation_graph,  # undirected edges (symmetric adjacency matrix). size=(bsz, n_node, n_node)
                extra_factor=None):
        ''' '''
        '''
        Current: 5 relation patterns.
            - argument edge type=1. 
            - argument edge type=2.
            - argument edge type=3.
            - argument edge type=4.
            - punctuation edges. type=5.
        '''

        node_len = node.size(1)

        diagmat = torch.diagflat(torch.ones(node.size(1), dtype=torch.long,
                                            device=node.device))  # diagonal matrix. (n_node, n_node)
        diagmat = diagmat.unsqueeze(0).expand(node.size(0), -1, -1)  # (bsz, n_node, n_node).
        dd_graph = node_mask.unsqueeze(1) * node_mask.unsqueeze(-1) * (1 - diagmat)  # the last item: acyclic graph.

        # graph_argument = dd_graph * argument_graph  # dep
        graphs_argument = [dd_graph * item for item in argument_graphs]  # list of Long tensor.
        graph_punctuation = dd_graph * punctuation_graph

        # node_neighbor_num = graph_argument.sum(-1) + graph_punctuation.sum(-1)  # dep
        node_neighbor_num = sum([item.sum(-1) for item in graphs_argument]) + graph_punctuation.sum(-1)
        node_neighbor_num_mask = (node_neighbor_num >= 1).long()
        node_neighbor_num = util.replace_masked_values(node_neighbor_num.float(), node_neighbor_num_mask, 1)

        all_weight = []
        for step in range(self.iteration_steps):


            ''' (1) Node Relatedness Measure '''
            if extra_factor is None:
                d_node_weight = torch.sigmoid(self._node_weight_fc(node)).squeeze(-1)  # Eleanor. eq (10). alpha_i.
            else:
                d_node_weight = torch.sigmoid(self._node_weight_fc(torch.cat((node, extra_factor), dim=-1))).squeeze(-1)  # alpha_i.

            all_weight.append(d_node_weight)  # Eleanor. Here record the weights.

            self_node_info = self._self_node_fc(node)  # v_i.


            ''' (2) Message Propagation (each relation type) '''
            assert len(graphs_argument) == 4

            # type 1. argument edges 1.
            node_info_argument_1 = self._node_fc_argument_1(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graphs_argument[0],
                0)  # filtering weights not type 1
            node_info_argument_1 = torch.matmul(node_weight, node_info_argument_1)

            # type 2. argument edges 2.
            node_info_argument_2 = self._node_fc_argument_2(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graphs_argument[1],
                0)  # filtering weights not type 1
            node_info_argument_2 = torch.matmul(node_weight, node_info_argument_2)

            # type 3. argument edges 3.
            node_info_argument_3 = self._node_fc_argument_3(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graphs_argument[2],
                0)  # filtering weights not type 1
            node_info_argument_3 = torch.matmul(node_weight, node_info_argument_3)

            # type 4. argument edges 4.
            node_info_argument_4 = self._node_fc_argument_4(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graphs_argument[3],
                0)  # filtering weights not type 1
            node_info_argument_4 = torch.matmul(node_weight, node_info_argument_4)

            # type 5. punctuation edges.
            node_info_punctuation = self._node_fc_punctuation(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graph_punctuation,
                0)  # filtering weights not type 2
            node_info_punctuation = torch.matmul(node_weight, node_info_punctuation)

            agg_node_info = (node_info_argument_1 + node_info_argument_2 + node_info_argument_3 + node_info_argument_4
                             + node_info_punctuation) / node_neighbor_num.unsqueeze(-1)


            ''' (3) Node Representation Update '''
            node = F.relu(self_node_info + agg_node_info)


        all_weight = [weight.unsqueeze(1) for weight in all_weight]
        all_weight = torch.cat(all_weight, dim=1)

        return node, all_weight



class ArgumentGCN_3(nn.Module):
    '''
    A variant of class ArgumentGCN_2.
        - Add argument "argument_graph_reverse" and "punctuation_graph_reverse" to forward(). The former
            is a list of adjacency matrices, while the latter is a single adjacency matrix.
        - The calculation within the forward() method includes all adjacency matrices.

    '''

    def __init__(self, node_dim, extra_factor_dim=0, iteration_steps=1):
        super(ArgumentGCN_3, self).__init__()

        self.node_dim = node_dim
        self.iteration_steps = iteration_steps

        self._node_weight_fc = torch.nn.Linear(node_dim + extra_factor_dim, 1, bias=True)

        self._self_node_fc = torch.nn.Linear(node_dim, node_dim, bias=True)

        self._node_fc_argument_1 = torch.nn.Linear(node_dim, node_dim,
                                                   bias=False)  # relation-specific transform matrices.
        self._node_fc_argument_2 = torch.nn.Linear(node_dim, node_dim,
                                                 bias=False)  # relation-specific transform matrices.
        self._node_fc_argument_3 = torch.nn.Linear(node_dim, node_dim,
                                                 bias=False)  # relation-specific transform matrices.
        self._node_fc_argument_4 = torch.nn.Linear(node_dim, node_dim,
                                                 bias=False)  # relation-specific transform matrices.
        self._node_fc_punctuation = torch.nn.Linear(node_dim, node_dim, bias=False)  # relation-specific transform matrices.

        self._node_fc_argument_1prime = torch.nn.Linear(node_dim, node_dim,
                                                   bias=False)  # relation-specific transform matrices.
        self._node_fc_argument_2prime = torch.nn.Linear(node_dim, node_dim,
                                                   bias=False)  # relation-specific transform matrices.
        self._node_fc_argument_3prime = torch.nn.Linear(node_dim, node_dim,
                                                   bias=False)  # relation-specific transform matrices.
        self._node_fc_argument_4prime = torch.nn.Linear(node_dim, node_dim,
                                                   bias=False)  # relation-specific transform matrices.
        self._node_fc_punctuationprime = torch.nn.Linear(node_dim, node_dim,
                                                    bias=False)  # relation-specific transform matrices.

    def forward(self,
                node, # span features. 1 node type. Homogeneous graph.
                node_mask,  # 1=is_node, 0=not_node. size=(bsz, n_node)
                adj_matrices: [list, list, torch.Tensor, torch.Tensor],
                # [argument_graphs, argument_graphs_reverse, punctuation_graph, punctuation_graph_reverse]
                # argument_graphs:list, # list of directed graph (asymmetric adjacency matrix). size=(bsz, n_node, n_node)
                # argument_graphs_reverse: list,
                # # list of directed graph (asymmetric adjacency matrix). size=(bsz, n_node, n_node)
                # punctuation_graph: torch.Tensor,  # directed edges (asymmetric adjacency matrix). size=(bsz, n_node, n_node)
                # punctuation_graph_reverse: torch.Tensor,  # directed edges (asymmetric adjacency matrix). size=(bsz, n_node, n_node)
                extra_factor=None):
        ''' '''
        '''
        10 Edge Types:
            1 - Comparison.*
            2 - Contingency.*
            3 - Expansion.*
            4 - Temporal.*
            5 - punctuation
            1' - reversed Comparison.*  (t, h, r)
            2' - reversed Contingency.*  (t, h, r)
            3' - reversed Expansion.*  (t, h, r)
            4' - reversed Temporal.*  (t, h, r)
            5' - reversed punctuation  (t, h, r)
        '''

        argument_graphs, argument_graphs_reverse, punctuation_graph, punctuation_graph_reverse = adj_matrices

        node_len = node.size(1)

        diagmat = torch.diagflat(torch.ones(node.size(1), dtype=torch.long,
                                            device=node.device))  # diagonal matrix. (n_node, n_node)
        diagmat = diagmat.unsqueeze(0).expand(node.size(0), -1, -1)  # (bsz, n_node, n_node).
        dd_graph = node_mask.unsqueeze(1) * node_mask.unsqueeze(-1) * (1 - diagmat)  # the last item: acyclic graph.

        # graph_argument = dd_graph * argument_graph  # dep
        graphs_argument = [dd_graph * item for item in argument_graphs]  # list of Long tensor.
        graphs_argument_re = [dd_graph * item for item in argument_graphs_reverse]  # list of Long tensor.
        graph_punctuation = dd_graph * punctuation_graph
        graph_punctuation_re = dd_graph * punctuation_graph_reverse

        # node_neighbor_num = graph_argument.sum(-1) + graph_punctuation.sum(-1)  # dep
        node_neighbor_num = sum([item.sum(-1) for item in graphs_argument]) + graph_punctuation.sum(-1) + \
                            sum([item.sum(-1) for item in graphs_argument_re]) + graph_punctuation_re.sum(-1)
        node_neighbor_num_mask = (node_neighbor_num >= 1).long()
        node_neighbor_num = util.replace_masked_values(node_neighbor_num.float(), node_neighbor_num_mask, 1)

        all_weight = []
        for step in range(self.iteration_steps):


            ''' (1) Node Relatedness Measure '''
            if extra_factor is None:
                d_node_weight = torch.sigmoid(self._node_weight_fc(node)).squeeze(-1)  # Eleanor. eq (10). alpha_i.
            else:
                d_node_weight = torch.sigmoid(self._node_weight_fc(torch.cat((node, extra_factor), dim=-1))).squeeze(-1)  # alpha_i.

            all_weight.append(d_node_weight)  # Eleanor. Here record the weights.

            self_node_info = self._self_node_fc(node)  # v_i.


            ''' (2) Message Propagation (each edge type) '''
            assert len(graphs_argument) == len(graphs_argument_re) == 4

            # type 1. argument edges 1.
            node_info_argument_1 = self._node_fc_argument_1(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graphs_argument[0],
                0)  # filtering out weights not type 1.
            node_info_argument_1 = torch.matmul(node_weight, node_info_argument_1)

            # type 1'. argument edge 1'.
            node_info_argument_1prime = self._node_fc_argument_1prime(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graphs_argument_re[0],
                0)  # filtering out weights not type 1'.
            node_info_argument_1prime = torch.matmul(node_weight, node_info_argument_1prime)

            # type 2. argument edges 2.
            node_info_argument_2 = self._node_fc_argument_2(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graphs_argument[1],
                0)  # filtering out weights not type 2.
            node_info_argument_2 = torch.matmul(node_weight, node_info_argument_2)

            # type 2'. argument edge 2'.
            node_info_argument_2prime = self._node_fc_argument_2prime(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graphs_argument_re[1],
                0)  # filtering out weights not type 2'.
            node_info_argument_2prime = torch.matmul(node_weight, node_info_argument_2prime)

            # type 3. argument edges 3.
            node_info_argument_3 = self._node_fc_argument_3(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graphs_argument[2],
                0)  # filtering out weights not type 3.
            node_info_argument_3 = torch.matmul(node_weight, node_info_argument_3)

            # type 3'. argument edge 3'.
            node_info_argument_3prime = self._node_fc_argument_3prime(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graphs_argument_re[2],
                0)  # filtering out weights not type 3'.
            node_info_argument_3prime = torch.matmul(node_weight, node_info_argument_3prime)

            # type 4. argument edges 4.
            node_info_argument_4 = self._node_fc_argument_4(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graphs_argument[3],
                0)  # filtering out weights not type 4
            node_info_argument_4 = torch.matmul(node_weight, node_info_argument_4)

            # type 4'. argument edge 4'.
            node_info_argument_4prime = self._node_fc_argument_4prime(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graphs_argument_re[3],
                0)  # filtering out weights not type 4'.
            node_info_argument_4prime = torch.matmul(node_weight, node_info_argument_4prime)

            # type 5. punctuation edges.
            node_info_punctuation = self._node_fc_punctuation(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graph_punctuation,
                0)  # filtering out weights not type 5.
            node_info_punctuation = torch.matmul(node_weight, node_info_punctuation)

            # type 5'. argument edge 5'.
            node_info_punctuationprime = self._node_fc_punctuationprime(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graph_punctuation_re,
                0)  # filtering out weights not type 5'.
            node_info_punctuationprime = torch.matmul(node_weight, node_info_punctuationprime)

            agg_node_info = (node_info_argument_1 + node_info_argument_2 + node_info_argument_3 + node_info_argument_4 + \
                             node_info_argument_1prime + node_info_argument_2prime + node_info_argument_3prime + \
                             node_info_argument_4prime + \
                             node_info_punctuation + node_info_punctuationprime) / node_neighbor_num.unsqueeze(-1)


            ''' (3) Node Representation Update '''
            node = F.relu(self_node_info + agg_node_info)

        all_weight = [weight.unsqueeze(1) for weight in all_weight]
        all_weight = torch.cat(all_weight, dim=1)

        return node, all_weight


class ArgumentGCN_31(nn.Module):
    '''
    A variant of class ArgumentGCN_3 in that the _reverse matrices are obtained by transpose of their counterparts.
        - Add argument "argument_graph_reverse" and "punctuation_graph_reverse" to forward(). The former
            is a list of adjacency matrices, while the latter is a single adjacency matrix.
        - The calculation within the forward() method includes all adjacency matrices.

    '''

    def __init__(self, node_dim, extra_factor_dim=0, iteration_steps=1):
        super(ArgumentGCN_31, self).__init__()

        self.node_dim = node_dim
        self.iteration_steps = iteration_steps

        self._node_weight_fc = torch.nn.Linear(node_dim + extra_factor_dim, 1, bias=True)

        self._self_node_fc = torch.nn.Linear(node_dim, node_dim, bias=True)

        self._node_fc_argument_1 = torch.nn.Linear(node_dim, node_dim,
                                                   bias=False)  # relation-specific transform matrices.
        self._node_fc_argument_2 = torch.nn.Linear(node_dim, node_dim,
                                                 bias=False)  # relation-specific transform matrices.
        self._node_fc_argument_3 = torch.nn.Linear(node_dim, node_dim,
                                                 bias=False)  # relation-specific transform matrices.
        self._node_fc_argument_4 = torch.nn.Linear(node_dim, node_dim,
                                                 bias=False)  # relation-specific transform matrices.
        self._node_fc_punctuation = torch.nn.Linear(node_dim, node_dim, bias=False)  # relation-specific transform matrices.

        self._node_fc_argument_1prime = torch.nn.Linear(node_dim, node_dim,
                                                   bias=False)  # relation-specific transform matrices.
        self._node_fc_argument_2prime = torch.nn.Linear(node_dim, node_dim,
                                                   bias=False)  # relation-specific transform matrices.
        self._node_fc_argument_3prime = torch.nn.Linear(node_dim, node_dim,
                                                   bias=False)  # relation-specific transform matrices.
        self._node_fc_argument_4prime = torch.nn.Linear(node_dim, node_dim,
                                                   bias=False)  # relation-specific transform matrices.
        self._node_fc_punctuationprime = torch.nn.Linear(node_dim, node_dim,
                                                    bias=False)  # relation-specific transform matrices.

    def forward(self,
                node, # span features. 1 node type. Homogeneous graph.
                node_mask,  # 1=is_node, 0=not_node. size=(bsz, n_node)
                adj_matrices: [list, list, torch.Tensor, torch.Tensor],
                # argument_graphs:list, # list of directed graph (asymmetric adjacency matrix). size=(bsz, n_node, n_node)
                # argument_graphs_reverse: list,
                # # list of directed graph (asymmetric adjacency matrix). size=(bsz, n_node, n_node)
                # punctuation_graph,  # directed edges (asymmetric adjacency matrix). size=(bsz, n_node, n_node)
                # punctuation_graph_reverse,  # directed edges (asymmetric adjacency matrix). size=(bsz, n_node, n_node)
                extra_factor=None):
        ''' '''
        '''
        10 Edge Types:
            1 - Comparison.*
            2 - Contingency.*
            3 - Expansion.*
            4 - Temporal.*
            5 - punctuation
            1' - Comparison.*  (t, h, r)
            2' - Contingency.*  (t, h, r)
            3' - Expansion.*  (t, h, r)
            4' - Temporal.*  (t, h, r)
            5' - punctuation  (t, h, r)
        '''

        argument_graphs, argument_graphs_reverse, punctuation_graph, punctuation_graph_reverse = adj_matrices

        node_len = node.size(1)

        diagmat = torch.diagflat(torch.ones(node.size(1), dtype=torch.long,
                                            device=node.device))  # diagonal matrix. (n_node, n_node)
        diagmat = diagmat.unsqueeze(0).expand(node.size(0), -1, -1)  # (bsz, n_node, n_node).
        dd_graph = node_mask.unsqueeze(1) * node_mask.unsqueeze(-1) * (1 - diagmat)  # the last item: acyclic graph.

        # graph_argument = dd_graph * argument_graph  # dep
        graphs_argument = [dd_graph * item for item in argument_graphs]  # list of Long tensor.
        # graphs_argument_re = [dd_graph * item for item in argument_graphs_reverse]  # list of Long tensor.
        graphs_argument_re = [dd_graph * item.permute(0, 2, 1) for item in argument_graphs]
        graph_punctuation = dd_graph * punctuation_graph
        # graph_punctuation_re = dd_graph * punctuation_graph_reverse
        graph_punctuation_re = dd_graph * punctuation_graph.permute(0, 2, 1)

        # node_neighbor_num = graph_argument.sum(-1) + graph_punctuation.sum(-1)  # dep
        node_neighbor_num = sum([item.sum(-1) for item in graphs_argument]) + graph_punctuation.sum(-1) + \
                            sum([item.sum(-1) for item in graphs_argument_re]) + graph_punctuation_re.sum(-1)
        node_neighbor_num_mask = (node_neighbor_num >= 1).long()
        node_neighbor_num = util.replace_masked_values(node_neighbor_num.float(), node_neighbor_num_mask, 1)

        all_weight = []
        for step in range(self.iteration_steps):


            ''' (1) Node Relatedness Measure '''
            if extra_factor is None:
                d_node_weight = torch.sigmoid(self._node_weight_fc(node)).squeeze(-1)  # Eleanor. eq (10). alpha_i.
            else:
                d_node_weight = torch.sigmoid(self._node_weight_fc(torch.cat((node, extra_factor), dim=-1))).squeeze(-1)  # alpha_i.

            all_weight.append(d_node_weight)  # Eleanor. Here record the weights.

            self_node_info = self._self_node_fc(node)  # v_i.


            ''' (2) Message Propagation (each edge type) '''
            assert len(graphs_argument) == len(graphs_argument_re) == 4

            # type 1. argument edges 1.
            node_info_argument_1 = self._node_fc_argument_1(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graphs_argument[0],
                0)  # filtering out weights not type 1.
            node_info_argument_1 = torch.matmul(node_weight, node_info_argument_1)

            # type 1'. argument edge 1'.
            node_info_argument_1prime = self._node_fc_argument_1prime(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graphs_argument_re[0],
                0)  # filtering out weights not type 1'.
            node_info_argument_1prime = torch.matmul(node_weight, node_info_argument_1prime)

            # type 2. argument edges 2.
            node_info_argument_2 = self._node_fc_argument_2(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graphs_argument[1],
                0)  # filtering out weights not type 2.
            node_info_argument_2 = torch.matmul(node_weight, node_info_argument_2)

            # type 2'. argument edge 2'.
            node_info_argument_2prime = self._node_fc_argument_2prime(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graphs_argument_re[1],
                0)  # filtering out weights not type 2'.
            node_info_argument_2prime = torch.matmul(node_weight, node_info_argument_2prime)

            # type 3. argument edges 3.
            node_info_argument_3 = self._node_fc_argument_3(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graphs_argument[2],
                0)  # filtering out weights not type 3.
            node_info_argument_3 = torch.matmul(node_weight, node_info_argument_3)

            # type 3'. argument edge 3'.
            node_info_argument_3prime = self._node_fc_argument_3prime(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graphs_argument_re[2],
                0)  # filtering out weights not type 3'.
            node_info_argument_3prime = torch.matmul(node_weight, node_info_argument_3prime)

            # type 4. argument edges 4.
            node_info_argument_4 = self._node_fc_argument_4(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graphs_argument[3],
                0)  # filtering out weights not type 4
            node_info_argument_4 = torch.matmul(node_weight, node_info_argument_4)

            # type 4'. argument edge 4'.
            node_info_argument_4prime = self._node_fc_argument_4prime(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graphs_argument_re[3],
                0)  # filtering out weights not type 4'.
            node_info_argument_4prime = torch.matmul(node_weight, node_info_argument_4prime)

            # type 5. punctuation edges.
            node_info_punctuation = self._node_fc_punctuation(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graph_punctuation,
                0)  # filtering out weights not type 5.
            node_info_punctuation = torch.matmul(node_weight, node_info_punctuation)

            # type 5'. argument edge 5'.
            node_info_punctuationprime = self._node_fc_punctuationprime(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graph_punctuation_re,
                0)  # filtering out weights not type 5'.
            node_info_punctuationprime = torch.matmul(node_weight, node_info_punctuationprime)

            agg_node_info = (node_info_argument_1 + node_info_argument_2 + node_info_argument_3 + node_info_argument_4 + \
                             node_info_argument_1prime + node_info_argument_2prime + node_info_argument_3prime + \
                             node_info_argument_4prime + \
                             node_info_punctuation + node_info_punctuationprime) / node_neighbor_num.unsqueeze(-1)


            ''' (3) Node Representation Update '''
            node = F.relu(self_node_info + agg_node_info)

        all_weight = [weight.unsqueeze(1) for weight in all_weight]
        all_weight = torch.cat(all_weight, dim=1)

        return node, all_weight


class ArgumentGCN_32(nn.Module):
    '''
    A variant of class ArgumentGCN_3 in that
        - no_edges and no_edges_reverse as two adjacency matrices are included.

    '''

    def __init__(self, node_dim, extra_factor_dim=0, iteration_steps=1):
        super(ArgumentGCN_32, self).__init__()

        self.node_dim = node_dim
        self.iteration_steps = iteration_steps

        self._node_weight_fc = torch.nn.Linear(node_dim + extra_factor_dim, 1, bias=True)

        self._self_node_fc = torch.nn.Linear(node_dim, node_dim, bias=True)

        self._node_fc_argument_1 = torch.nn.Linear(node_dim, node_dim,
                                                   bias=False)  # relation-specific transform matrices.
        self._node_fc_argument_2 = torch.nn.Linear(node_dim, node_dim,
                                                 bias=False)  # relation-specific transform matrices.
        self._node_fc_argument_3 = torch.nn.Linear(node_dim, node_dim,
                                                 bias=False)  # relation-specific transform matrices.
        self._node_fc_argument_4 = torch.nn.Linear(node_dim, node_dim,
                                                 bias=False)  # relation-specific transform matrices.
        self._node_fc_punctuation = torch.nn.Linear(node_dim, node_dim, bias=False)  # relation-specific transform matrices.

        self._node_fc_no_edges = torch.nn.Linear(node_dim, node_dim, bias=False)  # relation-specific transform matrices.

        self._node_fc_argument_1prime = torch.nn.Linear(node_dim, node_dim,
                                                   bias=False)  # relation-specific transform matrices.
        self._node_fc_argument_2prime = torch.nn.Linear(node_dim, node_dim,
                                                   bias=False)  # relation-specific transform matrices.
        self._node_fc_argument_3prime = torch.nn.Linear(node_dim, node_dim,
                                                   bias=False)  # relation-specific transform matrices.
        self._node_fc_argument_4prime = torch.nn.Linear(node_dim, node_dim,
                                                   bias=False)  # relation-specific transform matrices.
        self._node_fc_punctuationprime = torch.nn.Linear(node_dim, node_dim,
                                                    bias=False)  # relation-specific transform matrices.
        self._node_fc_no_edges_prime = torch.nn.Linear(node_dim, node_dim,
                                                    bias=False)  # relation-specific transform matrices.

    def forward(self,
                node, # span features. 1 node type. Homogeneous graph.
                node_mask,  # 1=is_node, 0=not_node. size=(bsz, n_node)
                adj_matrices: [list, list, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                # argument_graphs:list, # list of directed graph (asymmetric adjacency matrix). size=(bsz, n_node, n_node)
                # argument_graphs_reverse: list,
                # # list of directed graph (asymmetric adjacency matrix). size=(bsz, n_node, n_node)
                # punctuation_graph: torch.Tensor,  # directed edges (asymmetric adjacency matrix). size=(bsz, n_node, n_node)
                # punctuation_graph_reverse: torch.Tensor,  # directed edges (asymmetric adjacency matrix). size=(bsz, n_node, n_node)
                extra_factor=None):
        ''' '''
        '''
        10 Edge Types:
            1 - Comparison.*
            2 - Contingency.*
            3 - Expansion.*
            4 - Temporal.*
            5 - punctuation
            6 - no_edges
            1' - reversed Comparison.*  (t, h, r)
            2' - reversed Contingency.*  (t, h, r)
            3' - reversed Expansion.*  (t, h, r)
            4' - reversed Temporal.*  (t, h, r)
            5' - reversed punctuation  (t, h, r)
            6' - reversed no_edges
        '''

        argument_graphs, argument_graphs_reverse, punctuation_graph, punctuation_graph_reverse,\
            no_edges_graph, no_edges_graph_reverse = adj_matrices

        node_len = node.size(1)

        diagmat = torch.diagflat(torch.ones(node.size(1), dtype=torch.long,
                                            device=node.device))  # diagonal matrix. (n_node, n_node)
        diagmat = diagmat.unsqueeze(0).expand(node.size(0), -1, -1)  # (bsz, n_node, n_node).
        dd_graph = node_mask.unsqueeze(1) * node_mask.unsqueeze(-1) * (1 - diagmat)  # the last item: acyclic graph.

        # graph_argument = dd_graph * argument_graph  # dep
        graphs_argument = [dd_graph * item for item in argument_graphs]  # list of Long tensor.
        graphs_argument_re = [dd_graph * item for item in argument_graphs_reverse]  # list of Long tensor.
        graph_punctuation = dd_graph * punctuation_graph
        graph_punctuation_re = dd_graph * punctuation_graph_reverse
        graph_noedges = dd_graph * no_edges_graph
        graph_noedges_re = dd_graph * no_edges_graph_reverse

        # node_neighbor_num = graph_argument.sum(-1) + graph_punctuation.sum(-1)  # dep
        node_neighbor_num = sum([item.sum(-1) for item in graphs_argument]) + graph_punctuation.sum(-1) + \
                            sum([item.sum(-1) for item in graphs_argument_re]) + graph_punctuation_re.sum(-1) + \
                            graph_noedges.sum(-1) + graph_noedges_re.sum(-1)
        node_neighbor_num_mask = (node_neighbor_num >= 1).long()
        node_neighbor_num = util.replace_masked_values(node_neighbor_num.float(), node_neighbor_num_mask, 1)

        all_weight = []
        for step in range(self.iteration_steps):


            ''' (1) Node Relatedness Measure '''
            if extra_factor is None:
                d_node_weight = torch.sigmoid(self._node_weight_fc(node)).squeeze(-1)  # Eleanor. eq (10). alpha_i.
            else:
                d_node_weight = torch.sigmoid(self._node_weight_fc(torch.cat((node, extra_factor), dim=-1))).squeeze(-1)  # alpha_i.

            all_weight.append(d_node_weight)  # Eleanor. Here record the weights.

            self_node_info = self._self_node_fc(node)  # v_i.


            ''' (2) Message Propagation (each edge type) '''
            assert len(graphs_argument) == len(graphs_argument_re) == 4

            # type 1. argument edges 1.
            node_info_argument_1 = self._node_fc_argument_1(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graphs_argument[0],
                0)  # filtering out weights not type 1.
            node_info_argument_1 = torch.matmul(node_weight, node_info_argument_1)

            # type 1'. argument edge 1'.
            node_info_argument_1prime = self._node_fc_argument_1prime(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graphs_argument_re[0],
                0)  # filtering out weights not type 1'.
            node_info_argument_1prime = torch.matmul(node_weight, node_info_argument_1prime)

            # type 2. argument edges 2.
            node_info_argument_2 = self._node_fc_argument_2(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graphs_argument[1],
                0)  # filtering out weights not type 2.
            node_info_argument_2 = torch.matmul(node_weight, node_info_argument_2)

            # type 2'. argument edge 2'.
            node_info_argument_2prime = self._node_fc_argument_2prime(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graphs_argument_re[1],
                0)  # filtering out weights not type 2'.
            node_info_argument_2prime = torch.matmul(node_weight, node_info_argument_2prime)

            # type 3. argument edges 3.
            node_info_argument_3 = self._node_fc_argument_3(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graphs_argument[2],
                0)  # filtering out weights not type 3.
            node_info_argument_3 = torch.matmul(node_weight, node_info_argument_3)

            # type 3'. argument edge 3'.
            node_info_argument_3prime = self._node_fc_argument_3prime(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graphs_argument_re[2],
                0)  # filtering out weights not type 3'.
            node_info_argument_3prime = torch.matmul(node_weight, node_info_argument_3prime)

            # type 4. argument edges 4.
            node_info_argument_4 = self._node_fc_argument_4(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graphs_argument[3],
                0)  # filtering out weights not type 4
            node_info_argument_4 = torch.matmul(node_weight, node_info_argument_4)

            # type 4'. argument edge 4'.
            node_info_argument_4prime = self._node_fc_argument_4prime(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graphs_argument_re[3],
                0)  # filtering out weights not type 4'.
            node_info_argument_4prime = torch.matmul(node_weight, node_info_argument_4prime)

            # type 5. punctuation edges.
            node_info_punctuation = self._node_fc_punctuation(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graph_punctuation,
                0)  # filtering out weights not type 5.
            node_info_punctuation = torch.matmul(node_weight, node_info_punctuation)

            # type 5'. argument edge 5'.
            node_info_punctuationprime = self._node_fc_punctuationprime(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graph_punctuation_re,
                0)  # filtering out weights not type 5'.
            node_info_punctuationprime = torch.matmul(node_weight, node_info_punctuationprime)

            # type 6. no_edges.
            node_info_noedges = self._node_fc_no_edges(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graph_noedges,
                0)  # filtering out weights not type 6.
            node_info_noedges = torch.matmul(node_weight, node_info_noedges)

            # type 6'. no_edges reverse.
            node_info_noedgesprime = self._node_fc_no_edges_prime(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graph_noedges_re,
                0)  # filtering out weights not type 6'.
            node_info_noedgesprime = torch.matmul(node_weight, node_info_noedgesprime)

            agg_node_info = (node_info_argument_1 + node_info_argument_2 + node_info_argument_3 + node_info_argument_4 + \
                             node_info_argument_1prime + node_info_argument_2prime + node_info_argument_3prime + \
                             node_info_argument_4prime + \
                             node_info_punctuation + node_info_punctuationprime + \
                             node_info_noedges + node_info_noedgesprime) / node_neighbor_num.unsqueeze(-1)


            ''' (3) Node Representation Update '''
            node = F.relu(self_node_info + agg_node_info)

        all_weight = [weight.unsqueeze(1) for weight in all_weight]
        all_weight = torch.cat(all_weight, dim=1)

        return node, all_weight


class ArgumentGCN_33(nn.Module):
    '''
    A variant of class ArgumentGCN_3 in that
        - no_edges and no_edges_reverse as two adjacency matrices are included.

    '''

    def __init__(self, node_dim, extra_factor_dim=0, iteration_steps=1):
        super(ArgumentGCN_33, self).__init__()

        self.node_dim = node_dim
        self.iteration_steps = iteration_steps

        self._node_weight_fc = torch.nn.Linear(node_dim + extra_factor_dim, 1, bias=True)

        self._self_node_fc = torch.nn.Linear(node_dim, node_dim, bias=True)

        self._node_fc_argument_1 = torch.nn.Linear(node_dim, node_dim,
                                                   bias=False)  # relation-specific transform matrices.
        # self._node_fc_argument_2 = torch.nn.Linear(node_dim, node_dim,
        #                                          bias=False)  # relation-specific transform matrices.
        # self._node_fc_argument_3 = torch.nn.Linear(node_dim, node_dim,
        #                                          bias=False)  # relation-specific transform matrices.
        # self._node_fc_argument_4 = torch.nn.Linear(node_dim, node_dim,
        #                                          bias=False)  # relation-specific transform matrices.
        self._node_fc_punctuation = torch.nn.Linear(node_dim, node_dim, bias=False)  # relation-specific transform matrices.

        self._node_fc_no_edges = torch.nn.Linear(node_dim, node_dim, bias=False)  # relation-specific transform matrices.

        self._node_fc_argument_1prime = torch.nn.Linear(node_dim, node_dim,
                                                   bias=False)  # relation-specific transform matrices.
        # self._node_fc_argument_2prime = torch.nn.Linear(node_dim, node_dim,
        #                                            bias=False)  # relation-specific transform matrices.
        # self._node_fc_argument_3prime = torch.nn.Linear(node_dim, node_dim,
        #                                            bias=False)  # relation-specific transform matrices.
        # self._node_fc_argument_4prime = torch.nn.Linear(node_dim, node_dim,
        #                                            bias=False)  # relation-specific transform matrices.
        self._node_fc_punctuationprime = torch.nn.Linear(node_dim, node_dim,
                                                    bias=False)  # relation-specific transform matrices.
        self._node_fc_no_edges_prime = torch.nn.Linear(node_dim, node_dim,
                                                    bias=False)  # relation-specific transform matrices.

    def forward(self,
                node, # span features. 1 node type. Homogeneous graph.
                node_mask,  # 1=is_node, 0=not_node. size=(bsz, n_node)
                adj_matrices: [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                # argument_graphs:list, # list of directed graph (asymmetric adjacency matrix). size=(bsz, n_node, n_node)
                # argument_graphs_reverse: list,
                # # list of directed graph (asymmetric adjacency matrix). size=(bsz, n_node, n_node)
                # punctuation_graph: torch.Tensor,  # directed edges (asymmetric adjacency matrix). size=(bsz, n_node, n_node)
                # punctuation_graph_reverse: torch.Tensor,  # directed edges (asymmetric adjacency matrix). size=(bsz, n_node, n_node)
                extra_factor=None):
        ''' '''
        '''
        6 Edge Types:
            1 - Comparison.* & Contingency.* & Expansion.* & Temporal.*
            2 - punctuation
            3 - no_edges
            1' - reversed Comparison.* & Contingency.* & Expansion.* & Temporal.* (t, h, r)
            2' - reversed punctuation  (t, h, r)
            3' - reversed no_edges
        '''

        argument_graph_unite, argument_graph_unite_reverse, \
            punctuation_graph, punctuation_graph_reverse,\
            no_edges_graph, no_edges_graph_reverse = adj_matrices

        node_len = node.size(1)

        diagmat = torch.diagflat(torch.ones(node.size(1), dtype=torch.long,
                                            device=node.device))  # diagonal matrix. (n_node, n_node)
        diagmat = diagmat.unsqueeze(0).expand(node.size(0), -1, -1)  # (bsz, n_node, n_node).
        dd_graph = node_mask.unsqueeze(1) * node_mask.unsqueeze(-1) * (1 - diagmat)  # the last item: acyclic graph.

        # graph_argument = dd_graph * argument_graph  # dep
        # graphs_argument = [dd_graph * item for item in argument_graphs]  # list of Long tensor.
        # graphs_argument_re = [dd_graph * item for item in argument_graphs_reverse]  # list of Long tensor.
        graph_argument = dd_graph * argument_graph_unite
        graph_argument_re = dd_graph * argument_graph_unite_reverse
        graph_punctuation = dd_graph * punctuation_graph
        graph_punctuation_re = dd_graph * punctuation_graph_reverse
        graph_noedges = dd_graph * no_edges_graph
        graph_noedges_re = dd_graph * no_edges_graph_reverse

        # node_neighbor_num = graph_argument.sum(-1) + graph_punctuation.sum(-1)  # dep
        # node_neighbor_num = sum([item.sum(-1) for item in graphs_argument]) + graph_punctuation.sum(-1) + \
        #                     sum([item.sum(-1) for item in graphs_argument_re]) + graph_punctuation_re.sum(-1)
        node_neighbor_num = graph_argument.sum(-1) + graph_argument_re.sum(-1) + graph_punctuation.sum(-1) + \
                            graph_punctuation_re.sum(-1) + graph_noedges.sum(-1) + graph_noedges_re.sum(-1)
        node_neighbor_num_mask = (node_neighbor_num >= 1).long()
        node_neighbor_num = util.replace_masked_values(node_neighbor_num.float(), node_neighbor_num_mask, 1)

        all_weight = []
        for step in range(self.iteration_steps):


            ''' (1) Node Relatedness Measure '''
            if extra_factor is None:
                d_node_weight = torch.sigmoid(self._node_weight_fc(node)).squeeze(-1)  # Eleanor. eq (10). alpha_i.
            else:
                d_node_weight = torch.sigmoid(self._node_weight_fc(torch.cat((node, extra_factor), dim=-1))).squeeze(-1)  # alpha_i.

            all_weight.append(d_node_weight)  # Eleanor. Here record the weights.

            self_node_info = self._self_node_fc(node)  # v_i.


            ''' (2) Message Propagation (each edge type) '''
            # assert len(graphs_argument) == len(graphs_argument_re) == 4

            # type 1. argument edges 1.
            node_info_argument = self._node_fc_argument_1(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graph_argument,
                0)  # filtering out weights not type 1.
            node_info_argument = torch.matmul(node_weight, node_info_argument)

            # type 1'. argument edge 1'.
            node_info_argument_prime = self._node_fc_argument_1prime(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graph_argument_re,
                0)  # filtering out weights not type 1'.
            node_info_argument_prime = torch.matmul(node_weight, node_info_argument_prime)

            # type 2. punctuation edges.
            node_info_punctuation = self._node_fc_punctuation(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graph_punctuation,
                0)  # filtering out weights not type 2.
            node_info_punctuation = torch.matmul(node_weight, node_info_punctuation)

            # type 2'. argument edge 5'.
            node_info_punctuationprime = self._node_fc_punctuationprime(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graph_punctuation_re,
                0)  # filtering out weights not type 2'.
            node_info_punctuationprime = torch.matmul(node_weight, node_info_punctuationprime)

            # type 3. no_edges.
            node_info_noedges = self._node_fc_no_edges(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graph_noedges,
                0)  # filtering out weights not type 3.
            node_info_noedges = torch.matmul(node_weight, node_info_noedges)

            # type 3'. no_edges reverse.
            node_info_noedgesprime = self._node_fc_no_edges_prime(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graph_noedges_re,
                0)  # filtering out weights not type 3'.
            node_info_noedgesprime = torch.matmul(node_weight, node_info_noedgesprime)

            agg_node_info = (node_info_argument + node_info_argument_prime +
                             node_info_punctuation + node_info_punctuationprime + \
                             node_info_noedges + node_info_noedgesprime) / node_neighbor_num.unsqueeze(-1)


            ''' (3) Node Representation Update '''
            node = F.relu(self_node_info + agg_node_info)

        all_weight = [weight.unsqueeze(1) for weight in all_weight]
        all_weight = torch.cat(all_weight, dim=1)

        return node, all_weight


class ArgumentGCN_34(nn.Module):
    '''
    A variant of class ArgumentGCN_3 in that
        - no_edges and no_edges_reverse as two adjacency matrices are included.

    '''

    def __init__(self, node_dim, extra_factor_dim=0, iteration_steps=1):
        super(ArgumentGCN_34, self).__init__()

        self.node_dim = node_dim
        self.iteration_steps = iteration_steps

        self._node_weight_fc = torch.nn.Linear(node_dim + extra_factor_dim, 1, bias=True)

        self._self_node_fc = torch.nn.Linear(node_dim, node_dim, bias=True)

        self._node_fc_argument_1 = torch.nn.Linear(node_dim, node_dim,
                                                   bias=False)  # relation-specific transform matrices.
        self._node_fc_no_edges = torch.nn.Linear(node_dim, node_dim, bias=False)  # relation-specific transform matrices.

        self._node_fc_argument_1prime = torch.nn.Linear(node_dim, node_dim,
                                                   bias=False)  # relation-specific transform matrices.
        self._node_fc_no_edges_prime = torch.nn.Linear(node_dim, node_dim,
                                                    bias=False)  # relation-specific transform matrices.

    def forward(self,
                node, # span features. 1 node type. Homogeneous graph.
                node_mask,  # 1=is_node, 0=not_node. size=(bsz, n_node)
                adj_matrices: [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                extra_factor=None):
        ''' '''
        '''
        4 Edge Types:
            1 - Comparison.* & Contingency.* & Expansion.* & Temporal.* & punctutaion 
            2 - no_edges
            1' - reversed Comparison.* & Contingency.* & Expansion.* & Temporal.* & punctuation (t, h, r)
            2' - reversed no_edges
        '''

        argument_graph_unite, argument_graph_unite_reverse, \
            no_edges_graph, no_edges_graph_reverse = adj_matrices

        node_len = node.size(1)

        diagmat = torch.diagflat(torch.ones(node.size(1), dtype=torch.long,
                                            device=node.device))  # diagonal matrix. (n_node, n_node)
        diagmat = diagmat.unsqueeze(0).expand(node.size(0), -1, -1)  # (bsz, n_node, n_node).
        dd_graph = node_mask.unsqueeze(1) * node_mask.unsqueeze(-1) * (1 - diagmat)  # the last item: acyclic graph.

        # graph_argument = dd_graph * argument_graph  # dep
        # graphs_argument = [dd_graph * item for item in argument_graphs]  # list of Long tensor.
        # graphs_argument_re = [dd_graph * item for item in argument_graphs_reverse]  # list of Long tensor.
        graph_argument = dd_graph * argument_graph_unite
        graph_argument_re = dd_graph * argument_graph_unite_reverse
        graph_noedges = dd_graph * no_edges_graph
        graph_noedges_re = dd_graph * no_edges_graph_reverse

        # node_neighbor_num = graph_argument.sum(-1) + graph_punctuation.sum(-1)  # dep
        # node_neighbor_num = sum([item.sum(-1) for item in graphs_argument]) + graph_punctuation.sum(-1) + \
        #                     sum([item.sum(-1) for item in graphs_argument_re]) + graph_punctuation_re.sum(-1)
        node_neighbor_num = graph_argument.sum(-1) + graph_argument_re.sum(-1) + \
                            graph_noedges.sum(-1) + graph_noedges_re.sum(-1)
        node_neighbor_num_mask = (node_neighbor_num >= 1).long()
        node_neighbor_num = util.replace_masked_values(node_neighbor_num.float(), node_neighbor_num_mask, 1)

        all_weight = []
        for step in range(self.iteration_steps):


            ''' (1) Node Relatedness Measure '''
            if extra_factor is None:
                d_node_weight = torch.sigmoid(self._node_weight_fc(node)).squeeze(-1)  # Eleanor. eq (10). alpha_i.
            else:
                d_node_weight = torch.sigmoid(self._node_weight_fc(torch.cat((node, extra_factor), dim=-1))).squeeze(-1)  # alpha_i.

            all_weight.append(d_node_weight)  # Eleanor. Here record the weights.

            self_node_info = self._self_node_fc(node)  # v_i.


            ''' (2) Message Propagation (each edge type) '''
            # assert len(graphs_argument) == len(graphs_argument_re) == 4

            # type 1. argument edges 1.
            node_info_argument = self._node_fc_argument_1(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graph_argument,
                0)  # filtering out weights not type 1.
            node_info_argument = torch.matmul(node_weight, node_info_argument)

            # type 1'. argument edge 1'.
            node_info_argument_prime = self._node_fc_argument_1prime(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graph_argument_re,
                0)  # filtering out weights not type 1'.
            node_info_argument_prime = torch.matmul(node_weight, node_info_argument_prime)

            # type 3. no_edges.
            node_info_noedges = self._node_fc_no_edges(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graph_noedges,
                0)  # filtering out weights not type 3.
            node_info_noedges = torch.matmul(node_weight, node_info_noedges)

            # type 3'. no_edges reverse.
            node_info_noedgesprime = self._node_fc_no_edges_prime(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graph_noedges_re,
                0)  # filtering out weights not type 3'.
            node_info_noedgesprime = torch.matmul(node_weight, node_info_noedgesprime)

            agg_node_info = (node_info_argument + node_info_argument_prime +
                             node_info_noedges + node_info_noedgesprime) / node_neighbor_num.unsqueeze(-1)


            ''' (3) Node Representation Update '''
            node = F.relu(self_node_info + agg_node_info)

        all_weight = [weight.unsqueeze(1) for weight in all_weight]
        all_weight = torch.cat(all_weight, dim=1)

        return node, all_weight


class ArgumentGCN_5(nn.Module):
    '''
        Considering both relation patterns and edge types.

    '''
    def __init__(self, node_dim, extra_factor_dim=0, iteration_steps=1):
        super(ArgumentGCN_5, self).__init__()

        self.node_dim = node_dim
        self.iteration_steps = iteration_steps

        self._node_weight_fc = torch.nn.Linear(node_dim + extra_factor_dim, 1, bias=True)

        self._self_node_fc = torch.nn.Linear(node_dim, node_dim, bias=True)

        self._node_fc_argument_1 = torch.nn.Linear(node_dim, node_dim,
                                                   bias=False)  # relation-specific transform matrices.
        self._node_fc_argument_2 = torch.nn.Linear(node_dim, node_dim,
                                                 bias=False)  # relation-specific transform matrices.
        self._node_fc_argument_3 = torch.nn.Linear(node_dim, node_dim,
                                                 bias=False)  # relation-specific transform matrices.
        self._node_fc_argument_4 = torch.nn.Linear(node_dim, node_dim,
                                                 bias=False)  # relation-specific transform matrices.
        self._node_fc_punctuation = torch.nn.Linear(node_dim, node_dim, bias=False)  # relation-specific transform matrices.

        self._node_fc_punctuationprime = torch.nn.Linear(node_dim, node_dim,
                                                    bias=False)  # relation-specific transform matrices.

    def forward(self,
                node, # span features. 1 node type. Homogeneous graph.
                node_mask,  # 1=is_node, 0=not_node. size=(bsz, n_node)
                argument_graphs:list, # list of directed graph (asymmetric adjacency matrix). size=(bsz, n_node, n_node)
                punctuation_graph,  # directed edges (asymmetric adjacency matrix). size=(bsz, n_node, n_node)
                punctuation_graph_reverse,  # directed edges (asymmetric adjacency matrix). size=(bsz, n_node, n_node)
                extra_factor=None):
        ''' '''
        '''
        10 Edge Types:
            1 - Comparison.*
            2 - Contingency.*
            3 - Expansion.*
            4 - Temporal.*
            5 - punctuation
            1' - Comparison.*  (t, h, r)
            2' - Contingency.*  (t, h, r)
            3' - Expansion.*  (t, h, r)
            4' - Temporal.*  (t, h, r)
            5' - punctuation  (t, h, r)
        '''

        node_len = node.size(1)

        diagmat = torch.diagflat(torch.ones(node.size(1), dtype=torch.long,
                                            device=node.device))  # diagonal matrix. (n_node, n_node)
        diagmat = diagmat.unsqueeze(0).expand(node.size(0), -1, -1)  # (bsz, n_node, n_node).
        dd_graph = node_mask.unsqueeze(1) * node_mask.unsqueeze(-1) * (1 - diagmat)  # the last item: acyclic graph.

        # graph_argument = dd_graph * argument_graph  # dep
        graphs_argument = [dd_graph * item for item in argument_graphs]  # list of Long tensor. The last 2 are of type=7.
        graph_punctuation = dd_graph * punctuation_graph
        graph_punctuation_re = dd_graph * punctuation_graph_reverse

        # node_neighbor_num = graph_argument.sum(-1) + graph_punctuation.sum(-1)  # dep
        node_neighbor_num = sum([item.sum(-1) for item in graphs_argument]) + graph_punctuation.sum(-1) + \
                            + graph_punctuation_re.sum(-1)
        node_neighbor_num_mask = (node_neighbor_num >= 1).long()
        node_neighbor_num = util.replace_masked_values(node_neighbor_num.float(), node_neighbor_num_mask, 1)

        all_weight = []
        for step in range(self.iteration_steps):


            ''' (1) Node Relatedness Measure '''
            if extra_factor is None:
                d_node_weight = torch.sigmoid(self._node_weight_fc(node)).squeeze(-1)  # Eleanor. eq (10). alpha_i.
            else:
                d_node_weight = torch.sigmoid(self._node_weight_fc(torch.cat((node, extra_factor), dim=-1))).squeeze(-1)  # alpha_i.

            all_weight.append(d_node_weight)  # Eleanor. Here record the weights.

            self_node_info = self._self_node_fc(node)  # v_i.


            ''' (2) Message Propagation (each edge type) '''
            assert len(graphs_argument) == 4

            # type 1. argument edges 1.
            node_info_argument_1 = self._node_fc_argument_1(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graphs_argument[0],
                0)  # filtering out weights not type 1.
            node_info_argument_1 = torch.matmul(node_weight, node_info_argument_1)

            # type 2. argument edges 2.
            node_info_argument_2 = self._node_fc_argument_2(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graphs_argument[1],
                0)  # filtering out weights not type 2.
            node_info_argument_2 = torch.matmul(node_weight, node_info_argument_2)

            # type 3. argument edges 3.
            node_info_argument_3 = self._node_fc_argument_3(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graphs_argument[2],
                0)  # filtering out weights not type 3.
            node_info_argument_3 = torch.matmul(node_weight, node_info_argument_3)

            # type 4. argument edges 4.
            node_info_argument_4 = self._node_fc_argument_4(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graphs_argument[3],
                0)  # filtering out weights not type 4
            node_info_argument_4 = torch.matmul(node_weight, node_info_argument_4)

            # type 5. punctuation edges.
            node_info_punctuation = self._node_fc_punctuation(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graph_punctuation,
                0)  # filtering out weights not type 5.
            node_info_punctuation = torch.matmul(node_weight, node_info_punctuation)

            # type 5'. argument edge 5'.
            node_info_punctuationprime = self._node_fc_punctuationprime(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graph_punctuation_re,
                0)  # filtering out weights not type 5'.
            node_info_punctuationprime = torch.matmul(node_weight, node_info_punctuationprime)

            agg_node_info = (node_info_argument_1 + node_info_argument_2 + node_info_argument_3 + node_info_argument_4 + \
                             node_info_punctuation + node_info_punctuationprime) / node_neighbor_num.unsqueeze(-1)

            ''' (3) Node Representation Update '''
            node = F.relu(self_node_info + agg_node_info)

        all_weight = [weight.unsqueeze(1) for weight in all_weight]
        all_weight = torch.cat(all_weight, dim=1)

        return node, all_weight