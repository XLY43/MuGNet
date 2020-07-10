"""
    Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs
    http://arxiv.org/abs/1711.09869
    2017 Loic Landrieu, Martin Simonovsky
"""
from __future__ import division
from __future__ import print_function
from builtins import range

import torch
import torch.nn as nn
import torch.nn.init as init
from learning import ecc
from learning.modules import RNNGraphConvModule, ECC_CRFModule, GRUCellEx, LSTMCellEx
from torch_geometric.nn import GatedGraphConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import uniform
import torch.nn.functional as nnf


class GRUCellEx(nn.GRUCell):
    """ Usual GRU cell extended with layer normalization and input gate.
    """
    def __init__(self, input_size, hidden_size, bias=True, layernorm=True, ingate=True):
        super(GRUCellEx, self).__init__(input_size, hidden_size, bias)
        self._layernorm = layernorm
        self._ingate = ingate
        if layernorm:
            self.add_module('ini', nn.InstanceNorm1d(1, eps=1e-5, affine=False, track_running_stats=False))
            self.add_module('inh', nn.InstanceNorm1d(1, eps=1e-5, affine=False, track_running_stats=False))
        if ingate:
            self.add_module('ig', nn.Linear(hidden_size, input_size, bias=True))

    def _normalize(self, gi, gh):
        if self._layernorm: # layernorm on input&hidden, as in https://arxiv.org/abs/1607.06450 (Layer Normalization)
            gi = self._modules['ini'](gi.unsqueeze(1)).squeeze(1)
            gh = self._modules['inh'](gh.unsqueeze(1)).squeeze(1)
        return gi, gh

    def forward(self, input, hidden):
        if self._ingate:
            input = torch.sigmoid(self._modules['ig'](hidden)) * input

        # GRUCell in https://github.com/pytorch/pytorch/blob/master/torch/nn/_functions/rnn.py extended with layer normalization
        if input.is_cuda and torch.__version__.split('.')[0]=='0':
            gi = nnf.linear(input, self.weight_ih)
            gh = nnf.linear(hidden, self.weight_hh)
            gi, gh = self._normalize(gi, gh)
            state = torch.nn._functions.thnn.rnnFusedPointwise.GRUFused
            try: #pytorch >=0.3
                return state.apply(gi, gh, hidden) if self.bias_ih is None else state.apply(gi, gh, hidden, self.bias_ih, self.bias_hh)
            except: #pytorch <=0.2
                return state()(gi, gh, hidden) if self.bias_ih is None else state()(gi, gh, hidden, self.bias_ih, self.bias_hh)
        gi = nnf.linear(input, self.weight_ih)
        gh = nnf.linear(hidden, self.weight_hh)
        gi, gh = self._normalize(gi, gh)
        i_r, i_i, i_n = gi.chunk(3, 1)
        h_r, h_i, h_n = gh.chunk(3, 1)
        bih_r, bih_i, bih_n = self.bias_ih.chunk(3)
        bhh_r, bhh_i, bhh_n = self.bias_hh.chunk(3)

        resetgate = torch.sigmoid(i_r + bih_r + h_r + bhh_r)
        inputgate = torch.sigmoid(i_i + bih_i + h_i + bhh_i)
        newgate = torch.tanh(i_n + bih_n + resetgate * (h_n + bhh_n))
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def __repr__(self):
        s = super(GRUCellEx, self).__repr__() + '('
        if self._ingate:
            s += 'ingate'
        if self._layernorm:
            s += ' layernorm'
        return s + ')'


class NNConv(MessagePassing):
    r"""The continuous kernel-based convolutional operator from the
    `"Neural Message Passing for Quantum Chemistry"
    <https://arxiv.org/abs/1704.01212>`_ paper.
    This convolution is also known as the edge-conditioned convolution from the
    `"Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on
    Graphs" <https://arxiv.org/abs/1704.02901>`_ paper (see
    :class:`torch_geometric.nn.conv.ECConv` for an alias):

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta} \mathbf{x}_i +
        \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \cdot
        h_{\mathbf{\Theta}}(\mathbf{e}_{i,j}),

    where :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.*
    a MLP.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps edge features :obj:`edge_attr` of shape :obj:`[-1,
            num_edge_features]` to shape
                :obj:`[-1, in_channels * out_channels]`, *e.g.*, defined by
                :class:`torch.nn.Sequential`.
            aggr (string, optional): The aggregation scheme to use
                (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
                (default: :obj:`"add"`)
            root_weight (bool, optional): If set to :obj:`False`, the layer will
                not add the transformed root node features to the output.
                (default: :obj:`True`)
            bias (bool, optional): If set to :obj:`False`, the layer will not learn
                an additive bias. (default: :obj:`True`)
            **kwargs (optional): Additional arguments of
                :class:`torch_geometric.nn.conv.MessagePassing`.
        """
    def __init__(self,
                in_channels,
                out_channels,
                aggr='mean',
                root_weight=False,
                bias=False,
                vv=True,
                flow="target_to_source",
                negative_slope=0.2,
                softmax=False,
                **kwargs):
        super(NNConv, self).__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggr = aggr
        self.vv = vv
        self.negative_slope = negative_slope
        self.softmax = softmax

        if root_weight:
            self.root = Parameter(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.in_channels, self.root)
        uniform(self.in_channels, self.bias)

    def forward(self, x, edge_index, weights):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        return self.propagate(edge_index, x=x, weights=weights)

    def message(self, edge_index_i, x_j, size_i, weights):
        if not self.vv:
            weight = weights.view(-1, self.in_channels, self.out_channels)
            if self.softmax: # APPLY A TWO DIMENSIONAL NON-DEPENDENT SPARSE SOFTMAX
                weight = F.leaky_relu(weight, self.negative_slope)
                weight = torch.cat([softmax(weight[:, k, :], edge_index_i, size_i).unsqueeze(1) for k in range(self.out_channels)], dim=1)
            return torch.matmul(x_j.unsqueeze(1), weight).squeeze(1)
        else:
            weight = weights.view(-1, self.in_channels)
            if self.softmax:
                weight = F.leaky_relu(weight, self.negative_slope)
                weight = torch.cat([softmax(w.unsqueeze(-1), edge_index_i, size_i).t() for w in weight.t()], dim=0).t()
            return x_j *  weight

    def update(self, aggr_out, x):
        if self.root is not None:
            aggr_out = aggr_out + torch.mm(x, self.root)
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                    self.out_channels)



def create_fnet(widths, orthoinit, llbias, bnidx=-1):
    """ Creates feature-generating network, a multi-layer perceptron.
    Parameters:
    widths: list of widths of layers (including input and output widths)
    orthoinit: whether to use orthogonal weight initialization
    llbias: whether to use bias in the last layer
    bnidx: index of batch normalization (-1 if not used)
    """
    fnet_modules = []
    for k in range(len(widths)-2):
        fnet_modules.append(nn.Linear(widths[k], widths[k+1]))
        if orthoinit: init.orthogonal_(fnet_modules[-1].weight, gain=init.calculate_gain('relu'))
        if bnidx==k: fnet_modules.append(nn.BatchNorm1d(widths[k+1]))
        fnet_modules.append(nn.ReLU(True))
    fnet_modules.append(nn.Linear(widths[-2], widths[-1], bias=llbias))
    if orthoinit: init.orthogonal_(fnet_modules[-1].weight)
    if bnidx==len(widths)-1: fnet_modules.append(nn.BatchNorm1d(fnet_modules[-1].weight.size(0)))
    return nn.Sequential(*fnet_modules)



class GraphNetwork(nn.Module):
    """ It is constructed in a flexible way based on `config` string, which contains sequence of comma-delimited layer definiton tokens layer_arg1_arg2_... See README.md for examples.
    """
    def __init__(self, config, nfeat, fnet_widths, fnet_orthoinit=True, fnet_llbias=True, fnet_bnidx=-1, edge_mem_limit=1e20, use_pyg = True, cuda = True):
        super(GraphNetwork, self).__init__()
        self.gconvs = []
        self.net = nn.Sequential()
        self._nrepeats = 10
        for d, conf in enumerate(config.split(',')):
            conf = conf.strip().split('_')
            #print(d, conf)
            if conf[0]=='f':    #Fully connected layer;  args: output_feats
                self.net.add_module('f', nn.Linear(96, int(conf[1])))
                #self.gconvs.append(nn.Linear(nfeat, int(conf[1])))
                nfeat = int(conf[1])
            elif conf[0]=='b':  #Batch norm;             args: not_affine
                self.add_module(str(d), nn.BatchNorm1d(nfeat, eps=1e-5, affine=len(conf)==1))
            elif conf[0]=='r':  #ReLU;
                self.add_module(str(d), nn.ReLU(True))
            elif conf[0]=='d':  #Dropout;                args: dropout_prob
                self.add_module(str(d), nn.Dropout(p=float(conf[1]), inplace=False))
            
            elif conf[0]=='gru' or conf[0]=='lstm': #RNN-ECC     args: repeats, mv=False, layernorm=True, ingate=True, cat_all=True
                nrepeats = int(conf[1])
                vv = bool(int(conf[2])) if len(conf)>2 else True # whether ECC does matrix-value mult or element-wise mult
                layernorm = bool(int(conf[3])) if len(conf)>3 else True
                ingate = bool(int(conf[4])) if len(conf)>4 else True
                cat_all = bool(int(conf[5])) if len(conf)>5 else True
                self.layers = 4

                self.egru = nn.ModuleDict({"egru{}".format(i):EGRU(fnet_widths, nfeat, vv, fnet_orthoinit, fnet_llbias, fnet_bnidx, layernorm, ingate, 10) for i in range(self.layers)})

                self.egru_bi = nn.ModuleDict({"egru_bi{}".format(i):EGRU(fnet_widths, nfeat, vv, fnet_orthoinit, fnet_llbias, fnet_bnidx, layernorm, ingate, 10) for i in range(self.layers-2)})


    def set_info(self, gc_infos, cuda):
         
        #    Provides convolution modules with graph structure information for the current batch.
        
        gc_infos = gc_infos if isinstance(gc_infos,(list,tuple)) else [gc_infos]
        #for i,gc in enumerate(self.gconvs):
        #    #print(i, gc)
        #    if cuda: gc_infos[i].cuda()
        #    gc.set_info(gc_infos[i])
        self._gci = gc_infos[0]
    #def set_info(self, gc_info):


    def forward(self, hx, edge_index, edge_attr):
        idxn, idxe, degs, degs_gpu, edgefeats = self._gci.get_buffers()
        edge_indexes = self._gci.get_pyg_buffers()
        hx, edge_indexes, edge_attr , edgefeats= hx.cuda(), edge_indexes.cuda(), edge_attr.cuda(), edgefeats.cuda()

        hxs_out, hxbis_out = [], []
        for i in range(self.layers):
            hxs, hx = self.egru["egru{}".format(i)](hx, edge_indexes, edgefeats)
            hxs_out.append(hx)
        

        for i in range(self.layers-2):
            if i == 0:
                hxbis, hxbi = self.egru_bi["egru_bi{}".format(i)](hxs_out[self.layers-2-i]+hxs_out[self.layers-2-i+1], edge_indexes, edgefeats)
            else:
                hxbis, hxbi = self.egru_bi["egru_bi{}".format(i)](hxs_out[self.layers-2-i]+hxbis_out[i-1], edge_indexes, edgefeats)
            hxbis_out.append(hxbi)    
        hxbis_out = torch.cat(hxbis_out, 1)
        #hxs_out = torch.cat(hxs_out, 1)
        outs = [hx, hxbis_out]
        outs = torch.cat(outs, 1)
        #eval
        out = self.net.f(outs)
        return out


class EGRU(nn.Module):

    def __init__(self, fnet_widths, nfeat, vv, fnet_orthoinit, fnet_llbias, fnet_bnidx, layernorm, ingate, nrepeats):
        super(EGRU, self).__init__()
        self.fnet = create_fnet(fnet_widths + [nfeat**2 if not vv else nfeat], fnet_orthoinit, fnet_llbias, fnet_bnidx)
        self.gru = GRUCellEx(nfeat, nfeat, bias=True, layernorm=layernorm, ingate=ingate)
        self.nn = NNConv(nfeat, nfeat, vv=vv)
        self.nrepeats = nrepeats
    def forward(self, hx, edge_indexes, edgefeats):
        weights = self.fnet(edgefeats)
        hxs = [hx]
        nc = hx.size(1)      
        assert hx.dim()==2 and weights.dim()==2 and weights.size(1) in [nc, nc*nc]  
        if weights.size(1) != nc:
            weights = weights.view(-1, nc, nc)

        for r in range(self.nrepeats):
            input = self.nn(hx, edge_indexes, weights)
            hx = self.gru(input, hx)
            hxs.append(hx)
        hxs = torch.cat(hxs,1)
        return hxs, hx





























