import torch
from torch.nn import Linear as Lin
import torch_geometric as tg
from gcn_lib.sparse import MultiSeq, MLP, GraphConv, ResDynBlock, DenseDynBlock, DilatedKnnGraph, BiFPN, BiFPNModule

class SparseDeepGCN(torch.nn.Module):
	def __init__(self):
		super(SparseDeepGCN, self).__init__()
		channels = 32 #32
		k = 16
		act = 'relu'
		norm = 'batch'
		bias = True
		epsilon = 0.8
		stochastic = True
		conv = 'edge'
		c_growth = channels
		in_channels = 32
		block = 'res'
		dropout = 0.8 
		n_classes = 13
		levels = 3
		self.levels = levels
		self.n_blocks = 5 #7

		# self.knn = DilatedKnnGraph(k, 1, stochastic, epsilon)
		self.head = GraphConv(in_channels, channels, conv, act, norm, bias)

		if block == 'res':
			self.backbone = MultiSeq(*[ResDynBlock(channels, k, 1, conv, act, norm, bias, stochastic=stochastic, epsilon=epsilon)
									   for i in range(self.n_blocks-1)])
		else:
			raise NotImplementedError('{} is not implemented. Please check.\n'.format(block))

		# complete the bifpn code
		#self.bifpn = BiFPN(channels, stack = 1, conv_cfg = None, norm_cfg = None, activation=None, levels=levels, init = 0.5, eps = 0.0001)

		self.bifpn = BiFPNModule(channels, levels, init = 0.5, conv_cfg = None, norm_cfg = None, activation = None, eps = 0.0001)

		self.fusion_block = MLP([channels * levels, 512], act, norm, bias)
		"""
		self.prediction = MultiSeq(*[MLP([1024, 512], act, norm, bias),
							 MLP([512, 256], act, norm, bias),
							 torch.nn.Dropout(p=dropout),
							 MLP([256, n_classes], None, None, bias)])
		"""
		"""
		self.prediction = MultiSeq(*[MLP([channels+c_growth*(self.n_blocks-1)+32, 256], act, norm, bias),  #64 128
									 MLP([256, 64], act, norm, bias),										#128 64
									 torch.nn.Dropout(p=dropout),
									 MLP([64, n_classes], None, None, bias)])
		"""
		self.prediction = MultiSeq(*[MLP([512, 256], act, norm, bias),  #64 128
									 MLP([256, 64], act, norm, bias),										#128 64
									 torch.nn.Dropout(p=dropout),
									 MLP([64, n_classes], None, None, bias)])
		self.model_init()

	def model_init(self):
		for m in self.modules():
			if isinstance(m, Lin):
				torch.nn.init.kaiming_normal_(m.weight)
				m.weight.requires_grad = True
				if m.bias is not None:
					m.bias.data.zero_()
					m.bias.requires_grad = True

	def forward(self, x, edge_index, edge_attr):
		# corr, color, batch = data.pos, data.x, data.batch
		# batch is index
		# corr is coordinate of points
		# x is color information
		# here x becomes feature matrix

		feats = [self.head(x, edge_index)]
		for i in range(self.n_blocks-1):
			feats.append(self.backbone[i](feats[-1])[0])
		
		fusion = self.bifpn(feats[self.n_blocks-self.levels:])
		#print('fusion', fusion[0].shape, fusion[2].shape, fusion[3].shape, len(fusion))
		fusion = torch.cat(fusion, dim=1)
		#print('fusion', fusion[0].shape, len(fusion))
		fusion = self.fusion_block(fusion)
		#feats = torch.cat(feats, dim=1)

		# print("fusion", self.prediction(torch.cat((fusion, feats), dim=1)).shape)
		#fusion = torch.repeat_interleave(fusion, repeats=feats.shape[0]//fusion.shape[0], dim=0)
		#return self.prediction(torch.cat((fusion, feats), dim=1))
		return self.prediction(fusion)

