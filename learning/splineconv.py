from __future__ import division
from __future__ import print_function
from builtins import range

import torch
import torch.nn.functional as F
from torch_geometric.datasets import MNISTSuperpixels
from torch.nn import BatchNorm1d, Linear
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.nn import SplineConv, voxel_grid, max_pool, max_pool_x, GATConv, GatedGraphConv, GINConv, TopKPooling, SAGEConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from learning import ecc
from torch.nn import init
#from dgl.nn.pytorch import GATConv, GatedGraphConv
import dgl
"""
class splineconv(torch.nn.Module):
	def __init__(self):
		super(Net, self).__init__()
 
		self.conv1 = SAGEConv(32, 128)
		self.pool1 = TopKPooling(128, ratio=0.8)
		self.conv2 = SAGEConv(128, 128)
		self.pool2 = TopKPooling(128, ratio=0.8)
		self.conv3 = SAGEConv(128, 128)
		self.pool3 = TopKPooling(128, ratio=0.8)
		self.item_embedding = torch.nn.Embedding(num_embeddings=df.item_id.max() +1, embedding_dim=embed_dim)
		self.lin1 = torch.nn.Linear(256, 128)
		self.lin2 = torch.nn.Linear(128, 64)
		self.lin3 = torch.nn.Linear(64, 1)
		self.bn1 = torch.nn.BatchNorm1d(128)
		self.bn2 = torch.nn.BatchNorm1d(64)
		self.act1 = torch.nn.ReLU()
		self.act2 = torch.nn.ReLU()        

	def set_info(self, gc_infos, cuda):
		#Provides convolution modules with graph structure information for the current batch.
		
		gc_infos = gc_infos if isinstance(gc_infos,(list,tuple)) else [gc_infos]
		for i,gc in enumerate(self.gconvs):
			if cuda: gc_infos[i].cuda()
			gc.set_info(gc_infos[i])
			#print("gc_info[i]",i, gc_infos[i])
			#print("gc", gc)
  
	def forward(self,  x, edge_index, edge_attr):
		x, edge_index, batch = data.x, data.edge_index, data.batch
		x = self.item_embedding(x)
		x = x.squeeze(1)        
 
		x = F.relu(self.conv1(x, edge_index))
 
		x, edge_index, _, batch, _ = self.pool1(x, edge_index, None, batch)
		x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
 
		x = F.relu(self.conv2(x, edge_index))
	 
		x, edge_index, _, batch, _ = self.pool2(x, edge_index, None, batch)
		x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
 
		x = F.relu(self.conv3(x, edge_index))
 
		x, edge_index, _, batch, _ = self.pool3(x, edge_index, None, batch)
		x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
 
		x = x1 + x2 + x3
 
		x = self.lin1(x)
		x = self.act1(x)
		x = self.lin2(x)
		x = self.act2(x)      
		x = F.dropout(x, p=0.5, training=self.training)
 
		x = torch.sigmoid(self.lin3(x)).squeeze(1)
 
		return x



Mar.24th
class splineconv(torch.nn.Module):
	def __init__(self):
		super(splineconv, self).__init__()
		self.conv1 = SplineConv(32, 32, dim=6, kernel_size=3)
		self.conv2 = SplineConv(32, 64, dim=6, kernel_size=3)
		self.bn1 = BatchNorm1d(64)

		self.conv3 = SplineConv(64, 128, dim=6, kernel_size=5)
		self.conv4 = SplineConv(128, 64, dim=6, kernel_size=5)
		self.bn2 = BatchNorm1d(64)

		self.conv5 = SplineConv(64, 32, dim=6, kernel_size=5)
		self.conv6 = SplineConv(32, 32, dim=6, kernel_size=5)
		self.conv7 = SplineConv(32, 13, dim=6, kernel_size=7)

	def forward(self, x, edge_index, edge_attr):
		x = F.elu(self.conv1(x, edge_index, edge_attr))
		x = self.bn1(self.conv2(x, edge_index, edge_attr))
		x = F.elu(self.conv3(x, edge_index, edge_attr))
		x = self.bn2(self.conv4(x, edge_index, edge_attr))
		x = F.elu(self.conv5(x, edge_index, edge_attr))
		x = self.conv6(x, edge_index, edge_attr)
		x = F.elu(self.conv7(x, edge_index, edge_attr))
		# x = F.dropout(x, training = self.training)
		return F.log_softmax(x, dim=1)

Mar. 27th GPU0
class splineconv(torch.nn.Module):
	def __init__(self):
		super(splineconv, self).__init__()
		self.conv1 = GATConv(32, 256, heads=3, dropout=0.2)
		self.conv2 = GATConv(768, 256, heads=3, dropout=0.2)

		self.conv3 = GATConv(768, 256, heads=3, dropout=0.2)
		self.conv4 = GATConv(768, 256, heads=1)
		self.conv5 = GATConv(256, 256, heads=1, dropout=0.2)
		self.classify_layer = Linear(256, 13)
		init.xavier_uniform_(self.classify_layer.weight)

	def forward(self, x, edge_index):
		x = self.conv1(x, edge_index)
		#print(x.shape)
		x = F.elu(self.conv2(x, edge_index))
		#print(x.shape)
		x = self.conv3(x, edge_index)
		#print(x.shape)
		x = F.elu(self.conv4(x, edge_index))
		x = self.conv5(x, edge_index)

		scores = self.classify_layer(x)
		# x = F.dropout(x, training = self.training)
		return F.log_softmax(scores, dim=1)


class splineconv(torch.nn.Module):
	def __init__(self):
		super(splineconv, self).__init__()
		self.conv1 = GATConv(32, 256, heads=3, dropout=0.4)
		self.conv2 = GATConv(768, 256, heads=3, dropout=0.4)
		self.conv3 = GATConv(768, 256, heads=3, dropout=0.4)
		self.conv4 = GATConv(768, 256, heads=1)
		self.conv5 = GATConv(256, 256, heads=1, dropout=0.4)
		self.bn1 = BatchNorm1d(256)
		self.lin1 = Linear(256, 128)
		self.lin2 = Linear(128, 128)
		self.classify_layer = Linear(128, 13)
		init.xavier_uniform_(self.classify_layer.weight)

	def forward(self, x, edge_index):
		x = self.conv1(x, edge_index)
		#print(x.shape)
		x = F.elu(self.conv2(x, edge_index))
		#print(x.shape)
		x = self.conv3(x, edge_index)
		#print(x.shape)
		x = F.elu(self.conv4(x, edge_index))
		x = self.conv5(x, edge_index)
		x = self.lin1(F.relu(self.bn1(x)))
		x = self.lin2(F.relu(x))
		scores = self.classify_layer(x)
		# x = F.dropout(x, training = self.training)
		return F.log_softmax(scores, dim=1)

class splineconv(torch.nn.Module):
	def __init__(self):
		super(splineconv, self).__init__()
		self.conv1 = GATConv(32, 256, heads=4, dropout=0.2)
		self.conv2 = GATConv(1024, 256, heads=4, dropout=0.2)
		self.conv3 = GATConv(1024, 256, heads=4, dropout=0.2)
		self.conv4 = GATConv(1024, 256, heads=1)
		self.conv5 = GATConv(256, 256, heads=1, dropout=0.2)
		self.classify_layer = Linear(256, 13)
		init.xavier_uniform_(self.classify_layer.weight)

	def forward(self, x, edge_index):
		x = self.conv1(x, edge_index)
		#print(x.shape)
		x = F.elu(self.conv2(x, edge_index))
		#print(x.shape)
		x = self.conv3(x, edge_index)
		#print(x.shape)
		x = F.elu(self.conv4(x, edge_index))
		x = self.conv5(x, edge_index)

		scores = self.classify_layer(x)
		# x = F.dropout(x, training = self.training)
		return F.log_softmax(scores, dim=1)
# saturates at 67% and iou at 27%

Mar. 27th GPU1
class splineconv(torch.nn.Module):
	def __init__(self):
		super(splineconv, self).__init__()
		self.conv1 = GATConv(32, 256, heads=3, dropout=0.2)
		self.conv2 = GATConv(768, 256, heads=3, dropout=0.2)

		self.conv3 = GATConv(768, 256, heads=3, dropout=0.2)
		self.conv4 = GATConv(768, 256, heads=1)
		self.conv5 = GATConv(256, 256, heads=1, dropout=0.2)
		self.classify_layer = Linear(256, 13)
		init.xavier_uniform_(self.classify_layer.weight)

	def forward(self, x, edge_index):
		x = self.conv1(x, edge_index)
		#print(x.shape)
		x = F.elu(self.conv2(x, edge_index))
		#print(x.shape)
		x = self.conv3(x, edge_index)
		#print(x.shape)
		x = F.elu(self.conv4(x, edge_index))
		x = self.conv5(x, edge_index)
		scores = self.classify_layer(x)
		# x = F.dropout(x, training = self.training)
		return F.log_softmax(scores, dim=1)

# Mar.27th GPU1
# iou stuck at 28% not good enough
class splineconv(torch.nn.Module):
	def __init__(self):
		super(splineconv, self).__init__()
		self.conv1 = GATConv(32, 256, heads=3, dropout=0.2)
		self.conv2 = GATConv(768, 256, heads=3, dropout=0.2)

		self.conv3 = GATConv(768, 256, heads=3, dropout=0.2)
		self.conv4 = GATConv(768, 256, heads=1, dropout=0.2)
		self.conv5 = GATConv(256, 256, heads=1, dropout=0.2)
		self.conv6 = GATConv(256, 256, heads=1, dropout=0.2)
		self.classify_layer = Linear(256, 13)
		init.xavier_uniform_(self.classify_layer.weight)

	def forward(self, x, edge_index):
		x = self.conv1(x, edge_index)
		#print(x.shape)
		x = F.elu(self.conv2(x, edge_index))
		#print(x.shape)
		x = self.conv3(x, edge_index)
		#print(x.shape)
		x = F.elu(self.conv4(x, edge_index))
		x = self.conv5(x, edge_index)
		x = F.elu(self.conv6(x, edge_index))
		scores = self.classify_layer(x)
		# x = F.dropout(x, training = self.training)
		return F.log_softmax(scores, dim=1)

"""
"""
class splineconv(torch.nn.Module):
	def __init__(self):
		super(splineconv, self).__init__()
		self.conv1 = SplineConv(32, 64, dim=6, kernel_size=3)
		self.pool1 = TopKPooling(64, ratio=0.5)
		self.conv2 = SplineConv(64, 64, dim=6, kernel_size=3)
		self.pool2 = TopKPooling(64, ratio=0.5)
		self.conv3 = SplineConv(128, 64, dim=6, kernel_size=3)
		self.conv4 = SplineConv(64, 64, dim=6, kernel_size=3)
		self.conv5 = SplineConv(64, 13, dim=6, kernel_size=3)

		#self.lin1 = torch.nn.Linear(256, 128)
		#self.lin2 = torch.nn.Linear(128, 64)
		#self.lin3 = torch.nn.Linear(64, 13)
		#self.bn1 = torch.nn.BatchNorm1d(128)
		#self.bn2 = torch.nn.BatchNorm1d(64)
		#self.act1 = torch.nn.ReLU()
		#self.act2 = torch.nn.ReLU()    
	
	def forward(self, x, edge_index, edge_attr):
		x = F.elu(self.conv1(x, edge_index, edge_attr))
		d1, d2 = x.shape
		#print(x.shape)
		x1 = x
		x, edge_index, edge_attr, batch, _, _ = self.pool1(x, edge_index, edge_attr)
		x = F.elu(self.conv2(x, edge_index, edge_attr))
		#print(x.shape)

		x2 = x
		# n/4 x 64
		x, edge_index, edge_attr, batch, _, _ = self.pool2(x, edge_index, edge_attr)
		#print(x.shape)
		x3 = x
		m, n = x3.shape
		x = gmp(x, batch).repeat(m,1)
		x = torch.cat([x, x3], dim=1)
		#print(x.shape)
		x = F.elu(self.conv3(x, edge_index, edge_attr))
		x = torch.cat([x.repeat(2,1), x2])
		#print(x.shape)
		x = F.elu(self.conv4(x, edge_index, edge_attr))
		x = torch.cat([x.repeat(2,1), x1])
		#print(x.shape)
		x = F.elu(self.conv5(x, edge_index, edge_attr))
		x = x[:d1, :]
		x = F.log_softmax(x, dim=1)
		return x




		x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

		x = F.relu(self.conv2(x, edge_index))
	 
		x, edge_index, _, batch, _, _ = self.pool2(x, edge_index)
		x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

		x = F.relu(self.conv3(x, edge_index))
 
		x, edge_index, _, batch, _, _ = self.pool3(x, edge_index)
		x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

		x = x1 + x2 + x3

		x = self.lin1(x)
		x = self.act1(x)
		x = self.lin2(x)
		x = self.act2(x)      
		#print(x.shape)
		x = F.dropout(x, p=0.5, training=self.training)
 
		x = F.log_softmax(x, dim=1)
 
		return x
"""

class splineconv(torch.nn.Module):
	def __init__(self):
		super(splineconv, self).__init__()
		self.conv1 = SplineConv(32, 128, dim=6, kernel_size=3)
		#self.conv2 = SplineConv(32, 64, dim=6, kernel_size=3)
		#self.bn1 = BatchNorm1d(64)

		self.conv3 = SplineConv(128, 128, dim=6, kernel_size=3)
		#self.conv4 = SplineConv(128, 64, dim=6, kernel_size=5)
		#self.bn2 = BatchNorm1d(64)

		#self.conv5 = SplineConv(64, 32, dim=6, kernel_size=5)
		self.conv6 = SplineConv(128, 64, dim=6, kernel_size=5)
		self.conv7 = SplineConv(64, 13, dim=6, kernel_size=7)

	def forward(self, x, edge_index, edge_attr):
		#x = F.elu(self.conv1(x, edge_index, edge_attr))
		x = self.conv1(x, edge_index, edge_attr)
		#x = F.elu(self.conv3(x, edge_index, edge_attr))
		x = self.conv3(x, edge_index, edge_attr)
		#x = F.elu(self.conv5(x, edge_index, edge_attr))
		x = self.conv6(x, edge_index, edge_attr)
		x = F.elu(self.conv7(x, edge_index, edge_attr))
		# x = F.dropout(x, training = self.training)
		return F.log_softmax(x, dim=1)