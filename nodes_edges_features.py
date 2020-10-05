import dgl
import numpy as np
from mxnet import nd

g = dgl.graph(([0,0,1,5],[1,2,2,0]))
print('g:\n',g)

g.ndata['x'] = nd.ones((g.num_nodes(),3)) # 长度为3的特征
g.edata['x'] = nd.ones(g.num_edges(),dtype=np.int32)
print('g:',g)
# 不同的名字可以有不同的特征
g.ndata['y'] = nd.random.uniform(shape=(g.num_nodes(),5))
print('g:',g)

print('the feature of node 1 in x',g.ndata['x'][1])  # 获取节点1 的特征
print('\n the feature of edge 0 and 3 in x:',g.edata['x'][nd.array([0,3],dtype=np.int32)])

# 对于加权图，可以将权重存储为边缘特征
edges = nd.array([0,0,0,1],dtype=np.int), nd.array([1,2,3,3],dtype=np.int)
weights = nd.array([0.1, 0.6, 0.9, 0.7]) # 权重
g = dgl.graph(edges)
g.edata['w'] = weights  # w 代表权重特征
print('\n g with weight:',g)