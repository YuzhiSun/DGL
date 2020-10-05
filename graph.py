import dgl
import numpy as np
from mxnet import nd
u, v = nd.array([0,0,0,1],dtype=np.int32), nd.array([1,2,3,3],dtype=np.int32)
print(u.dtype, v)
g = dgl.graph((u,v))
print('g:',g,'\n')
print('nodes:',g.nodes(),'\n')
print('edges:',g.edges(),'\n')
print('form=all:',g.edges(form='all'))
# 如果ID最大的节点被隔离(意味着没有边)，
# 然后需要显式地设置节点的数量
g = dgl.graph((u,v), num_nodes=8)
print('g with isolate nodes:\n edges:',g.edges(),'\n nodes:',g.nodes())

u2, v2 = nd.array([0,1,0,1],dtype=np.int32), \
         nd.array([1,0,2,2],dtype=np.int32)
g2 = dgl.graph((u2,v2))
print('g2 with two direct edges:\n',g2.edges(form='all'),'nodes:\n',g2.nodes())

# 无向图需要节点之间建立两条边
bg = dgl.to_bidirected(g)
print('no direct edge:',bg.edges(form='all'))