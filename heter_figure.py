import dgl
from mxnet import nd

graph_data = {
    ('drug', 'interacts', 'drug'): ([0,1], [1,2]),  # 第一个数组是源节点 第二个数组是目标节点
    ('drug', 'interacts', 'gene'): ([0,1], [2,3]),
    ('drug', 'treats', 'disease'): ([1,2], [2,0])
}

g = dgl.heterograph(graph_data)
# print(g.ntypes, g.etypes,g.canonical_etypes)
print(g.nodes('disease'))
print(g)
# 齐次图
g1 = dgl.heterograph({('node_type', 'edge_type', 'node_type'): ([1,2], [3,4])})
print(g1)
# 二部图
g2 = dgl.heterograph({('source_type', 'edge_type', 'destination_type'): ([1,2], [3,4])})
print(g2)
print(g.metagraph().edges())
g.nodes['drug'].data['hv'] = nd.ones((3,1))
print(g.nodes['drug'].data['hv'])
g.edges['treats'].data['he'] = nd.zeros((2,1))
print(g.edges['treats'].data['he'])

g3 = dgl.heterograph({
    ('drug','interacts', 'drug'): ([0,1], [1, 2]),
    ('drug','is similar', 'drug'): ([0,1],[1, 3])
})
g3.ndata['hv'] = nd.ones((4,1))
print(g)

# 边类型子图
eg = dgl.edge_type_subgraph(g,[('drug', 'interacts', 'drug'),('drug', 'treats', 'disease')])
print(eg,eg.nodes['drug'].data['hv'])

print('-----------------------------')
g = dgl.heterograph({
   ('drug', 'interacts', 'drug'): ([0, 1], [1, 2]),
   ('drug', 'treats', 'disease'): ([1], [2])})
g.nodes['drug'].data['hv'] = nd.ones((3, 1))
g.nodes['disease'].data['hv'] = nd.ones((3, 1))
g.edges['interacts'].data['he'] = nd.zeros((2, 1))
g.edges['treats'].data['he'] = nd.zeros((1, 2))

hg = dgl.to_homogeneous(g)
print('hv' in hg.ndata)
# hg = dgl.to_homogeneous(g, edata=['he'])   复制特征必须保证边/节点拥有相同的尺寸和数据类型
hg = dgl.to_homogeneous(g, ndata=['hv'])
print(hg.ndata['hv'],g.ntypes)
print(hg.ndata[dgl.NTYPE],hg.ndata[dgl.NID])
print(g.etypes,hg.ndata[dgl.ETYPE],hg.ndata[dgl.EID])

g0 = dgl.heterograph({
   ('drug', 'interacts', 'drug'): ([0, 1],[1, 2]),
   ('drug', 'interacts', 'gene'): ([0, 1],[2, 3]),
   ('drug', 'treats', 'disease'): ([1], [2])
})
sub_g = dgl.edge_type_subgraph(g0, [('drug', 'interacts', 'drug'),
                                   ('drug', 'interacts', 'gene')])
h_sub_g = dgl.to_homogeneous(sub_g)
print(h_sub_g)
print('==================================')


