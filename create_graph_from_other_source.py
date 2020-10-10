import dgl
from mxnet import nd
import scipy.sparse as sp

spmat = sp.rand(4, 4,format='csr', density=0.5) # 50% nonzero entries  一半的边是存在的
# dgl 必须接受方形矩阵 但是 scipy可以生成矩形矩阵
print(dgl.from_scipy(spmat),'\n matric of spmat: \n',spmat)

from scipy.sparse import rand
matrix = rand(3, 4, density=0.25, format="csr", random_state=42) # 生成一个矩形矩阵
print(matrix.todense()) # todense  生成稀疏矩阵的密集表示

import networkx as nx
nx_g = nx.path_graph(5) # 生成一个单链  0-1-2-3-4
print(dgl.from_networkx(nx_g))  # 这里有八条边 因为Networkx生成的是无向图

nxg = nx.DiGraph([(2,1),(1,2),(2,3),(0,0)]) # 使用networkx中的 DiGraph方法可以避免上面的问题
print(dgl.from_networkx(nxg))  # 在github收藏了很多示例代码  



