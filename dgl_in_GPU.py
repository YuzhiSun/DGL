import dgl
from mxnet import nd
import mxnet


ctx = mxnet.gpu()
# 方法1  将图复制到GPU上
u, v = nd.array([0,1,2],dtype='int'), nd.array([2,3,4],dtype='int', )
print()
g = dgl.graph((u, v))
print(g)
g.ndata['x'] = nd.ones((5,3))
g_cuda = g.to(mxnet.gpu())
print(g_cuda.device)

# 方法2  在GPU上建立数组  然后直接在GPU上构建图
u, v = nd.array([0,1,2],dtype='int',ctx=ctx), nd.array([2,3,4],dtype='int', ctx=ctx)
g1 = dgl.graph((u, v))
g1.ndata['x'] = nd.ones((5,3),ctx=ctx)
print(g1.device)
print(g1.in_degrees(),
      g1.in_edges([2,3,4]),  # 非NDarray格式 可以直接赋值  自动放到GPU上
      g1.in_edges(nd.array([2,3,4],dtype='int',ctx=ctx)))
g1.ndata['x'] = nd.random.normal((5,4),ctx=ctx)       # NDarray格式数据，必须指定context
