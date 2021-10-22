import torch
import torch.nn.functional as F
import dgl
import dgl.function as fn

class MLP(torch.nn.Module):
    def __init__(self, *sizes):
        super().__init__()
        fcs = []
        for i in range(1, len(sizes)):
            fcs.append(torch.nn.Linear(sizes[i - 1], sizes[i]))
            if i < len(sizes) - 1:
                fcs.append(torch.nn.LeakyReLU(negative_slope=0.2))
                # fcs.append(torch.nn.Dropout(p=0.5))
        self.layers = torch.nn.Sequential(*fcs)

    def forward(self, x):
        return self.layers(x)

class AllConv(torch.nn.Module):
    def __init__(self, in_nf, in_ef, h1, h2, out_nf, mlp_h1=64, mlp_h2=128, mlp_h3=64):
        super().__init__()
        self.h1 = h1
        self.h2 = h2
        self.MLP_msg = MLP(in_nf * 2 + in_ef, mlp_h1, mlp_h2, mlp_h3, 1 + h1 + h2)
        self.MLP_reduce = MLP(in_nf + h1 + h2, mlp_h1, mlp_h2, mlp_h3, out_nf)

    def edge_udf(self, edges):
        x = self.MLP_msg(torch.cat([edges.src['nf'], edges.dst['nf'], edges.data['ef']], dim=1))
        k, f1, f2 = torch.split(x, [1, self.h1, self.h2], dim=1)
        k = torch.sigmoid(k)
        return {'ef1': f1 * k, 'ef2': f2 * k}

    def forward(self, g, nf, ef):
        with g.local_scope():
            g.ndata['nf'] = nf
            g.edata['ef'] = ef
            g.apply_edges(self.edge_udf)
            g.update_all(fn.copy_e('ef1', 'ef1'), fn.sum('ef1', 'nf1'))
            g.update_all(fn.copy_e('ef2', 'ef2'), fn.max('ef2', 'nf2'))
            x = torch.cat([g.ndata['nf'], g.ndata['nf1'], g.ndata['nf2']], dim=1)
            x = self.MLP_reduce(x)
            return x

class TimingGCN(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, num_node_outputs):
        super().__init__()
        self.conv1 = AllConv(num_node_features, num_edge_features, 64, 64, 64)
        self.conv2 = AllConv(64, num_edge_features, 64, 64, 64)
        self.conv3 = AllConv(64, num_edge_features, 32, 32, num_node_outputs)

    def forward(self, g):
        x = self.conv1(g, g.ndata['node_features'], g.edata['edge_features'])
        x = self.conv2(g, x, g.edata['edge_features'])
        x = self.conv3(g, x, g.edata['edge_features'])
        return x
