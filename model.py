import torch
import torch.nn.functional as F
import dgl
import dgl.function as fn
import functools
import pdb

class MLP(torch.nn.Module):
    def __init__(self, *sizes, batchnorm=False, dropout=False):
        super().__init__()
        fcs = []
        for i in range(1, len(sizes)):
            fcs.append(torch.nn.Linear(sizes[i - 1], sizes[i]))
            if i < len(sizes) - 1:
                fcs.append(torch.nn.LeakyReLU(negative_slope=0.2))
                if dropout: fcs.append(torch.nn.Dropout(p=0.2))
                if batchnorm: fcs.append(torch.nn.BatchNorm1d(sizes[i]))
        self.layers = torch.nn.Sequential(*fcs)

    def forward(self, x):
        return self.layers(x)

class NetConv(torch.nn.Module):
    def __init__(self, in_nf, in_ef, out_nf, h1=32, h2=32):
        super().__init__()
        self.in_nf = in_nf
        self.in_ef = in_ef
        self.out_nf = out_nf
        self.h1 = h1
        self.h2 = h2
        
        self.MLP_msg_i2o = MLP(self.in_nf * 2 + self.in_ef, 64, 64, 64, 1 + self.h1 + self.h2)
        self.MLP_reduce_o = MLP(self.in_nf + self.h1 + self.h2, 64, 64, 64, self.out_nf)
        self.MLP_msg_o2i = MLP(self.in_nf * 2 + self.in_ef, 64, 64, 64, 64, self.out_nf)

    def edge_msg_i(self, edges):
        x = torch.cat([edges.src['nf'], edges.dst['nf'], edges.data['ef']], dim=1)
        x = self.MLP_msg_o2i(x)
        return {'efi': x}

    def edge_msg_o(self, edges):
        x = torch.cat([edges.src['nf'], edges.dst['nf'], edges.data['ef']], dim=1)
        x = self.MLP_msg_i2o(x)
        k, f1, f2 = torch.split(x, [1, self.h1, self.h2], dim=1)
        k = torch.sigmoid(k)
        return {'efo1': f1 * k, 'efo2': f2 * k}

    def node_reduce_o(self, nodes):
        x = torch.cat([nodes.data['nf'], nodes.data['nfo1'], nodes.data['nfo2']], dim=1)
        x = self.MLP_reduce_o(x)
        return {'new_nf': x}
        
    def forward(self, g, ts, nf):
        with g.local_scope():
            g.ndata['nf'] = nf
            # input nodes
            g.update_all(self.edge_msg_i, fn.sum('efi', 'new_nf'), etype='net_out')
            # output nodes
            g.apply_edges(self.edge_msg_o, etype='net_in')
            g.update_all(fn.copy_e('efo1', 'efo1'), fn.sum('efo1', 'nfo1'), etype='net_in')
            g.update_all(fn.copy_e('efo2', 'efo2'), fn.max('efo2', 'nfo2'), etype='net_in')
            g.apply_nodes(self.node_reduce_o, ts['output_nodes'])
            
            return g.ndata['new_nf']

class SignalProp(torch.nn.Module):
    def __init__(self, in_nf, in_cell_num_luts, in_cell_lut_sz, h1=32, h2=32, lut_dup=4):
        super().__init__()
        self.in_nf = in_nf
        self.in_cell_num_luts = in_cell_num_luts
        self.in_cell_lut_sz = in_cell_lut_sz
        self.out_nf = 4 + 4
        self.h1 = h1
        self.h2 = h2
        self.lut_dup = lut_dup
        
        self.MLP_netprop = MLP(self.out_nf + 2 * self.in_nf, 64, 64, 64, 64, self.out_nf)
        self.MLP_lut_query = MLP(self.out_nf + 2 * self.in_nf, 64, 64, 64, self.in_cell_num_luts * lut_dup * 2)
        self.MLP_lut_attention = MLP(1 + 2 + self.in_cell_lut_sz * 2, 64, 64, 64, self.in_cell_lut_sz * 2)
        self.MLP_cellarc_msg = MLP(self.out_nf + 2 * self.in_nf + self.in_cell_num_luts * self.lut_dup, 64, 64, 64, 1 + self.h1 + self.h2)
        self.MLP_cellreduce = MLP(self.in_nf + self.h1 + self.h2, 64, 64, 64, self.out_nf)

    def edge_msg_net(self, edges, groundtruth=False):
        if groundtruth:
            last_nf = edges.src['n_atslew']
        else:
            last_nf = edges.src['new_nf']
        
        x = torch.cat([last_nf, edges.src['nf'], edges.dst['nf']], dim=1)
        x = self.MLP_netprop(x)
        return {'efn': x}

    def edge_msg_cell(self, edges, groundtruth=False):
        # generate lut axis query
        if groundtruth:
            last_nf = edges.src['n_atslew']
        else:
            last_nf = edges.src['new_nf']
            
        q = torch.cat([last_nf, edges.src['nf'], edges.dst['nf']], dim=1)
        q = self.MLP_lut_query(q)
        q = q.reshape(-1, 2)
        
        # answer lut axis query
        axis_len = self.in_cell_num_luts * (1 + 2 * self.in_cell_lut_sz)
        axis = edges.data['ef'][:, :axis_len]
        axis = axis.reshape(-1, 1 + 2 * self.in_cell_lut_sz)
        axis = axis.repeat(1, self.lut_dup).reshape(-1, 1 + 2 * self.in_cell_lut_sz)
        a = self.MLP_lut_attention(torch.cat([q, axis], dim=1))
        
        # transform answer to answer mask matrix
        a = a.reshape(-1, 2, self.in_cell_lut_sz)
        ax, ay = torch.split(a, [1, 1], dim=1)
        a = torch.matmul(ax.reshape(-1, self.in_cell_lut_sz, 1), ay.reshape(-1, 1, self.in_cell_lut_sz))  # batch tensor product

        # look up answer matrix in lut
        tables_len = self.in_cell_num_luts * self.in_cell_lut_sz ** 2
        tables = edges.data['ef'][:, axis_len:axis_len + tables_len]
        tables = tables.reshape(-1, self.in_cell_lut_sz ** 2)
        tables = tables.repeat(1, self.lut_dup)
        r = torch.matmul(tables.reshape(-1, 1, self.in_cell_lut_sz ** 2), a.reshape(-1, self.in_cell_lut_sz ** 2, 1))   # batch dot product

        # construct final msg
        r = r.reshape(len(edges), self.in_cell_num_luts * self.lut_dup)
        x = torch.cat([last_nf, edges.src['nf'], edges.dst['nf'], r], dim=1)
        x = self.MLP_cellarc_msg(x)
        k, f1, f2 = torch.split(x, [1, self.h1, self.h2], dim=1)
        k = torch.sigmoid(k)
        return {'efc1': f1 * k, 'efc2': f2 * k}

    def node_reduce_o(self, nodes):
        x = torch.cat([nodes.data['nf'], nodes.data['nfc1'], nodes.data['nfc2']], dim=1)
        x = self.MLP_cellreduce(x)
        return {'new_nf': x}

    def node_skip_level_o(self, nodes):
        return {'new_nf': nodes.data['n_atslew']}
        
    def forward(self, g, ts, nf, groundtruth=False, level_limit=None):
        assert len(ts['topo']) % 2 == 0, 'The number of logic levels must be even (net, cell, net)'
        
        with g.local_scope():
            # init level 0 with ground truth features
            g.ndata['nf'] = nf
            g.ndata['new_nf'] = torch.zeros(g.num_nodes(), self.out_nf, device='cuda', dtype=nf.dtype)
            g.apply_nodes(self.node_skip_level_o, ts['pi_nodes'])

            def prop_net(nodes, groundtruth):
                g.pull(nodes, functools.partial(self.edge_msg_net, groundtruth=groundtruth), fn.sum('efn', 'new_nf'), etype='net_out')

            def prop_cell(nodes, groundtruth):
                es = g.in_edges(nodes, etype='cell_out')
                g.apply_edges(functools.partial(self.edge_msg_cell, groundtruth=groundtruth), es, etype='cell_out')
                g.send_and_recv(es, fn.copy_e('efc1', 'efc1'), fn.sum('efc1', 'nfc1'), etype='cell_out')
                g.send_and_recv(es, fn.copy_e('efc2', 'efc2'), fn.max('efc2', 'nfc2'), etype='cell_out')
                g.apply_nodes(self.node_reduce_o, nodes)
            
            if groundtruth:
                # don't need to propagate.
                prop_net(ts['input_nodes'], groundtruth)
                prop_cell(ts['output_nodes_nonpi'], groundtruth)
                valid_nodes = slice(None)  # [:], all

            else:
                # propagate
                valid_nodes = [ts['pi_nodes']]
                for i in range(1, len(ts['topo'])):
                    if level_limit is not None and i >= level_limit:
                        # g.apply_nodes(self.node_skip_level_o, ts['topo'][i])
                        break
                    valid_nodes.append(ts['topo'][i])
                    if i % 2 == 1:
                        prop_net(ts['topo'][i], groundtruth)
                    else:
                        prop_cell(ts['topo'][i], groundtruth)
                        
                valid_nodes, _ = torch.sort(torch.cat(valid_nodes))
                valid_nodes = valid_nodes.type(torch.long)
                
            return g.ndata['new_nf'], valid_nodes

class TimingGCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.nc1 = NetConv(10, 2, 32)
        self.nc2 = NetConv(32, 2, 32)
        self.nc3 = NetConv(32, 2, 16)  # 16 = 4x delay + 12x arbitrary (might include cap, beta)
        self.prop = SignalProp(10 + 16, 8, 7)

    def forward(self, g, ts, groundtruth=False, level_limit=None):
        nf0 = g.ndata['nf']
        x = self.nc1(g, ts, nf0)
        x = self.nc2(g, ts, x)
        x = self.nc3(g, ts, x)
        net_delays = x[:, :4]
        nf1 = torch.cat([nf0, x], dim=1)
        nf2, at_nodes = self.prop(g, ts, nf1, groundtruth=groundtruth, level_limit=level_limit)
        return net_delays, nf2, at_nodes
