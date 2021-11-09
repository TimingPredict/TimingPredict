import torch
import dgl
import random

random.seed(8026728)

available_data = 'blabla usb_cdc_core BM64 jpeg_encoder salsa20 usbf_device aes128 wbqspiflash aes192 cic_decimator xtea aes256 des spm y_huff aes_cipher picorv32a synth_ram zipdiv genericfir usb'.split()

train_data_keys = random.sample(available_data, 14)

def gen_topo(g_hetero):
    na, nb = g_hetero.edges(etype='net_out', form='uv')
    ca, cb = g_hetero.edges(etype='cell_out', form='uv')
    g = dgl.graph((torch.cat([na, ca]).cpu(), torch.cat([nb, cb]).cpu()))
    topo = dgl.topological_nodes_generator(g)
    return [t.cuda() for t in topo]

data = {}
for k in available_data:
    g = dgl.load_graphs('data/6_cellatslew/{}.graph.bin'.format(k))[0][0].to('cuda')
    g.ndata['n_net_delays_log'] = torch.log(0.0001 + g.ndata['n_net_delays']) + 7.6
    invalid_nodes = torch.abs(g.ndata['n_ats']) > 1e20   # ignore all uninitialized stray pins
    g.ndata['n_ats'][invalid_nodes] = 0
    g.ndata['n_slews'][invalid_nodes] = 0
    g.ndata['n_atslew'] = torch.cat([g.ndata['n_ats'], g.ndata['n_slews']], dim=1)
    g.edges['cell_out'].data['ef'] = g.edges['cell_out'].data['ef'].type(torch.float32)
    g.edges['cell_out'].data['e_cell_delays'] = g.edges['cell_out'].data['e_cell_delays'].type(torch.float32)
    ts = {'input_nodes': (g.ndata['nf'][:, 1] < 0.5).nonzero().flatten().type(torch.int32),
          'output_nodes': (g.ndata['nf'][:, 1] > 0.5).nonzero().flatten().type(torch.int32),
          'output_nodes_nonpi': torch.logical_and(g.ndata['nf'][:, 1] > 0.5, g.ndata['nf'][:, 0] < 0.5).nonzero().flatten().type(torch.int32),
          'pi_nodes': torch.logical_and(g.ndata['nf'][:, 1] > 0.5, g.ndata['nf'][:, 0] > 0.5).nonzero().flatten().type(torch.int32),
          'po_nodes': torch.logical_and(g.ndata['nf'][:, 1] < 0.5, g.ndata['nf'][:, 0] > 0.5).nonzero().flatten().type(torch.int32),
          'topo': gen_topo(g)}
    data[k] = g, ts

data_train = {k: t for k, t in data.items() if k in train_data_keys}
data_test = {k: t for k, t in data.items() if k not in train_data_keys}

if __name__ == '__main__':
    print('Graph statistics: (total {} graphs)'.format(len(data)))
    print('{:15} {:>10} {:>10}'.format('NAME', '#NODES', '#EDGES'))
    for k, (g, ts) in data.items():
        print('{:15} {:>10} {:>10}'.format(k, g.num_nodes(), g.num_edges()))
