import torch
import dgl
import random
import pdb

random.seed(8026728)

available_data = 'blabla usb_cdc_core BM64 jpeg_encoder salsa20 usbf_device aes128 wbqspiflash aes192 cic_decimator xtea aes256 des spm y_huff aes_cipher picorv32a synth_ram zipdiv genericfir usb'.split()

train_data_keys = random.sample(available_data, 14)

data = {}
for k in available_data:
    g = dgl.load_graphs('data/5_cellat/{}.graph.bin'.format(k))[0][0].to('cuda')
    g.ndata['n_net_delays_log'] = torch.log(0.0001 + g.ndata['n_net_delays']) + 7.6
    ts = {'input_nodes': (g.ndata['nf'][:, 1] < 0.5).nonzero().flatten().type(torch.int32),
          'output_nodes': (g.ndata['nf'][:, 1] > 0.5).nonzero().flatten().type(torch.int32)}
    data[k] = g, ts

data_train = {k: t for k, t in data.items() if k in train_data_keys}
data_test = {k: t for k, t in data.items() if k not in train_data_keys}

if __name__ == '__main__':
    print('Graph statistics: (total {} graphs)'.format(len(data)))
    print('{:15} {:>10} {:>10}'.format('NAME', '#NODES', '#EDGES'))
    for k, (g, ts) in data.items():
        print('{:15} {:>10} {:>10}'.format(k, g.num_nodes(), g.num_edges()))
