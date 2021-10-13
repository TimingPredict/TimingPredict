import torch
import dgl
import random

random.seed(8026728)

available_data = 'blabla usb_cdc_core BM64 jpeg_encoder salsa20 usbf_device aes128 wbqspiflash aes192 cic_decimator xtea aes256 des spm y_huff aes_cipher picorv32a synth_ram zipdiv genericfir usb'.split()

data = {k: dgl.load_graphs('data/{}.graph.bin'.format(k))[0][0].to('cuda')
        for k in available_data}

for k, g in data.items():
    g.ndata['node_net_delays_log'] = torch.log(0.0001 + g.ndata['node_net_delays']) + 7.6

example_g = next(iter(data.values()))
num_node_features = example_g.ndata['node_features'].shape[1]
num_edge_features = example_g.edata['edge_features'].shape[1]
num_node_outputs = example_g.ndata['node_net_delays'].shape[1]

train_data_keys = random.sample(available_data, 14)
data_train = {k: g for k, g in data.items() if k in train_data_keys}
data_test = {k: g for k, g in data.items() if k not in train_data_keys}

if __name__ == '__main__':
    print('Graph statistics: (total {} graphs)'.format(len(data)))
    print('{:15} {:>10} {:>10}'.format('NAME', '#NODES', '#EDGES'))
    for k, g in data.items():
        print('{:15} {:>10} {:>10}'.format(k, g.num_nodes(), g.num_edges()))
