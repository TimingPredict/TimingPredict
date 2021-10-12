import dgl
import torch

available_data = 'blabla usb_cdc_core BM64 jpeg_encoder salsa20 usbf_device aes128 wbqspiflash aes192 cic_decimator xtea aes256 des spm y_huff aes_cipher picorv32a synth_ram zipdiv genericfir usb'

data = {k: dgl.load_graphs('data/{}.graph.bin'.format(k))[0][0].to('cuda')
        for k in available_data.split()}

if __name__ == '__main__':
    print('Graph statistics: (total {} graphs)'.format(len(data)))
    print('{:15} {:>10} {:>10}'.format('NAME', '#NODES', '#EDGES'))
    for k, g in data.items():
        print('{:15} {:>10} {:>10}'.format(k, g.num_nodes(), g.num_edges()))
