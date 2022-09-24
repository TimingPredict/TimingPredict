import torch
import numpy as np
import random

random.seed(8026728)

available_data = 'blabla usb_cdc_core BM64 jpeg_encoder salsa20 usbf_device aes128 wbqspiflash aes192 cic_decimator xtea aes256 des spm y_huff aes_cipher picorv32a synth_ram zipdiv genericfir usb'.split()

data = {k: list(map(lambda t: t.numpy(), torch.load('data/4_netstat/{}.netstat.pt'.format(k))))
        for k in available_data}

for k, (x, y) in data.items():
    y = np.log(0.0001 + y) + 7.6
    data[k] = x, y

example_d = next(iter(data.values()))
num_input_features = example_d[0].shape[1]
num_outputs = example_d[1].shape[1]

train_data_keys = random.sample(available_data, 14)
data_train = {k: g for k, g in data.items() if k in train_data_keys}
data_test = {k: g for k, g in data.items() if k not in train_data_keys}

if __name__ == '__main__':
    print('Netstat total {} benchmarks'.format(len(data)))
