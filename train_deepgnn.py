import torch
import dgl
import torch.nn.functional as F
import random
import pdb
import time
import argparse
import os
from sklearn.metrics import r2_score
import tee

from data_homograph import data_train, data_test
from model import DeepGCNII

parser = argparse.ArgumentParser()
parser.add_argument(
    '--save_to', type=str,
    help='If specified, the log and model would be saved to that checkpoint directory')
parser.add_argument(
    '--nlayers', type=int, required=True,
    help='The number of deep GCN layers to use. suggest max 19 to avoid out of GPU memory')

def test(model):    # at
    model.eval()
    with torch.no_grad():
        print('======= Training dataset ======')
        for k, g in data_train.items():
            torch.cuda.synchronize()
            time_s = time.time()
            pred = model(g)[:, :4]
            torch.cuda.synchronize()
            time_t = time.time()
            truth = g.ndata['n_atslew'][:, :4]
            print(k, r2_score(pred.cpu().numpy().reshape(-1),
                              truth.cpu().numpy().reshape(-1)), '\ttime', time_t - time_s)
        print('======= Test dataset ======')
        for k, g in data_test.items():
            torch.cuda.synchronize()
            time_s = time.time()
            pred = model(g)[:, :4]
            torch.cuda.synchronize()
            time_t = time.time()
            truth = g.ndata['n_atslew'][:, :4]
            print(k, r2_score(pred.cpu().numpy().reshape(-1),
                              truth.cpu().numpy().reshape(-1)), '\ttime', time_t - time_s)

def train(model, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    batch_size = 7

    # create a function scope so that memory can be freed
    def test_iter(e, train_loss_tot_ats):
        with torch.no_grad():
            model.eval()
            test_loss_tot_ats = 0

            for k, g in data_test.items():
                pred_atslew = model(g)
                test_loss_tot_ats += F.mse_loss(pred_atslew, g.ndata['n_atslew']).item()
                del pred_atslew

            print('Epoch {}, at {:.6f}/{:.6f}'.format(
                e,
                train_loss_tot_ats / batch_size,
                test_loss_tot_ats / len(data_test)))

        if e == 0 or e % 200 == 199 or (e > 6000 and test_loss_tot_ats_prop / len(data_test) < 6):
            if args.save_to:
                save_path = './checkpoints/{}/{}.pth'.format(args.save_to, e)
                torch.save(model.state_dict(), save_path)
                print('saved model to', save_path)
            try:
                test(model)
            except ValueError as e:
                print(e)
                print('Error testing, but ignored')

    def train_iter(e):
        model.train()
        train_loss_tot_ats = 0
        optimizer.zero_grad()
        
        for k, g in random.sample(data_train.items(), batch_size):
            pred_atslew = model(g)
            loss_ats = F.mse_loss(pred_atslew, g.ndata['n_atslew'])
            train_loss_tot_ats += loss_ats.item()
            loss_ats.backward()
            del loss_ats, pred_atslew
            
        optimizer.step()
        return train_loss_tot_ats

    for e in range(100000):
        train_loss_tot_ats = train_iter(e)
        if e == 0 or e % 20 == 19:
            test_iter(e, train_loss_tot_ats)

if __name__ == '__main__':
    args = parser.parse_args()
    model = DeepGCNII(n_layers=args.nlayers)
    model.cuda()
    if args.save_to:
        print('saving logs and models to ./checkpoints/{}'.format(args.save_to))
        os.makedirs('./checkpoints/{}'.format(args.save_to))  # exist not ok
        stdout_f = './checkpoints/{}/stdout.log'.format(args.save_to)
        stderr_f = './checkpoints/{}/stderr.log'.format(args.save_to)
        with tee.StdoutTee(stdout_f), tee.StderrTee(stderr_f):
            train(model, args)
    else:
        print('No save_to is specified. abandoning all model checkpoints and logs')
        train(model, args)
        
    # model.load_state_dict(torch.load('./checkpoints/08_atcd_specul/11799.pth'))
    # test(model)
