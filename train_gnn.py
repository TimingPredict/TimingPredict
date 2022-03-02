import torch
import numpy as np
import dgl
import torch.nn.functional as F
import random
import pdb
import time
import argparse
import os
from sklearn.metrics import r2_score
import tee

from data_graph import data_train, data_test
from model import TimingGCN

parser = argparse.ArgumentParser()
parser.add_argument(
    '--test_iter', type=int,
    help='If specified, test a saved model instead of training')
parser.add_argument(
    '--checkpoint', type=str,
    help='If specified, the log and model would be saved to/loaded from that checkpoint directory')
parser.set_defaults(netdelay=True, celldelay=True, groundtruth=True)
parser.add_argument(
    '--no_netdelay', dest='netdelay', action='store_false',
    help='Disable the net delay training supervision (default enabled)')
parser.add_argument(
    '--no_celldelay', dest='celldelay', action='store_false',
    help='Disable the cell delay training supervision (default enabled)')
parser.add_argument(
    '--no_groundtruth', dest='groundtruth', action='store_false',
    help='Disable ground-truth breakdown in training (default enabled)')

model = TimingGCN()
model.cuda()

def test(model):    # at
    model.eval()
    with torch.no_grad():
        def test_dict(data):
            for k, (g, ts) in data.items():
                torch.cuda.synchronize()
                time_s = time.time()
                pred = model(g, ts, groundtruth=False)[2][:, :4]
                torch.cuda.synchronize()
                time_t = time.time()
                truth = g.ndata['n_atslew'][:, :4]
                r2 = r2_score(pred.cpu().numpy().reshape(-1),
                              truth.cpu().numpy().reshape(-1))
                print('{:15} r2 {:1.5f}, time {:2.5f}'.format(k, r2, time_t - time_s))
                
                # print('{}'.format(time_t - time_s + ts['topo_time']))
                
        print('======= Training dataset ======')
        test_dict(data_train)
        print('======= Test dataset ======')
        test_dict(data_test)

def test_netdelay(model):    # net delay
    model.eval()
    with torch.no_grad():
        def test_dict(data):
            for k, (g, ts) in data.items():
                pred = model(g, ts, groundtruth=False)[0]
                truth = g.ndata['n_net_delays_log']
                r2 = r2_score(pred.cpu().numpy().reshape(-1),
                              truth.cpu().numpy().reshape(-1))
                print('{:15} {}'.format(k, r2))
                
        print('======= Training dataset ======')
        test_dict(data_train)
        print('======= Test dataset ======')
        test_dict(data_test)

def train(model, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    batch_size = 7

    for e in range(100000):
        model.train()
        train_loss_tot_net_delays, train_loss_tot_cell_delays, train_loss_tot_ats = 0, 0, 0
        train_loss_tot_cell_delays_prop, train_loss_tot_ats_prop = 0, 0
        optimizer.zero_grad()
        
        for k, (g, ts) in random.sample(data_train.items(), batch_size):
            pred_net_delays, pred_cell_delays, pred_atslew = model(g, ts, groundtruth=args.groundtruth)
            loss_net_delays, loss_cell_delays = 0, 0

            if args.netdelay:
                loss_net_delays = F.mse_loss(pred_net_delays, g.ndata['n_net_delays_log'])
                train_loss_tot_net_delays += loss_net_delays.item()

            if args.celldelay:
                loss_cell_delays = F.mse_loss(pred_cell_delays, g.edges['cell_out'].data['e_cell_delays'])
                train_loss_tot_cell_delays += loss_cell_delays.item()
            else:
                # Workaround for a dgl bug...
                # It seems that if some forward propagation channel is not used in backward graph, the GPU memory would BOOM. so we just create a fake gradient channel for this cell delay fork and make sure it does not contribute to gradient by *0.
                loss_cell_delays = torch.sum(pred_cell_delays) * 0.0
            
            loss_ats = F.mse_loss(pred_atslew, g.ndata['n_atslew'])
            train_loss_tot_ats += loss_ats.item()
            
            (loss_net_delays + loss_cell_delays + loss_ats).backward()
            
        optimizer.step()

        if e == 0 or e % 20 == 19:
            with torch.no_grad():
                model.eval()
                test_loss_tot_net_delays, test_loss_tot_cell_delays, test_loss_tot_ats = 0, 0, 0
                test_loss_tot_cell_delays_prop, test_loss_tot_ats_prop = 0, 0
                
                for k, (g, ts) in data_test.items():
                    pred_net_delays, pred_cell_delays, pred_atslew = model(g, ts, groundtruth=True)
                    _, pred_cell_delays_prop, pred_atslew_prop = model(g, ts, groundtruth=False)

                    if args.netdelay:
                        test_loss_tot_net_delays += F.mse_loss(pred_net_delays, g.ndata['n_net_delays_log']).item()
                    if args.celldelay:
                        test_loss_tot_cell_delays += F.mse_loss(pred_cell_delays, g.edges['cell_out'].data['e_cell_delays']).item()
                    test_loss_tot_ats += F.mse_loss(pred_atslew, g.ndata['n_atslew']).item()
                    test_loss_tot_ats_prop += F.mse_loss(pred_atslew_prop, g.ndata['n_atslew']).item()
                    
                print('Epoch {}, net delay {:.6f}/{:.6f}, cell delay {:.6f}/{:.6f}, at {:.6f}/({:.6f}, {:.6f})'.format(
                    e,
                    train_loss_tot_net_delays / batch_size,
                    test_loss_tot_net_delays / len(data_test),
                    train_loss_tot_cell_delays / batch_size,
                    test_loss_tot_cell_delays / len(data_test),
                    train_loss_tot_ats / batch_size,
                    # train_loss_tot_ats_prop / batch_size,
                    test_loss_tot_ats / len(data_test),
                    test_loss_tot_ats_prop / len(data_test)))

            if e == 0 or e % 200 == 199 or (e > 6000 and test_loss_tot_ats_prop / len(data_test) < 6):
                if args.checkpoint:
                    save_path = './checkpoints/{}/{}.pth'.format(args.checkpoint, e)
                    torch.save(model.state_dict(), save_path)
                    print('saved model to', save_path)
                try:
                    test(model)
                except ValueError as e:
                    print(e)
                    print('Error testing, but ignored')

if __name__ == '__main__':
    args = parser.parse_args()
    if args.test_iter:
        assert args.checkpoint, 'no checkpoint dir specified'
        model.load_state_dict(torch.load('./checkpoints/{}/{}.pth'.format(args.checkpoint, args.test_iter)))
        test(model)
        test_netdelay(model)
        
    else:
        if args.checkpoint:
            print('saving logs and models to ./checkpoints/{}'.format(args.checkpoint))
            os.makedirs('./checkpoints/{}'.format(args.checkpoint))  # exist not ok
            stdout_f = './checkpoints/{}/stdout.log'.format(args.checkpoint)
            stderr_f = './checkpoints/{}/stderr.log'.format(args.checkpoint)
            with tee.StdoutTee(stdout_f), tee.StderrTee(stderr_f):
                train(model, args)
        else:
            print('No checkpoint is specified. abandoning all model checkpoints and logs')
            train(model, args)

