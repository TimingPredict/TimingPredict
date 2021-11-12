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

from data_graph import data_train, data_test
from model import TimingGCN

parser = argparse.ArgumentParser()
parser.add_argument(
    '--save_to', type=str,
    help='If specified, the log and model would be saved to that checkpoint directory')
parser.set_defaults(netdelay=True, celldelay=True)
parser.add_argument(
    '--no_netdelay', dest='netdelay', action='store_false',
    help='Disable the net delay training supervision (default enabled)')
parser.add_argument(
    '--no_celldelay', dest='celldelay', action='store_false',
    help='Disable the cell delay training supervision (default enabled)')

model = TimingGCN()
model.cuda()

def test(model):    # at
    model.eval()
    with torch.no_grad():
        print('======= Training dataset ======')
        for k, (g, ts) in data_train.items():
            torch.cuda.synchronize()
            time_s = time.time()
            pred = model(g, ts, groundtruth=False)[2][:, :4]
            torch.cuda.synchronize()
            time_t = time.time()
            truth = g.ndata['n_atslew'][:, :4]
            print(k, r2_score(pred.cpu().numpy().reshape(-1),
                              truth.cpu().numpy().reshape(-1)), '\ttime', time_t - time_s)
        print('======= Test dataset ======')
        for k, (g, ts) in data_test.items():
            torch.cuda.synchronize()
            time_s = time.time()
            pred = model(g, ts, groundtruth=False)[2][:, :4]
            torch.cuda.synchronize()
            time_t = time.time()
            truth = g.ndata['n_atslew'][:, :4]
            print(k, r2_score(pred.cpu().numpy().reshape(-1),
                              truth.cpu().numpy().reshape(-1)), '\ttime', time_t - time_s)

def train(model, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    batch_size = 7

    for e in range(100000):
        model.train()
        train_loss_tot_net_delays, train_loss_tot_cell_delays, train_loss_tot_ats = 0, 0, 0
        train_loss_tot_cell_delays_prop, train_loss_tot_ats_prop = 0, 0
        optimizer.zero_grad()
        
        for k, (g, ts) in random.sample(data_train.items(), batch_size):
            pred_net_delays, pred_cell_delays, pred_atslew = model(g, ts, groundtruth=True)
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
                if args.save_to:
                    save_path = './checkpoints/{}/{}.pth'.format(args.save_to, e)
                    torch.save(model.state_dict(), save_path)
                    print('saved model to', save_path)
                try:
                    test(model)
                except ValueError as e:
                    print(e)
                    print('Error testing, but ignored')

if __name__ == '__main__':
    args = parser.parse_args()
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
