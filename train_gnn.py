import torch
import dgl
import torch.nn.functional as F
import random
import pdb

from data_graph import data_train, data_test
from model import TimingGCN

from sklearn.metrics import r2_score

model = TimingGCN()
model.cuda()

def test(model):    # at
    model.eval()
    with torch.no_grad():
        print('======= Training dataset ======')
        for k, (g, ts) in data_train.items():
            pred = model(g, ts, groundtruth=False)[1][:, :4]
            truth = g.ndata['n_atslew'][:, :4]
            print(k, r2_score(pred.cpu().numpy().reshape(-1),
                              truth.cpu().numpy().reshape(-1)))
        print('======= Test dataset ======')
        for k, (g, ts) in data_test.items():
            pred = model(g, ts, groundtruth=False)[1][:, :4]
            truth = g.ndata['n_atslew'][:, :4]
            print(k, r2_score(pred.cpu().numpy().reshape(-1),
                              truth.cpu().numpy().reshape(-1)))

def train(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    for e in range(100000):
        model.train()
        train_loss_tot_net_delays, train_loss_tot_ats, train_loss_tot_ats_prop = 0, 0, 0
        optimizer.zero_grad()
        for k, (g, ts) in data_train.items():
            pred_net_delays, pred_atslew, _ = model(g, ts, groundtruth=True)
            loss_net_delays = F.mse_loss(pred_net_delays, g.ndata['n_net_delays_log'])
            train_loss_tot_net_delays += loss_net_delays.item()
            loss_ats = F.mse_loss(pred_atslew, g.ndata['n_atslew'])
            train_loss_tot_ats += loss_ats.item()
            (loss_net_delays + loss_ats).backward()
            if e == 0 or e % 10 == 9:
                _, pred_atslew_prop, _ = model(g, ts, groundtruth=False)
                loss_ats_prop = F.mse_loss(pred_atslew_prop, g.ndata['n_atslew'])
                train_loss_tot_ats_prop += loss_ats_prop.item()
                loss_ats_prop.backward()
        optimizer.step()

        if e == 0 or e % 10 == 9:
            with torch.no_grad():
                model.eval()
                test_loss_tot_net_delays, test_loss_tot_ats, test_loss_tot_ats_prop = 0, 0, 0
                for k, (g, ts) in data_test.items():
                    pred_net_delays, pred_atslew, _ = model(g, ts, groundtruth=True)
                    _, pred_atslew_prop, _ = model(g, ts, groundtruth=False)
                    test_loss_tot_net_delays += F.mse_loss(pred_net_delays, g.ndata['n_net_delays_log']).item()
                    test_loss_tot_ats += F.mse_loss(pred_atslew, g.ndata['n_atslew']).item()
                    test_loss_tot_ats_prop += F.mse_loss(pred_atslew_prop, g.ndata['n_atslew']).item()
                    
                print('Epoch {}, net delay {:.6f} / {:.6f}, at ({:.6f}, {:.6f}) / ({:.6f}, {:.6f})'.format(
                    e,
                    train_loss_tot_net_delays / len(data_train),
                    test_loss_tot_net_delays / len(data_test),
                    train_loss_tot_ats / len(data_train),
                    train_loss_tot_ats_prop / len(data_train),
                    test_loss_tot_ats / len(data_test),
                    test_loss_tot_ats_prop / len(data_test)))

            if e == 0 or e % 100 == 99:
                # torch.save(model.state_dict(), './checkpoints/06_atprop/{}.pth'.format(e))
                # print('saved model')
                test(model)

if __name__ == '__main__':
    # model.load_state_dict(torch.load('./checkpoints/03_largermlp/1799.pth'))
    # test(model)
    train(model)
