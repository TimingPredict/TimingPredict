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

def test(model):
    model.eval()
    with torch.no_grad():
        print('======= Training dataset ======')
        for k, (g, ts) in data_train.items():
            pred = model(g, ts)
            truth = g.ndata['n_net_delays_log']
            print(k, r2_score(pred.cpu().numpy().reshape(-1),
                              truth.cpu().numpy().reshape(-1)))
        print('======= Test dataset ======')
        for k, (g, ts) in data_test.items():
            pred = model(g, ts)
            truth = g.ndata['n_net_delays_log']
            print(k, r2_score(pred.cpu().numpy().reshape(-1),
                              truth.cpu().numpy().reshape(-1)))

def train(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for e in range(100000):
        model.train()
        level_limit = e // 64
        train_loss_tot_net_delays, train_loss_tot_ats = 0, 0
        optimizer.zero_grad()
        for k, (g, ts) in data_train.items():
            pred_net_delays, pred_ats, at_nodes = model(g, ts, level_limit=level_limit)
            loss_net_delays = F.mse_loss(pred_net_delays, g.ndata['n_net_delays_log'])
            train_loss_tot_net_delays += loss_net_delays.item()
            if at_nodes.shape:
                loss_ats = F.mse_loss(pred_ats, g.ndata['n_ats'][at_nodes])
                train_loss_tot_ats += loss_ats.item()
            else:
                loss_ats = 0
            (loss_net_delays + loss_ats * min(1, level_limit / 100)).backward()
        optimizer.step()

        if e == 0 or e % 10 == 9:
            with torch.no_grad():
                test_loss_tot_net_delays, test_loss_tot_ats = 0, 0
                for k, (g, ts) in data_test.items():
                    pred_net_delays, pred_ats, at_nodes = model(g, ts, level_limit=level_limit)
                    test_loss_tot_net_delays += F.mse_loss(pred_net_delays, g.ndata['n_net_delays_log']).item()
                    test_loss_tot_ats += F.mse_loss(pred_ats, g.ndata['n_ats'][at_nodes]).item()
                    
                print('Epoch {}, level limit {}, net delay {:.6f}/{:.6f}, at {:.6f}/{:.6f}'.format(
                    e, level_limit,
                    train_loss_tot_net_delays / len(data_train),
                    test_loss_tot_net_delays / len(data_test),
                    train_loss_tot_ats / len(data_train),
                    test_loss_tot_ats / len(data_test)))

            # if e == 0 or e % 100 == 99:
            #     torch.save(model.state_dict(), './checkpoints/05_heterond/{}.pth'.format(e))
            #     print('saved model')
            #     test(model)

if __name__ == '__main__':
    # model.load_state_dict(torch.load('./checkpoints/03_largermlp/1799.pth'))
    # test(model)
    train(model)
