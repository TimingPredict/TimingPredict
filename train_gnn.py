import torch
import dgl
import torch.nn.functional as F
import random

from data_graph import data_train, data_test, num_node_features, num_edge_features, num_node_outputs
from model import TimingGCN

from sklearn.metrics import r2_score

model = TimingGCN(num_node_features, num_edge_features, num_node_outputs)
model.cuda()

def test(model):
    with torch.no_grad():
        print('======= Training dataset ======')
        for k, g in data_train.items():
            pred = model(g)
            truth = g.ndata['node_net_delays_log']
            print(k, r2_score(pred.cpu().numpy().reshape(-1),
                              truth.cpu().numpy().reshape(-1)))
        print('======= Test dataset ======')
        for k, g in data_test.items():
            pred = model(g)
            truth = g.ndata['node_net_delays_log']
            print(k, r2_score(pred.cpu().numpy().reshape(-1),
                              truth.cpu().numpy().reshape(-1)))

def train(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()

    for e in range(10000):
        train_loss_tot = 0
        optimizer.zero_grad()
        for k, g in data_train.items():
            pred = model(g)
            loss = F.mse_loss(pred, g.ndata['node_net_delays_log'])
            train_loss_tot += loss.item()
            loss.backward()
        optimizer.step()

        if e == 0 or e % 10 == 9:
            with torch.no_grad():
                test_loss = sum(F.mse_loss(model(g), g.ndata['node_net_delays_log']).item()
                                for k, g in data_test.items())
                print('Epoch {}, train loss {:.6f}, test loss {:.6f}'.format(e, train_loss_tot / len(data_train), test_loss / len(data_test)))

        if e == 0 or e % 100 == 99:
            torch.save(model.state_dict(), './checkpoints/02_xxx/{}.pth'.format(e))
            print('saved model')

if __name__ == '__main__':
    # model.load_state_dict(torch.load('./checkpoints/01_netdelaylog/1599.pth'))
    # test(model)
    train(model)
