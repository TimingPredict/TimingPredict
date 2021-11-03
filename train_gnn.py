import torch
import dgl
import torch.nn.functional as F
import random

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

    for e in range(10000):
        model.train()
        train_loss_tot = 0
        optimizer.zero_grad()
        for k, (g, ts) in data_train.items():
            pred = model(g, ts)
            loss = F.mse_loss(pred, g.ndata['n_net_delays_log'])
            train_loss_tot += loss.item()
            loss.backward()
        optimizer.step()

        if e == 0 or e % 10 == 9:
            with torch.no_grad():
                test_loss = sum(F.mse_loss(model(g, ts), g.ndata['n_net_delays_log']).item()
                                for k, (g, ts) in data_test.items())
                print('Epoch {}, train loss {:.6f}, test loss {:.6f}'.format(e, train_loss_tot / len(data_train), test_loss / len(data_test)))

            if e == 0 or e % 100 == 99:
                torch.save(model.state_dict(), './checkpoints/05_heterond/{}.pth'.format(e))
                print('saved model')
                test(model)

if __name__ == '__main__':
    # model.load_state_dict(torch.load('./checkpoints/03_largermlp/1799.pth'))
    # test(model)
    train(model)
