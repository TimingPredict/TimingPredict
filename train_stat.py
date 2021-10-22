import torch
import torch.nn.functional as F
import random

from data_stat import data_train, data_test, num_input_features, num_outputs
from model import MLP

from sklearn.metrics import r2_score

model = MLP(num_input_features, 64, 128, 128, 128, 64, 32, num_outputs)
model.cuda()

def test(model):
    with torch.no_grad():
        print('======= Training dataset ======')
        for k, (x, y) in data_train.items():
            pred = model(x)
            print(k, r2_score(pred.cpu().numpy().reshape(-1),
                              y.cpu().numpy().reshape(-1)))
        print('======= Test dataset ======')
        for k, (x, y) in data_test.items():
            pred = model(x)
            print(k, r2_score(pred.cpu().numpy().reshape(-1),
                              y.cpu().numpy().reshape(-1)))

def train(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()

    for e in range(10000):
        train_loss_tot = 0
        optimizer.zero_grad()
        for k, (x, y) in data_train.items():
            pred = model(x)
            loss = F.mse_loss(pred, y)
            train_loss_tot += loss.item()
            loss.backward()
        optimizer.step()

        if e == 0 or e % 10 == 9:
            with torch.no_grad():
                test_loss = sum(F.mse_loss(model(x), y).item()
                                for k, (x, y) in data_test.items())
                print('Epoch {}, train loss {:.6f}, test loss {:.6f}'.format(e, train_loss_tot / len(data_train), test_loss / len(data_test)))

        if e == 0 or e % 100 == 99:
            torch.save(model.state_dict(), './checkpoints/04_netstat_largermlp/{}.pth'.format(e))
            print('saved model')

if __name__ == '__main__':
    model.load_state_dict(torch.load('./checkpoints/04_netstat_largermlp/6899.pth'))
    test(model)
    # train(model)
