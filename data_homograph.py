import torch

def gen_homograph():
    from data_graph import data_train, data_test, gen_homobigraph_with_features
    # replace hetero graph with homographs
    # do not execute this in other modules, as it would modify
    # the global data in a dirty way
    for dic in [data_train, data_test]:
        for k in dic:
            g, ts = dic[k]
            dic[k] = gen_homobigraph_with_features(g)

    torch.save([data_train, data_test], './data/7_homotest/train_test.pt')

data_train, data_test = torch.load('./data/7_homotest/train_test.pt')
    
if __name__ == '__main__':
    # gen_homograph()
    pass

