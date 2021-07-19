import sys
sys.path.append("..")
import argparse
from torch import optim
from torch.utils.data import DataLoader
import os
import logging
from dirichlet_data import *
from model import *

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
print('Cuda: ', os.environ["CUDA_VISIBLE_DEVICES"])
cuda = torch.cuda.is_available()
device = torch.device('cuda') if cuda else torch.device('cpu')

def main(n_round, n_client,activate_rate, n_epoch, lr,lr_decay, alpha,csd_importance, batch_size=32, p =0, time =1):
    n_client = n_client
    alpha = alpha
    activate_rate = activate_rate
    n_epoch = n_epoch
    train_batch_size = batch_size
    pruing_p = p
    n_round = n_round
    csd_importance = csd_importance
    eps = 1e-5
    lr = lr
    lr_decay = lr_decay
    time = time

    client_idx_list = [i for i in range(n_client)]
    activate_client_num = int(activate_rate * n_client)
    assert activate_client_num > 1

    server_model = BasicCNN().to(device)
    print(server_model)
    client_model_set = [BasicCNN() for _ in range(n_client)]
    server_omega = dict()
    client_omega_set = [dict() for _ in range(n_client)]
    train_set_group, test_set = dirichlet_data(data_name='cifar', num_users=n_client, alpha=alpha)

    train_loader_group = []
    test_loader = DataLoader(test_set, batch_size=4096,
                             **({'num_workers': 2, 'shuffle': True, 'pin_memory': False} if cuda else {}))

    client_alpha = get_client_alpha(train_set_group)

    criterion = nn.CrossEntropyLoss()

    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        filename='./hyperlog/{}_alpha{}_pruning{}_lambda{}_e{}_b{}_lr{}_{}.log'.format(args.logname, alpha,
                                                                                          pruing_p, csd_importance,
                                                                                          n_epoch, train_batch_size, lr,
                                                                                          time), filemode='w')

    def server_excute(lr):
        for train_set in train_set_group:
            train_loader_group.append(DataLoader(train_set, batch_size=train_batch_size,
                                                 **({'num_workers': 2, 'shuffle': True,
                                                     'pin_memory': False} if cuda else {})))

        for client_idx in range(n_client):
            get_client_omega(client_idx)

        # lr = args.lr
        # lr_decay = args.decay  # 0.997

        for r in range(n_round):
            round_num = r + 1
            activate_clients = random.sample(client_idx_list, activate_client_num)
            alpha_sum = sum([client_alpha[idx] for idx in activate_clients]) + eps
            for client_idx in activate_clients:
                # print(f'client = {client_idx + 1}')
                client_update(client_idx, round_num, lr)

            local_acc, self_acc = 0.1,0.1 # test_client_model()
            if pruing_p > 0:
                pruning_clients(p=pruing_p)
            new_param = {}
            new_omega = {}

            with torch.no_grad():
                for name, param in server_model.named_parameters():
                    new_param[name] = param.data.zero_()
                    new_omega[name] = server_omega[name].data.zero_()
                    for client_idx in activate_clients:
                        new_param[name] += (client_alpha[client_idx]/alpha_sum) * client_omega_set[client_idx][name] * \
                                           client_model_set[client_idx].state_dict()[name].to(device)
                        new_omega[name] += (client_alpha[client_idx]/alpha_sum) * client_omega_set[client_idx][name]
                    new_param[name] /= (new_omega[name] + eps)

                if pruing_p > 0:
                    pruning_server(pruing_p, new_param, new_omega)
                for name, param in server_model.named_parameters():
                    server_model.state_dict()[name].data.copy_(new_param[
                                                                   name])  # https://discuss.pytorch.org/t/how-can-i-modify-certain-layers-weight-and-bias/11638
                    server_omega[name] = new_omega[name]
                    for client_idx in range(n_client):
                        client_model_set[client_idx].state_dict()[name].data.copy_(new_param[name].cpu())
                        client_omega_set[client_idx][name].data.copy_(new_omega[name])

            acc = test_server_model()
            print(f'******* round = {r + 1} | acc = {round(acc, 4)} |  local acc: {round(local_acc,4)} | self acc: {round(self_acc, 4)}*******')
            logging.info( f'round: {r + 1}, acc: {round(acc, 4)}, local acc: {round(local_acc, 4)}, self acc: {round(self_acc,4)}')

            lr = lr * lr_decay

    def get_csd_loss(client_idx, mu, omega, round_num):
        loss_set = []
        for name, param in client_model_set[client_idx].named_parameters():
            theta = client_model_set[client_idx].state_dict()[name]
            # omega_dropout = torch.rand(omega[name].size()).cuda() if cuda else torch.rand(omega[name].size())
            # omega_dropout[omega_dropout>0.5] = 1.0
            # omega_dropout[omega_dropout <= 0.5] = 0.0

            loss_set.append((0.5 / round_num) * (omega[name] * ((theta - mu[name]) ** 2)).sum())

        return sum(loss_set)

    def client_update(client_idx, round_num, lr):
        log_ce_loss = 0
        log_csd_loss = 0
        client_model_set[client_idx] = client_model_set[client_idx].to(device)
        optimizer = optim.SGD(client_model_set[client_idx].parameters(), lr=lr)

        new_omega = dict()
        new_mu = dict()
        data_alpha = client_alpha[client_idx]
        server_model_state_dict = server_model.state_dict()
        for name, param in client_model_set[client_idx].named_parameters():
            # new_omega[name] = 1 / (1 - data_alpha) * (server_omega[name] - data_alpha * client_omega_set[client_idx][name])
            # new_mu[name] = 1 / (1 - data_alpha) * (server_omega[name] * server_model_state_dict[name] -
            #                 data_alpha * client_omega_set[client_idx][name] * client_model_set[client_idx].state_dict()[name]) /\
            #                (new_omega[name] + args.eps)
            new_omega[name] = deepcopy(server_omega[name])
            new_mu[name] = deepcopy(server_model_state_dict[name])

        client_model_set[client_idx].train()
        for epoch in range(n_epoch):
            for batch_idx, (data, target) in enumerate(train_loader_group[client_idx]):

                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = client_model_set[client_idx](data)
                ce_loss = criterion(output, target)

                csd_loss = get_csd_loss(client_idx, new_mu, new_omega, round_num) if csd_importance > 0 else 0

                ce_loss.backward(retain_graph=True)

                # for name, param in client_model_set[client_idx].named_parameters():
                #     if param.grad is not None:
                #         client_omega_set[client_idx][name] += (len(target) / len(
                #             train_set_group[client_idx])) * param.grad.data.clone() ** 2

                optimizer.zero_grad()
                loss = ce_loss + csd_importance * csd_loss
                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(client_model_set[client_idx].parameters(), args.clip)
                optimizer.step()

                log_ce_loss += ce_loss.item()
                log_csd_loss += csd_loss.item() if csd_importance > 0 else 0

        log_ce_loss /= n_epoch
        log_csd_loss /= (n_epoch / csd_importance) if csd_importance > 0 else 1
        print(
            f'client_idx = {client_idx + 1} | loss = {log_ce_loss + log_csd_loss} (ce: {log_ce_loss} + csd: {log_csd_loss})')
        logging.info(
            f'client: {client_idx + 1}, log_ce_loss: {round(log_ce_loss, 8)}, log_csd_loss: {round(log_csd_loss, 8)}')
        # client_model_set[client_idx] = client_model_set[client_idx].cpu()

        for name, param in client_model_set[client_idx].named_parameters():
            client_omega_set[client_idx][name] = torch.zeros_like(param.data).to(device)
        # client_model_set[client_idx] = client_model_set[client_idx].to(device)
        for batch_idx, (data, target) in enumerate(train_loader_group[client_idx]):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = client_model_set[client_idx](data)
            ce_loss = criterion(output, target)
            ce_loss.backward(retain_graph=True)
            for name, param in client_model_set[client_idx].named_parameters():
                if param.grad is not None:
                    client_omega_set[client_idx][name] += (len(target) / len(
                        train_set_group[client_idx])) * param.grad.data.clone() ** 2
            optimizer.zero_grad()

        client_model_set[client_idx] = client_model_set[client_idx].cpu()

    def get_client_omega(client_idx):
        for name, param in deepcopy(client_model_set[client_idx]).named_parameters():
            param.data.zero_()
            client_omega_set[client_idx][name] = param.data.to(device)

    def test_server_model():
        server_model.eval()
        correct = 0
        n_test = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            scores = server_model(data)
            _, predicted = scores.max(1)
            correct += predicted.eq(target.view_as(predicted)).sum().item()
            n_test += data.size(0)
        return correct / n_test

    def test_client_model():
        local_accuacy = 0
        for client_idx in range(n_client):
            client_model_set[client_idx] = client_model_set[client_idx].to(device)
            correct = 0
            n_test = 0
            client_model_set[client_idx].eval()
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                # data = data.view(len(data), -1)
                scores = client_model_set[client_idx].eval()(data)
                _, predicted = scores.max(1)
                correct += predicted.eq(target.view_as(predicted)).sum().item()
                n_test += data.size(0)

            local_accuacy += client_alpha[client_idx] * correct / n_test
            client_model_set[client_idx].train()
            client_model_set[client_idx] = client_model_set[client_idx].cpu()

        return local_accuacy, 0.1 #local_correct / local_n_test

    def get_threshold(p, matrix):
        rank_matrix, _ = matrix.view(-1).sort(descending=True)
        threshold = rank_matrix[int(p * len(rank_matrix))]
        return threshold

    def pruning_clients(p):
        for client_idx in range(n_client):
            for name, param in deepcopy(client_model_set[client_idx]).named_parameters():
                unpruing_omega = client_omega_set[client_idx][name]
                threshold = get_threshold(p, unpruing_omega)
                client_omega_set[client_idx][name][unpruing_omega < threshold] = unpruing_omega[
                    unpruing_omega < threshold].mean().item()
                # unpruing_param = deepcopy(param.data.detach())
                # unpruing_param[unpruing_omega < threshold] = 1e-5
                # param.data.copy_(unpruing_param)

    def pruning_server(p, new_param, new_omega):
        for name in new_param.keys():
            unpruing_omega = new_omega[name]
            threshold = get_threshold(p, unpruing_omega)
            new_omega[unpruing_omega < threshold] = unpruing_omega[unpruing_omega < threshold].mean().item()
            # unpruing_param = deepcopy(server_model.state_dict()[name].data.detach())
            # new_param[name][unpruing_omega < threshold] = unpruing_param[unpruing_omega < threshold]

    def init():
        server_model.apply(weight_init)
        init_state_dict = server_model.state_dict()
        for client_idx in range(n_client):
            client_model_set[client_idx].load_state_dict(init_state_dict)
            client_model_set[client_idx].load_state_dict(init_state_dict)
            for name, param in deepcopy(client_model_set[client_idx]).named_parameters():
                client_omega_set[client_idx][name] = torch.zeros_like(param.data).to(device)
        for name, param in deepcopy(server_model).named_parameters():
            server_omega[name] = torch.zeros_like(param.data).to(device)

    init()
    server_excute(lr=lr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_round', type=int, default=100)
    parser.add_argument('--n_client', type=int, default=20)
    parser.add_argument('--activate_rate', type=float, default=1.0)
    parser.add_argument('--n_epoch', type=int, default=5)
    parser.add_argument('--gpu', type=str, default='2')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--alpha', type=float, default=100)
    parser.add_argument('--decay', type=float, default=1.0)
    parser.add_argument('--pruning_p', type=float, default=0)
    parser.add_argument('--csd_importance', type=float, default=0)
    parser.add_argument('--eps', type=float, default=1e-5)
    parser.add_argument('--clip', type=float, default=10)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--logname', type=str, default='BasicCNN_ours+offline')
    parser.add_argument('--time', type=str, default='1')
    # mnist_noniid_mlp_pds

    args = parser.parse_args()

    print(f"n_round {args.n_round}, n_client {args.n_client}, activate rate {args.activate_rate}, n_epoch {args.n_epoch}, lr {args.lr}, alpha {args.alpha}, batch_size {args.train_batch_size}, pruning_p {args.pruning_p}")

    main(args.n_round, args.n_client, args.activate_rate, args.n_epoch, args.lr, args.decay, args.alpha, args.csd_importance, batch_size=args.train_batch_size, p =args.pruning_p, time =1)
