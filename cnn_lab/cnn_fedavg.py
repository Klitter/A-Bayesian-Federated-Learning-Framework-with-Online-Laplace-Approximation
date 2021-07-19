import argparse
import sys
sys.path.append("..")
import wandb
wandb.login(key="put your wandb key here")

import argparse
import os
import logging
import torch

from copy import deepcopy
from torch import optim
from torch import nn

from dirichlet_data import *
from model import BasicCNN as Model
from model import weight_init

global device
global eps

eps = 1e-5
checkpoint_interval = 10

class FedSystem(object):
    """Simulate the environment and learning process of federated learning.

    Set a given number of clients and training data with a given degree
    of heterogeneity.
    Use SGD to optimize the client's local model and use different methods
    to aggregate models
    Record the accuracy of the aggregate model and local model for each
    round.

    Attributes:
        None
    """

    def __init__(self, args, filename_suffix):
        """Initialize the federated learning environment.

        Set a given number of clients and training datasets with a given
        degree of heterogeneity.

        Args:
             --n_round, type=int, default=20
             --n_client, type=int, default=5
             --activate_rate, type=float, default=1.0
             --n_epoch, type=int, default=1
             --gpu, type=str, default='2'
             --lr, type=float, default=1e-2
             --alpha, type=float, default=0.01
             --decay, type=float, default=1.0
             --pruning_p, type=float, default=0
             --csd_importance, type=float, default=0
             --eps, type=float, default=1e-5
             --clip, type=float, default=10
             --train_batch_size, type=int, default=32
             --test_batch_size', type=int, default=64
             --logname, type=str, default='BasicCNN_ours'
             --time, type=str, default='1'
        """
        self.args = args

        self.server_model = Model().to(device)
        self.client_model_set = [Model() for _ in range(args.n_client)]
        self.server_omega = dict()
        self.client_omega_set = [dict() for _ in range(args.n_client)]

        self.train_set_group, self.test_set = dirichlet_data(data_name=args.data, num_users=args.n_client, alpha=args.alpha)
        self.train_loader_group = [DataLoader(train_set, batch_size=args.train_batch_size,
                                              **({'num_workers': 2,
                                                  'shuffle': True,
                                                  'pin_memory': False}
                                                 if cuda else {}))
                                   for train_set in self.train_set_group]
        self.test_loader = DataLoader(self.test_set, batch_size=4096,
                                      **({'num_workers': 2,
                                          'shuffle': True,
                                          'pin_memory': False}
                                         if cuda else {}))
        self.client_alpha = get_client_alpha(self.train_set_group)

        self.criterion = nn.CrossEntropyLoss()

        if not os.path.exists("./hyperlog/"):
            os.mkdir("./hyperlog/")

        logging.basicConfig(level=logging.INFO,
                            format='%(message)s',
                            filename="./hyperlog/{}_{}.log".format(args.logname, filename_suffix),
                            filemode='w')

        if not os.path.exists("./../checkpoint/"):
            os.mkdir("./../checkpoint/")

        self.checkpoint_path = "./../checkpoint/{}_{}.pth".format(args.logname, filename_suffix)

    def server_excute(self):
        start = 0
        acc = 0
        local_acc = 0
        best_acc = 0
        best_local_acc = 0

        self.server_model.apply(weight_init)
        init_state_dict = self.server_model.state_dict()
        for client_idx in range(args.n_client):
            self.client_model_set[client_idx].load_state_dict(init_state_dict)
            self.client_model_set[client_idx].load_state_dict(init_state_dict)
            for name, param in deepcopy(self.client_model_set[client_idx]).named_parameters():
                self.client_omega_set[client_idx][name] = torch.zeros_like(param.data).to(device)
        for name, param in deepcopy(self.server_model).named_parameters():
            self.server_omega[name] = torch.zeros_like(param.data).to(device)

        if os.path.exists(self.checkpoint_path) and args.resume:
            checkpoint = torch.load(self.checkpoint_path)
            start = max(0, checkpoint['start'])
            print('Start Round -- ', start)
            if start > 0:
                print("Loading checkpoint from ", self.checkpoint_path)
                self.client_omega_set = deepcopy(checkpoint['client_omega_set'])
                self.server_omega = deepcopy(checkpoint['server_omega'])
                for c_idx, client_model in enumerate(self.client_model_set):
                    client_model.load_state_dict(checkpoint['client_model_set'][c_idx])
                self.server_model.load_state_dict(checkpoint['server_model'])
            del checkpoint

        for client_idx in range(args.n_client):
            self.get_client_omega(client_idx)

        # lr = args.lr
        # lr_decay = args.decay  # 0.997
        client_idx_list = [i for i in range(args.n_client)]
        activate_client_num = int(args.activate_rate * args.n_client)
        assert activate_client_num > 1

        for r in range(start, args.n_round):
            round_num = r + 1
            print('round_num -- ',round_num)

            if args.activate_rate<1:
                activate_clients = random.sample(client_idx_list, activate_client_num)
            else:
                activate_clients = client_idx_list
            alpha_sum = sum([self.client_alpha[idx] for idx in activate_clients])
            for client_idx in activate_clients:
                # print(f'client = {client_idx + 1}')
                self.client_update(client_idx, round_num, args.lr)

            local_acc, self_acc = self.test_client_model()
            if args.pruing_p > 0:
                self.pruning_clients(p=args.pruing_p)
            new_param = {}

            with torch.no_grad():
                for name, param in self.server_model.named_parameters():
                    new_param[name] = param.data.zero_()

                    for client_idx in activate_clients:
                        new_param[name] += (self.client_alpha[client_idx]/alpha_sum) * self.client_model_set[client_idx].state_dict()[name].to(device)

                    self.server_model.state_dict()[name].data.copy_(
                        new_param[name])  # https://discuss.pytorch.org/t/how-can-i-modify-certain-layers-weight-and-bias/11638

                    for client_idx in range(args.n_client):
                        self.client_model_set[client_idx].state_dict()[name].data.copy_(new_param[name].cpu())

                for name, param in self.server_model.named_parameters():
                    self.server_model.state_dict()[name].data.copy_(new_param[name])  # https://discuss.pytorch.org/t/how-can-i-modify-certain-layers-weight-and-bias/11638
                    for client_idx in range(args.n_client):
                        self.client_model_set[client_idx].state_dict()[name].data.copy_(new_param[name].cpu())


            acc = self.test_server_model()
            if acc > best_acc: best_acc = acc
            if local_acc > best_local_acc: best_local_acc = local_acc
            print(f'******* round = {r + 1} | acc = {round(acc, 4)} |  local acc: {round(local_acc,4)}*******')
            wandb.log({'round': r+1, 'accuracy': acc, 'local_accuracy': local_acc})
            logging.info( f'round: {r + 1}, acc: {round(acc, 4)}, local acc: {round(local_acc, 4)}, self acc: {round(self_acc,4)}')

            # lr = lr * args.lr_decay

            if round_num % checkpoint_interval == 0 and args.resume:
                torch.save({
                    'start': round_num,
                    'client_model_set': [client_m.state_dict() for client_m in self.client_model_set],
                    'client_omega_set': self.client_omega_set,
                    'server_model': self.server_model.state_dict(),
                    'server_omega': self.server_omega,
                },
                    self.checkpoint_path)

                print('Saving checkpoint to ', self.checkpoint_path)

        return acc, local_acc, best_acc, best_local_acc

    def get_csd_loss(self, client_idx, mu, omega, round_num):
        loss_set = []
        for name, param in self.client_model_set[client_idx].named_parameters():
            theta = self.client_model_set[client_idx].state_dict()[name]
            # omega_dropout = torch.rand(omega[name].size()).cuda() if cuda else torch.rand(omega[name].size())
            # omega_dropout[omega_dropout>0.5] = 1.0
            # omega_dropout[omega_dropout <= 0.5] = 0.0

            loss_set.append((0.5 / round_num) * (omega[name] * ((theta - mu[name]) ** 2)).sum())

        return sum(loss_set)

    def client_update(self, client_idx, round_num, lr):
        log_ce_loss = 0
        log_csd_loss = 0
        self.client_model_set[client_idx] = self.client_model_set[client_idx].to(device)
        optimizer = optim.SGD(self.client_model_set[client_idx].parameters(), lr=lr)

        new_omega = dict()
        new_mu = dict()
        data_alpha = self.client_alpha[client_idx]
        server_model_state_dict = self.server_model.state_dict()
        # for name, param in self.client_model_set[client_idx].named_parameters():
        #     new_omega[name] = deepcopy(self.server_omega[name])
        #     new_mu[name] = deepcopy(server_model_state_dict[name])

        self.client_model_set[client_idx].train()
        for epoch in range(args.n_epoch):
            for batch_idx, (data, target) in enumerate(self.train_loader_group[client_idx]):

                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = self.client_model_set[client_idx](data)
                ce_loss = self.criterion(output, target)

                # for name, param in self.client_model_set[client_idx].named_parameters():
                #     if param.grad is not None:
                #         self.client_omega_set[client_idx][name] += (len(target) / len(
                #             self.train_set_group[client_idx])) * param.grad.data.clone() ** 2

                loss = ce_loss
                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.client_model_set[client_idx].parameters(), args.clip)
                optimizer.step()

                log_ce_loss += ce_loss.item()
                log_csd_loss +=  0

        log_ce_loss /= args.n_epoch
        log_csd_loss /= (args.n_epoch / args.csd_importance) if args.csd_importance > 0 else 1
        print(
            f'client_idx = {client_idx + 1} | loss = {log_ce_loss + log_csd_loss} (ce: {log_ce_loss} + csd: {log_csd_loss})')
        logging.info(
            f'client: {client_idx + 1}, log_ce_loss: {round(log_ce_loss, 8)}, log_csd_loss: {round(log_csd_loss, 8)}')
        self.client_model_set[client_idx] = self.client_model_set[client_idx].cpu()

    def get_client_omega(self, client_idx):
        for name, param in deepcopy(self.client_model_set[client_idx]).named_parameters():
            param.data.zero_()
            self.client_omega_set[client_idx][name] = param.data.to(device)

    def test_server_model(self):
        self.server_model.eval()
        correct = 0
        n_test = 0
        for data, target in self.test_loader:
            data, target = data.to(device), target.to(device)
            scores = self.server_model(data)
            _, predicted = scores.max(1)
            correct += predicted.eq(target.view_as(predicted)).sum().item()
            n_test += data.size(0)
        return correct / n_test

    def test_client_model(self):
        local_accuacy = 0
        for client_idx in range(args.n_client):
            self.client_model_set[client_idx] = self.client_model_set[client_idx].to(device)
            correct = 0
            n_test = 0
            self.client_model_set[client_idx].eval()
            for data, target in self.test_loader:
                data, target = data.to(device), target.to(device)
                # data = data.view(len(data), -1)
                scores = self.client_model_set[client_idx].eval()(data)
                _, predicted = scores.max(1)
                correct += predicted.eq(target.view_as(predicted)).sum().item()
                n_test += data.size(0)

            local_accuacy += self.client_alpha[client_idx] * correct / n_test
            self.client_model_set[client_idx].train()
            self.client_model_set[client_idx] = self.client_model_set[client_idx].cpu()

        return local_accuacy, 0.1 #local_correct / local_n_test

    def get_threshold(self, p, matrix):
        rank_matrix, _ = matrix.view(-1).sort(descending=True)
        threshold = rank_matrix[int(p * len(rank_matrix))]
        return threshold

    def pruning_clients(self, p):
        for client_idx in range(args.n_client):
            for name, param in deepcopy(self.client_model_set[client_idx]).named_parameters():
                unpruing_omega = self.client_omega_set[client_idx][name]
                threshold = self.get_threshold(p, unpruing_omega)
                self.client_omega_set[client_idx][name][unpruing_omega < threshold] = unpruing_omega[
                    unpruing_omega < threshold].mean().item()
                # unpruing_param = deepcopy(param.data.detach())
                # unpruing_param[unpruing_omega < threshold] = 1e-5
                # param.data.copy_(unpruing_param)

    def pruning_server(self, p, new_param, new_omega):
        for name in new_param.keys():
            unpruing_omega = new_omega[name]
            threshold = self.get_threshold(p, unpruing_omega)
            new_omega[unpruing_omega < threshold] = unpruing_omega[unpruing_omega < threshold].mean().item()
            # unpruing_param = deepcopy(server_model.state_dict()[name].data.detach())
            # new_param[name][unpruing_omega < threshold] = unpruing_param[unpruing_omega < threshold]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='cifar10')
    parser.add_argument('--n_round', type=int, default=8)
    parser.add_argument('--n_client', type=int, default=5)
    parser.add_argument('--activate_rate', type=float, default=1.0)
    parser.add_argument('--n_epoch', type=int, default=1)
    parser.add_argument('--gpu', type=str, default='2')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--decay', type=float, default=1.0)
    parser.add_argument('--pruing_p', type=float, default=0)
    parser.add_argument('--csd_importance', type=float, default=0)
    parser.add_argument('--eps', type=float, default=1e-5)
    parser.add_argument('--clip', type=float, default=10)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--logname', type=str, default='Avg_Cifar10')
    parser.add_argument('--method', type=str, default='Avg')
    parser.add_argument('--projectname', type=str, default='Cifar10')
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--time', type=int, default=1)

    args = parser.parse_args()

    config = {'n_client': args.n_client,
              'alpha': args.alpha,
              'n_epoch': args.n_epoch,
              'csd_importance': args.csd_importance,
              'lr': args.lr,
              'decay': args.decay,
              'pruing_p': args.pruing_p,
              'train_batch_size': args.train_batch_size,
              'n_round': args.n_round,
              'activate_rate': args.activate_rate,
              'logname': args.logname,
              'method': args.method,
              'projectname': args.projectname
              }

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print("Cuda: ", os.environ['CUDA_VISIBLE_DEVICES'])
    cuda = torch.cuda.is_available()
    device = torch.device('cuda') if cuda else torch.device('cpu')

    print(f"n_round {args.n_round}, n_client {args.n_client}, activate rate {args.activate_rate}, n_epoch {args.n_epoch}, lr {args.lr}, alpha {args.alpha}, batch_size {args.train_batch_size}, pruing_p {args.pruing_p}")

    filename_suffix = "{}_N{}a{}p{}l{}e{}b{}lr{}t{}".format(
        args.logname,
        args.n_client,
        args.alpha,
        args.pruing_p,
        args.csd_importance,
        args.n_epoch,
        args.train_batch_size,
        args.lr,
        args.time)
    tags = [filename_suffix.split('_t')[0]]
    wandb.init(config=config, project=args.projectname, reinit=True, resume='allow',
               id=filename_suffix, tags=tags)

    fed_sys = FedSystem(args, filename_suffix)
    acc, local_acc, best_acc, best_local_acc = fed_sys.server_excute()
    wandb.run.summary["final_accuracy"] = acc
    wandb.run.summary["final_local_accuracy"] = local_acc
    wandb.run.summary["bset_accuracy"] = best_acc
    wandb.run.summary["best_local__accuracy"] = best_local_acc


    # main(args.n_round, args.n_client, args.activate_rate, args.n_epoch, args.lr, args.decay, args.alpha, args.csd_importance, batch_size=args.train_batch_size, p =args.pruning_p, time =1)
