import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.init as init

import numpy as np
import csv
import random
import copy
import math

import n_output
import topm

from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        init.xavier_uniform_(self.linear.weight)
        init.zeros_(self.linear.bias) 
    
    def forward(self, x):
        return self.linear(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='gamma, shuttle')
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--runs', type=int, default=50)

    parser.add_argument('--mech', help='nonpriv, topm, pm_sub, nopm')
    parser.add_argument('--eps', type=float, help='the privacy budget')
    parser.add_argument('--range', type=float, default=0.3, help='range of gradient')

    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0, help='regularization factor')
    parser.add_argument('--epoch', type=int, default=5, help='local epoch')

    parser.add_argument('--device', default='cpu')
    parser.add_argument('--rounds', type=int, default=50)
    parser.add_argument('--total_clients', type=int, default=500)
    parser.add_argument('--fraction', type=float, default=0.1, help='fraction of total nodes for each round')
    parser.add_argument('--batch_size', type=int, default=50, help='the batch size for each node')

    args = parser.parse_args()

    rng = np.random.default_rng(seed=args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    performance_lst = []

    if args.dataset == 'shuttle':
        data = np.loadtxt('data/shuttle.tst')
        X = data[:, :-1]
        y = data[:, -1].astype(int) - 1  # Convert labels to zero-based index
    elif args.dataset == 'gamma':
        with open('data/gamma.data', 'r') as file:
            lines = file.readlines()

        data = []
        labels = []
        label_map = {'g': 0, 'h': 1} 

        for line in lines:
            parts = line.strip().split(',')
            features = list(map(float, parts[:-1]))
            label = label_map[parts[-1]]
            data.append(features)
            labels.append(label)

        X = np.array(data)
        y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    total_clients = args.total_clients
    train_dataset = TensorDataset(X_train, y_train)
    adjusted_length = (len(train_dataset) // total_clients) * total_clients
    train_dataset = torch.utils.data.Subset(train_dataset, range(adjusted_length))
    client_datasets = random_split(train_dataset, [len(train_dataset) // total_clients] * total_clients)

    input_dim = X_train.shape[1]
    output_dim = len(set(y_train.numpy())) 

    batch_size = args.batch_size
    frac = args.fraction
    clients_per_round = int(total_clients * frac)
    num_items_per_client = len(train_dataset) // total_clients

    # Construct LDP mechanism
    d = input_dim * output_dim
    k = max(1,min(d, math.floor(args.eps/2.5)))
    if args.mech == 'topm':
        if args.eps > 1.7:
            mech = n_output.NOUTPUT(d, args.eps, rng, 3, args.range, k)
        else:
            mech = topm.TOPM(d, args.eps, rng, range=args.range, k=k)
    elif args.mech == 'pm_sub':
        mech = topm.TOPM(d, args.eps, rng, range=args.range)
    elif args.mech == 'nopm':
        mech = n_output.NOUTPUT(d, args.eps, rng, 3, args.range, 1)

    total_accuracies = []
    for run in range(args.runs):
        accuracies = []
        clients = []
        for i in range(total_clients):
            clients.append({})
            clients[i]['dataset'] = client_datasets[i]
            clients[i]['train_loader'] = DataLoader(client_datasets[i], batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)

        test_dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, pin_memory=True, num_workers=4)
        global_model = LogisticRegressionModel(input_dim, output_dim)

        device = torch.device(args.device)
        global_model.to(device)

        tmp_client_model = LogisticRegressionModel(input_dim, output_dim)
        tmp_client_model.to(device)

        # train
        for r in range(args.rounds):
            sampled_clients = random.sample(range(len(clients)), clients_per_round)
            gradient_differences_lst = []
            noisy_gradient_differences_lst = []
            initial_state_dict = copy.deepcopy(global_model.state_dict())
            for i in sampled_clients:
                tmp_client_model = copy.deepcopy(global_model)
                tmp_client_model.train()
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.SGD(tmp_client_model.parameters(), lr = args.lr, weight_decay=args.weight_decay)
                train_loader = clients[i]['train_loader']
                for e in range(args.epoch):
                    for batch_idx, (data, target) in enumerate(train_loader):
                        data, target = data.to(device), target.to(device)
                        optimizer.zero_grad()
                        output = tmp_client_model(data)
                        loss = criterion(output, target)
                        loss.backward()
                        optimizer.step()
                gradient_difference = {}
                for key in initial_state_dict.keys():
                    gradient_difference[key] = tmp_client_model.state_dict()[key] - initial_state_dict[key]
                gradient_differences_lst.append(gradient_difference)
            
            # Perturbation
            if args.mech == 'nonpriv':
                # Aggregate noisy gradient differences and update the global model
                new_global_params = copy.deepcopy(global_model.state_dict())
                for key in new_global_params.keys():
                    avg_grad_diff = sum([grad_diff[key] for grad_diff in gradient_differences_lst]) / len(gradient_differences_lst)
                    new_global_params[key] += avg_grad_diff  # Update parameters using the average gradient difference

                global_model.load_state_dict(new_global_params)
            else:
                for grad_diff in gradient_differences_lst:
                    w_shape = grad_diff['linear.weight'].shape
                    flat_w_grad_diff = grad_diff['linear.weight'].reshape(-1)
                    len_w = len(flat_w_grad_diff)
                    
                    b_shape = grad_diff['linear.bias'].shape
                    flat_b_grad_diff = grad_diff['linear.bias'].reshape(-1)
                    len_b = len(flat_b_grad_diff)
                    
                    concat_grad_diff = torch.cat([flat_w_grad_diff, flat_b_grad_diff], dim=0)
                    concat_grad_diff = concat_grad_diff.clamp(-args.range, args.range)
                    
                    # Convert to numpy for noise addition, then back to torch tensor
                    concat_grad_diff_np = np.array(concat_grad_diff)
                    noisy_grad_diff_np = mech.multi(args.mech, concat_grad_diff_np)
                    noisy_grad_diff = torch.tensor(noisy_grad_diff_np)
                    
                    # Reconstruct the noisy gradient difference
                    noisy_grad_diff_dict = copy.deepcopy(grad_diff)
                    noisy_grad_diff_dict['linear.weight'] = noisy_grad_diff[:len_w].reshape(w_shape)
                    noisy_grad_diff_dict['linear.bias'] = noisy_grad_diff[len_w:].reshape(b_shape)
                    noisy_gradient_differences_lst.append(noisy_grad_diff_dict)

                # Aggregate noisy gradient differences and update the global model
                new_global_params = copy.deepcopy(global_model.state_dict())
                for key in new_global_params.keys():
                    avg_grad_diff = sum([grad_diff[key] for grad_diff in noisy_gradient_differences_lst]) / len(noisy_gradient_differences_lst)
                    new_global_params[key] += avg_grad_diff  # Update parameters using the average gradient difference

                global_model.load_state_dict(new_global_params)

            # test
            global_model.eval()
            test_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = global_model(data)
                    test_loss += criterion(output, target).item()
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(test_loader.dataset)
            accuracy = 100. * correct / len(test_loader.dataset)
            print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.1f}%)\n')
            accuracies.append(accuracy)
        total_accuracies.append(np.array(accuracies))
    total_accuracies = np.array(total_accuracies)
    total_accuracies = np.mean(total_accuracies, axis=0)

    # Specify the CSV file name
    csv_file = args.dataset + '_' + args.mech + '_' + str(args.weight_decay) + '_' + str(args.seed) + '.csv'
    # Generate round numbers
    rounds = np.arange(1, len(total_accuracies) + 1)

    # Save the round numbers and accuracies to a CSV file
    with open('fedl/'+csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Round', 'Accuracy'])  # Header
        for round_num, accuracy in zip(rounds, total_accuracies):
            writer.writerow([round_num, accuracy])

        