import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim import Adam
import numpy as np
import math


def flatten(x):
    return x.view(x.size(0), -1)


class C(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.multiplier = 32
        self.conv = nn.Sequential(nn.Conv2d(in_channel, self.multiplier, kernel_size=3, stride=2, padding=1),
                                  nn.BatchNorm2d(self.multiplier),
                                  nn.ReLU(),
                                  nn.Conv2d(self.multiplier, 2 * self.multiplier, kernel_size=3, stride=2, padding=1),
                                  nn.BatchNorm2d(2 * self.multiplier),
                                  nn.ReLU(),
                                  nn.Conv2d(2 * self.multiplier, 4 * self.multiplier, kernel_size=3, stride=2, padding=1),
                                  # nn.BatchNorm2d(4 * self.multiplier),
                                  # nn.ReLU(),
                                  # nn.Conv2d(4 * self.multiplier, 4 * self.multiplier, kernel_size=3, stride=2,
                                  #           padding=1)
                                  )

    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, in_channel, global_dim=64):
        super().__init__()
        self.c = C(in_channel)
        # self.f = nn.Sequential(nn.Linear(128*2*2, 512),
        #                        nn.ReLU(),
        #                        nn.Linear(512, global_dim))
        multiplier = 128
        self.f = nn.Sequential(nn.Conv2d(multiplier, multiplier, kernel_size=3, stride=2, padding=1),
                               nn.BatchNorm2d(multiplier),
                               nn.ReLU(),
                               nn.Conv2d(multiplier, multiplier, kernel_size=3, stride=2, padding=1),
                               )

    def forward(self, x):
        local_feature_map = self.c(x)
        #print('c size ', local_feature_map.size())
        #local_ = flatten(local_feature_map)
        global_feature = self.f(local_feature_map)
        return local_feature_map, global_feature


class Critic(nn.Module):
    def __init__(self, global_dim=64):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(global_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.f(x)


def log_unit_gaussian(x):
    return -0.5 * (math.log(2 * math.pi) + x.pow(2).sum(dim=1))


class ConvT(nn.Module):
    def __init__(self, in_channel):
        self.multiplier = in_channel
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channel, self.multiplier, kernel_size=1),
                                  nn.BatchNorm2d(self.multiplier),
                                  nn.ReLU(),
                                  nn.Conv2d(self.multiplier, self.multiplier, kernel_size=1),
                                  nn.BatchNorm2d(self.multiplier),
                                  nn.ReLU(),
                                  nn.Conv2d(self.multiplier, 1, kernel_size=1)
                                  )

    def forward(self, global_, local_):
        #b, c = global_.size()
        b, c, h, w = local_.size()
        global_ = global_.repeat(1, 1, h, w)

        cat = torch.cat([global_, local_], dim=1)
        out = self.conv(cat)
        assert (b, 1, h, w) == out.size()
        return out


class DIM(nn.Module):
    def __init__(self, in_channel, global_dim=64):
        super().__init__()
        self.global_dim = global_dim
        self.enc = Encoder(in_channel, global_dim)
        # self.t = nn.Sequential(nn.Linear(128*2*2 + global_dim, 512),
        #                        nn.ReLU(),
        #                        nn.Linear(512, 128),
        #                        nn.ReLU(),
        #                        nn.Linear(128, 1))
        self.t = ConvT(128*2)

        self.critic = Critic(global_dim)

    def forward(self, x):
        local_, global_ = self.enc(x)
        # pair = torch.cat([local_, global_], dim=1)
        paired_scores = self.t(global_, local_)


        global_shuffle = global_[torch.randperm(global_.size()[0])]
        unpaired_scores = self.t(global_shuffle, local_)

        mi = donsker_varadhan_loss(paired_scores, unpaired_scores)

        # real_samples = torch.randn(x.size(0), self.global_dim).to(device)
        # fake_samples = global_
        #
        # real_probs = self.critic(real_samples)
        # fake_probs = self.critic(fake_samples)
        #
        # log_probs = torch.log(real_probs) + torch.log(1. - fake_probs)
        nll = -log_unit_gaussian(global_)  # Gaussianize
        regulation = nll.mean()

        loss = mi + 0.1 * regulation
        return loss, global_


def log_mean_exp(x):
    """Stable log mean exp."""
    max_ = x.max()
    return max_ + (x-max_).exp().mean().log()


def donsker_varadhan_loss(paired_scores, unpaired_scores):
    mi = paired_scores.mean() - log_mean_exp(unpaired_scores)
    return -mi


class Classifier(nn.Module):
    def __init__(self, dim=128, n_classes=10):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, n_classes),
        )

    def forward(self, x):
        return self.f(x.squeeze())


if __name__ == '__main__':

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    dataset = datasets.MNIST('data/MNIST', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.Resize((32, 32)),
                                 transforms.ToTensor(),
                                 # transforms.Normalize((0.1307,), (0.3081,))
                             ]))

    test_dataset = datasets.MNIST('data/MNIST', train=False, download=True,
                             transform=transforms.Compose([
                                 transforms.Resize((32, 32)),
                                 transforms.ToTensor(),
                                 # transforms.Normalize((0.1307,), (0.3081,))
                             ]))

    train_loader = DataLoader(dataset=dataset, batch_size=100, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=200, shuffle=True)

    model = DIM(in_channel=1, global_dim=64).to(device)

    if False:
        optimizer = Adam(model.parameters(), lr=1e-3)

        loss_optim = 1e4
        for epoch in range(20):
            mean_loss = []
            for batch_id, (x, y) in enumerate(train_loader):
                x = x.to(device)
                loss, _ = model(x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                mean_loss.append(loss.item())
            print('Epoch: {}, loss: {:.4f}'.format(epoch+1, np.mean(mean_loss)))

            if np.mean(mean_loss) < loss_optim:
                print('Saving checkpoint ...')
                loss_optim = np.mean(mean_loss)
                check_point = {"loss_optim": loss_optim,
                               'model_state': model.state_dict()}

                torch.save(check_point, 'model.ckpt')
    else:
        check_point = torch.load('model.ckpt')
        model.load_state_dict(check_point['model_state'])
        model.eval()

        classifier = Classifier(dim=128, n_classes=10).to(device)
        classifier.train()
        c_optimizer = Adam(classifier.parameters(), lr=1e-3)

        for i in range(10):
            losses = []
            for batch_id, (x, y) in enumerate(train_loader):
                x = x.to(device)
                y = y.to(device)
                _, x = model(x)
                logits = classifier(x)
                c_optimizer.zero_grad()
                loss = F.cross_entropy(logits, y)
                loss.backward()
                c_optimizer.step()
                losses.append(loss.item())
            print('epoch: {}, mean_loss: {:.4f}'.format(i+1, np.mean(losses)))

        classifier.eval()
        acc_list = []
        for batch_id, (x, y) in enumerate(test_loader):
            x = x.to(device)
            y = y.to(device)
            _, x = model(x)
            logits = classifier(x)
            idx = logits.argmax(dim=1)
            acc = (idx == y).float().mean().item()
            acc_list.append(acc)
        print('Test set acc: {:.4f}'.format(np.mean(acc_list)))

