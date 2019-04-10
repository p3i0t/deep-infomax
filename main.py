import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim import Adam


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
                                  nn.Conv2d(2 * self.multiplier, 4 * self.multiplier, kernel_size=3, stride=2, padding=1)
                                  )

    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, in_channel, global_dim=64):
        super().__init__()
        self.c = C(in_channel)
        self.f = nn.Sequential(nn.Linear(128*4*4, 1024),
                               nn.ReLU(),
                               nn.Linear(1024, global_dim))

    def forward(self, x):
        local_feature_map = self.c(x)
        local_ = flatten(local_feature_map)
        global_feature = self.f(local_)
        return local_, global_feature


class T(nn.Module):
    def __init__(self, in_channel, global_dim=64):
        super().__init__()
        self.enc = Encoder(in_channel, global_dim)
        self.t = nn.Sequential(nn.Linear(128*4*4 + global_dim, 512),
                               nn.ReLU(),
                               nn.Linear(512, 128),
                               nn.ReLU(),
                               nn.Linear(128, 1))

    def forward(self, x):
        local_, global_ = self.enc(x)
        pair = torch.cat([local_, global_], dim=1)
        paired_scores = self.t(pair)

        global_shuffle = global_[torch.randperm(global_.size()[0])]
        unpairs = torch.cat([local_, global_shuffle])
        unpaired_scores = self.t(unpairs)

        return paired_scores, unpaired_scores


class Critic(nn.Module):
    def __init__(self, global_dim=64):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(global_dim, global_dim),
            nn.ReLU(),
            nn.Linear(global_dim, global_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.f(x)


class DIM(nn.Module):
    def __init__(self, in_channel, global_dim=64):
        super().__init__()
        self.global_dim = global_dim
        self.enc = Encoder(in_channel, global_dim)
        self.t = nn.Sequential(nn.Linear(128*4*4 + global_dim, 512),
                               nn.ReLU(),
                               nn.Linear(512, 128),
                               nn.ReLU(),
                               nn.Linear(128, 1))

        self.critic = Critic(global_dim)

    def forward(self, x):
        local_, global_ = self.enc(x)
        pair = torch.cat([local_, global_], dim=1)
        paired_scores = self.t(pair)

        global_shuffle = global_[torch.randperm(global_.size()[0])]
        unpairs = torch.cat([local_, global_shuffle])
        unpaired_scores = self.t(unpairs)

        mi = donsker_varadhan_loss(paired_scores, unpaired_scores)

        real_samples = torch.randn(x.size(0), self.global_dim).to(device)
        fake_samples = global_

        real_probs = self.critic(real_samples)
        fake_probs = self.critic(fake_samples)

        log_probs = torch.log(real_probs) + torch.log(1. - fake_probs)
        regulation = -log_probs.mean()

        loss = mi + 0.1 * regulation
        return loss


def log_mean_exp(x):
    """Stable log mean exp."""
    max_ = x.max()
    return max_ + (x-max_).exp().mean().log()


def donsker_varadhan_loss(paired_scores, unpaired_scores):
    mi = paired_scores.mean() - log_mean_exp(unpaired_scores)
    return -mi


if __name__ == '__main__':

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    dataset = datasets.MNIST('data/MNIST', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.Resize((32, 32)),
                                 transforms.ToTensor(),
                                 # transforms.Normalize((0.1307,), (0.3081,))
                             ]))

    train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)

    model = DIM(in_channel=1, global_dim=64).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        for batch_id, (x, y) in enumerate(train_loader):
            x = x.to(device)
            loss = model(x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_id % 100 == 1:
                print('step {}, mi: {:.4f}'.format(batch_id + 1, -loss.item()))