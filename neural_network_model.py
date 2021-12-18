import torch
import torch.nn.functional as nn_func

from torch import nn
from torch import sigmoid
from torch.optim import Adam
from torchsummary import summary
from torchvision.models import vgg16
from torch.nn import Sequential, Linear


class LicensePlateDetectionNN(nn.Module):
    def __init__(self):
        super(LicensePlateDetectionNN, self).__init__()
        # super(vgg16, self).__init__()
        # self.linear = nn.Linear(2, 1)
        self.vgg16 = vgg16(pretrained=True)
        # Disincluding the last 3 fully connected layers (Not including the 7x7x512 Layer)
        # self.classifier = Sequential(*list(self.vgg16.classifier.children())[:-8])
        # self.classifier = Sequential(*list(self.vgg16.features.children())[:-1])
        self.classifier = Sequential(*list(self.vgg16.features.children()),
                                     self.vgg16.avgpool)
        # Freezing the weights in the transferred VGG16 Neural Network
        for param in self.classifier.parameters():
            param.requires_grad = False
        # self.classifier.train(mode=False)
        # self.fc1 = Linear(in_features=(14*14*512), out_features=128)
        # self.fc2 = Linear(in_features=128, out_features=64)
        # self.fc3 = Linear(in_features=64, out_features=64)
        # self.fc4 = Linear(in_features=64, out_features=4)  # Our output coordinates (x_t, y_t, x_b, y_b)
        self.fc1 = Linear((7 * 7 * 512), 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = Linear(64, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = Linear(64, 4)  # Our output coordinates (x_t, y_t, x_b, y_b)

    def forward(self, x):
        x = self.classifier(x)
        # x = Flatten(x)
        x = torch.flatten(x, 1)
        x = nn_func.relu(self.bn1(self.fc1(x)))
        x = nn_func.relu(self.bn2(self.fc2(x)))
        x = nn_func.relu(self.bn3(self.fc3(x)))
        x = sigmoid(self.fc4(x))

        return x


def make_train_step(model, loss_fn, optimizer):
    def train_step(x, y):
        model.train()
        yhat = model(x)
        loss = loss_fn(model, y, yhat)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        return loss.item()

    return train_step


# Defining the loss function of the Neural Network Model
def loss_function(model, y, yhat, lambda1=1e-6, lambda2=1e-6, lambda3=1e-6, lambda4=1e-6):
    loss_func = nn.MSELoss(reduction='mean')
    all_linear1_params = torch.cat([x.view(-1) for x in model.fc1.parameters()])
    all_linear2_params = torch.cat([x.view(-1) for x in model.fc2.parameters()])
    all_linear3_params = torch.cat([x.view(-1) for x in model.fc3.parameters()])
    all_linear4_params = torch.cat([x.view(-1) for x in model.fc4.parameters()])
    l1_regularization = lambda1 * torch.norm(all_linear1_params, p=2)
    l2_regularization = lambda2 * torch.norm(all_linear2_params, p=2)
    l3_regularization = lambda3 * torch.norm(all_linear3_params, p=2)
    l4_regularization = lambda4 * torch.norm(all_linear4_params, p=2)

    total_loss = loss_func(y, yhat) + \
                 l1_regularization + l2_regularization + l3_regularization + l4_regularization

    return total_loss


def neural_network(channel, width, height):
    """
    Q: Why seed(42)?
    A: It's a pop-culture reference! In Douglas Adams's popular 1979 science-fiction novel The Hitchhiker's Guide to
    the Galaxy, towards the end of the book, the supercomputer Deep Thought reveals that the answer to the great
    question of “life, the universe and everything” is 42.
    """
    torch.manual_seed(42)

    model = LicensePlateDetectionNN()  # In case of cuda GPU usage, add: .to(device)
    print('\nSummary of our Neural Network Model:')
    summary(model, (channel, width, height))
    print('\n')

    # loss_fn = nn.MSELoss(reduction='mean')
    loss_fn = loss_function
    optimizer = Adam(model.parameters(), lr=3e-4)  # Adam's good learning rate (1/3 of 1e-3)
    train_step = make_train_step(model, loss_fn, optimizer)

    return model, train_step, loss_fn
