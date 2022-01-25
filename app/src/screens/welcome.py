from abc import ABC

from kivy.uix.screenmanager import Screen
import numpy as np
import torch as T


class WelcomeScreen(Screen):

    def __init__(self, **kwargs):
        super(WelcomeScreen, self).__init__(**kwargs)

    def demo(self, *args):
        self.ids.demo.text = ""
        logs = "Begin minimal PyTorch Iris demo\n"
        T.manual_seed(1)
        np.random.seed(1)
        device = T.device("cpu")
        logs += "Loading Iris train data\n"
        train_x = np.array([
            [5.0, 3.5, 1.3, 0.3],
            [4.5, 2.3, 1.3, 0.3],
            [5.5, 2.6, 4.4, 1.2],
            [6.1, 3.0, 4.6, 1.4],
            [6.7, 3.1, 5.6, 2.4],
            [6.9, 3.1, 5.1, 2.3]], dtype=np.float32)
        train_y = np.array([0, 0, 1, 1, 2, 2], dtype=np.long)
        train_x = T.tensor(train_x, dtype=T.float32).to(device)
        train_y = T.tensor(train_y, dtype=T.long).to(device)
        net = Net().to(device)
        max_epochs = 100
        lrn_rate = 0.04
        loss_func = T.nn.CrossEntropyLoss()  # applies softmax()
        optimizer = T.optim.SGD(net.parameters(), lr=lrn_rate)
        logs += "Starting training\n"
        net.train()
        indices = np.arange(6)
        for epoch in range(0, max_epochs):
            np.random.shuffle(indices)
            for i in indices:
                X = train_x[i].reshape(1, 4)  # device inherited
                Y = train_y[i].reshape(1, )
                optimizer.zero_grad()
                oupt = net(X)
                loss_obj = loss_func(oupt, Y)
                loss_obj.backward()
                optimizer.step()
        logs += "Done training\n"
        net.eval()
        logs += "Predicting species for [5.8, 2.8, 4.5, 1.3]\n"
        unk = np.array([[5.8, 2.8, 4.5, 1.3]], dtype=np.float32)
        unk = T.tensor(unk, dtype=T.float32).to(device)
        logits = net(unk).to(device)
        probs = T.softmax(logits, dim=1)
        probs = probs.detach().numpy()
        np.set_printoptions(precision=4)
        logs += f"{probs}\n"
        logs += "End Iris demo\n"
        self.ids.demo.text = logs


class Net(T.nn.Module, ABC):
    def __init__(self):
        super(Net, self).__init__()
        self.hid1 = T.nn.Linear(4, 7)  # 4-7-3
        self.oupt = T.nn.Linear(7, 3)

    def forward(self, x):
        z = T.tanh(self.hid1(x))
        z = self.oupt(z)  # no softmax. see CrossEntropyLoss()
        return z
