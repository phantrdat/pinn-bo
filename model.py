import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
class MlpMNIST(nn.Module):
    def __init__(self, dropout = 0.2, num_classes=10, n_hidden = 100):
        super(MlpMNIST, self).__init__()
        self.layer1 = nn.Linear(in_features=784, out_features=n_hidden)
        self.layer2 = nn.Linear(in_features=n_hidden, out_features=num_classes)
        self.activation = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=1)
    def forward(self, x):
        x = x.view(-1, 784)
        out = self.layer1(x)
        out = self.activation(out)
        out = self.layer2(out)
        out = self.log_softmax(out)
        return out
class Cifar10():
    def __init__(self, data_path):
        self.data_path = data_path
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None
        
        self.X_val = None
        self.Y_val = None
        
    def load_CIFAR_batch(self, filename):
        """ load single batch of cifar """
        with open(filename, 'rb') as f:
            datadict = pkl.load(f, encoding='bytes')
            X = datadict[b'data']
            X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float32")
            X = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X])

            Y = datadict[b'labels']
            Y = np.array(Y)
            return X, Y
    def load_CIFAR10(self):
        """ load all of cifar """
        xs = []
        ys = []
        for b in range(1,6):
            f = os.path.join(self.data_path, 'data_batch_%d' % (b, ))
            X, Y = self.load_CIFAR_batch(f)
            xs.append(X)
            ys.append(Y)    
        Xtr = np.concatenate(xs)
        Ytr = np.concatenate(ys)
        del X, Y
        Xte, Yte = self.load_CIFAR_batch(os.path.join(self.data_path, 'test_batch'))
        return Xtr, Ytr, Xte, Yte
    def get_CIFAR10_data(self, num_training=49000, num_val=1000, num_test=10000, show_sample=True):
        """
        Load the CIFAR-10 dataset, and divide the sample into training set, validation set and test set
        """

        
        self.X_train, self.Y_train, self.X_test, self.Y_test = self.load_CIFAR10()
            
        # subsample the data for validation set
        mask = range(num_training, num_training + num_val)
        self.X_val = self.X_train[mask]
        self.Y_val = self.Y_train[list(mask)]

        mask = range(num_training)
        self.X_train = self.X_train[mask]
        self.Y_train = self.Y_train[mask]

        mask = range(num_test)
        self.X_test = self.X_test[mask]
        self.Y_test = self.Y_test[mask]

        self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], -1)) # [49000, 3072]
        self.X_val = np.reshape(self.X_val, (self.X_val.shape[0], -1)) # [1000, 3072]
        self.X_test = np.reshape(self.X_test, (self.X_test.shape[0], -1)) # [10000, 3072]
        
        # Normalize the data: subtract the mean image
        mean_image = np.mean(self.X_train, axis = 0)
        self.X_train -= mean_image
        self.X_val -= mean_image
        self.X_test -= mean_image
        
        # Add bias dimension and transform into columns
        # self.X_train = np.hstack([self.X_train, np.ones((self.X_train.shape[0], 1))]).T
        # self.X_val = np.hstack([self.X_val, np.ones((self.X_val.shape[0], 1))]).T
        # self.X_test = np.hstack([self.X_test, np.ones((self.X_test.shape[0], 1))]).T
        return self
    

    
class FashionMNIST(nn.Module):
    def __init__(self, dropout = 0.2, num_classes=10, num_channels_1 = 32, num_channels_2 = 64):
    # def __init__(self):
        super(FashionMNIST, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=num_channels_1, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels_1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=num_channels_1, out_channels=num_channels_2, kernel_size=3),
            nn.BatchNorm2d(num_channels_2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.fc1 = nn.Linear(in_features=64*6*6, out_features=600)
        dropout = max(dropout, 0)
        self.drop = nn.Dropout2d(p=dropout)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=num_classes) 
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        
        return out

if __name__ == '__main__':
    # hsz = 30
    # model = MlpMNIST(dropout = 0.2, num_classes=10, n_hidden = hsz)
    # torch.save(model.state_dict(),f'init_weights/MlpMNISTsample_{hsz}x10.pth')

    cifar10 = Cifar10('cifar10_data/cifar-10-batches-py')
    cifar10.get_CIFAR10_data()


# def model_mnist(params, num_epochs = 1):
    
    # dropout, learning_rate
    # model = ConvNet(dropout).to(device)

    # # Loss and optimizer
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # # Train the model
    # total_step = len(train_loader)
    # for epoch in range(num_epochs):
    #     for i, (images, labels) in enumerate(train_loader):
    #         images, labels = images.to(device), labels.to(device)
    #         # Forward pass
    #         outputs = model(images)
    #         loss = criterion(outputs, labels)

    #         # Backward and optimize
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #         if (i+1) % 100 == 0:
    #             print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
    #                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    # # Test the model
    # with torch.no_grad():
    #     correct = 0
    #     total = 0
    #     for images, labels in test_loader:
    #         images, labels = images.to(device), labels.to(device)
    #         outputs = model(images)
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()

    #     print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

    # return correct / total

# if __name__ =='__main__':
#     model_mnist(dropout=0.3, learning_rate=0.01, num_epochs=2)

# Save the model checkpoint
#torch.save(model.state_dict(), 'model.ckpt')