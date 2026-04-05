# CSC580 Applying Machine Learning and Neural Networks
# Module 4 - Portfolio Milestone
# 
# Implementation of CIFAR10 with CNNs
# Train a network to classify images from the CIFAR10 dataset using a Convolutional Neural Network (CNN)
# Provide an analysis of the veracity of the model

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

# 1. Define the CNN Architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        
        # 32x32 -> Pool -> 16x16 -> Pool -> 8x8
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def main():
    # 2. Pre-processing with Data Augmentation
    print("Loading data with augmentation...")
    
    # Training transforms include augmentation to help generalization
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5), # 50% chance to flip horizontally
        transforms.RandomCrop(32, padding=4),   # Pad by 4 pixels then crop back to 32x32
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Test transforms do not include random augmentation
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    batch_size = 64

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 3. Initialize Network, Loss, and Optimizer
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # 4. Train the Network
    epochs = 10 # Increased epochs slightly as augmentation takes longer to converge
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):  
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 199:
                print(f'[Epoch {epoch + 1}, Batch {i + 1:5d}] loss: {running_loss / 200:.3f}')
                running_loss = 0.0

    print('Finished Training')

    # 5. Veracity Analysis
    print("\nEvaluating model veracity with augmentation...")
    correct = 0
    total = 0
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
            
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    print(f'\nOverall Accuracy: {100 * correct // total} %')

    print('\n--- Per-Class Veracity Analysis ---')
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for {classname:5s}: {accuracy:.1f} %')

if __name__ == '__main__':
    main()