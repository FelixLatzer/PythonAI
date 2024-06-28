import torch
import torch.nn as nn
import torch.optim as optim

# Define the dataset for training
X = torch.tensor([
    [0, 0, 0, 0],
    [0, 1, 0, 0],
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 1],
    [0, 1, 1, 1],
    [1, 0, 1, 1],
    [1, 1, 1, 1],
], dtype=torch.float32)

# AND gate output considering only the first two inputs
Y_and = torch.tensor([
    [0],
    [0],
    [0],
    [0],
    [0],
    [0],
    [0],
    [0],
    [0],
    [1],
], dtype=torch.float32)

# OR gate output considering only the first two inputs
Y_or = torch.tensor([
    [0],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
], dtype=torch.float32)

# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(4, 4)  # 2 input features, 2 neurons in the hidden layer
        self.fc2 = nn.Linear(4, 1)  # 2 neurons in the hidden layer, 1 output feature

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))  # Sigmoid activation for hidden layer
        x = torch.sigmoid(self.fc2(x))  # Sigmoid activation for output layer
        return x

# Instantiate the network, define the loss function and the optimizer
andModel = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.SGD(andModel.parameters(), lr=0.1)

# Training the network
def train(model, criterion, optimizer, X, Y, epochs=10000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 1000 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

print("Training for AND gate:")
train(andModel, criterion, optimizer, X, Y_and)

# Test the network
def test(model, X):
    with torch.no_grad():
        outputs = model(X)
        predicted = (outputs > 0.5).float()
        print("Predicted outputs:")
        print(predicted)

print("Testing for AND gate:")
testData = torch.tensor([
    [1, 1, 0, 0],
    [1, 1, 1, 1],
], dtype=torch.float32)
test(andModel, testData)

# Re-initialize the network for OR gate
orModel = SimpleNN()
optimizer = optim.SGD(orModel.parameters(), lr=0.1)

print("\nTraining for OR gate:")
train(orModel, criterion, optimizer, X, Y_or)

print("Testing for OR gate:")
testData = torch.tensor([
    [1, 1, 0, 0],
    [1, 1, 1, 1],
], dtype=torch.float32)
test(orModel, testData)
