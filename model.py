import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, hidden_size) 
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)  # adding dropout layer to prevent overfitting
        self.softmax = nn.Softmax(dim=1)  # adding a softmax layer at the end

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)  # apply dropout to the output of the first fully connected layer
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)  # apply dropout to the output of the second fully connected layer
        out = self.fc3(out)
        out = self.softmax(out)  # apply softmax to the output of the third fully connected layer
        return out
