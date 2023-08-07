import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import math

def nn_eval(nn_model, train, test, input_size, hidden_size, num_classes, device, learning_rate=0.001, batch_size=100, num_epochs=1000, print_point=5):
    # Set Params
    total_samples = len(train)
    n_iterations = math.ceil(total_samples/batch_size)

    # Data loader
    loader = DataLoader(dataset=train,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=1)

    model = nn_model(input_size, hidden_size, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(loader):
              
            #inputs = inputs.reshape(-1, input_size)
            inputs = inputs.to(device)
            labels = labels.squeeze().type(torch.LongTensor).to(device)

            #print(inputs.shape, labels.shape)
            outputs = model(inputs.float())
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print(i, (i+1) % print_point)
            if (epoch+1) % print_point == 0:#
                print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_iterations}], Loss: {loss.item():.4f}')
                #print(f'Epoch: {epoch+1}/{num_epochs}, Step {i+1}/{n_iterations}| Inputs {inputs.shape} | Labels {labels.shape}')

    PATH = './models/nn_1.pth'
    torch.save(model.state_dict(), PATH)

    # Data loader Test
    loader = DataLoader(dataset=test,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=1) 
    
    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for inputs, labels in loader:
            print("loader")
            inputs = inputs.to(device)
            labels = labels.squeeze().type(torch.LongTensor).to(device)
            outputs = model(inputs.float())     
            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()
            acc = 100.0 * n_correct / n_samples
            print(f'Accuracy of the network: {acc} %')