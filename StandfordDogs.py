import torch
import random
import pandas as pd
from torch import nn
from torch.utils import data
from helper_functions import accuracy_fn
from tqdm.auto import tqdm
from timeit import default_timer as timer
from torch.nn.modules import transformer
from torch.utils.data import DataLoader, dataloader
import torchvision
from sklearn.metrics import confusion_matrix
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.transforms import transforms
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# device agnostic code
device = torch.device('mps')
print(f'Device: {device}')

# Define data transformation
transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.ToTensor()
    ])

# Use our transform variable while downloading our images
train_data = datasets.CIFAR10(root='img_data',
                              train=True,
                              download=True,
                              transform=transform)
test_data = datasets.CIFAR10(root='img_data',
                             train=False,
                             download=True,
                             transform=transform)

# Make a list of our class names
class_names = train_data.classes

# create pie chart to visualize class distribution to check for unbalance.
class_counts = {label: class_names.count(label) for label in set(class_names)}
plt.pie(class_counts.values(), labels=class_counts.keys(), autopct='%1.1f%%')
plt.title('Class Distribution')
# plt.show()

# View our data 
fig = plt.figure(figsize=(10,10))
row, col = 4, 4
for sample in range(1, (row*col)+1):
    # Pick a random index in our train_data
    rand_idx = torch.randint(1, len(train_data), size=[1]).item()
    # Add subsplots in our figure (4x4) with our sample (image)
    fig.add_subplot(row, col, sample)
    # Store random image and label from our train data
    img, label = train_data[rand_idx]
    plt.imshow(img[0,:,:], cmap='gray')
    plt.title(class_names[label])
    plt.axis(False)

BATCH_SIZE=54
# Set our data into mini-batches
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_data,
                              batch_size=BATCH_SIZE,
                              shuffle=False)

print(f'train_dataloader: {train_dataloader}, length: {len(train_dataloader)}')
print(f'test_dataloader: {test_dataloader}, length: {len(test_dataloader)}')

# Make model
class CNNModelV0(nn.Module):
    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
                nn.Conv2d(in_channels=1,
                          out_channels=9,
                          kernel_size=3,
                          padding=1,
                          stride=1),
                nn.BatchNorm2d(9),
                nn.ReLU(),
                nn.Conv2d(in_channels=9,
                          out_channels=18,
                          kernel_size=3,
                          padding=1,
                          stride=1),
                nn.BatchNorm2d(18, eps=1e-05, momentum=0.1, affine=True),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, ceil_mode=False)
                    )
        self.conv_block_2 = nn.Sequential(
                nn.Conv2d(in_channels=18,
                          out_channels=18,
                          kernel_size=3,
                          padding=1,
                          stride=1),
                nn.BatchNorm2d(18),
                nn.ReLU(),
                nn.Conv2d(in_channels=18,
                          out_channels=36,
                          kernel_size=3,
                          padding=1,
                          stride=1),
                nn.BatchNorm2d(36, eps=1e-05, momentum=0.1, affine=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, ceil_mode=False)
                    )
              self.conv_block_3 = nn.Sequential(
                nn.Conv2d(in_channels=36,
                          out_channels=36,
                          kernel_size=3,
                          padding=1,
                          stride=1),
                nn.BatchNorm2d(36, eps=1e-05, momentum=0.1, affine=True),
                nn.ReLU(),
                nn.Conv2d(in_channels=36,
                          out_channels=45,
                          kernel_size=3,
                          padding=1,
                          stride=1),
                nn.BatchNorm2d(45),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, ceil_mode=False)
                    )
        self.conv_block_4 = nn.Sequential(
                nn.Conv2d(in_channels=45,
                          out_channels=45,
                          kernel_size=3,
                          padding=1,
                          stride=1),
                nn.BatchNorm2d(45),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=45,
                          out_channels=45,
                          kernel_size=3,
                          padding=1,
                          stride=1),
                nn.BatchNorm2d(45),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, ceil_mode=False)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=180,
                      out_features=output_shape,
                      bias=True)
                )

    # Passing our data through all the layers
    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)
        x = self.classifier(x)
        return x

model_0 = CNNModelV0(input_shape=1,
                     output_shape=len(class_names)).to(device)

# Setup loss function and optimizers
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_0.parameters(),
                            lr=0.01)

# Create train and test step
def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn):
    train_loss, train_acc = 0, 0

    # Loop throgh batches 
    for batch, (X,y) in enumerate(dataloader):
        X = X.type(torch.float32).to(device)
        y = y.to(device)
        # Set model to train mode
        model.train()
        # Forward pass
        y_pred = model(X)
        # Calculate the loss 
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y, y_pred.argmax(dim=1))
        # Set the gradients to zero for next iteration
        optimizer.zero_grad()
        # backpropagation
        loss.backward()
        # step
        optimizer.step()

    # Calculate the average loss 
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    # Print out our loss
    print(f'Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%')

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data,
              loss_fn: torch.nn.Module,
              accuracy_fn):

    test_loss, test_acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in dataloader:
            X = X.type(torch.float32).to(device)
            y = y.to(device)
            test_pred = model(X)
            loss = loss_fn(test_pred, y)
            test_loss += loss
            test_acc += accuracy_fn(y,
                                    test_pred.argmax(dim=1))

        # Calculate the average
        test_loss /= len(dataloader)
        test_acc /= len(dataloader)

    print(f'Test loss: {test_loss:.4f}, Test acc: {test_acc:.2f}%')

train_time_start = timer()
epochs = 25
for epoch in tqdm(range(epochs)):
    print(f'Epoch: {epoch}\n------------')
    train_step(model=model_0,
               dataloader=train_dataloader,
               loss_fn=loss_fn,
               optimizer=optimizer,
               accuracy_fn=accuracy_fn)
    test_step(model=model_0,
              dataloader=test_dataloader,
              loss_fn=loss_fn,
              accuracy_fn=accuracy_fn)
train_time_end = timer()
total_train_time = train_time_end-train_time_start
print(f'Total train time: {total_train_time}')

# create confusion matrix
def conv_matrix(self, dataloader):
    model_0.eval()
    true_labels = []
    pred_labels = []
    for img, labels in test_dataloader:
        dfdf
        img, = img.to(device)
        labels = labels.to(device)
        outputs = model_0(img)
        _, preds = torch.max(outputs.data, 1)
        true_labels.extend(labels.cpu().numpy())
        pred_labels.extend(preds.cpu().numpy())
    
    # convert list to numpy 
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    # compute cm
    cm = confusion_matrix(true_labels, pred_labels)
    print(cm)
    
def make_predictions(model: torch.nn.Module,
                     data: list,
                     device: torch.device=device):
    pred_probs = []
    model.eval()
    with torch.inference_mode():
        for sample in data:
            sample = torch.unsqueeze(sample, dim=0)
            pred_logits = model(sample)
            pred_prob = torch.softmax(pred_logits.squeeze(), dim=0)
            pred_probs.append(pred_prob.cpu())
            
    return torch.stack(pred_probs)

# get test samples
test_samples, test_labels = [], []
for sample, label in random.sample(list(test_data), k=9):
    test_samples.append(sample)
    test_labels.append(label)
    
pred_probs = make_predictions(model=model_0.cpu(),
                              data=test_samples)

pred_classes = pred_probs.argmax(dim=1)

# show 3x3 plot with predicted labels next to true labels
plt.figure(figsize=(9,9))
nrows = 3
nclows = 3
for i, sample in enumerate(test_samples):
    plt.subplot(nrows, ncols, i+1)
    plt.imshow(sample[0,:,:], cmap='gray')
    pred_label = class_names[pred_classes[i]]
    truth_label = class_names[test_labels[i]]
    title_text = f'Pred: {pred_label} | Truth: {truth_label}'
    
if pred_label == truth_label:
    plt.title(title_text, fontsize=10, c='g')
else:
    plt.title(title_text, fontsize=10, c='r')
    
plt.show()













