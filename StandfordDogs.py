import torch
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

# Define a custom transform function that standardizes the data
class StandardizeTransform(object):
    # creates a 'StandardScaler' object called self.scaler
    # this object is uded to compute the mean and standard deviation of the input data
    def __init__(self):
        self.scaler = StandardScaler()

    # used to apply the transformation to the input data 'x' 
    def __call__(self, x):
        # we reshape it in order to standardize our data and then
        # we reshape it back to its original form before returning the data.
        x = self.scaler.fit_transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)
        return x

# Define data transformation
transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.Grayscale(1),
    transforms.RandomHorizontalFlip(p=0.1),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    transforms.ToTensor(),
    StandardizeTransform() # add the standardization transform
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
                nn.Conv2d(in_channels=input_shape,
                          out_channels=hidden_units,
                          kernel_size=3,
                          padding=1,
                          stride=1),
                nn.BatchNorm2d(hidden_units),
                nn.ReLU(),
                nn.Conv2d(in_channels=hidden_units,
                          out_channels=hidden_units,
                          kernel_size=3,
                          padding=1,
                          stride=1),
                nn.BatchNorm2d(hidden_units),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
                    )
        self.conv_block_2 = nn.Sequential(
                nn.Conv2d(in_channels=hidden_units,
                          out_channels=hidden_units,
                          kernel_size=3,
                          padding=1,
                          stride=1),
                nn.BatchNorm2d(hidden_units),
                nn.ReLU(),
                nn.Conv2d(in_channels=hidden_units,
                          out_channels=hidden_units,
                          kernel_size=3,
                          padding=1,
                          stride=1),
                nn.BatchNorm2d(hidden_units),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
                    )
              self.conv_block_3 = nn.Sequential(
                nn.Conv2d(in_channels=hidden_units,
                          out_channels=hidden_units,
                          kernel_size=3,
                          padding=1,
                          stride=1),
                nn.BatchNorm2d(hidden_units),
                nn.ReLU(),
                nn.Conv2d(in_channels=hidden_units,
                          out_channels=hidden_units,
                          kernel_size=3,
                          padding=1,
                          stride=1),
                nn.BatchNorm2d(hidden_units),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
                    )
        self.conv_block_4 = nn.Sequential(
                nn.Conv2d(in_channels=hidden_units,
                          out_channels=hidden_units,
                          kernel_size=3,
                          padding=1,
                          stride=1),
                nn.BatchNorm2d(hidden_units),
                nn.ReLU(),
                nn.Conv2d(in_channels=hidden_units,
                          out_channels=hidden_units,
                          kernel_size=3,
                          padding=1,
                          stride=1),
                nn.BatchNorm2d(hidden_units),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
        )
        
        self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=hidden_units*64,
                          out_features=output_shape)
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
                     hidden_units=16,
                     output_shape=len(class_names)).to(device)

# Setup loss function and optimizers
l1_lambda = 0.001
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.01)

# Create train and test step
def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               l1_lambda):
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
        # L1 regularization
        l1_reg=0
        for param in model_0.parameters():
            l1_reg += torch.norm(param, p=2)
        loss += l1_lambda * l1_reg
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
epochs = 20
for epoch in tqdm(range(epochs)):
    print(f'Epoch: {epoch}\n------------')
    train_step(model=model_0,
               dataloader=train_dataloader,
               loss_fn=loss_fn,
               optimizer=optimizer,
               accuracy_fn=accuracy_fn,
               l1_lambda=l1_lambda)
    test_step(model=model_0,
              dataloader=test_dataloader,
              loss_fn=loss_fn,
              accuracy_fn=accuracy_fn)
train_time_end = timer()
total_train_time = train_time_end-train_time_start
print(f'Total train time: {total_train_time}')

# create confusion matrix to visualize true values with labels
model_0.eval()
true_labels = []
pred_labels = []
for img, labels in test_dataloader:
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













