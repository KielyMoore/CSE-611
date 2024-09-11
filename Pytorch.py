import deeplake
from PIL import Image
import numpy as np
import os, time
import torch
from torchvision import transforms, models
import tensorrt
from torchsummary import summary
import torch2trt

# Connect to the training and testing datasets
ds_train = deeplake.load('hub://activeloop/fashion-mnist-train')
ds_test = deeplake.load('hub://activeloop/fashion-mnist-test')

tform = transforms.Compose([
    transforms.RandomRotation(20), # Image augmentation
    transforms.ToTensor(), # Must convert to pytorch tensor for subsequent operations to run
    transforms.Normalize([0.5], [0.5]),
])

batch_size = 100

# Since torchvision transforms expect PIL images, we use the 'pil' decode_method for the 'images' tensor. This is much faster than running ToPILImage inside the transform
train_loader = ds_train.pytorch(num_workers = 0, shuffle = True, transform = {'images': tform, 'labels': None}, batch_size = batch_size, decode_method = {'images': 'pil'})
test_loader = ds_test.pytorch(num_workers = 0, transform = {'images': tform, 'labels': None}, batch_size = batch_size, decode_method = {'images': 'pil'})

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def print_model_size(model, model_name):
    print(f"{model_name} - Estimated Model Size:")
    print("===================================")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params}")
    total_size = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"Total Size: {total_size / (1024 ** 2):.2f} MB\n")  # Convert to MB for readability

def test_model(model, data_loader, device):

    model.eval()

    total = 0
    correct = 0
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs = data['images']
            labels = torch.squeeze(data['labels'])

            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs.float())

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
            
        print('Finished Testing')
        print('Testing accuracy: %.1f %%' %(accuracy))

# loaded_torch_model = models.resnet50()
loaded_torch_model = models.resnet50()
# Convert model to grayscale
loaded_torch_model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

# Update the fully connected layer based on the number of classes in the dataset
loaded_torch_model.fc = torch.nn.Linear(loaded_torch_model.fc.in_features, len(ds_train.labels.info.class_names))

loaded_torch_model.to(device)

# Specity the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(loaded_torch_model.parameters(), lr=0.001, momentum=0.1)
loaded_torch_model.load_state_dict(torch.load('torch_model.pth'))
loaded_torch_model.eval()

data = torch.ones((100,1,28,28)).cuda()
trt_torch_model = torch2trt.torch2trt(loaded_torch_model, [data])

# Save optimized PyTorch TensorRT model
trt_torch_model.load_state_dict(torch.load('optimized_model.pth'))

trt_torch_model.to(device)

start_time = time.time()

test_model(loaded_torch_model, test_loader, device)
print("Non-Optimized Model Run Time: ", (time.time() - start_time))

print("TensorRT Optimized Model Summary:")

start_time = time.time()

test_model(trt_torch_model, test_loader, device)
print("TensorRT Optimized Model Run Time: ", (time.time() - start_time))