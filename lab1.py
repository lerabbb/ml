import torch
import numpy as np
import pandas

df = pandas.read_csv("train.csv")
features = df.drop(columns=['label'])
target = pandas.Categorical(df['label'])

X = torch.tensor(features.to_numpy(), dtype=torch.float32) / 255
y = torch.tensor(target.codes, dtype=torch.int64)


dataset = torch.utils.data.TensorDataset(X, y)
trainset, valset = torch.utils.data.random_split(dataset, [40000, 2000])

train_loader = torch.utils.data.DataLoader(trainset, batch_size=50, shuffle=True)
val_loader = torch.utils.data.DataLoader(valset, batch_size=50)


class NumberClassifier(torch.nn.Module):
  def __init__(self, num_hidden: int):
    super().__init__()

    #self.flat = torch.nn.Flatten()
    self.first_layer = torch.nn.Linear(28*28, num_hidden)
    self.relu1 = torch.nn.ReLU()
    self.second_layer = torch.nn.Linear(num_hidden, 10)
    self.sigmoid = torch.nn.Sigmoid()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    #x = self.flat(x)
    x = self.first_layer(x)
    x = self.relu1(x)
    x = self.second_layer(x)
    x = self.sigmoid(x)

    return x
  

net = NumberClassifier(300).cuda()
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

errors_num = 0
epsilon = 0.01
max_num = 4

for epoch in range(100):
  train_loss = 0.0
  prev_loss = 0.0
  
  for img, label in train_loader:
    optimizer.zero_grad()
    img = img.cuda()
    label = label.cuda()
    loss_val = loss(net(img), label)
    loss_val.backward()

    train_loss += loss_val.item()

    optimizer.step()

  train_loss /= len(train_loader)

  if(abs(train_loss - prev_loss) <= epsilon):
    errors_num += 1
  else:
    errors_num = 0
  if(errors_num > max_num): 
    break
  prev_loss = train_loss
  
  val_loss, val_acc = 0.0, 0.0

  for img, label in val_loader:
    img = img.cuda()
    label = label.cuda()
    with torch.no_grad():
      label_hat = net(img)
        
    val_loss += loss(label_hat, label).item()
    val_acc += (label_hat.argmax(axis=-1) == label).type(torch.float32).mean().item()

  val_loss /= len(val_loader)
  val_acc /= len(val_loader)

  print(f'[Epoch #{epoch + 1}] train_loss={train_loss}, val_loss={val_loss}, val_acc={val_acc}')
  

test_features = pandas.read_csv("test.csv")
X_test = torch.tensor(test_features.to_numpy(), dtype=torch.float32) / 255

predictions = net.forward(X_test)
predictions = predictions.detach().numpy()
predictions = np.argmax(predictions, axis=1)

ImageId = test_features.index+1
output = pandas.DataFrame({'ImageId': ImageId, 'Label': predictions})
output.to_csv('submission.csv', index=False)
print("Submission successfully saved!")