import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn import BCELoss, MSELoss, CrossEntropyLoss, ELU
from torch.optim import SGD, Adam, Adadelta, Adamax
from torch import sigmoid,tanh,relu, prelu
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error,mean_squared_error,accuracy_score,roc_auc_score,precision_score, recall_score
from sklearn.preprocessing import StandardScaler


clean_data = pd.read_csv("clean_data.csv")

clean_data = pd.read_csv('clean_data.csv', index_col=None)

clean_data[['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16',
'x17','x18','x19','x20','x21','x22','x23','x24','x25','x26','x27','x28','x29','x30','x31','x32','x33','x34',
'x35','x36','x37','x38','x39','x40','x41','x42','x43','x44','x45','x46','x47','x48','x49','x50']] = clean_data[['x1',
'x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16',
'x17','x18','x19','x20','x21','x22','x23','x24','x25','x26','x27','x28','x29','x30','x31','x32','x33','x34',
'x35','x36','x37','x38','x39','x40','x41','x42','x43','x44','x45','x46','x47','x48','x49','x50']].astype(float)

clean_data.x34.fillna(0, inplace=True)
clean_data.x35.fillna(0, inplace=True)
clean_data.x36.fillna(0, inplace=True)
clean_data.x37.fillna(0, inplace=True)
clean_data.x38.fillna(0, inplace=True)
clean_data.x39.fillna(0, inplace=True)
clean_data.x40.fillna(0, inplace=True)
clean_data.x41.fillna(0, inplace=True)
clean_data.x42.fillna(0, inplace=True)
clean_data.x43.fillna(0, inplace=True)
clean_data.x44.fillna(0, inplace=True)
clean_data.x45.fillna(0, inplace=True)
clean_data.x46.fillna(0, inplace=True)
clean_data.x47.fillna(0, inplace=True)
clean_data.x48.fillna(0, inplace=True)
clean_data.x49.fillna(0, inplace=True)
clean_data.x50.fillna(0, inplace=True)

std_scaler = StandardScaler()

clean_data[['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16',
'x17','x18','x19','x20','x21','x22','x23','x24','x25','x26','x27','x28','x29','x30','x31','x32','x33','x34',
'x35','x36','x37','x38','x39','x40','x41','x42','x43','x44','x45','x46','x47','x48','x49',
'x50']] = std_scaler.fit_transform(clean_data[['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16',
'x17','x18','x19','x20','x21','x22','x23','x24','x25','x26','x27','x28','x29','x30','x31','x32','x33','x34',
'x35','x36','x37','x38','x39','x40','x41','x42','x43','x44','x45','x46','x47','x48','x49','x50']])


le = LabelEncoder()
clean_data['Known_Allergen']= le.fit_transform(clean_data['Known_Allergen'])

x_train,x_val = train_test_split(clean_data,test_size=0.05)


## Dataset constructor - Input is a dataframe, output is a dataset object with values encoded into tensors, with the
## required methods for length and retrieving an item by index.
class MyDataset(Dataset):
  
  def __init__(self,clean_data):

    # Separate our explanatory and dependent variables
    y = clean_data['Known_Allergen'].values
    x = clean_data[['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16',
'x17','x18','x19','x20','x21','x22','x23','x24','x25','x26','x27','x28','x29','x30','x31','x32','x33','x34',
'x35','x36','x37','x38','x39','x40','x41','x42','x43','x44','x45','x46','x47','x48','x49','x50']].values

    self.x_train = torch.tensor(x,dtype=torch.float32)
    self.y_train = torch.tensor(y,dtype=torch.float32)
 
  def __len__(self):
    return len(self.y_train)
   
  def __getitem__(self,idx):
    return self.x_train[idx],self.y_train[idx]

## Neural Net constructor - Input is a batch of tensors with explanatory variables with model specifications, and output are class predictions
## To avoid the vanishing gradient problem, we use rectified linear unit activations on the two hidden layers.
## We pass the linear hidden layers through ReLU activations, and the final linear output layer through a sigmoid activation.
## This in effect is a multi-equation logistic regression for predicting class probabilities. 

elu = nn.ELU()
hardswish = nn.Hardswish()
silu = nn.SiLU()

class Net(nn.Module):
    def __init__(self, D_in,H1,H2,H3,D_out):
        super(Net,self).__init__()
        self.linear1 = nn.Linear(D_in,H1)
        self.init = torch.nn.init.kaiming_normal_(self.linear1.weight)
        self.linear2 = nn.Linear(H1,H2)
        self.linear3 = nn.Linear(H2,H3)
        self.linear4 = nn.Linear(H3,D_out)
    
    def forward(self,x):
        #x = prelu(self.linear1(x), torch.tensor(.3, dtype = torch.float))
        #self.init
        #x = prelu(self.linear2(x), torch.tensor(.1, dtype = torch.float))
        #x = prelu(self.linear3(x), torch.tensor(.05, dtype = torch.float))
        #x = prelu(self.linear4(x), torch.tensor(.02, dtype = torch.float))
        #x = sigmoid(self.linear5(x))

        x = silu(self.linear1(x))
        self.init
        x = silu(self.linear2(x))
        x = silu(self.linear3(x))
        x = sigmoid(self.linear4(x))
        return x

## We call the "Net" class to initialize the model. Net(Input_Dim, Hidden_Layer_1_Neurons, Hidden_Layer_2_Neurons, Output_Dim)
model = Net(50,210,140,65,1)

## We use binary cross entropy loss for measuring model performance. This is analogous to minimizing MSE in OLS.
## Description:(https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a)
criterion = BCELoss()

## After computing the gradients for all tensors in the model, calling optimizer. step() makes the optimizer 
## iterate over all parameters (tensors)it is supposed to update and use their internally stored grad to update their values.
## Learning rate is a key hyperparameter that determines how fast the network moves weights to gradient minima
## Weight decay is an optional hyperparameter which progressivly reduces |weights| each epoch, in effect penalizing overfitting.
optimizer = Adam(model.parameters(), lr=0.00183, weight_decay=0.00155, amsgrad=True)
## amsgrad!

#optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0)

## We call our dataset classes from our train/test split, and returns two dataset objects
train_data = MyDataset(x_train)
val_data = MyDataset(x_val)

## DataLoader takes our dataset objects, and turns them into iterables. 
## Batch-size determines how many row tensors are passed to the model in each epoch.
## Setting shuffle to true, ensures that each batch is a random sample.
train_loader = DataLoader(dataset = train_data, batch_size = 64, shuffle=True)
val_loader = DataLoader(dataset = val_data, batch_size = 64, shuffle = True)


# 50 is optimal???
## ---------------- training the model  ---------------- ##
loss_list = []                      ## We initialize two empty lists to append loss from each epoch to
val_loss_list = []
acc = []
for epoch in range(50):             ## By inputing the range(x), we are choosing 'x' epochs to iterate over the training data
    for x,y in train_loader:        ## Obtain samples for each batch
        optimizer.zero_grad()       ## Zero out the gradient
        y = y.unsqueeze(1)          ## Take targets tensor of shape [150] and coerces it to [150,1] 
        yhat = model(x)             ## Make a prediction
        loss = criterion(yhat,y)    ## Calculate loss
        loss.backward()             ## Differentiate loss w.r.t parameters
        optimizer.step()            ## Update parameters

 ## Testing the updated parameters on the held validation data...   
    val_steps = 0
    total = 0
    correct = 0
    val_loss_total = 0.0

    with torch.no_grad():
        for w,z in enumerate(val_loader,0):
            inputs,labels = z 
            outputs = model(inputs)
            labels = labels.unsqueeze(1)
            val_loss = criterion(outputs,labels)
            val_loss_total += loss.numpy()
            val_steps += 1
    
            aller_probs = outputs.detach().numpy()
            labels = labels.detach().numpy()

            pred_list = []
            for item in aller_probs:
                if item > .5:
                    pred_list.append(1)
                else:
                    pred_list.append(0)

            i=0
            while i < len(aller_probs):
                if pred_list[i] == labels[i]:
                    correct +=1
                    total += 1
                    i += 1
                else:
                    total +=1
                    i +=1
            acc.append(correct/total)
## At each epoch, we append the calculated loss to a list, so we can graph it's change over time...
    if epoch:
        loss_list.append(loss.item())
        val_loss_list.append(val_loss.item())
print("Finished Training!")

## A simple plotting function for showing loss changes over time as parameters are updated...
plt.plot(acc, linewidth =.5)
plt.legend("Validation Accuracy")
plt.ylabel("Validation Accuracy")
plt.xlabel("Batch")
plt.show()


plt.plot(loss_list, linewidth=.5)
plt.plot(val_loss_list, linewidth =.5)
plt.legend(("Training Loss", "Validation Loss"))
plt.xlabel("Epoch")
plt.ylabel("BCE Loss")
plt.show()

y_test = x_val['Known_Allergen'].values
x_test = x_val[['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16',
'x17','x18','x19','x20','x21','x22','x23','x24','x25','x26','x27','x28','x29','x30','x31','x32','x33','x34',
'x35','x36','x37','x38','x39','x40','x41','x42','x43','x44','x45','x46','x47','x48','x49','x50']].values

## Transforming the explanatory variables to a tensor.
x_test_var = Variable(torch.FloatTensor(x_test), requires_grad=False) 


## We call the model to make the predictions. The parameters from training are saved in "state_dict":
## Which is a dictionary with the optimizer settings, and weights for each parameters. 
# The state_dict is maintained in the background, but can be exported to use in other scripts or in production
test_result = model(x_test_var)

aller_probs = test_result.detach().numpy()

thresholds = [.35,.4,.45,.5,.55,.9]

for th in thresholds:
    print('The metrics for cutoff value:', th)
    test_pred = []
    i=0
    while i < len(aller_probs):
        if aller_probs[i] > th:
            test_pred.append(1)
            i += 1
        else:
            test_pred.append(0)
            i += 1

    meanAbErr = mean_absolute_error(y_test, test_pred)
    meanSqErr = mean_squared_error(y_test, test_pred)
    rootMeanSqErr = np.sqrt(mean_squared_error(y_test, test_pred))
    precision = precision_score(y_test, test_pred)
    recall = recall_score(y_test, test_pred)
    AUC_nn = roc_auc_score(y_test, test_pred)

    print('[1] Neural Network Testing Accuracy: ', accuracy_score(y_test,test_pred))
    print('Precision Score', precision)
    print('Recall Score', recall)
    print('Mean Absolute Error:', meanAbErr)
    print('Mean Square Error:', meanSqErr)
    print('Root Mean Square Error:', rootMeanSqErr)
    print('AUC:', AUC_nn)
    print('\t')