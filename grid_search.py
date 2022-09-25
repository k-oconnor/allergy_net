import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn import BCELoss, MSELoss, CrossEntropyLoss, ELU
from torch.optim import SGD, Adam, Adadelta, Adamax
from torch import sigmoid,tanh,relu, prelu
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error,mean_squared_error,accuracy_score,roc_auc_score,precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from ray import tune as tn
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import os
from functools import partial


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

x_train,x_val = train_test_split(clean_data,test_size=0.2)


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

elu = nn.ELU()
hardswish = nn.Hardswish()
silu = nn.SiLU()

class Net(nn.Module):
    def __init__(self,H1=240,H2=120,H3=60):
        super(Net,self).__init__()
        self.linear1 = nn.Linear(50,H1)
        self.init = torch.nn.init.kaiming_normal_(self.linear1.weight)
        self.linear2 = nn.Linear(H1,H2)
        self.linear3 = nn.Linear(H2,H3)
        self.linear4 = nn.Linear(H3,1)
    
    def forward(self,x):
        x = prelu(self.linear1(x), torch.tensor(.6, dtype = torch.float))
        self.init
        x = prelu(self.linear2(x), torch.tensor(.3, dtype = torch.float))
        x = prelu(self.linear3(x), torch.tensor(.05, dtype = torch.float))
        x = sigmoid(self.linear4(x))
        return x

def train_tune(config, checkpoint_dir=None):
    model = Net(H1 = config["H1"],H2 = config["H2"],H3 = config["H3"])
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=config['wd'], amsgrad=True)
    #optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0)
    criterion = BCELoss()

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    train_data = MyDataset(x_train)
    val_data = MyDataset(x_val)
    train_loader = DataLoader(dataset = train_data, batch_size = config["bs"], shuffle=True)
    val_loader = DataLoader(dataset = val_data, batch_size = config["bs"], shuffle = True)

    
    for epoch in range(200):  
        running_loss = 0.0
        epoch_steps = 0
        for x,y in train_loader:        ## Obtain samples for each batch
            optimizer.zero_grad()       ## Zero out the gradient
            y = y.unsqueeze(1)          ## Take targets tensor of shape [150] and coerces it to [150,1] 
            yhat = model(x)             ## Make a prediction
            loss = criterion(yhat,y)    ## Calculate loss
            loss.backward()             ## Differentiate loss w.r.t parameters
            optimizer.step()
            running_loss += loss.item()
            epoch_steps += 1

        val_steps = 0
        total = 0
        correct = 0
        val_loss_total = 0.0
        with torch.no_grad():
            for w,z in enumerate(val_loader,0):
                inputs,labels = z 
                outputs = model(inputs)
                labels = labels.unsqueeze(1)
                loss = criterion(outputs,labels)
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

        with tn.checkpoint_dir(epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((model.state_dict(), optimizer.state_dict()), path)
        tn.report(loss=(val_loss_total / val_steps), accuracy= correct/total)
    
    print("Finished Training!")

def test_accuracy(net):
    
    val_data = MyDataset(x_val)
    val_loader = DataLoader(dataset = val_data, batch_size = 64, shuffle = True)
    correct = 0
    total = 0
    with torch.no_grad():
        for w,z in val_loader:          ## Obtain samples for each batch
                z = z.unsqueeze(1)          ## Take targets tensor of shape [150] and coerces it to [150,1]
                y_val_hat = net(w)        ## Make a prediction
                if y_val_hat > .5:
                    pred = 1
                else:
                    pred = 0
                total += z
                correct += (pred == z).sum().item()
        return correct / total


def main(num_samples=10, max_num_epochs=10):
    config ={
    "H1": tn.choice([200,205,210,211,212,213,214,215,216,220,240]),
    "H2": tn.choice([120,130,135,140,145,150,155,160]),
    "H3": tn.choice([30,35,40,45,50,55,60,65]),
    "lr": tn.loguniform(1e-3,1e-2),
    "wd": tn.loguniform(1e-4,1e-2),
    "bs": tn.choice([64,128])}

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])
    result = tn.run(
        partial(train_tune),
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    best_trained_model = Net(H1 = best_trial.config["H1"],H2 = best_trial.config["H2"],H3 = best_trial.config["H3"])
    
    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    test_acc = test_accuracy(best_trained_model)
    print("Best trial test set accuracy: {}".format(test_acc))

main(num_samples=50,max_num_epochs=150)

#if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
#   main(num_samples=1, max_num_epochs=1)