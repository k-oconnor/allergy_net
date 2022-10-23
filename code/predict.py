import os
import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.autograd import Variable
from torch import sigmoid
import numpy as np

model_in = os.path.join('models','best_model_94_9')
data_in = os.path.join('data_intermediate','clean_na_data.csv')


clean_data = pd.read_csv(data_in, index_col=None)

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

silu = nn.SiLU()


class Net(nn.Module):
    def __init__(self, D_in,H1,H2,H3,H4,D_out):
        super(Net,self).__init__()
        self.linear1 = nn.Linear(D_in,H1)
        self.init = torch.nn.init.kaiming_normal_(self.linear1.weight)
        self.linear2 = nn.Linear(H1,H2)
        self.linear3 = nn.Linear(H2,H3)
        self.linear4 = nn.Linear(H3,H4)
        self.linear5 = nn.Linear(H4,D_out)
    
    def forward(self,x):
        #x = prelu(self.linear1(x), torch.tensor(.6, dtype = torch.float))
        #self.init
        #x = prelu(self.linear2(x), torch.tensor(.3, dtype = torch.float))
        #x = prelu(self.linear3(x), torch.tensor(.15, dtype = torch.float))
        #x = prelu(self.linear4(x), torch.tensor(.02, dtype = torch.float))
        #x = sigmoid(self.linear5(x))

        x = silu(self.linear1(x))
        self.init
        x = silu(self.linear2(x))
        x = silu(self.linear3(x))
        x = silu(self.linear4(x))
        x = sigmoid(self.linear5(x))
        return x

model = Net(50,220,141,50,50,1)
model.load_state_dict(torch.load(model_in))
model.eval()

name = clean_data['Name'].values
name = list(name)
x_test = clean_data[['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16',
'x17','x18','x19','x20','x21','x22','x23','x24','x25','x26','x27','x28','x29','x30','x31','x32','x33','x34',
'x35','x36','x37','x38','x39','x40','x41','x42','x43','x44','x45','x46','x47','x48','x49','x50']].values

## Transforming the explanatory variables to a tensor.
x_test_var = Variable(torch.FloatTensor(x_test), requires_grad=False) 


## We call the model to make the predictions. The parameters from training are saved in "state_dict":
## Which is a dictionary with the optimizer settings, and weights for each parameters. 
# The state_dict is maintained in the background, but can be exported to use in other scripts or in production
test_result = model(x_test_var)

aller_probs = test_result.tolist()
clean_list = []
for ele in aller_probs:
    clean_list.extend(ele)

tup = list(zip(name,clean_list))

count = 0
for x,y in tup:
    if y > .99999999:
        print(x)
        print(y)
        count +=1
print(count)