import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from clstm import ConvLSTM
import joblib
from PIL import Image
from target_relevance import TargetRelevance
from tqdm import tqdm
from collections import OrderedDict
torch.set_default_tensor_type('torch.cuda.FloatTensor')
import pandas as pd

def initialize_seed(seed):
    torch.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

class DenseLoss(nn.Module):
    def __init__(self,y,alpha=1.0):
        super().__init__()
        self.target_relevance = TargetRelevance(y,alpha)

    def forward(self, labels, preds):
        try:
            self.relevance = torch.from_numpy(self.target_relevance(labels))
        except AttributeError:
            print(
                'WARNING: self.target_relevance does not exist yet! (This is normal once at the beginning when\
                lightning tests things)'
            )
            self.relevance = torch.ones_like(labels)
        #print(preds)
        #print("--------------------")
        #print(labels)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.relevance = self.relevance.to(device)
        err = torch.pow(preds - labels, 2)
        err_weighted = self.relevance * err
        mse = err_weighted.mean()
        return mse

class Model(nn.Module):

    def __init__(self):

        super(Model, self).__init__()
        self.rain_encoder = ConvLSTM(input_dim=48,
                                     hidden_dim=[32,32,32],
                                     kernel_size=(3,3),
                                     num_layers=3
                                     )

        self.geo_encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32,kernel_size=3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.temp_connection = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32,kernel_size=3,padding=1),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=32,kernel_size=3,padding=1),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=16,kernel_size=3,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.full_connection = nn.Sequential(
            nn.Linear(in_features=2*(32*32)+9, out_features=4096), # '+3' in_oneDの3ユニット分を追加
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_features=4096, out_features=1024),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_features=1024 , out_features=1)
        )

    def forward(self,x_rain,x_geo,x_river):

        # ひとつめのデータを用いて畳み込みの計算
        geo_x = self.geo_encoder(x_geo)
        rain_x,_ = self.rain_encoder(x_rain)
        rain_x = rain_x[0]
        # 畳み込み層からの出力を1次元化
        #geo_x = geo_x.view(geo_x.size(0),32*32*32)
        #rain_x = rain_x.view(rain_x.size(1),32*32*32)
        rain_x = torch.squeeze(rain_x)
        #print(geo_x.shape)
        x_2d=torch.stack([rain_x,geo_x],dim=0)
        x_2d=x_2d.view(geo_x.size(0)*2,geo_x.size(1),geo_x.size(2),geo_x.size(3))
        temp_x = self.temp_connection(x_2d)
        temp_x = temp_x.view(geo_x.size(0),2*(32*32))
        # 1次元化した畳み込み層のからの出力と2つめの入力を結合
        x = torch.cat([temp_x,x_river], dim=1)
        # 全結合層に入力して計算
        y = self.full_connection(x)
        
        return y

    # 学習用関数
def train(data_loader, model, optimizer, loss_fn, device):
    model.train()
    losses=[]
    for rain_x, geo_x, river_x, target in data_loader:

        rain_x = rain_x.to(device, dtype=torch.float)
        geo_x = geo_x.to(device, dtype=torch.float)
        river_x = river_x.to(device, dtype=torch.float)

        optimizer.zero_grad()
        output = model(rain_x, geo_x,river_x)
        loss = loss_fn(output, target)

        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return sum(losses)/len(losses)

def reshape_rain(rain_list,window):
    new_rains=[]
    for rains in rain_list:
        temp_rain = []
        for rain in rains:
            img = Image.fromarray(np.array(rain))
            img = img.resize((32,32))
            temp_rain.append(np.array(img))
        new_rains.append(np.array(temp_rain))
    return new_rains

def reshape_geo(geo_list):
    new_geos=[]
    for geo in geo_list:
        img = Image.fromarray(np.array(geo))
        img = img.resize((32,32))
        new_geos.append(np.array(img))
    return new_geos

def create_dataloader():
    window=5
    new_path="/home/student/e17/e175764/"
    locations=["hiwatari","inbara","ohito","nagano","hinodebashi"]
    all_x_rain=np.array([])
    all_x_river=np.array([])
    all_x_geo=np.array([])
    all_y=np.array([])
    for location in locations:
        with open(new_path + "train_future/x_rain/" + location+".jb",mode="rb") as f:
            x_rain = joblib.load(f)
        with open(new_path + "train_future/x_river/" + location+".jb",mode="rb") as f:
            x_river = joblib.load(f)
        with open(new_path + "train_future/x_geo/" + location+".jb",mode="rb") as f:
            x_geo = joblib.load(f)    
        with open(new_path + "train_future/y_data/" + location+".jb",mode="rb") as f:
            y = joblib.load(f)
        x_rain = reshape_rain(x_rain,window)
        print(np.array(x_rain).shape)
        x_geo = reshape_geo(x_geo)

        if len(all_y)==0 and len(all_x_rain)==0 and len(all_x_river)==0 and len(all_y)==0:
            all_x_rain=np.array(x_rain)
            all_x_river=np.array(x_river)
            all_x_geo=np.array(x_geo)
            all_y=np.array(y)
        else:
            all_x_rain=np.concatenate([all_x_rain,np.array(x_rain)],axis=0)
            all_x_river=np.concatenate([all_x_river,np.array(x_river)],axis=0)
            all_x_geo=np.concatenate([all_x_geo,np.array(x_geo)],axis=0)
            all_y=np.concatenate([all_y,np.array(y)],axis=0)
    
    all_x_rain = np.expand_dims(all_x_rain,1)/255
    all_x_river = np.squeeze(all_x_river).astype(float)
    all_x_geo = np.expand_dims(all_x_geo,1)
    
    #all_y=np.log(all_y+5)
    
    all_x_rain = torch.Tensor(all_x_rain)
    all_x_river = torch.Tensor(all_x_river)
    all_x_geo = torch.Tensor(all_x_geo)
    all_y_t = torch.Tensor(all_y)

    ds_train = TensorDataset(all_x_rain, all_x_geo, all_x_river, all_y_t)
    return DataLoader(ds_train, batch_size=16, shuffle=True), all_y

def create_testloader(location):
    window=5
    new_path="/home/student/e17/e175764/"

    with open(new_path + "test_future/"+location+"/x_rain/" + location+".jb",mode="rb") as f:
        x_rain = joblib.load(f)
    with open(new_path + "test_future/"+location+"/x_river/" + location+".jb",mode="rb") as f:
        x_river = joblib.load(f)
    with open(new_path + "test_future/"+location+"/x_geo/" + location+".jb",mode="rb") as f:
        x_geo = joblib.load(f)
    with open(new_path + "test_future/"+location+"/y_data/" + location+".jb",mode="rb") as f:
        y = joblib.load(f)

    x_rain = reshape_rain(x_rain,window)
    x_geo = reshape_geo(x_geo)

    x_rain = np.expand_dims(x_rain,1)/255
    x_river = np.squeeze(x_river).astype(float)
    x_geo = np.expand_dims(x_geo,1)
    
    x_rain = torch.Tensor(x_rain)
    x_river = torch.Tensor(x_river)
    x_geo = torch.Tensor(x_geo)
    y_t = torch.Tensor(y)

    ds_test = TensorDataset(x_rain, x_geo, x_river, y_t)
    return DataLoader(ds_test, batch_size=16, shuffle=False)

def estimate(data_loader, model, device):
    model.eval()
    obs = []
    pred = []
    for rain_x, geo_x, river_x, target in data_loader:
        rain_x = rain_x.to(device, dtype=torch.float)
        geo_x = geo_x.to(device, dtype=torch.float)
        river_x = river_x.to(device, dtype=torch.float)
        target=target.cpu()
        obs.extend(target.numpy().flatten())
        with torch.no_grad():
            output = model(rain_x, geo_x,river_x)
            output = output.cpu()
            pred.extend(output.numpy().flatten())
    return obs,pred

def main():
    initialize_seed(11)
    # DataLoaderを作成
    train_loader, train_y= create_dataloader()
    # デバイス（GPU/CPU）の設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 初期モデルを作成
    model = Model().to(device)
    #yの値を用いて損失関数を定義
    loss_fn = DenseLoss(train_y)
    train_y = torch.Tensor(train_y).to(device)
    
    # 損失関数とOptimizerを作成
    #loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001,eps=1e-3, amsgrad=True)
    # 学習
    for epoch in range(1,201):
        train_loss = train(train_loader, model, optimizer, loss_fn, device)
        print(f'epoch:{epoch}, loss:{train_loss:.5f}')
        # 以下，省略
    print("Finish Training!")
    print("Estimation from here.")
    print("--------------------")
    testloader = create_testloader("ishida")
    obs,pred = estimate(testloader,model,device)
    res=[]
    #pred = np.exp(np.array(pred))-5
    for i in range(len(obs)):
        res.append([obs[i],pred[i]])
    result = pd.DataFrame(res,columns=["obs","pred"])
    
    print(pred)
    print(result)
    model = model.to('cpu')
    torch.save(model.state_dict(), '/home/student/e17/e175764/ML_models/model_cpu.pth')

if __name__ == '__main__':
    main()
