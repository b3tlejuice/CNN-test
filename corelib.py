import torch
import pandas as pd
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from torchvision.io import read_image, ImageReadMode

class Manipulator():
    __device__ = "cpu"
    __model__ = None
    __abc__ = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ" #"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    __transform__ = transforms.Compose([
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), shear=10, scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
        transforms.RandomResizedCrop((64,96), scale=(0.8,1.0)),
        #transforms.ToTensor()
    ])

    def getTransform(self):
        return self.__transform__

    def getDevice(self):
        return self.__device__

    def setDefaultDevice(self, name=""):
        torch.set_default_device(name)
        self.__device__ = name

    def genData(self, table="", saveFiles=False, debug=False):
        df = pd.read_csv(table)
        N = len(df)
        img_tensor = torch.empty((N, 1, 64, 96), dtype=torch.float32)
        ans_tensor = torch.empty((N,), dtype=torch.long)
        for i, row in df.iterrows():
            if debug and not (i+1)%(N//20):
                print(f"{(i+1)*100//N}%")
            img = read_image("res/"+row["filename"], ImageReadMode.GRAY)
            img_tensor[i] = self.__transform__(img) / 255.0
            ans_tensor[i] = self.__abc__.index(row["label"].replace("_caps", "").upper())
        if saveFiles:
            torch.save(img_tensor, "out/ImgData.pt")
            torch.save(ans_tensor, "out/AnsData.pt")
        return img_tensor, ans_tensor

    def loadData(self, X, y):
        return tuple(torch.load(f) for f in [X, y])

    def trainModel(self, data, percentage=80, fn_loss=nn.CrossEntropyLoss(), epochs=50, batch_size=128,
                   saveFiles=False, debug=False):
        device = torch.device(self.getDevice())
        X, y = data
        N = X.shape[0]
        index = torch.randperm(N)
        X, y = X[index], y[index]

        Np = N*percentage//100 
        X_train, y_train = X[:Np], y[:Np]
        X_val, y_val = X[Np:], y[Np:]

        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        g = torch.Generator(device=self.getDevice())
        train_loader = DataLoader(train_dataset, batch_size=batch_size, generator=g, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        model = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(256*4*6, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 36)
        ).to(device)
        optimizer = Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
        
        for i in range(epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                y_hat = model(batch_X)
                loss = fn_loss(y_hat, batch_y)
                loss.backward()
                optimizer.step()
            
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    val_pred = model(batch_X)
                    val_correct += (val_pred.argmax(1) == batch_y).sum().item()
                    val_total += batch_y.size(0)
            val_acc = val_correct / val_total * 100
            scheduler.step(val_acc)
            if debug:
                print(f"{i} loss: {loss:.4f}")
                print(f"Accuracy: {val_acc:.2f}%")

        if saveFiles:
            torch.save(model.state_dict(), "out/model_weights.pth")
        
        return
    
    def uploadModel(self, path):
        self.__model__ = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(256*4*6, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 36)
        ).to(self.getDevice())
        self.__model__.load_state_dict(torch.load(path, map_location=self.getDevice()))
        self.__model__.eval()
    
    def eval(self, img):
        tc = transforms.Compose([transforms.ToTensor()])
        img_tensor = torch.empty(1, 1, 64, 96, dtype=torch.float32)
        img_tensor[0] = tc(img)
        ans = self.__model__(img_tensor).argmax(1)[0]
        result = self.__abc__[ans]
        return result
        
    
    
if __name__ == "main":
    dm = Manipulator()
    dm.setDefaultDevice("cuda:0")
    #dm.genData("res/image_labels.csv",True, True)
    data = dm.loadData("out/ImgData.pt", "out/AnsData.pt")
    dm.trainModel(data, 80, epochs=50, batch_size=256, saveFiles=True, debug=True)