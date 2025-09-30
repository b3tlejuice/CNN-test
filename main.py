import torch
import pandas as pd
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from torchvision.io import read_image, ImageReadMode

class Manipulator():
    __abc = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    __transform = transforms.Compose([
        transforms.RandomRotation(15),                # small rotation
        transforms.RandomAffine(0, translate=(0.1,0.1)),  # small shift
        transforms.RandomResizedCrop((64,96), scale=(0.9,1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # if grayscale, optional
        #transforms.ToTensor()
    ])

    def getTransform(self):
        return self.__transform

    def setDefaultDevice(self, name=""):
        torch.set_default_device(name)

    def genData(self, table="", saveFiles=False, debug=False):
        df = pd.read_csv(table)
        N = len(df)
        img_tensor = torch.empty((N, 1, 64, 96), dtype=torch.float32)
        ans_tensor = torch.empty((N,), dtype=torch.long)
        for i, row in df.iterrows():
            if debug and not (i+1)%(N//20):
                print(f"{(i+1)*100//N}%")
            img = read_image("res/"+row["filename"], ImageReadMode.GRAY)
            img_tensor[i] = self.__transform(img)
            ans_tensor[i] = self.__abc.index(row["label"].replace("_caps", ""))
        if saveFiles:
            torch.save(img_tensor, "ImgData.pt")
            torch.save(ans_tensor, "AnsData.pt")
        return img_tensor, ans_tensor

    def loadData(self, X, y):
        return tuple(torch.load(f) for f in [X, y])

    def trainModel(self, data, percentage=80, fn_loss=nn.CrossEntropyLoss(), epochs=50, batch_size=128,
                   saveFiles=False, debug=False):
        X, y = data
        N = X.shape[0]
        index = torch.randperm(N)
        X, y = X[index], y[index]

        Np = N*percentage//100 
        X_train, y_train = X[:Np], y[:Np]
        X_val, y_val = X[Np:], y[Np:]

        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        g = torch.Generator(device="cuda")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, generator=g, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        model = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(256*6, 512), nn.Dropout(0.5), nn.ReLU(),
            nn.Linear(512, 62)
        )

        optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
        
        for i in range(epochs):
            model.train()
            for batch_X, batch_y in train_loader:
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
            torch.save(model.state_dict(), "model_weights.pth")
        
        return model
    
dm = Manipulator()
dm.setDefaultDevice("cuda")
#dm.genData("res/image_labels.csv",True, True)
data = dm.loadData("ImgData.pt", "AnsData.pt")
dm.trainModel(data, 80, epochs=50, batch_size=256, saveFiles=True, debug=True)