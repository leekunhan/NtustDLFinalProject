import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 設定隨機種子以確保可重現結果
torch.manual_seed(42)
np.random.seed(42)

# 設定設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用設備: {device}")

# 資料路徑
TRAIN_DIR = 'train'
TEST_DIR = 'test'
LABELS_PATH = 'labels.csv'

# 讀取標籤
labels_df = pd.read_csv(LABELS_PATH)

# 獲取所有品種類別
all_breeds = sorted(labels_df['breed'].unique())
breed_to_idx = {breed: idx for idx, breed in enumerate(all_breeds)}
idx_to_breed = {idx: breed for idx, breed in enumerate(all_breeds)}
num_classes = len(all_breeds)

print(f"總共有 {num_classes} 種狗品種")

# 自定義資料集
class DogBreedDataset(Dataset):
    def __init__(self, image_dir, df=None, transform=None, is_test=False):
        self.image_dir = image_dir
        self.df = df
        self.transform = transform
        self.is_test = is_test
        
        if is_test:
            self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        else:
            self.image_files = df['id'].values
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        # 修正路徑處理，確保文件名包含.jpg擴展名
        if self.is_test or img_name.endswith('.jpg'):
            img_path = os.path.join(self.image_dir, img_name)
        else:
            img_path = os.path.join(self.image_dir, f"{img_name}.jpg")
        
        # 調試信息
        if not os.path.exists(img_path):
            print(f"找不到文件: {img_path}")
            available_files = os.listdir(self.image_dir)[:5]
            print(f"目錄中前5個文件: {available_files}")
            raise FileNotFoundError(f"找不到文件: {img_path}")
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        if self.is_test:
            return image, img_name
        else:
            breed = self.df.loc[self.df['id'] == img_name, 'breed'].values[0]
            label = breed_to_idx[breed]
            return image, label

# 資料增強和預處理
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 分割訓練集和驗證集
train_df, valid_df = train_test_split(labels_df, test_size=0.2, random_state=42, stratify=labels_df['breed'])

# 創建資料集和資料載入器
train_dataset = DogBreedDataset(TRAIN_DIR, train_df, transform=train_transform)
valid_dataset = DogBreedDataset(TRAIN_DIR, valid_df, transform=valid_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4)

# 使用預訓練的ResNet-50模型
class DogBreedClassifier(nn.Module):
    def __init__(self, num_classes):
        super(DogBreedClassifier, self).__init__()
        self.model = models.resnet50(weights='IMAGENET1K_V1')
        
        # 凍結大部分的層
        for param in list(self.model.parameters())[:-20]:
            param.requires_grad = False
            
        # 修改最後的全連接層以符合我們的類別數量
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

# 創建模型
model = DogBreedClassifier(num_classes).to(device)

# 定義損失函數和優化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam([
    {'params': list(model.model.fc.parameters()), 'lr': 1e-3},
    {'params': list(model.model.parameters())[:-20], 'lr': 1e-5}
])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

# 訓練功能
def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, num_epochs=10):
    best_val_acc = 0.0
    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # 訓練階段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = correct / total
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)
        
        print(f'訓練損失: {epoch_train_loss:.4f}, 訓練準確率: {epoch_train_acc:.4f}')
        
        # 驗證階段
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(valid_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_valid_loss = running_loss / len(valid_loader.dataset)
        epoch_valid_acc = correct / total
        valid_losses.append(epoch_valid_loss)
        valid_accs.append(epoch_valid_acc)
        
        print(f'驗證損失: {epoch_valid_loss:.4f}, 驗證準確率: {epoch_valid_acc:.4f}')
        
        # 調整學習率
        scheduler.step(epoch_valid_loss)
        
        # 保存最佳模型
        if epoch_valid_acc > best_val_acc:
            best_val_acc = epoch_valid_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'模型已保存，驗證準確率: {best_val_acc:.4f}')
    
    # 繪製訓練過程
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(valid_accs, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.savefig('training_history.png')
    plt.close()
    
    return model

# 訓練模型
trained_model = train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, num_epochs=15)

# 預測測試集
def predict_test_data():
    # 載入最佳模型
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    # 創建測試集資料集和載入器
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_dataset = DogBreedDataset(TEST_DIR, transform=test_transform, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # 創建結果DataFrame
    all_predictions = []
    all_image_ids = []
    
    with torch.no_grad():
        for inputs, img_names in tqdm(test_loader, desc="預測測試資料"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probabilities = F.softmax(outputs, dim=1)
            
            all_predictions.append(probabilities.cpu().numpy())
            all_image_ids.extend(img_names)
    
    all_predictions = np.vstack(all_predictions)
    
    # 準備提交格式
    submission = pd.DataFrame(all_predictions, columns=all_breeds)
    submission['id'] = all_image_ids
    submission['id'] = submission['id'].apply(lambda x: x.split('.')[0])  # 移除.jpg副檔名
    submission = submission[['id'] + all_breeds]
    
    submission.to_csv('submission.csv', index=False)
    print("已生成提交檔案 submission.csv")

# 預測測試集並生成提交文件
predict_test_data()

print("完成！") 