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
import random
import shutil

# 確保目錄存在
os.makedirs('model', exist_ok=True)
os.makedirs('src/images', exist_ok=True)

# 設定隨機種子以確保可重現結果
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

NUM_EPOCHS = 50

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
            return image, label, img_name

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
def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, num_epochs=10, patience=5):
    best_val_acc = 0.0
    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []
    
    # 早停機制
    patience_counter = 0
    early_stopped = False
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # 訓練階段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels, _ in tqdm(train_loader):
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
            for inputs, labels, _ in tqdm(valid_loader):
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
            torch.save(model.state_dict(), 'model/best_model.pth')
            print(f'模型已保存，驗證準確率: {best_val_acc:.4f}')
            # 重置耐心計數器
            patience_counter = 0
        else:
            # 增加耐心計數器
            patience_counter += 1
            print(f'驗證準確率未提升，耐心計數: {patience_counter}/{patience}')
            
            # 檢查是否需要早停
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                early_stopped = True
                break
        
        # 每五個epoch保存一次模型
        if (epoch + 1) % 5 == 0:
            model_save_path = os.path.join('model', f'model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), model_save_path)
            print(f'Epoch {epoch+1} 模型已保存至 {model_save_path}')
    
    # 記錄訓練結果
    training_info = {
        'epochs_completed': epoch + 1,
        'best_val_acc': best_val_acc,
        'early_stopped': early_stopped,
        'train_losses': train_losses,
        'valid_losses': valid_losses,
        'train_accs': train_accs,
        'valid_accs': valid_accs
    }
    
    # 分別繪製損失和準確率曲線
    # 損失曲線
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('src/images/loss_curve.png')
    plt.close()
    
    # 準確率曲線
    plt.figure(figsize=(10, 5))
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(valid_accs, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.savefig('src/images/accuracy_curve.png')
    plt.close()
    
    return model, training_info

# 分析驗證集上的模型性能
def analyze_model_performance():
    print("分析模型在驗證集上的性能...")
    
    # 載入最佳模型
    model.load_state_dict(torch.load('model/best_model.pth'))
    model.eval()
    
    # 用於收集每個品種的預測結果
    breed_results = {breed: {'correct': 0, 'total': 0, 'images': []} for breed in all_breeds}
    
    # 收集所有驗證集圖像和其預測結果
    all_images = []
    all_labels = []
    all_predictions = []
    all_image_ids = []
    
    with torch.no_grad():
        for inputs, labels, img_names in tqdm(valid_loader, desc="評估驗證集"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            # 將資料移到CPU並轉換為numpy陣列
            labels_np = labels.cpu().numpy()
            predicted_np = predicted.cpu().numpy()
            
            # 收集結果
            for i in range(len(img_names)):
                img_id = img_names[i]
                label = labels_np[i]
                pred = predicted_np[i]
                breed = idx_to_breed[label]
                pred_breed = idx_to_breed[pred]
                
                # 記錄結果
                breed_results[breed]['total'] += 1
                if label == pred:
                    breed_results[breed]['correct'] += 1
                
                # 記錄圖片資訊
                breed_results[breed]['images'].append({
                    'img_id': img_id,
                    'predicted': pred_breed,
                    'correct': (label == pred)
                })
                
                all_images.append(inputs[i].cpu())
                all_labels.append(label)
                all_predictions.append(pred)
                all_image_ids.append(img_id)
    
    # 計算每個品種的準確率
    for breed in breed_results:
        if breed_results[breed]['total'] > 0:
            breed_results[breed]['accuracy'] = breed_results[breed]['correct'] / breed_results[breed]['total']
        else:
            breed_results[breed]['accuracy'] = 0
    
    # 找出錯誤最多的十種狗品種
    error_rates = {breed: 1 - results['accuracy'] for breed, results in breed_results.items() if results['total'] > 0}
    worst_breeds = sorted(error_rates.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # 找出準確率最高的十種狗品種
    accuracies = {breed: results['accuracy'] for breed, results in breed_results.items() if results['total'] > 0}
    best_breeds = sorted(accuracies.items(), key=lambda x: x[1], reverse=True)[:10]
    
    print("\n錯誤率最高的十種狗品種:")
    for breed, error_rate in worst_breeds:
        print(f"{breed}: {error_rate:.4f}")
    
    print("\n準確率最高的十種狗品種:")
    for breed, accuracy in best_breeds:
        print(f"{breed}: {accuracy:.4f}")
    
    # 可視化最差的10種狗品種
    visualize_breed_samples(worst_breeds, breed_results, "worst")
    
    # 可視化最好的10種狗品種
    visualize_breed_samples(best_breeds, breed_results, "best")
    
    return worst_breeds, best_breeds

def visualize_breed_samples(breed_list, breed_results, category):
    """可視化指定品種的樣本圖像"""
    for i, (breed, _) in enumerate(breed_list):
        # 獲取品種的圖像列表
        breed_images = breed_results[breed]['images']
        
        # 分離正確和錯誤分類的圖像
        correct_images = [img for img in breed_images if img['correct']]
        incorrect_images = [img for img in breed_images if not img['correct']]
        
        # 選擇要顯示的圖像（最多5張）
        if category == "worst":
            # 對於最差的品種，顯示錯誤分類的圖像
            sample_images = incorrect_images[:5] if len(incorrect_images) >= 5 else incorrect_images
        else:
            # 對於最好的品種，顯示正確分類的圖像
            sample_images = correct_images[:5] if len(correct_images) >= 5 else correct_images
        
        # 如果樣本不足5張，從另一類中補充
        if len(sample_images) < 5:
            if category == "worst":
                additional = correct_images[:5-len(sample_images)]
            else:
                additional = incorrect_images[:5-len(sample_images)]
            sample_images.extend(additional)
        
        # 樣本仍然不足5張，則跳過
        if len(sample_images) == 0:
            continue
        
        # 創建一張圖表來顯示這個品種的樣本
        fig, axes = plt.subplots(1, min(5, len(sample_images)), figsize=(15, 4))
        if len(sample_images) == 1:
            axes = [axes]
        
        for j, img_info in enumerate(sample_images[:5]):
            img_id = img_info['img_id']
            img_path = os.path.join(TRAIN_DIR, f"{img_id}.jpg")
            img = Image.open(img_path).convert('RGB')
            
            # 顯示圖像
            axes[j].imshow(img)
            axes[j].axis('off')
            
            # 設置標題顯示真實品種和預測品種
            status = "Correct" if img_info['correct'] else "Wrong"
            title = f"{status}\nTrue: {breed}\nPred: {img_info['predicted']}"
            axes[j].set_title(title, fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f"src/images/{category}_{i+1}_{breed.replace(' ', '_')}.png")
        plt.close()
    
    # 創建一個彙總圖，顯示所有品種的表現
    plt.figure(figsize=(12, 8))
    
    if category == "worst":
        title = "Top 10 Dog Breeds with Highest Error Rate"
        values = [error_rate for _, error_rate in breed_list]
        ylabel = "Error Rate"
    else:
        title = "Top 10 Dog Breeds with Highest Accuracy"
        values = [accuracy for _, accuracy in breed_list]
        ylabel = "Accuracy"
    
    breeds = [breed for breed, _ in breed_list]
    
    plt.barh(range(len(breeds)), values, align='center')
    plt.yticks(range(len(breeds)), breeds)
    plt.xlabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"src/images/{category}_breeds_summary.png")
    plt.close()

# 訓練模型
trained_model, training_info = train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS, patience=10)

# 輸出訓練結果摘要
print("\n訓練摘要:")
print(f"完成的訓練周期: {training_info['epochs_completed']}/{NUM_EPOCHS}")
print(f"最佳驗證準確率: {training_info['best_val_acc']:.4f}")
if training_info['early_stopped']:
    print("訓練因早停機制提前結束")

# 分析模型性能
worst_breeds, best_breeds = analyze_model_performance()

# 保存分析結果
with open('src/images/performance_summary.txt', 'w') as f:
    f.write("訓練摘要:\n")
    f.write(f"完成的訓練周期: {training_info['epochs_completed']}/{NUM_EPOCHS}\n")
    f.write(f"最佳驗證準確率: {training_info['best_val_acc']:.4f}\n")
    if training_info['early_stopped']:
        f.write("訓練因早停機制提前結束\n")
    
    f.write("\n錯誤率最高的十種狗品種:\n")
    for breed, error_rate in worst_breeds:
        f.write(f"{breed}: {error_rate:.4f}\n")
    
    f.write("\n準確率最高的十種狗品種:\n")
    for breed, accuracy in best_breeds:
        f.write(f"{breed}: {accuracy:.4f}\n")

# 預測測試集
def predict_test_data(model=None):
    # 如果沒有提供模型，則載入最佳模型
    if model is None:
        # 創建一個新的模型實例並載入狀態字典
        loaded_model = DogBreedClassifier(num_classes).to(device)
        loaded_model.load_state_dict(torch.load('model/best_model.pth'))
        model = loaded_model
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
predict_test_data(trained_model)

print("完成！") 