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
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tqdm import tqdm
import random
import shutil
import cv2

# Ensure directories exist
os.makedirs('model', exist_ok=True)
os.makedirs('src/images', exist_ok=True)

# Set random seeds for reproducible results
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

NUM_EPOCHS = 100

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data paths
TRAIN_DIR = 'train'
TEST_DIR = 'test'
LABELS_PATH = 'labels.csv'

# Read labels
labels_df = pd.read_csv(LABELS_PATH)

# Get all breed categories
all_breeds = sorted(labels_df['breed'].unique())
breed_to_idx = {breed: idx for idx, breed in enumerate(all_breeds)}
idx_to_breed = {idx: breed for idx, breed in enumerate(all_breeds)}
num_classes = len(all_breeds)

print(f"Total {num_classes} dog breeds")

# Custom data set
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
        # Correct path handling, ensuring file name includes .jpg extension
        if self.is_test or img_name.endswith('.jpg'):
            img_path = os.path.join(self.image_dir, img_name)
        else:
            img_path = os.path.join(self.image_dir, f"{img_name}.jpg")
        
        # Debug information
        if not os.path.exists(img_path):
            print(f"File not found: {img_path}")
            available_files = os.listdir(self.image_dir)[:5]
            print(f"First 5 files in directory: {available_files}")
            raise FileNotFoundError(f"File not found: {img_path}")
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        if self.is_test:
            return image, img_name
        else:
            breed = self.df.loc[self.df['id'] == img_name, 'breed'].values[0]
            label = breed_to_idx[breed]
            return image, label, img_name

# Data enhancement and preprocessing
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

# Split training set and validation set
train_df, valid_df = train_test_split(labels_df, test_size=0.2, random_state=42, stratify=labels_df['breed'])

# Create dataset and data loader
train_dataset = DogBreedDataset(TRAIN_DIR, train_df, transform=train_transform)
valid_dataset = DogBreedDataset(TRAIN_DIR, valid_df, transform=valid_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4)

# Use pre-trained ResNet-50 model
class DogBreedClassifier(nn.Module):
    def __init__(self, num_classes):
        super(DogBreedClassifier, self).__init__()
        self.model = models.resnet50(weights='IMAGENET1K_V1')
        
        # Freeze most layers
        for param in list(self.model.parameters())[:-20]:
            param.requires_grad = False
            
        # Modify the last fully connected layer to fit our class count
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

# Create model
model = DogBreedClassifier(num_classes).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam([
    {'params': list(model.model.fc.parameters()), 'lr': 1e-3},
    {'params': list(model.model.parameters())[:-20], 'lr': 1e-5}
])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

# Training function
def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, num_epochs=100, patience=10):
    best_val_acc = 0.0
    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []
    
    # Early stopping mechanism
    patience_counter = 0
    early_stopped = False
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Training stage
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
        
        print(f'Training loss: {epoch_train_loss:.4f}, Training accuracy: {epoch_train_acc:.4f}')
        
        # Validation stage
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
        
        print(f'Validation loss: {epoch_valid_loss:.4f}, Validation accuracy: {epoch_valid_acc:.4f}')
        
        # Adjust learning rate
        scheduler.step(epoch_valid_loss)
        
        # Save best model
        if epoch_valid_acc > best_val_acc:
            best_val_acc = epoch_valid_acc
            torch.save(model.state_dict(), 'model/best_model.pth')
            print(f'Model saved, Validation accuracy: {best_val_acc:.4f}')
            # Reset patience counter
            patience_counter = 0
        else:
            # Increase patience counter
            patience_counter += 1
            print(f'Validation accuracy did not improve, Patience counter: {patience_counter}/{patience}')
            
            # Check if early stopping is triggered
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                early_stopped = True
                break
        
        # Save model every 5 epochs
        if (epoch + 1) % 5 == 0:
            model_save_path = os.path.join('model', f'model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), model_save_path)
            print(f'Epoch {epoch+1} model saved to {model_save_path}')
    
    # Record training results
    training_info = {
        'epochs_completed': epoch + 1,
        'best_val_acc': best_val_acc,
        'early_stopped': early_stopped,
        'train_losses': train_losses,
        'valid_losses': valid_losses,
        'train_accs': train_accs,
        'valid_accs': valid_accs
    }
    
    # Plot loss and accuracy curves separately
    # Loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('src/images/loss_curve.png')
    plt.close()
    
    # Accuracy curve
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

# Analyze model performance on validation set
def analyze_model_performance():
    print("Analyzing model performance on validation set...")
    
    # Load best model
    model.load_state_dict(torch.load('model/best_model.pth'))
    model.eval()
    
    # Used to collect prediction results for each breed
    breed_results = {breed: {'correct': 0, 'total': 0, 'images': []} for breed in all_breeds}
    
    # Collect all validation set images and their prediction results
    all_images = []
    all_labels = []
    all_predictions = []
    all_image_ids = []
    
    with torch.no_grad():
        for inputs, labels, img_names in tqdm(valid_loader, desc="Evaluating validation set"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            # Move data to CPU and convert to numpy array
            labels_np = labels.cpu().numpy()
            predicted_np = predicted.cpu().numpy()
            
            # Collect results
            for i in range(len(img_names)):
                img_id = img_names[i]
                label = labels_np[i]
                pred = predicted_np[i]
                breed = idx_to_breed[label]
                pred_breed = idx_to_breed[pred]
                
                # Record results
                breed_results[breed]['total'] += 1
                if label == pred:
                    breed_results[breed]['correct'] += 1
                
                # Record image information
                breed_results[breed]['images'].append({
                    'img_id': img_id,
                    'predicted': pred_breed,
                    'correct': (label == pred)
                })
                
                all_images.append(inputs[i].cpu())
                all_labels.append(label)
                all_predictions.append(pred)
                all_image_ids.append(img_id)
    
    # Calculate accuracy for each breed
    for breed in breed_results:
        if breed_results[breed]['total'] > 0:
            breed_results[breed]['accuracy'] = breed_results[breed]['correct'] / breed_results[breed]['total']
        else:
            breed_results[breed]['accuracy'] = 0
    
    # Find top 10 dog breeds with highest error rate
    error_rates = {breed: 1 - results['accuracy'] for breed, results in breed_results.items() if results['total'] > 0}
    worst_breeds = sorted(error_rates.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # Find top 10 dog breeds with highest accuracy
    accuracies = {breed: results['accuracy'] for breed, results in breed_results.items() if results['total'] > 0}
    best_breeds = sorted(accuracies.items(), key=lambda x: x[1], reverse=True)[:10]
    
    print("\nTop 10 dog breeds with highest error rate:")
    for breed, error_rate in worst_breeds:
        print(f"{breed}: {error_rate:.4f}")
    
    print("\nTop 10 dog breeds with highest accuracy:")
    for breed, accuracy in best_breeds:
        print(f"{breed}: {accuracy:.4f}")
    
    # Visualize worst 10 dog breeds
    visualize_breed_samples(worst_breeds, breed_results, "worst")
    
    # Visualize best 10 dog breeds
    visualize_breed_samples(best_breeds, breed_results, "best")
    
    # Generate confusion matrix for the worst performing breed only
    generate_single_worst_breed_confusion_matrix(worst_breeds[0][0], model)
    
    # Generate heatmap for worst breed
    generate_heatmaps_for_worst_breed(worst_breeds[0][0], breed_results, model)
    
    return worst_breeds, best_breeds

def visualize_breed_samples(breed_list, breed_results, category):
    """Visualize samples of specified breed"""
    for i, (breed, _) in enumerate(breed_list):
        # Get breed image list
        breed_images = breed_results[breed]['images']
        
        # Separate correct and incorrect classified images
        correct_images = [img for img in breed_images if img['correct']]
        incorrect_images = [img for img in breed_images if not img['correct']]
        
        # Select images to display (up to 5 images)
        if category == "worst":
            # For worst breeds, display incorrect classified images
            sample_images = incorrect_images[:5] if len(incorrect_images) >= 5 else incorrect_images
        else:
            # For best breeds, display correct classified images
            sample_images = correct_images[:5] if len(correct_images) >= 5 else correct_images
        
        # If sample images are less than 5, supplement from the other category
        if len(sample_images) < 5:
            if category == "worst":
                additional = correct_images[:5-len(sample_images)]
            else:
                additional = incorrect_images[:5-len(sample_images)]
            sample_images.extend(additional)
        
        # If sample images are still less than 5, skip
        if len(sample_images) == 0:
            continue
        
        # Create a figure to display this breed's samples
        fig, axes = plt.subplots(1, min(5, len(sample_images)), figsize=(15, 4))
        if len(sample_images) == 1:
            axes = [axes]
        
        for j, img_info in enumerate(sample_images[:5]):
            img_id = img_info['img_id']
            img_path = os.path.join(TRAIN_DIR, f"{img_id}.jpg")
            img = Image.open(img_path).convert('RGB')
            
            # Display image
            axes[j].imshow(img)
            axes[j].axis('off')
            
            # Set title to display true breed and predicted breed
            status = "Correct" if img_info['correct'] else "Wrong"
            title = f"{status}\nTrue: {breed}\nPred: {img_info['predicted']}"
            axes[j].set_title(title, fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f"src/images/{category}_{i+1}_{breed.replace(' ', '_')}.png")
        plt.close()
    
    # Create a summary figure to display performance of all breeds
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

def generate_single_worst_breed_confusion_matrix(worst_breed, model):
    """Generate confusion matrix for the worst performing breed only"""
    print(f"Generating confusion matrix for {worst_breed}...")
    
    model.eval()
    worst_breed_idx = breed_to_idx[worst_breed]
    
    # Create a data loader for the entire training set
    full_train_dataset = DogBreedDataset(TRAIN_DIR, labels_df, transform=valid_transform)
    full_train_loader = DataLoader(full_train_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Collect predictions for the worst breed
    true_labels = []
    predicted_labels = []
    misclassified_breeds = set()
    
    with torch.no_grad():
        for inputs, labels, img_names in tqdm(full_train_loader, desc=f"Evaluating {worst_breed} samples"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            # Move data to CPU and convert to numpy array
            labels_np = labels.cpu().numpy()
            predicted_np = predicted.cpu().numpy()
            
            # Collect only samples where true label is the worst breed
            for true_label, pred_label in zip(labels_np, predicted_np):
                if true_label == worst_breed_idx:
                    true_labels.append(true_label)
                    predicted_labels.append(pred_label)
                    if true_label != pred_label:
                        misclassified_breeds.add(pred_label)
    
    if len(true_labels) == 0:
        print(f"No samples found for {worst_breed}")
        return
    
    print(f"Found {len(true_labels)} samples for {worst_breed}")
    
    # Get top misclassified breeds (limit to top 10 for readability)
    misclass_counts = {}
    for pred_label in predicted_labels:
        if pred_label != worst_breed_idx:
            if pred_label not in misclass_counts:
                misclass_counts[pred_label] = 0
            misclass_counts[pred_label] += 1
    
    top_misclass = sorted(misclass_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # Include the correct breed and top misclassified breeds
    relevant_indices = [worst_breed_idx] + [idx for idx, _ in top_misclass]
    relevant_names = [idx_to_breed[idx] for idx in relevant_indices]
    
    print(f"Including {len(relevant_names)} breeds in confusion matrix:")
    print(f"- Target breed: {worst_breed}")
    print(f"- Top misclassified targets: {len(relevant_names) - 1}")
    
    # Create label mapping
    label_mapping = {idx: i for i, idx in enumerate(relevant_indices)}
    mapped_true = [label_mapping.get(label, -1) for label in true_labels]
    mapped_pred = [label_mapping.get(pred, -1) for pred in predicted_labels]
    
    # Filter valid mappings
    valid_indices = [i for i, (t, p) in enumerate(zip(mapped_true, mapped_pred)) if t != -1 and p != -1]
    mapped_true = [mapped_true[i] for i in valid_indices]
    mapped_pred = [mapped_pred[i] for i in valid_indices]
    
    # Generate confusion matrix (only one row since we only have one true breed)
    cm = confusion_matrix(mapped_true, mapped_pred, labels=list(range(len(relevant_names))))
    
    # Create visualization
    plt.figure(figsize=(max(10, len(relevant_names) * 0.8), 6))
    
    # Only show the first row (our target breed)
    cm_row = cm[0:1, :]  # Keep as 2D array for heatmap
    
    sns.heatmap(cm_row, annot=True, fmt='d', cmap='Blues', 
                xticklabels=relevant_names, 
                yticklabels=[worst_breed],
                cbar_kws={'label': 'Number of Predictions'})
    
    plt.title(f'Confusion Matrix for {worst_breed}\n(Shows where {worst_breed} samples are classified)')
    plt.xlabel('Predicted Breed')
    plt.ylabel('True Breed')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('src/images/worst_breed_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print detailed analysis
    total_samples = len(true_labels)
    correct_predictions = sum(1 for t, p in zip(true_labels, predicted_labels) if t == p)
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    
    print(f"\nDetailed analysis for {worst_breed}:")
    print(f"Total samples: {total_samples}")
    print(f"Correct predictions: {correct_predictions} ({accuracy:.1%})")
    print(f"Misclassifications: {total_samples - correct_predictions}")
    
    print("\nTop misclassification targets:")
    for i, (breed_name, count) in enumerate(zip(relevant_names[1:], cm_row[0, 1:]), 1):
        if count > 0:
            percentage = count / total_samples * 100
            print(f"  {i}. {breed_name}: {count} times ({percentage:.1f}%)")
    
    print(f"Confusion matrix saved to src/images/worst_breed_confusion_matrix.png")

def generate_heatmaps_for_worst_breed(worst_breed, breed_results, model):
    """Generate heatmap analysis for worst breed"""
    print(f"Generating heatmap analysis for {worst_breed}...")
    
    # Get incorrect classified images for this breed
    breed_images = breed_results[worst_breed]['images']
    incorrect_images = [img for img in breed_images if not img['correct']]
    
    # If incorrect classified images are less than 5, supplement correct classified images
    if len(incorrect_images) < 5:
        correct_images = [img for img in breed_images if img['correct']]
        sample_images = incorrect_images + correct_images[:5-len(incorrect_images)]
    else:
        sample_images = incorrect_images[:5]
    
    if len(sample_images) == 0:
        print(f"No sample images found for {worst_breed}")
        return
    
    # Generate Grad-CAM heatmap for each image
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    
    for i, img_info in enumerate(sample_images[:5]):
        img_id = img_info['img_id']
        img_path = os.path.join(TRAIN_DIR, f"{img_id}.jpg")
        
        # Load and preprocess image
        original_img = Image.open(img_path).convert('RGB')
        img_array = np.array(original_img)
        
        # Preprocess image for model
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(original_img).unsqueeze(0).to(device)
        
        # Generate Grad-CAM
        heatmap = generate_gradcam(model, input_tensor, breed_to_idx[worst_breed])
        
        # Original image
        axes[0, i].imshow(original_img)
        axes[0, i].set_title(f'Original\nTrue: {worst_breed}\nPred: {img_info["predicted"]}', fontsize=10)
        axes[0, i].axis('off')
        
        # Heatmap overlay
        superimposed_img = superimpose_heatmap(img_array, heatmap)
        axes[1, i].imshow(superimposed_img)
        axes[1, i].set_title('Attention Heatmap', fontsize=10)
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'src/images/{worst_breed.replace(" ", "_")}_attention_heatmaps.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Heatmap saved to src/images/{worst_breed.replace(' ', '_')}_attention_heatmaps.png")

def generate_gradcam(model, input_tensor, target_class):
    """Generate Grad-CAM heatmap"""
    model.eval()
    input_tensor.requires_grad_(True)
    
    # Get feature maps and gradients
    gradients = []
    activations = []
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    
    def forward_hook(module, input, output):
        activations.append(output)
    
    # Register hook to ResNet layer4 last layer
    target_layer = model.model.layer4[-1].conv3
    backward_handle = target_layer.register_backward_hook(backward_hook)
    forward_handle = target_layer.register_forward_hook(forward_hook)
    
    # Forward propagation
    output = model(input_tensor)
    
    # Zero gradients
    model.zero_grad()
    
    # Backward propagation to target class
    class_score = output[0, target_class]
    class_score.backward(retain_graph=True)
    
    # Calculate weights
    if gradients and activations:
        grads = gradients[0]
        acts = activations[0]
        
        # Calculate weights for each channel (Global Average Pooling)
        weights = torch.mean(grads, dim=(2, 3), keepdim=True)
        
        # Weighted average
        cam = torch.sum(weights * acts, dim=1).squeeze()
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize to 0-1
        if torch.max(cam) > 0:
            cam = cam / torch.max(cam)
        
        # Adjust size to input image size
        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), 
                          size=(224, 224), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().detach().numpy()
    else:
        # If gradients cannot be obtained, return empty heatmap
        cam = np.zeros((224, 224))
    
    # Remove hook
    backward_handle.remove()
    forward_handle.remove()
    
    return cam

def superimpose_heatmap(img, heatmap, alpha=0.6):
    """Overlay heatmap on original image"""
    # Adjust image size to match heatmap
    img_resized = cv2.resize(img, (224, 224))
    
    # Convert heatmap to color mapping
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay image
    superimposed = heatmap_colored * alpha + img_resized * (1 - alpha)
    superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)
    
    return superimposed

# Train model
trained_model, training_info = train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS, patience=10)

# Output training results summary
print("\nTraining summary:")
print(f"Completed training cycle: {training_info['epochs_completed']}/{NUM_EPOCHS}")
print(f"Best validation accuracy: {training_info['best_val_acc']:.4f}")
if training_info['early_stopped']:
    print("Training stopped early due to early stopping mechanism")

# Analyze model performance
worst_breeds, best_breeds = analyze_model_performance()

# Save analysis results
with open('src/images/performance_summary.txt', 'w') as f:
    f.write("Training summary:\n")
    f.write(f"Completed training cycle: {training_info['epochs_completed']}/{NUM_EPOCHS}\n")
    f.write(f"Best validation accuracy: {training_info['best_val_acc']:.4f}\n")
    if training_info['early_stopped']:
        f.write("Training stopped early due to early stopping mechanism\n")
    
    f.write("\nTop 10 dog breeds with highest error rate:\n")
    for breed, error_rate in worst_breeds:
        f.write(f"{breed}: {error_rate:.4f}\n")
    
    f.write("\nTop 10 dog breeds with highest accuracy:\n")
    for breed, accuracy in best_breeds:
        f.write(f"{breed}: {accuracy:.4f}\n")

# Predict test set
def predict_test_data(model=None):
    # If no model is provided, load best model
    if model is None:
        # Create a new model instance and load state dictionary
        loaded_model = DogBreedClassifier(num_classes).to(device)
        loaded_model.load_state_dict(torch.load('model/best_model.pth'))
        model = loaded_model
    model.eval()
    
    # Create test set dataset and loader
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_dataset = DogBreedDataset(TEST_DIR, transform=test_transform, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Create result DataFrame
    all_predictions = []
    all_image_ids = []
    
    with torch.no_grad():
        for inputs, img_names in tqdm(test_loader, desc="Predicting test data"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probabilities = F.softmax(outputs, dim=1)
            
            all_predictions.append(probabilities.cpu().numpy())
            all_image_ids.extend(img_names)
    
    all_predictions = np.vstack(all_predictions)
    
    # Prepare submission format
    submission = pd.DataFrame(all_predictions, columns=all_breeds)
    submission['id'] = all_image_ids
    submission['id'] = submission['id'].apply(lambda x: x.split('.')[0])  # Remove .jpg file extension
    submission = submission[['id'] + all_breeds]
    
    submission.to_csv('submission.csv', index=False)
    print("Submission file submission.csv generated")

# Predict test set and generate submission file
predict_test_data(trained_model)

print("Completed!") 