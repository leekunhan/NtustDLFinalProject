import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure the output directory exists
os.makedirs('src/images', exist_ok=True)

# Define the path to the labels file
LABELS_PATH = 'labels.csv'

def analyze_dataset(labels_path):
    """
    Analyzes the dataset to count occurrences and proportions of each breed.

    Args:
        labels_path (str): The path to the CSV file containing image IDs and breeds.

    Prints:
        Descriptive statistics of breed counts.
        Count and proportion for each breed.

    Generates:
        A bar plot of breed distribution saved to 'src/images/breed_distribution.png'.
    """
    print(f"讀取標籤檔案: {labels_path}")
    try:
        labels_df = pd.read_csv(labels_path)
    except FileNotFoundError:
        print(f"錯誤: 找不到標籤檔案 {labels_path}。請確保檔案存在於正確的路徑。")
        return

    print("\n資料集基本資訊:")
    labels_df.info()

    print("\n各犬種數量統計:")
    breed_counts = labels_df['breed'].value_counts()
    
    print("\n犬種數量描述性統計:")
    print(breed_counts.describe())

    print("\n各犬種數量與比例:")
    breed_proportions = breed_counts / len(labels_df)
    
    analysis_results = pd.DataFrame({
        'Count': breed_counts,
        'Proportion': breed_proportions
    })
    print(analysis_results)

    # 可視化犬種分佈
    plt.figure(figsize=(20, 12)) # 增加圖表大小以容納更多標籤
    sns.barplot(x=breed_counts.index, y=breed_counts.values, palette="viridis")
    plt.xlabel("Breed", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.title("Distribution of Dog Breeds in the Training Set", fontsize=16)
    plt.xticks(rotation=90, ha='right', fontsize=8) # 旋轉標籤並調整字型大小
    plt.yticks(fontsize=10)
    plt.tight_layout() # 自動調整子圖參數以提供緊湊的佈局
    
    plot_path = 'src/images/breed_distribution.png'
    try:
        plt.savefig(plot_path)
        print(f"\n犬種分佈圖已儲存至: {plot_path}")
    except Exception as e:
        print(f"儲存圖片時發生錯誤: {e}")
    
    plt.show() # 在腳本執行時顯示圖表 (如果環境支援)
    plt.close()

if __name__ == '__main__':
    analyze_dataset(LABELS_PATH)
