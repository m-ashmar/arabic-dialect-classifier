import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


from wordcloud import WordCloud
from config import PLOT_STYLE, COLOR_PALETTE

class DataVisualizer:
    """Handles data visualization tasks"""
    
    def __init__(self):
        sns.set_style('whitegrid')

        sns.set_palette(COLOR_PALETTE)
        
    def plot_distribution(self, df: pd.DataFrame, column: str, save_path: str = None):
        """Plot class distribution"""
        if column not in df.columns:
            print(f"Error: The column '{column}' does not exist in the DataFrame.")
            return 
        plt.figure(figsize=(12, 8))
        counts = df[column].value_counts() 
        
        ax = counts.plot.pie(autopct='%1.1f%%', startangle=90)
        ax.set_title(f'Distribution by {column.capitalize()}', fontsize=14)
        ax.set_ylabel('')
          
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def generate_wordcloud(self, text: str, save_path: str = None):
        """Generate Arabic word cloud"""
        wordcloud = WordCloud(
            font_path='arial',
            width=1600,
            height=800,
            background_color='white'
        ).generate(text)
        
        plt.figure(figsize=(20, 10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_heatmap(self, matrix, labels, title: str, save_path: str = None):
        """Plot similarity heatmap"""
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            matrix,
            annot=True,
            cmap="YlGnBu",
            fmt=".2f",
            xticklabels=labels,
            yticklabels=labels
        )
        plt.title(title, fontsize=14)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()