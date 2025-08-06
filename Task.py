from transformers import pipeline
import pandas as pd
import gradio as gr
import json
from collections import Counter
import numpy as np
import re
import arabic_reshaper
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

plt.style.use('default')
sns.set_palette("husl")

file_path = "C:\\Users\\soliman\\train.csv"  
df = pd.read_csv(file_path)

pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")


def parse_int(value):
    try:
        return int(float(str(value).strip()))
    except:
        return "unknown"
    

def row_to_invoice_line(row):
    name = row.get("Item_Name", "unknown")
    category = row.get("class", "unknown")
    quantity = parse_int(row.get("Number of units"))

    price_raw = row.get("Price") if pd.notnull(row.get("Price")) else row.get("T.Price")
    price = price_raw if isinstance(price_raw, int) else "unknown"

    return f"{name} - {category} - Qty: {quantity} - Price: {price}"


def build_prompt(text):
    return (
        "<|system|>\nYou are an expert invoice reader. Extract the following fields from a product line and reply ONLY in JSON:\n"
        "- product_name\n- category\n- quantity\n- unit_price\n"
        "<|user|>\n" +
        text +
        "\n<|assistant|>\n"
    )

def parse_model_output(output):
    match = re.search(r"\{.*\}", output, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except:
        return None
    

def is_arabic(text):
    return any("\u0600" <= c <= "\u06FF" or "\u0750" <= c <= "\u077F" for c in text)


def create_visualizations(df_stats):
    """Create and save visualization plots"""
    if df_stats.empty:
        return None
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Invoice Data Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Price vs Quantity Scatter Plot
    valid_data = df_stats.dropna(subset=['unit_price', 'quantity'])
    if not valid_data.empty:
        axes[0, 0].scatter(valid_data['quantity'], valid_data['unit_price'], alpha=0.6, s=50)
        axes[0, 0].set_xlabel('Quantity')
        axes[0, 0].set_ylabel('Unit Price')
        axes[0, 0].set_title('Price vs Quantity Relationship')
        axes[0, 0].grid(True, alpha=0.3)
    else:
        axes[0, 0].text(0.5, 0.5, 'No valid price/quantity data', 
                       ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('Price vs Quantity Relationship')
    
    # 2. Language Distribution Pie Chart
    english_count = sum(not is_arabic(str(p)) for p in df_stats["product_name"])
    arabic_count = len(df_stats) - english_count
    
    if english_count > 0 or arabic_count > 0:
        language_data = [english_count, arabic_count]
        language_labels = ['English', 'Arabic']
        colors = ['#ff9999', '#66b3ff']
        
        axes[0, 1].pie(language_data, labels=language_labels, autopct='%1.1f%%', 
                      colors=colors, startangle=90)
        axes[0, 1].set_title('Product Name Language Distribution')
    else:
        axes[0, 1].text(0.5, 0.5, 'No language data available', 
                       ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Product Name Language Distribution')
    
    # 3. Category Distribution Bar Chart
    category_counts = df_stats['category'].value_counts().head(10)
    if not category_counts.empty:
        bars = axes[0, 2].bar(range(len(category_counts)), category_counts.values)
        axes[0, 2].set_xlabel('Categories')
        axes[0, 2].set_ylabel('Count')
        axes[0, 2].set_title('Top 10 Categories Distribution')
        axes[0, 2].set_xticks(range(len(category_counts)))
        axes[0, 2].set_xticklabels(category_counts.index, rotation=45, ha='right')
        

        for bar in bars:
            height = bar.get_height()
            axes[0, 2].text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}', ha='center', va='bottom')
    else:
        axes[0, 2].text(0.5, 0.5, 'No category data available', 
                       ha='center', va='center', transform=axes[0, 2].transAxes)
        axes[0, 2].set_title('Categories Distribution')
    
    # 4. Price Distribution Histogram
    valid_prices = df_stats['unit_price'].dropna()
    if not valid_prices.empty:
        axes[1, 0].hist(valid_prices, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1, 0].set_xlabel('Unit Price')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Unit Price Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        mean_price = valid_prices.mean()
        axes[1, 0].axvline(mean_price, color='red', linestyle='--', 
                          label=f'Mean: {mean_price:.2f}')
        axes[1, 0].legend()
    else:
        axes[1, 0].text(0.5, 0.5, 'No valid price data', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Unit Price Distribution')
    
    # 5. Quantity Distribution Box Plot
    valid_quantities = df_stats['quantity'].dropna()
    if not valid_quantities.empty:
        box_plot = axes[1, 1].boxplot(valid_quantities, patch_artist=True)
        box_plot['boxes'][0].set_facecolor('lightcoral')
        axes[1, 1].set_ylabel('Quantity')
        axes[1, 1].set_title('Quantity Distribution (Box Plot)')
        axes[1, 1].grid(True, alpha=0.3)
        
        stats_text = f'Median: {valid_quantities.median():.1f}\nIQR: {valid_quantities.quantile(0.75) - valid_quantities.quantile(0.25):.1f}'
        axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes, 
                       fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
    else:
        axes[1, 1].text(0.5, 0.5, 'No valid quantity data', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Quantity Distribution (Box Plot)')
    
    # 6. Top Products by Frequency
    product_counts = df_stats['product_name'].value_counts().head(8)
    if not product_counts.empty:
        bars = axes[1, 2].barh(range(len(product_counts)), product_counts.values)
        axes[1, 2].set_xlabel('Frequency')
        axes[1, 2].set_ylabel('Products')
        axes[1, 2].set_title('Top 8 Most Frequent Products')
        axes[1, 2].set_yticks(range(len(product_counts)))
        

        truncated_labels = [label[:30] + '...' if len(label) > 30 else label 
                           for label in product_counts.index]
        axes[1, 2].set_yticklabels(truncated_labels)
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            axes[1, 2].text(width, bar.get_y() + bar.get_height()/2.,
                           f'{int(width)}', ha='left', va='center')
    else:
        axes[1, 2].text(0.5, 0.5, 'No product data available', 
                       ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Most Frequent Products')
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"invoice_analysis_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    
    return plot_filename


def generate_results(n_rows):
    if n_rows < 1:
        return "Please enter a number greater than 0.", None, None

    limited_df = df.head(n_rows)
    limited_df["invoice_line"] = limited_df.apply(row_to_invoice_line, axis=1)

    results = []
    raw_outputs = []

    for i, row in limited_df.iterrows():
        prompt = build_prompt(row["invoice_line"])
        output = pipe(prompt, max_new_tokens=200, do_sample=False, temperature=0.0)[0]['generated_text']
        json_output = output.split("<|assistant|>")[-1].strip()
        results.append(json_output)
        raw_outputs.append(json_output)

    parsed_outputs = [parse_model_output(r) for r in results if parse_model_output(r)]
    
    if not parsed_outputs:
        return "Model failed to return valid JSON outputs.", None, None
    
    df_stats = pd.DataFrame(parsed_outputs)


    df_stats["unit_price"] = pd.to_numeric(df_stats["unit_price"], errors="coerce")
    df_stats["quantity"] = pd.to_numeric(df_stats["quantity"], errors="coerce")

    if df_stats.empty:
        return "No valid data after cleaning.", None, None

  
    df_stats['processed_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df_stats['row_index'] = range(len(df_stats))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"extracted_invoice_data_{timestamp}.csv"
    df_stats.to_csv(csv_filename, index=False, encoding='utf-8')

  
    plot_filename = create_visualizations(df_stats)


    avg_price = df_stats["unit_price"].mean()
    avg_quantity = df_stats["quantity"].mean()
    most_common_product = df_stats["product_name"].mode().iloc[0] if not df_stats["product_name"].mode().empty else "N/A"
    most_common_category = df_stats["category"].mode().iloc[0] if not df_stats["category"].mode().empty else "N/A"


    english_count = sum(not is_arabic(str(p)) for p in df_stats["product_name"])
    arabic_count = len(df_stats) - english_count
    total = len(df_stats)
    english_ratio = english_count / total * 100 if total > 0 else 0
    arabic_ratio = arabic_count / total * 100 if total > 0 else 0

    unique_products = df_stats["product_name"].nunique()
    unique_categories = df_stats["category"].nunique()

    price_std = df_stats["unit_price"].std()
    quantity_std = df_stats["quantity"].std()
    median_price = df_stats["unit_price"].median()
    median_quantity = df_stats["quantity"].median()

    output_str = "\n\n".join(raw_outputs)
    stats_str = (
   
        f"Successfully processed {len(df_stats)} invoice lines\n"
        f"Data saved to: {csv_filename}\n"
        f"Visualizations saved to: {plot_filename}\n\n"
        

        f"Average unit price: {avg_price:.2f} (±{price_std:.2f})\n"
        f"Median unit price: {median_price:.2f}\n"
        f"Average quantity: {avg_quantity:.2f} (±{quantity_std:.2f})\n"
        f"Median quantity: {median_quantity:.2f}\n\n"
        
        
        f"Most common product: {most_common_product}\n"
        f"Most common category: {most_common_category}\n\n"
    
        f"English product names: {english_ratio:.1f}% ({english_count} items)\n"
        f"Arabic product names: {arabic_ratio:.1f}% ({arabic_count} items)\n"
        f"Unique products: {unique_products}\n"
        f"Unique categories: {unique_categories}\n\n"
    )

    return output_str + stats_str, csv_filename, plot_filename


demo = gr.Interface(
    fn=generate_results,
    inputs=gr.Number(
        label="Number of rows to process",
        value=10,
        minimum=1,
        maximum=1000,
        step=1
    ),
    outputs=[
        gr.Textbox(
            label="Processing Results & Statistics", 
            lines=30,
            max_lines=50
        ),
        gr.File(label="Download Extracted Data (CSV)"),
        gr.File(label="Download Analysis Charts (PNG)")
    ],
    title="Advanced Invoice Data Extractor & Analyzer",
    theme=gr.themes.Soft(),
    allow_flagging="never"
)


demo.launch()