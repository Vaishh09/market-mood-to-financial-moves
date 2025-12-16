"""
Real SHAP Analysis using the saved model
This script loads the trained model and computes actual SHAP values
"""
import sys
import os
import joblib
import json
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

print("=" * 60)
print("REAL SHAP ANALYSIS FROM SAVED MODEL")
print("=" * 60)

# Load the model
model_path = os.path.join(parent_dir, 'models', 'prisca_xgb_model.pkl')
print(f"Loading model from: {model_path}")
model = joblib.load(model_path)
print(f"✓ Model loaded: {type(model).__name__}")

# Load feature list
feature_path = os.path.join(parent_dir, 'models', 'feature_list.json')
with open(feature_path, 'r') as f:
    feature_list = json.load(f)
print(f"✓ Features loaded: {len(feature_list)} features")

# Load the dataset
data_path = os.path.join(parent_dir, 'data', 'final_modeling_dataset.csv')
print(f"Loading data from: {data_path}")
df = pd.read_csv(data_path, index_col=0, parse_dates=True)
df = df.sort_index()
print(f"✓ Data loaded: {df.shape}")

# Prepare the data just like in training
n = len(df)
holdout_size = int(n * 0.20)

# Get holdout set (last 20%)
holdout = df.iloc[-holdout_size:]

# Try to extract features that match the model
print("\n" + "=" * 60)
print("ATTEMPTING TO MATCH FEATURES")
print("=" * 60)

# Check which features exist in the dataset
missing_features = []
available_features = []
for feat in feature_list:
    if feat in df.columns:
        available_features.append(feat)
    else:
        missing_features.append(feat)

print(f"Available features: {len(available_features)}/{len(feature_list)}")
if missing_features:
    print(f"\n⚠️  Missing features ({len(missing_features)}):")
    for feat in missing_features[:10]:  # Show first 10
        print(f"   - {feat}")
    if len(missing_features) > 10:
        print(f"   ... and {len(missing_features) - 10} more")
    print("\n❌ Cannot compute SHAP: Dataset features don't match model expectations")
    print("The dataset appears to have raw features, but the model was trained")
    print("on engineered features (MA_5, MA_20, ema_12, etc.)")
    print("\nTo get real SHAP, you need to:")
    print("1. Re-run the full XGB_Regressor notebook to recreate engineered features")
    print("2. Or use the training data that was actually used to train the model")
    sys.exit(1)

# If we have all features, proceed with SHAP
X_holdout = holdout[feature_list]

print("\n" + "=" * 60)
print("COMPUTING SHAP VALUES")
print("=" * 60)
print(f"Using {len(X_holdout)} samples from holdout set")
print("This may take a minute...")

# Create SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_holdout)

print("✓ SHAP values computed")

# Calculate mean absolute SHAP values for each feature
mean_abs_shap = np.abs(shap_values).mean(axis=0)

# Create feature importance dataframe
shap_importance = pd.DataFrame({
    'feature': feature_list,
    'mean_abs_shap': mean_abs_shap
}).sort_values('mean_abs_shap', ascending=False)

# Normalize to percentages
total = shap_importance['mean_abs_shap'].sum()
shap_importance['percentage'] = (shap_importance['mean_abs_shap'] / total * 100)

print("\n" + "=" * 60)
print("TOP 10 FEATURES BY SHAP IMPORTANCE")
print("=" * 60)
for i, row in shap_importance.head(10).iterrows():
    print(f"{row['feature']:25s} {row['mean_abs_shap']:8.4f} ({row['percentage']:5.2f}%)")

print("\n" + "=" * 60)
print("TOP 3 FEATURES - DETAILED")
print("=" * 60)
for idx, (i, row) in enumerate(shap_importance.head(3).iterrows(), 1):
    feat = row['feature']
    feat_idx = feature_list.index(feat)
    feat_shap = shap_values[:, feat_idx]
    
    print(f"\n{idx}. {feat}")
    print(f"   Mean |SHAP|: {row['mean_abs_shap']:.4f}")
    print(f"   Percentage:  {row['percentage']:.2f}%")
    print(f"   SHAP range:  [{feat_shap.min():.2f}, {feat_shap.max():.2f}]")
# Generate SHAP summary plot
print("\n" + "=" * 60)
print("GENERATING SHAP SUMMARY PLOT")
print("=" * 60)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_holdout, plot_type="bar", show=False, max_display=15)
plt.title("SHAP Feature Importance (Real Analysis)", fontsize=14, fontweight='bold')
plt.tight_layout()

output_path = os.path.join(parent_dir, 'docs', 'real_shap_summary.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✓ Summary plot saved: {output_path}")

# Generate beautiful Plotly interactive chart
print("\n" + "=" * 60)
print("GENERATING INTERACTIVE PLOTLY CHARTS")
print("=" * 60)

# Get top 10 features
top_n = 10
top_features = shap_importance.head(top_n).copy()
top_features = top_features.iloc[::-1]  # Reverse for better visualization

# Color scheme - purple gradient
colors = ['#667eea', '#764ba2', '#9333ea', '#a855f7', '#c084fc', '#d8b4fe', '#e9d5ff', '#f3e8ff', '#faf5ff', '#fdf4ff']

# Create subplots: 1x2 layout (just top two charts)
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        '<b>SHAP Feature Importance</b>',
        '<b>Percentage Contribution</b>'
    ),
    specs=[
        [{'type': 'bar'}, {'type': 'pie'}]
    ],
    horizontal_spacing=0.15
)

# Chart 1: Horizontal bar chart - SHAP values
fig.add_trace(
    go.Bar(
        y=top_features['feature'],
        x=top_features['mean_abs_shap'],
        orientation='h',
        marker=dict(
            color=top_features['mean_abs_shap'],
            colorscale=[[0, '#667eea'], [0.5, '#764ba2'], [1, '#9333ea']],
            line=dict(color='white', width=1)
        ),
        text=[f"{val:.2f}" for val in top_features['mean_abs_shap']],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>SHAP: %{x:.4f}<extra></extra>',
        showlegend=False
    ),
    row=1, col=1
)

# Chart 2: Pie chart - Percentage contribution
fig.add_trace(
    go.Pie(
        labels=top_features['feature'],
        values=top_features['percentage'],
        marker=dict(
            colors=colors[:len(top_features)],
            line=dict(color='white', width=2)
        ),
        textinfo='label+percent',
        textposition='auto',
        hovertemplate='<b>%{label}</b><br>%{percent}<br>SHAP: %{value:.2f}%<extra></extra>',
        hole=0.4
    ),
    row=1, col=2
)

# Update layout
fig.update_xaxes(title_text="Mean |SHAP Value|", row=1, col=1, showgrid=True, gridcolor='lightgray')
fig.update_yaxes(title_text="Feature", row=1, col=1, showgrid=False)

# Overall layout
fig.update_layout(
    height=600,
    title=dict(
        text='<b>Real SHAP Analysis - Feature Importance for SPY Prediction</b>',
        font=dict(size=20, color='#1a202c'),
        x=0.5,
        xanchor='center'
    ),
    showlegend=False,
    template='plotly_white',
    font=dict(family='Arial, sans-serif', size=11),
    margin=dict(t=100, b=50, l=50, r=50)
)

# Save interactive HTML
html_output = os.path.join(parent_dir, 'docs', 'real_shap_analysis.html')
fig.write_html(html_output)
print(f"✓ Interactive chart saved: {html_output}")

# Open in browser
import webbrowser
webbrowser.open('file://' + html_output)
print(f"✓ Opening in browser...")

print("\n" + "=" * 60)
print("✅ REAL SHAP ANALYSIS COMPLETE")
print("=" * 60)
print(f"\nGenerated files:")
print(f"  1. {output_path}")
print(f"  2. {html_output}")