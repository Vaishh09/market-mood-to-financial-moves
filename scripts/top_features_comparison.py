"""
Top 3 Features Comparison Chart
Visualizes the most important features for SPY prediction
"""

import joblib
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def create_top_features_chart():
    """Create an interactive comparison chart for top 3 features"""
    
    # Load the trained model
    model = joblib.load('../models/prisca_xgb_model.pkl')
    
    # Load feature names
    with open('../models/feature_list.json', 'r') as f:
        feature_names = json.load(f)
    
    # Get feature importances from XGBoost
    importances = model.feature_importances_
    
    # Create a list of (feature_name, importance) tuples
    feature_importance_pairs = list(zip(feature_names, importances))
    
    # Sort by importance (descending)
    feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
    
    # Get top 3 features
    top_3 = feature_importance_pairs[:3]
    
    # Extract names and values
    top_names = [f[0] for f in top_3]
    top_values = [f[1] for f in top_3]
    
    # Calculate percentages
    total_importance = sum(importances)
    top_percentages = [(v / total_importance) * 100 for v in top_values]
    
    print("\n" + "="*60)
    print("TOP 3 MOST IMPORTANT FEATURES FOR SPY PREDICTION")
    print("="*60)
    for i, (name, value, pct) in enumerate(zip(top_names, top_values, top_percentages), 1):
        print(f"{i}. {name}")
        print(f"   Importance Score: {value:.4f}")
        print(f"   Contribution: {pct:.2f}% of total importance")
        print()
    
    # Create figure with subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Feature Importance Scores',
            'Contribution to Total Importance',
            'Relative Comparison',
            'Normalized Impact (0-100)'
        ),
        specs=[
            [{"type": "bar"}, {"type": "pie"}],
            [{"type": "bar"}, {"type": "bar"}]
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )
    
    # Define colors for consistency
    colors = ['#667eea', '#764ba2', '#9333ea']
    
    # Chart 1: Horizontal Bar Chart (Absolute Importance)
    fig.add_trace(
        go.Bar(
            y=top_names[::-1],  # Reverse for better visual (highest at top)
            x=top_values[::-1],
            orientation='h',
            marker=dict(
                color=colors[::-1],
                line=dict(color='white', width=2)
            ),
            text=[f'{v:.4f}' for v in top_values[::-1]],
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Chart 2: Pie Chart (Percentage Contribution)
    fig.add_trace(
        go.Pie(
            labels=top_names,
            values=top_percentages,
            marker=dict(colors=colors),
            hole=0.4,
            textinfo='label+percent',
            textposition='outside',
            hovertemplate='<b>%{label}</b><br>%{percent}<br>Score: %{value:.2f}%<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Chart 3: Relative Comparison (Stacked Bar)
    baseline = top_values[0]  # Use highest as baseline (100%)
    relative_values = [(v / baseline) * 100 for v in top_values]
    
    fig.add_trace(
        go.Bar(
            x=top_names,
            y=relative_values,
            marker=dict(
                color=colors,
                line=dict(color='white', width=2)
            ),
            text=[f'{v:.1f}%' for v in relative_values],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Relative to Top: %{y:.1f}%<extra></extra>',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Chart 4: Normalized Impact (0-100 scale)
    normalized_values = [(v - min(top_values)) / (max(top_values) - min(top_values)) * 100 for v in top_values]
    
    fig.add_trace(
        go.Bar(
            x=top_names,
            y=normalized_values,
            marker=dict(
                color=colors,
                line=dict(color='white', width=2),
                pattern_shape=["/", "\\", "x"]  # Different patterns for distinction
            ),
            text=[f'{v:.1f}' for v in normalized_values],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Normalized Score: %{y:.1f}/100<extra></extra>',
            showlegend=False
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='<b>Top 3 Feature Importance Comparison</b><br><sub>SPY Next-Day Opening Price Prediction</sub>',
            x=0.5,
            xanchor='center',
            font=dict(size=24, color='#1f2937')
        ),
        height=900,
        showlegend=False,
        paper_bgcolor='white',
        plot_bgcolor='rgba(249, 250, 251, 0.8)',
        font=dict(family='Inter, sans-serif', size=12),
        margin=dict(t=120, b=60, l=80, r=80)
    )
    
    # Update axes
    fig.update_xaxes(title_text="Importance Score", row=1, col=1, showgrid=True, gridcolor='lightgray')
    fig.update_yaxes(row=1, col=1, showgrid=False)
    
    fig.update_xaxes(title_text="Feature", row=2, col=1, showgrid=False)
    fig.update_yaxes(title_text="Relative % (Top=100%)", row=2, col=1, showgrid=True, gridcolor='lightgray')
    
    fig.update_xaxes(title_text="Feature", row=2, col=2, showgrid=False)
    fig.update_yaxes(title_text="Normalized Score (0-100)", row=2, col=2, showgrid=True, gridcolor='lightgray')
    
    # Save as HTML
    output_file = '../docs/top_features_comparison.html'
    fig.write_html(output_file)
    print(f"✓ Chart saved to: {output_file}")
    
    # Also display in browser
    fig.show()
    
    return fig, top_3

if __name__ == "__main__":
    fig, top_features = create_top_features_chart()
