"""
Generate beautiful model comparison chart for presentation
Compares Baseline, Random Forest, and XGBoost models
"""
import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

print("=" * 60)
print("MODEL COMPARISON CHART")
print("=" * 60)

# Model performance data from training
models_data = {
    'Model': ['Baseline\n(Previous Open)', 'Random Forest', 'XGBoost\n(Tuned)', 'XGBoost\n(Initial) ⭐'],
    'MAE': [4.596, 4.333, 3.869, 3.714],
    'RMSE': [6.223, 6.147, 5.356, 5.230],
    'MAPE': [1.745, 1.631, 1.460, 1.403],
    'R2': [0.000, 0.9395, 0.9541, 0.9562],
    'Type': ['Baseline', 'Ensemble', 'Gradient Boosting', 'Gradient Boosting'],
    'Color': ['#ef4444', '#f97316', '#3b82f6', '#10b981']  # Red (worst), Orange, Blue, Green (best)
}

df = pd.DataFrame(models_data)

# Calculate improvement over baseline
baseline_mae = df[df['Model'] == 'Baseline\n(Previous Open)']['MAE'].values[0]
df['Improvement'] = ((baseline_mae - df['MAE']) / baseline_mae * 100)

print("\nModel Performance Summary:")
for _, row in df.iterrows():
    print(f"  {row['Model'].strip()}: MAE=${row['MAE']:.2f}, R²={row['R2']:.4f}")

# ============================================================================
# CREATE COMPREHENSIVE COMPARISON CHART (2x2 Layout)
# ============================================================================
print("\n" + "=" * 60)
print("GENERATING COMPARISON VISUALIZATION")
print("=" * 60)

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        '<b>Mean Absolute Error (Lower is Better)</b>',
        '<b>R² Score (Higher is Better)</b>',
        '<b>Error Metrics Comparison</b>',
        '<b>Improvement Over Baseline</b>'
    ),
    specs=[
        [{'type': 'bar'}, {'type': 'bar'}],
        [{'type': 'bar'}, {'type': 'bar'}]
    ],
    vertical_spacing=0.25,
    horizontal_spacing=0.15
)

# Chart 1: MAE Comparison (Primary Metric)
fig.add_trace(
    go.Bar(
        x=df['Model'],
        y=df['MAE'],
        marker=dict(
            color=df['Color'],
            line=dict(color='white', width=2)
        ),
        text=[f"${val:.2f}" for val in df['MAE']],
        textposition='outside',
        textfont=dict(size=12, color='#1a202c', family='Arial Black'),
        hovertemplate='<b>%{x}</b><br>MAE: $%{y:.2f}<br>Lower is better<extra></extra>',
        showlegend=False,
        width=0.5
    ),
    row=1, col=1
)

# Add "WINNER" annotation
best_mae_idx = df['MAE'].idxmin()
fig.add_annotation(
    x=df.loc[best_mae_idx, 'Model'],
    y=df.loc[best_mae_idx, 'MAE'] + 0.4,
    text="🏆 WINNER",
    showarrow=False,
    font=dict(size=11, color='#10b981', family='Arial Black'),
    row=1, col=1
)

# Chart 2: R² Score Comparison
fig.add_trace(
    go.Bar(
        x=df['Model'],
        y=df['R2'],
        marker=dict(
            color=df['Color'],
            line=dict(color='white', width=2)
        ),
        text=[f"{val:.3f}" if val > 0 else "N/A" for val in df['R2']],
        textposition='outside',
        textfont=dict(size=12, color='#1a202c', family='Arial Black'),
        hovertemplate='<b>%{x}</b><br>R²: %{y:.4f}<br>Variance explained: %{y:.1%}<extra></extra>',
        showlegend=False,
        width=0.5
    ),
    row=1, col=2
)

# Chart 3: Grouped Bar - All Error Metrics
metrics = ['MAE', 'RMSE', 'MAPE']
colors_metrics = ['#10b981', '#eab308', '#f97316']  # Green, Gold, Orange

for i, metric in enumerate(metrics):
    fig.add_trace(
        go.Bar(
            x=df['Model'],
            y=df[metric],
            name=metric,
            marker=dict(color=colors_metrics[i]),
            hovertemplate=f'<b>%{{x}}</b><br>{metric}: %{{y:.2f}}<extra></extra>',
            legendgroup='metrics',
            width=0.5
        ),
        row=2, col=1
    )

# Chart 4: Improvement Over Baseline
fig.add_trace(
    go.Bar(
        x=df['Model'][1:],  # Exclude baseline from improvement chart
        y=df['Improvement'][1:],
        marker=dict(
            color=['#f97316', '#3b82f6', '#10b981'],  # Orange, Blue, Green
            line=dict(color='white', width=2),
            pattern=dict(shape=['/', '\\', 'x'])
        ),
        text=[f"+{val:.1f}%" for val in df['Improvement'][1:]],
        textposition='outside',
        textfont=dict(size=12, color='#1a202c', family='Arial Black'),
        hovertemplate='<b>%{x}</b><br>Improvement: %{y:.1f}%<br>vs Baseline<extra></extra>',
        showlegend=False,
        width=0.5
    ),
    row=2, col=2
)

# Add reference line at 0% improvement
fig.add_hline(y=0, line_dash='dash', line_color='#9ca3af', opacity=0.5, row=2, col=2)

# Update axes
fig.update_xaxes(title_text="Model", row=1, col=1, tickangle=-45)
fig.update_yaxes(title_text="MAE ($)", row=1, col=1, showgrid=True, gridcolor='lightgray')

fig.update_xaxes(title_text="Model", row=1, col=2, tickangle=-45)
fig.update_yaxes(title_text="R² Score", row=1, col=2, showgrid=True, gridcolor='lightgray', range=[0, 1])

fig.update_xaxes(title_text="Model", row=2, col=1, tickangle=-45)
fig.update_yaxes(title_text="Error Value", row=2, col=1, showgrid=True, gridcolor='lightgray')

fig.update_xaxes(title_text="Model", row=2, col=2, tickangle=-45)
fig.update_yaxes(title_text="Improvement (%)", row=2, col=2, showgrid=True, gridcolor='lightgray')

# Overall layout
fig.update_layout(
    height=950,
    title=dict(
        text='<b>Model Performance Comparison: SPY Price Prediction</b>',
        font=dict(size=22, color='#1a202c', family='Arial Black'),
        x=0.5,
        xanchor='center'
    ),
    template='plotly_white',
    font=dict(family='Arial, sans-serif', size=12),
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=-0.15,
        xanchor='center',
        x=0.25,
        title_text='Error Metrics'
    ),
    margin=dict(t=100, b=100, l=80, r=80),
    showlegend=True
)

# Save
output_path = os.path.join(parent_dir, 'docs', 'model_comparison.html')
fig.write_html(output_path)
print(f"✓ Saved: {output_path}")

# ============================================================================
# CREATE PERFORMANCE METRICS TABLE
# ============================================================================
print("\n" + "=" * 60)
print("GENERATING DETAILED METRICS TABLE")
print("=" * 60)

# Create a detailed table visualization
fig_table = go.Figure(data=[go.Table(
    columnwidth=[150, 80, 80, 80, 80, 100],
    header=dict(
        values=['<b>Model</b>', '<b>MAE ($)</b>', '<b>RMSE ($)</b>', '<b>MAPE (%)</b>', '<b>R²</b>', '<b>vs Baseline</b>'],
        fill_color='#10b981',
        align='center',
        font=dict(color='white', size=14, family='Arial Black'),
        height=40
    ),
    cells=dict(
        values=[
            df['Model'],
            [f"${val:.2f}" for val in df['MAE']],
            [f"${val:.2f}" for val in df['RMSE']],
            [f"{val:.2f}%" for val in df['MAPE']],
            [f"{val:.4f}" if val > 0 else "N/A" for val in df['R2']],
            ["—"] + [f"+{val:.1f}%" for val in df['Improvement'][1:]]
        ],
        fill_color=[
            ['#f8fafc', '#f1f5f9', '#e2e8f0', '#d1fae5'],  # Row colors (best model is light green)
        ],
        align='center',
        font=dict(color='#1a202c', size=13),
        height=35
    )
)])

fig_table.update_layout(
    title=dict(
        text='<b>Detailed Performance Metrics</b>',
        font=dict(size=20, color='#1a202c', family='Arial Black'),
        x=0.5,
        xanchor='center'
    ),
    height=400,
    margin=dict(t=80, b=20, l=20, r=20)
)

output_table = os.path.join(parent_dir, 'docs', 'model_comparison_table.html')
fig_table.write_html(output_table)
print(f"✓ Saved: {output_table}")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n" + "=" * 60)
print("KEY INSIGHTS")
print("=" * 60)

best_model = df.loc[df['MAE'].idxmin()]
print(f"\n🏆 BEST MODEL: {best_model['Model'].strip()}")
print(f"   • MAE: ${best_model['MAE']:.2f}")
print(f"   • R²: {best_model['R2']:.4f} ({best_model['R2']*100:.2f}% variance explained)")
print(f"   • Improvement vs Baseline: {best_model['Improvement']:.1f}%")

print(f"\n📊 PERFORMANCE RANKING (by MAE):")
for rank, (_, row) in enumerate(df.sort_values('MAE').iterrows(), 1):
    print(f"   {rank}. {row['Model'].strip()}: ${row['MAE']:.2f}")

print("\n" + "=" * 60)
print("✅ COMPARISON CHARTS GENERATED")
print("=" * 60)
print(f"\nGenerated files:")
print(f"  1. {output_path}")
print(f"  2. {output_table}")

# Open in browser
print(f"\n✓ Opening charts in browser...")
webbrowser.open('file://' + output_path)
