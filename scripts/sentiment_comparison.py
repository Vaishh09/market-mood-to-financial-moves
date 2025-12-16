"""
Generate focused chart showing VADER vs FinBERT predictive power
Correlation with returns shown via color-coding
"""
import os
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import webbrowser

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

print("=" * 60)
print("SENTIMENT PREDICTIVE POWER ANALYSIS")
print("=" * 60)

# Load data
data_path = os.path.join(parent_dir, 'data', 'final_modeling_dataset.csv')
df = pd.read_csv(data_path)

print(f"\nData loaded: {df.shape}")

# Calculate sentiment metrics
vader_compound = df['vader_compound'].values
finbert_sentiment = df['finbert_positive'] - df['finbert_negative']
returns = df['Next_Open_Return'].values

# Correlations
vader_return_corr = np.corrcoef(vader_compound, returns)[0, 1]
finbert_return_corr = np.corrcoef(finbert_sentiment, returns)[0, 1]
vader_finbert_corr = np.corrcoef(vader_compound, finbert_sentiment)[0, 1]

print(f"\n📊 Predictive Power (Correlation with Returns):")
print(f"  VADER: {vader_return_corr:.4f}")
print(f"  FinBERT: {finbert_return_corr:.4f}")
print(f"  VADER-FinBERT Agreement: {vader_finbert_corr:.4f}")

# ============================================================================
# CREATE PREDICTIVE POWER VISUALIZATION
# ============================================================================
print("\n" + "=" * 60)
print("GENERATING CHART")
print("=" * 60)

fig = go.Figure()

# Add scatter plot
fig.add_trace(
    go.Scatter(
        x=vader_compound,
        y=finbert_sentiment,
        mode='markers',
        name='Trading Days',
        marker=dict(
            color=returns * 100,  # Convert to percentage
            colorscale='RdYlGn',
            size=8,
            opacity=0.7,
            colorbar=dict(
                title=dict(
                    text='Next Day<br>Return (%)',
                    font=dict(size=14)
                ),
                thickness=20,
                len=0.7,
                x=1.12,
                tickformat='.1f'
            ),
            line=dict(width=0.5, color='white'),
            cmin=-8,
            cmax=8
        ),
        hovertemplate=(
            '<b>Date:</b> %{text}<br>' +
            '<b>VADER:</b> %{x:.3f}<br>' +
            '<b>FinBERT:</b> %{y:.3f}<br>' +
            '<b>Return:</b> %{marker.color:.2f}%<br>' +
            '<extra></extra>'
        ),
        text=df['Date']
    )
)

# Add trend line
z = np.polyfit(vader_compound, finbert_sentiment, 1)
p = np.poly1d(z)
x_trend = np.linspace(vader_compound.min(), vader_compound.max(), 100)
fig.add_trace(
    go.Scatter(
        x=x_trend,
        y=p(x_trend),
        mode='lines',
        name=f'Agreement Trend (r={vader_finbert_corr:.3f})',
        line=dict(color='#6366f1', width=3, dash='dash'),
        hoverinfo='skip'
    )
)

# Add quadrant lines
fig.add_hline(y=0, line_dash='dot', line_color='gray', opacity=0.4)
fig.add_vline(x=0, line_dash='dot', line_color='gray', opacity=0.4)

# Add quadrant labels
annotations = [
    dict(x=0.5, y=0.5, text='Both Positive', showarrow=False, 
         font=dict(size=12, color='gray'), opacity=0.5),
    dict(x=-0.5, y=-0.5, text='Both Negative', showarrow=False,
         font=dict(size=12, color='gray'), opacity=0.5),
    dict(x=-0.5, y=0.5, text='Disagreement', showarrow=False,
         font=dict(size=12, color='gray'), opacity=0.5),
    dict(x=0.5, y=-0.5, text='Disagreement', showarrow=False,
         font=dict(size=12, color='gray'), opacity=0.5)
]

for ann in annotations:
    fig.add_annotation(ann)

# Add correlation statistics box
stats_text = (
    f'<b>Predictive Power:</b><br>' +
    f'VADER → Return: r = {vader_return_corr:.4f}<br>' +
    f'FinBERT → Return: r = {finbert_return_corr:.4f}<br><br>' +
    f'<b>Winner:</b> {"FinBERT" if abs(finbert_return_corr) > abs(vader_return_corr) else "VADER"} ' +
    f'({max(abs(vader_return_corr), abs(finbert_return_corr)) / min(abs(vader_return_corr), abs(finbert_return_corr)):.2f}x stronger)'
)

fig.add_annotation(
    text=stats_text,
    xref='paper', yref='paper',
    x=0.02, y=0.98,
    showarrow=False,
    font=dict(size=13, color='#1a202c', family='Arial'),
    bgcolor='rgba(255, 255, 255, 0.95)',
    bordercolor='#3b82f6',
    borderwidth=3,
    borderpad=15,
    align='left',
    xanchor='left',
    yanchor='top'
)

# Layout
fig.update_layout(
    title=dict(
        text='<b>VADER vs FinBERT: Predictive Power Comparison</b><br><sub>Color indicates next-day return | Position shows sentiment agreement</sub>',
        font=dict(size=22, color='#1a202c', family='Arial Black'),
        x=0.5,
        xanchor='center',
        y=0.97
    ),
    xaxis=dict(
        title='<b>VADER Compound Score</b>',
        title_font=dict(size=16),
        showgrid=True,
        gridcolor='lightgray',
        zeroline=True,
        zerolinecolor='gray',
        zerolinewidth=2
    ),
    yaxis=dict(
        title='<b>FinBERT Net Sentiment</b><br>(Positive - Negative)',
        title_font=dict(size=16),
        showgrid=True,
        gridcolor='lightgray',
        zeroline=True,
        zerolinecolor='gray',
        zerolinewidth=2
    ),
    template='plotly_white',
    font=dict(family='Arial, sans-serif', size=12),
    showlegend=True,
    legend=dict(
        orientation='v',
        yanchor='top',
        y=0.35,
        xanchor='left',
        x=0.02,
        bgcolor='rgba(255, 255, 255, 0.9)',
        bordercolor='#cbd5e0',
        borderwidth=1
    ),
    height=700,
    margin=dict(t=120, b=80, l=80, r=150),
    hovermode='closest'
)

# Save
output_path = os.path.join(parent_dir, 'docs', 'sentiment_comparison.html')
fig.write_html(output_path)
print(f"✓ Saved: {output_path}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("KEY INSIGHTS")
print("=" * 60)

print(f"\n💰 PREDICTIVE POWER:")
print(f"  VADER → Return Correlation: {vader_return_corr:.4f}")
print(f"  FinBERT → Return Correlation: {finbert_return_corr:.4f}")
print(f"  Improvement: {abs(finbert_return_corr)/abs(vader_return_corr):.2f}x")

print(f"\n🤝 AGREEMENT:")
print(f"  VADER-FinBERT Correlation: {vader_finbert_corr:.4f} (weak agreement)")

# Analyze quadrants
both_positive = ((vader_compound > 0) & (finbert_sentiment > 0)).sum()
both_negative = ((vader_compound < 0) & (finbert_sentiment < 0)).sum()
total_agreement = both_positive + both_negative
agreement_pct = total_agreement / len(vader_compound) * 100

print(f"\n📊 SENTIMENT DISTRIBUTION:")
print(f"  Both Positive: {both_positive} days ({both_positive/len(vader_compound)*100:.1f}%)")
print(f"  Both Negative: {both_negative} days ({both_negative/len(vader_compound)*100:.1f}%)")
print(f"  Total Agreement: {agreement_pct:.1f}%")

print("\n" + "=" * 60)
print("✅ PREDICTIVE POWER CHART GENERATED")
print("=" * 60)

# Open in browser
print(f"\n✓ Opening chart in browser...")
webbrowser.open('file://' + output_path)
