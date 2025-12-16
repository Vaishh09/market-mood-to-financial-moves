"""
Generate presentation-ready charts from exploratory data analysis
Creates clean, professional visualizations for PRISCA presentation
"""
import os
import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

print("=" * 60)
print("PRISCA EDA PRESENTATION CHARTS")
print("=" * 60)

# Load the dataset
data_path = os.path.join(parent_dir, 'data', 'final_modeling_dataset.csv')
print(f"Loading data from: {data_path}")
df = pd.read_csv(data_path, index_col=0, parse_dates=True)
df = df.sort_index()
print(f"✓ Data loaded: {df.shape}")

# Calculate returns if not present
if 'Return' not in df.columns:
    df['Return'] = df['Close_SPY'].pct_change() * 100

# Calculate 20-day rolling volatility
df['Volatility_20'] = df['Return'].rolling(20).std()

print("\n" + "=" * 60)
print("GENERATING VISUALIZATIONS")
print("=" * 60)

# Color scheme - professional purple/blue
colors = {
    'primary': '#667eea',
    'secondary': '#764ba2',
    'accent': '#10b981',
    'danger': '#ef4444',
    'warning': '#f59e0b'
}

# ============================================================================
# CHART 1: SPY Opening Price Trend with COVID Crash Highlighted
# ============================================================================
print("\n1. Creating SPY Opening Price Trend chart...")

fig1 = go.Figure()

# Main price line
fig1.add_trace(go.Scatter(
    x=df.index,
    y=df['Open_SPY'],
    mode='lines',
    name='SPY Opening Price',
    line=dict(color=colors['primary'], width=2),
    hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Open: $%{y:.2f}<extra></extra>'
))

# Highlight COVID crash period (March-April 2020)
covid_period = df['2020-03':'2020-04']
fig1.add_trace(go.Scatter(
    x=covid_period.index,
    y=covid_period['Open_SPY'],
    mode='lines',
    name='COVID-19 Crash',
    line=dict(color=colors['danger'], width=3),
    hovertemplate='<b>COVID Period</b><br>%{x|%Y-%m-%d}<br>Open: $%{y:.2f}<extra></extra>'
))

fig1.update_layout(
    title=dict(
        text='<b>SPY Opening Price: 2018-2020 with COVID-19 Crash</b>',
        font=dict(size=20, color='#1a202c'),
        x=0.5,
        xanchor='center'
    ),
    xaxis=dict(
        title='Date',
        showgrid=True,
        gridcolor='lightgray'
    ),
    yaxis=dict(
        title='Opening Price ($)',
        showgrid=True,
        gridcolor='lightgray'
    ),
    hovermode='x unified',
    template='plotly_white',
    height=500,
    font=dict(family='Arial, sans-serif', size=12),
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='center',
        x=0.5
    ),
    margin=dict(t=80, b=50, l=60, r=50)
)

# Add annotation for key insights
fig1.add_annotation(
    x=df.index.min(),
    y=df['Open_SPY'].max(),
    text=f"Growth: ${df['Open_SPY'].iloc[0]:.0f} → ${df['Open_SPY'].max():.0f} (+{((df['Open_SPY'].max()/df['Open_SPY'].iloc[0])-1)*100:.1f}%)",
    showarrow=False,
    font=dict(size=11, color=colors['secondary']),
    align='left',
    xanchor='left',
    yanchor='top'
)

output1 = os.path.join(parent_dir, 'docs', 'eda_price_trend.html')
fig1.write_html(output1)
print(f"✓ Saved: {output1}")

# ============================================================================
# CHART 2: Volatility Over Time with COVID Spike
# ============================================================================
print("2. Creating Volatility Over Time chart...")

fig2 = go.Figure()

# Volatility line
fig2.add_trace(go.Scatter(
    x=df.index,
    y=df['Volatility_20'],
    mode='lines',
    name='20-Day Volatility',
    line=dict(color=colors['warning'], width=2),
    fill='tozeroy',
    fillcolor='rgba(245, 158, 11, 0.2)',
    hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Volatility: %{y:.2f}%<extra></extra>'
))

# Highlight extreme volatility period
extreme_vol = df[df['Volatility_20'] > df['Volatility_20'].quantile(0.95)]
fig2.add_trace(go.Scatter(
    x=extreme_vol.index,
    y=extreme_vol['Volatility_20'],
    mode='markers',
    name='Extreme Volatility',
    marker=dict(color=colors['danger'], size=8, symbol='diamond'),
    hovertemplate='<b>High Risk Period</b><br>%{x|%Y-%m-%d}<br>Volatility: %{y:.2f}%<extra></extra>'
))

fig2.update_layout(
    title=dict(
        text='<b>Market Volatility: Stability vs Crisis Periods</b>',
        font=dict(size=20, color='#1a202c'),
        x=0.5,
        xanchor='center'
    ),
    xaxis=dict(
        title='Date',
        showgrid=True,
        gridcolor='lightgray'
    ),
    yaxis=dict(
        title='20-Day Rolling Volatility (Std Dev %)',
        showgrid=True,
        gridcolor='lightgray'
    ),
    hovermode='x unified',
    template='plotly_white',
    height=500,
    font=dict(family='Arial, sans-serif', size=12),
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='center',
        x=0.5
    ),
    margin=dict(t=80, b=50, l=60, r=50)
)

# Add annotation
normal_vol = df[df.index < '2020-02-01']['Volatility_20'].mean()
peak_vol = df['Volatility_20'].max()
fig2.add_annotation(
    x=df['Volatility_20'].idxmax(),
    y=peak_vol,
    text=f"Peak: {peak_vol:.2f}% (COVID)<br>{peak_vol/normal_vol:.1f}x normal",
    showarrow=True,
    arrowhead=2,
    arrowcolor=colors['danger'],
    font=dict(size=11, color=colors['danger']),
    align='left'
)

output2 = os.path.join(parent_dir, 'docs', 'eda_volatility.html')
fig2.write_html(output2)
print(f"✓ Saved: {output2}")

# ============================================================================
# CHART 3: Extreme Events - Largest Gains and Losses
# ============================================================================
print("3. Creating Extreme Events chart...")

# Identify extreme events
worst_10 = df.nsmallest(10, 'Return')
best_10 = df.nlargest(10, 'Return')

fig3 = go.Figure()

# All returns (background)
fig3.add_trace(go.Scatter(
    x=df.index,
    y=df['Return'],
    mode='markers',
    name='Daily Returns',
    marker=dict(
        color=df['Return'],
        colorscale='RdYlGn',
        size=4,
        opacity=0.6,
        colorbar=dict(title='Return %', x=1.15)
    ),
    hovertemplate='%{x|%Y-%m-%d}<br>Return: %{y:.2f}%<extra></extra>'
))

# Largest drops (red)
fig3.add_trace(go.Scatter(
    x=worst_10.index,
    y=worst_10['Return'],
    mode='markers',
    name='10 Largest Drops',
    marker=dict(
        color=colors['danger'],
        size=12,
        symbol='triangle-down',
        line=dict(width=2, color='white')
    ),
    hovertemplate='<b>Major Drop</b><br>%{x|%Y-%m-%d}<br>Return: %{y:.2f}%<extra></extra>'
))

# Largest gains (green)
fig3.add_trace(go.Scatter(
    x=best_10.index,
    y=best_10['Return'],
    mode='markers',
    name='10 Largest Gains',
    marker=dict(
        color=colors['accent'],
        size=12,
        symbol='triangle-up',
        line=dict(width=2, color='white')
    ),
    hovertemplate='<b>Major Gain</b><br>%{x|%Y-%m-%d}<br>Return: %{y:.2f}%<extra></extra>'
))

# Zero line
fig3.add_hline(y=0, line_dash='dash', line_color='black', opacity=0.3)

fig3.update_layout(
    title=dict(
        text='<b>Extreme Market Events: Fat-Tailed Distribution</b>',
        font=dict(size=20, color='#1a202c'),
        x=0.5,
        xanchor='center'
    ),
    xaxis=dict(
        title='Date',
        showgrid=True,
        gridcolor='lightgray'
    ),
    yaxis=dict(
        title='Daily Return (%)',
        showgrid=True,
        gridcolor='lightgray'
    ),
    hovermode='closest',
    template='plotly_white',
    height=550,
    font=dict(family='Arial, sans-serif', size=12),
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='center',
        x=0.5
    ),
    margin=dict(t=80, b=50, l=60, r=50)
)

output3 = os.path.join(parent_dir, 'docs', 'eda_extreme_events.html')
fig3.write_html(output3)
print(f"✓ Saved: {output3}")

# ============================================================================
# CHART 4: Daily Returns Distribution (Bell Curve with Fat Tails)
# ============================================================================
print("4. Creating Returns Distribution chart...")

fig4 = go.Figure()

# Histogram
fig4.add_trace(go.Histogram(
    x=df['Return'].dropna(),
    nbinsx=50,
    name='Daily Returns',
    marker=dict(
        color=colors['primary'],
        line=dict(color='white', width=1)
    ),
    hovertemplate='Return: %{x:.2f}%<br>Count: %{y}<extra></extra>',
    opacity=0.8
))

# Add normal distribution overlay
returns_clean = df['Return'].dropna()
mean_return = returns_clean.mean()
std_return = returns_clean.std()
x_range = np.linspace(returns_clean.min(), returns_clean.max(), 100)
normal_dist = (1/(std_return * np.sqrt(2*np.pi))) * np.exp(-0.5*((x_range-mean_return)/std_return)**2)
# Scale to match histogram
normal_dist_scaled = normal_dist * len(returns_clean) * (returns_clean.max() - returns_clean.min()) / 50

fig4.add_trace(go.Scatter(
    x=x_range,
    y=normal_dist_scaled,
    mode='lines',
    name='Normal Distribution',
    line=dict(color=colors['danger'], width=3, dash='dash'),
    hovertemplate='%{x:.2f}%<extra></extra>'
))

fig4.update_layout(
    title=dict(
        text='<b>Daily Returns Distribution: Bell Curve with Fat Tails</b>',
        font=dict(size=20, color='#1a202c'),
        x=0.5,
        xanchor='center'
    ),
    xaxis=dict(
        title='Daily Return (%)',
        showgrid=True,
        gridcolor='lightgray'
    ),
    yaxis=dict(
        title='Frequency',
        showgrid=True,
        gridcolor='lightgray'
    ),
    template='plotly_white',
    height=500,
    font=dict(family='Arial, sans-serif', size=12),
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='center',
        x=0.5
    ),
    margin=dict(t=80, b=50, l=60, r=50),
    bargap=0.1
)

# Add statistics annotation
fig4.add_annotation(
    x=returns_clean.max() * 0.7,
    y=max(normal_dist_scaled) * 0.8,
    text=f"<b>Statistics:</b><br>Mean: {mean_return:.2f}%<br>Std Dev: {std_return:.2f}%<br>Min: {returns_clean.min():.2f}%<br>Max: {returns_clean.max():.2f}%",
    showarrow=False,
    font=dict(size=11, color='#1a202c'),
    align='left',
    bgcolor='rgba(255,255,255,0.8)',
    bordercolor=colors['secondary'],
    borderwidth=2,
    borderpad=10
)

output4 = os.path.join(parent_dir, 'docs', 'eda_returns_distribution.html')
fig4.write_html(output4)
print(f"✓ Saved: {output4}")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n" + "=" * 60)
print("KEY FINDINGS SUMMARY")
print("=" * 60)

print("\n📈 PRICE TRENDS:")
print(f"  • Starting Price: ${df['Open_SPY'].iloc[0]:.2f}")
print(f"  • Peak Price: ${df['Open_SPY'].max():.2f}")
print(f"  • Growth: {((df['Open_SPY'].max()/df['Open_SPY'].iloc[0])-1)*100:.1f}%")

print("\n📊 VOLATILITY:")
normal_vol_period = df[df.index < '2020-02-01']['Volatility_20'].mean()
print(f"  • Normal Volatility: {normal_vol_period:.2f}%")
print(f"  • Peak Volatility (COVID): {df['Volatility_20'].max():.2f}%")
print(f"  • Spike Magnitude: {df['Volatility_20'].max()/normal_vol_period:.1f}x normal")

print("\n⚠️ EXTREME EVENTS:")
print(f"  • Worst Single Day: {df['Return'].min():.2f}% ({df['Return'].idxmin().strftime('%Y-%m-%d')})")
print(f"  • Best Single Day: {df['Return'].max():.2f}% ({df['Return'].idxmax().strftime('%Y-%m-%d')})")
print(f"  • Days with |Return| > 5%: {len(df[abs(df['Return']) > 5])}")

print("\n🔔 DISTRIBUTION:")
print(f"  • Mean Return: {returns_clean.mean():.2f}%")
print(f"  • Std Deviation: {returns_clean.std():.2f}%")
print(f"  • Skewness: {returns_clean.skew():.2f}")
print(f"  • Kurtosis: {returns_clean.kurtosis():.2f} (excess kurtosis = fat tails)")

print("\n" + "=" * 60)
print("✅ ALL CHARTS GENERATED")
print("=" * 60)
print(f"\nGenerated files:")
print(f"  1. {output1}")
print(f"  2. {output2}")
print(f"  3. {output3}")
print(f"  4. {output4}")

# Open first chart in browser
print(f"\n✓ Opening charts in browser...")
webbrowser.open('file://' + output1)
