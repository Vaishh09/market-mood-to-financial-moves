"""
Enhanced Visualization Module for Prisca
Interactive and professional charts for stock price prediction dashboard
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class PriscaVisualizations:
    """Professional visualization suite for stock prediction dashboard"""
    
    def __init__(self, theme='dark'):
        """
        Initialize visualization suite
        
        Args:
            theme: 'dark' or 'light' color theme
        """
        self.theme = theme
        self.colors = self._get_theme_colors()
        
    def _get_theme_colors(self):
        """Define color schemes for themes"""
        if self.theme == 'dark':
            return {
                'background': '#0e1117',
                'paper': '#1a1d24',
                'text': '#ffffff',
                'grid': '#2d3139',
                'positive': '#26a69a',
                'negative': '#ef5350',
                'neutral': '#78909c',
                'accent': '#7c4dff',
                'warning': '#ffa726'
            }
        else:
            return {
                'background': '#ffffff',
                'paper': '#f8f9fa',
                'text': '#212529',
                'grid': '#dee2e6',
                'positive': '#28a745',
                'negative': '#dc3545',
                'neutral': '#6c757d',
                'accent': '#6f42c1',
                'warning': '#fd7e14'
            }
    
    def price_trend_interactive(self, df, predictions=None):
        """
        Interactive price trend chart with volume and predictions
        
        Args:
            df: DataFrame with OHLCV data
            predictions: Optional dict with prediction data
        """
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=('SPY Price & Predictions', 'Trading Volume')
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open_SPY'],
                high=df['High_SPY'],
                low=df['Low_SPY'],
                close=df['Close_SPY'],
                name='Price',
                increasing_line_color=self.colors['positive'],
                decreasing_line_color=self.colors['negative']
            ),
            row=1, col=1
        )
        
        # Add predictions if provided
        if predictions:
            fig.add_trace(
                go.Scatter(
                    x=predictions['dates'],
                    y=predictions['values'],
                    mode='markers+lines',
                    name='Predicted Opening',
                    line=dict(color=self.colors['accent'], width=2, dash='dash'),
                    marker=dict(size=8, symbol='diamond')
                ),
                row=1, col=1
            )
        
        # Volume bars
        colors = [self.colors['positive'] if df['Close_SPY'].iloc[i] >= df['Open_SPY'].iloc[i] 
                  else self.colors['negative'] for i in range(len(df))]
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['Volume_SPY'],
                name='Volume',
                marker_color=colors,
                opacity=0.5
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            template='plotly_dark' if self.theme == 'dark' else 'plotly_white',
            paper_bgcolor=self.colors['paper'],
            plot_bgcolor=self.colors['background'],
            font=dict(color=self.colors['text'], size=12),
            height=700,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode='x unified',
            xaxis_rangeslider_visible=False
        )
        
        fig.update_xaxes(showgrid=True, gridcolor=self.colors['grid'])
        fig.update_yaxes(showgrid=True, gridcolor=self.colors['grid'])
        
        return fig
    
    def sentiment_impact_chart(self, df):
        """
        Dual-axis chart showing sentiment scores and price movement
        
        Args:
            df: DataFrame with sentiment and price columns
        """
        fig = make_subplots(
            specs=[[{"secondary_y": True}]]
        )
        
        # Price line
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Close_SPY'],
                name='SPY Price',
                line=dict(color=self.colors['accent'], width=2),
                fill='tonexty',
                fillcolor=f'rgba(124, 77, 255, 0.1)'
            ),
            secondary_y=False
        )
        
        # Sentiment scores
        if 'vader_compound' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['vader_compound'].rolling(10).mean(),
                    name='VADER Sentiment (10-day MA)',
                    line=dict(color=self.colors['positive'], width=2)
                ),
                secondary_y=True
            )
        
        if 'finbert_positive' in df.columns and 'finbert_negative' in df.columns:
            finbert_signal = df['finbert_positive'] - df['finbert_negative']
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=finbert_signal.rolling(10).mean(),
                    name='FinBERT Sentiment (10-day MA)',
                    line=dict(color=self.colors['warning'], width=2, dash='dot')
                ),
                secondary_y=True
            )
        
        # Add zero line for sentiment
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color=self.colors['neutral'],
            opacity=0.5,
            secondary_y=True
        )
        
        # Update layout
        fig.update_layout(
            template='plotly_dark' if self.theme == 'dark' else 'plotly_white',
            paper_bgcolor=self.colors['paper'],
            plot_bgcolor=self.colors['background'],
            font=dict(color=self.colors['text']),
            height=500,
            title='News Sentiment vs Price Movement',
            hovermode='x unified'
        )
        
        fig.update_xaxes(showgrid=True, gridcolor=self.colors['grid'])
        fig.update_yaxes(title_text="SPY Price ($)", secondary_y=False, showgrid=True, gridcolor=self.colors['grid'])
        fig.update_yaxes(title_text="Sentiment Score", secondary_y=True, showgrid=False)
        
        return fig
    
    def volatility_heatmap(self, df, window=20):
        """
        Calendar heatmap showing daily volatility patterns
        
        Args:
            df: DataFrame with return data
            window: Rolling window for volatility calculation
        """
        df_copy = df.copy()
        df_copy['Volatility'] = df_copy['Return'].rolling(window).std() * 100
        df_copy['Year'] = df_copy.index.year
        df_copy['Month'] = df_copy.index.month
        df_copy['Day'] = df_copy.index.day
        
        # Aggregate by month
        monthly_vol = df_copy.groupby(['Year', 'Month'])['Volatility'].mean().reset_index()
        monthly_vol['Date'] = pd.to_datetime(monthly_vol[['Year', 'Month']].assign(Day=1))
        
        fig = go.Figure(data=go.Scatter(
            x=monthly_vol['Date'],
            y=monthly_vol['Volatility'],
            mode='markers',
            marker=dict(
                size=monthly_vol['Volatility'] * 100,
                color=monthly_vol['Volatility'],
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Volatility %")
            ),
            text=[f"Volatility: {v:.2f}%" for v in monthly_vol['Volatility']],
            hovertemplate='%{text}<br>Date: %{x}<extra></extra>'
        ))
        
        fig.update_layout(
            template='plotly_dark' if self.theme == 'dark' else 'plotly_white',
            paper_bgcolor=self.colors['paper'],
            plot_bgcolor=self.colors['background'],
            font=dict(color=self.colors['text']),
            title='Market Volatility Over Time',
            xaxis_title='Date',
            yaxis_title='Average Volatility (%)',
            height=400
        )
        
        return fig
    
    def feature_importance_chart(self, feature_names, importance_scores):
        """
        Horizontal bar chart for model feature importance
        
        Args:
            feature_names: List of feature names
            importance_scores: List of importance scores
        """
        # Sort by importance
        sorted_idx = np.argsort(importance_scores)[-20:]  # Top 20
        
        fig = go.Figure(go.Bar(
            x=np.array(importance_scores)[sorted_idx],
            y=np.array(feature_names)[sorted_idx],
            orientation='h',
            marker=dict(
                color=np.array(importance_scores)[sorted_idx],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Importance")
            ),
            text=[f"{x:.3f}" for x in np.array(importance_scores)[sorted_idx]],
            textposition='outside'
        ))
        
        fig.update_layout(
            template='plotly_dark' if self.theme == 'dark' else 'plotly_white',
            paper_bgcolor=self.colors['paper'],
            plot_bgcolor=self.colors['background'],
            font=dict(color=self.colors['text']),
            title='Top 20 Most Important Features',
            xaxis_title='Importance Score',
            yaxis_title='Features',
            height=600,
            showlegend=False
        )
        
        return fig
    
    def prediction_confidence_gauge(self, confidence_score, prediction_change):
        """
        Gauge chart showing prediction confidence
        
        Args:
            confidence_score: Float between 0 and 1
            prediction_change: Predicted price change percentage
        """
        # Determine color based on prediction
        if prediction_change > 0:
            gauge_color = self.colors['positive']
            direction = "UP"
        elif prediction_change < 0:
            gauge_color = self.colors['negative']
            direction = "DOWN"
        else:
            gauge_color = self.colors['neutral']
            direction = "FLAT"
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=confidence_score * 100,
            title={'text': f"Prediction Confidence<br><span style='font-size:0.8em'>Expected: {direction} {abs(prediction_change):.2f}%</span>"},
            delta={'reference': 75},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': gauge_color},
                'steps': [
                    {'range': [0, 50], 'color': self.colors['background']},
                    {'range': [50, 75], 'color': self.colors['neutral']},
                    {'range': [75, 100], 'color': self.colors['paper']}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': 85
                }
            }
        ))
        
        fig.update_layout(
            paper_bgcolor=self.colors['paper'],
            font=dict(color=self.colors['text'], size=16),
            height=300
        )
        
        return fig
    
    def correlation_network(self, corr_matrix, threshold=0.5):
        """
        Interactive network graph showing feature correlations
        
        Args:
            corr_matrix: Correlation matrix DataFrame
            threshold: Minimum correlation to display
        """
        import networkx as nx
        
        # Create network
        G = nx.Graph()
        
        # Add edges for correlations above threshold
        for i in range(len(corr_matrix)):
            for j in range(i+1, len(corr_matrix)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) > threshold:
                    G.add_edge(
                        corr_matrix.index[i],
                        corr_matrix.columns[j],
                        weight=abs(corr),
                        correlation=corr
                    )
        
        # Position nodes
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        
        # Create edge traces
        edge_traces = []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            corr = edge[2]['correlation']
            
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(
                    width=edge[2]['weight'] * 3,
                    color=self.colors['positive'] if corr > 0 else self.colors['negative']
                ),
                opacity=0.5,
                hoverinfo='none'
            )
            edge_traces.append(edge_trace)
        
        # Create node trace
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            text=list(G.nodes()),
            textposition='top center',
            marker=dict(
                size=20,
                color=self.colors['accent'],
                line=dict(color=self.colors['text'], width=2)
            ),
            textfont=dict(color=self.colors['text'], size=10)
        )
        
        # Create figure
        fig = go.Figure(data=edge_traces + [node_trace])
        
        fig.update_layout(
            template='plotly_dark' if self.theme == 'dark' else 'plotly_white',
            paper_bgcolor=self.colors['paper'],
            plot_bgcolor=self.colors['background'],
            title=f'Feature Correlation Network (|r| > {threshold})',
            showlegend=False,
            hovermode='closest',
            height=600,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        return fig
    
    def real_time_prediction_card(self, current_price, predicted_price, prediction_time):
        """
        Create prediction summary card data
        
        Args:
            current_price: Current SPY price
            predicted_price: Predicted next-day opening
            prediction_time: Timestamp of prediction
        """
        change = predicted_price - current_price
        change_pct = (change / current_price) * 100
        
        return {
            'current_price': current_price,
            'predicted_price': predicted_price,
            'change': change,
            'change_pct': change_pct,
            'prediction_time': prediction_time,
            'direction': 'up' if change > 0 else 'down' if change < 0 else 'flat',
            'color': self.colors['positive'] if change > 0 else self.colors['negative'] if change < 0 else self.colors['neutral']
        }


# Example usage
if __name__ == "__main__":
    print("Prisca Enhanced Visualizations Module")
    print("=" * 50)
    print("\nAvailable visualizations:")
    print("  - price_trend_interactive(): Candlestick chart with predictions")
    print("  - sentiment_impact_chart(): Sentiment vs price dual-axis")
    print("  - volatility_heatmap(): Calendar volatility visualization")
    print("  - feature_importance_chart(): ML feature rankings")
    print("  - prediction_confidence_gauge(): Confidence meter")
    print("  - correlation_network(): Interactive feature relationships")
    print("  - real_time_prediction_card(): Prediction summary data")
