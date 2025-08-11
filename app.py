"""
Interactive Portfolio Builder Streamlit App
A comprehensive tool for building, analyzing, and optimizing investment portfolios.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns
except ImportError as e:
    st.error(f"Failed to import PyPortfolioOpt: {e}")
    st.error("Please ensure PyPortfolioOpt is properly installed")
    st.stop()
from pypfopt.risk_models import CovarianceShrinkage


import json
import os
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Interactive Portfolio Builder",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .chart-container {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Available tickers - Tech-focused with famous ETFs and crypto
AVAILABLE_TICKERS = [
    # Major Tech Stocks
    'AAPL', 'MSFT', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD', 'CRM', 'NFLX',
    
    # Famous ETFs
    'SPY', 'GLD', 'TLT', 'VNQ',
    
    # Crypto
    'BTC-USD', 'ETH-USD'
]

# Risk-free rate for Sharpe ratio calculations
RISK_FREE_RATE = 0.02

# Data availability information for different assets
ASSET_DATA_AVAILABILITY = {
    'SPY': '1993-01-29',
    'AAPL': '1980-12-12',
    'MSFT': '1986-03-13',
    'GOOG': '2004-08-19',
    'AMZN': '1997-05-15',
    'NVDA': '1999-01-22',
    'META': '2012-05-18',
    'TSLA': '2010-06-29',
    'AMD': '1980-01-02',
    'CRM': '2004-06-23',
    'NFLX': '2002-05-23',
    'GLD': '2004-11-18',
    'TLT': '2002-07-30',
    'VNQ': '2004-09-23',
    'BTC-USD': '2014-09-17',
    'ETH-USD': '2017-11-09'
}

@st.cache_data
def fetch_stock_data(tickers, start_date, end_date):
    """
    Fetch historical stock data using yfinance with caching.
    
    Args:
        tickers (list): List of stock tickers
        start_date (datetime.date or pd.Timestamp): Start date
        end_date (datetime.date or pd.Timestamp): End date
    
    Returns:
        pd.DataFrame: Historical price data with only successful tickers
    """
    try:
        # Convert dates to string format
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # Download data with timeout and better error handling
        try:
            data = yf.download(tickers, start=start_str, end=end_str, progress=False, timeout=30)
        except Exception as download_error:
            st.warning(f"Download timeout or error, retrying with longer timeout: {download_error}")
            try:
                data = yf.download(tickers, start=start_str, end=end_str, progress=False, timeout=60)
            except Exception as retry_error:
                st.error(f"Failed to download data after retry: {retry_error}")
                return None
        
        # Handle different data structures
        if len(tickers) == 1:
            # Single ticker
            if 'Adj Close' in data.columns:
                return data[['Adj Close']]
            elif 'Close' in data.columns:
                return data[['Close']]
            else:
                st.error("No price data found in the downloaded data")
                return None
        else:
            # Multiple tickers - handle MultiIndex columns
            if isinstance(data.columns, pd.MultiIndex):
                # MultiIndex columns (multiple tickers)
                if 'Adj Close' in data.columns.get_level_values(0):
                    adj_close_data = data['Adj Close']
                elif 'Close' in data.columns.get_level_values(0):
                    adj_close_data = data['Close']
                else:
                    st.error("No price data found in the downloaded data")
                    return None
                
                # Filter out any columns that are all NaN (failed tickers)
                successful_tickers = []
                for col in adj_close_data.columns:
                    if not adj_close_data[col].isna().all():
                        successful_tickers.append(col)
                
                if not successful_tickers:
                    st.error("No successful tickers found. All selected assets failed to download.")
                    return None
                
                # Show warning for failed tickers
                failed_tickers = [t for t in tickers if t not in successful_tickers]
                if failed_tickers:
                    st.warning(f"Failed to fetch data for: {', '.join(failed_tickers)}")
                
                return adj_close_data[successful_tickers]
            else:
                # Single column (single ticker case)
                if 'Adj Close' in data.columns:
                    return data[['Adj Close']]
                elif 'Close' in data.columns:
                    return data[['Close']]
                else:
                    st.error("No price data found in the downloaded data")
                    return None
            
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def calculate_returns(prices):
    """
    Calculate daily returns from price data.
    
    Args:
        prices (pd.DataFrame): Historical price data
    
    Returns:
        pd.DataFrame: Daily returns
    """
    return prices.pct_change().dropna()

def calculate_portfolio_returns(returns, weights):
    """
    Calculate portfolio returns given asset returns and weights.
    
    Args:
        returns (pd.DataFrame): Asset returns
        weights (dict): Asset weights
    
    Returns:
        pd.Series: Portfolio returns
    """
    # Check if weights are valid
    if not weights or sum(weights.values()) == 0:
        # Return zero returns if no valid weights
        return pd.Series(0.0, index=returns.index)
    
    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    normalized_weights = {k: v/total_weight for k, v in weights.items()}
    
    # Calculate portfolio returns
    portfolio_returns = pd.Series(0.0, index=returns.index)
    for ticker, weight in normalized_weights.items():
        if ticker in returns.columns:
            portfolio_returns += weight * returns[ticker]
    
    return portfolio_returns

def calculate_portfolio_metrics(returns):
    """
    Calculate key portfolio performance metrics.
    
    Args:
        returns (pd.Series): Portfolio returns
    
    Returns:
        dict: Portfolio metrics
    """
    # Annualized return (assuming 252 trading days)
    annual_return = returns.mean() * 252
    
    # Annualized volatility
    annual_volatility = returns.std() * np.sqrt(252)
    
    # Sharpe ratio
    sharpe_ratio = (annual_return - RISK_FREE_RATE) / annual_volatility if annual_volatility > 0 else 0
    
    # Maximum drawdown
    cumulative_returns = (1 + returns).cumprod()
    rolling_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    return {
        'Annual Return': annual_return,
        'Annual Volatility': annual_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown
    }

def calculate_drawdown(returns):
    """
    Calculate drawdown series for plotting.
    
    Args:
        returns (pd.Series): Portfolio returns
    
    Returns:
        pd.Series: Drawdown series
    """
    cumulative_returns = (1 + returns).cumprod()
    rolling_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    return drawdown

def optimize_portfolio(returns, optimization_goal, target_return=None):
    """
    Optimize portfolio using Modern Portfolio Theory.
    
    Args:
        returns (pd.DataFrame): Asset returns
        optimization_goal (str): Optimization objective
        target_return (float): Target return for custom optimization
    
    Returns:
        dict: Optimal weights and metrics
    """
    try:
        # More aggressive data cleaning
        clean_returns = returns.copy()
        
        # Replace infinite values with NaN
        clean_returns = clean_returns.replace([np.inf, -np.inf], np.nan)
        
        # Remove rows with any NaN values
        clean_returns = clean_returns.dropna()
        
        # Additional check for extremely large values (beyond reasonable bounds)
        clean_returns = clean_returns.clip(-0.5, 0.5)  # Clip returns to ±50%
        
        if clean_returns.empty:
            st.error("No valid data available for optimization after cleaning.")
            return None
        
        # Check if we have enough data
        if len(clean_returns) < 30:
            st.error("Insufficient data for optimization. Need at least 30 days of data.")
            return None
        
        # Check for any remaining problematic values
        if clean_returns.isin([np.inf, -np.inf]).any().any():
            st.error("Infinite values still detected after cleaning.")
            return None
        
        if clean_returns.isna().any().any():
            st.error("NaN values still detected after cleaning.")
            return None
        
        # Try a simpler approach first - use sample covariance instead of shrinkage
        try:
            # Calculate expected returns manually to avoid PyPortfolioOpt issues
            mu = clean_returns.mean() * 252  # Annualize
            
            # Use sample covariance matrix instead of shrinkage
            S = clean_returns.cov() * 252  # Annualize
            
            # Check for any remaining infinite or NaN values
            if mu.isna().any() or S.isna().any().any():
                st.error("Invalid values detected in expected returns or covariance matrix.")
                st.error(f"mu has NaN: {mu.isna().sum()}")
                st.error(f"S has NaN: {S.isna().sum().sum()}")
                return None
            
            # Additional check for infinite values in mu and S
            if np.isinf(mu).any() or np.isinf(S.values).any():
                st.error("Infinite values detected in expected returns or covariance matrix.")
                st.error(f"mu has inf: {np.isinf(mu).sum()}")
                st.error(f"S has inf: {np.isinf(S.values).sum()}")
                return None
            
            # Create efficient frontier
            ef = EfficientFrontier(mu, S)
            
            if optimization_goal == "Max Sharpe":
                ef.max_sharpe()
            elif optimization_goal == "Min Volatility":
                ef.min_volatility()
            elif optimization_goal == "Max Return":
                # Find the asset with highest expected return and allocate 100% to it
                max_ret_asset = mu.idxmax()
                optimal_weights = {asset: 0.0 for asset in mu.index}
                optimal_weights[max_ret_asset] = 1.0
                
                # Calculate optimal portfolio metrics
                optimal_returns = calculate_portfolio_returns(clean_returns, optimal_weights)
                optimal_metrics = calculate_portfolio_metrics(optimal_returns)
                
                return {
                    'weights': optimal_weights,
                    'metrics': optimal_metrics,
                    'returns': optimal_returns
                }
            elif optimization_goal == "Custom Return Target" and target_return:
                ef.efficient_return(target_return)
            
            # Get optimal weights
            optimal_weights = ef.clean_weights()
            
            # Calculate optimal portfolio metrics
            optimal_returns = calculate_portfolio_returns(clean_returns, optimal_weights)
            optimal_metrics = calculate_portfolio_metrics(optimal_returns)
            
            return {
                'weights': optimal_weights,
                'metrics': optimal_metrics,
                'returns': optimal_returns
            }
            
        except Exception as e:
            st.warning(f"Sample covariance approach failed: {e}")
            st.info("Trying alternative optimization method...")
            
            # Fallback: Simple equal-weighted portfolio with manual optimization
            n_assets = len(clean_returns.columns)
            equal_weights = {asset: 1.0/n_assets for asset in clean_returns.columns}
            
            # Calculate portfolio metrics
            portfolio_returns = calculate_portfolio_returns(clean_returns, equal_weights)
            portfolio_metrics = calculate_portfolio_metrics(portfolio_returns)
            
            st.info("Using equal-weighted portfolio as fallback")
            
            return {
                'weights': equal_weights,
                'metrics': portfolio_metrics,
                'returns': portfolio_returns
            }
        
    except Exception as e:
        st.error(f"Optimization error: {e}")
        # Add debugging information
        st.error(f"Data shape: {returns.shape}")
        st.error(f"Data types: {returns.dtypes}")
        st.error(f"Sample data:\n{returns.head()}")
        
        # Check for infinite values in the original data
        inf_check = returns.isin([np.inf, -np.inf])
        if inf_check.any().any():
            st.error("Infinite values found in original data:")
            st.error(inf_check.sum())
        
        return None

def create_correlation_heatmap(returns):
    """
    Create correlation heatmap using seaborn and plotly.
    
    Args:
        returns (pd.DataFrame): Asset returns
    
    Returns:
        plotly.graph_objects.Figure: Correlation heatmap
    """
    corr_matrix = returns.corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Asset Correlation Matrix"
    )
    
    fig.update_layout(
        width=800,
        height=600,
        title_x=0.5
    )
    
    return fig





# Main application
def main():
    st.markdown('<h1 class="main-header">Interactive Portfolio Builder</h1>', unsafe_allow_html=True)
    
    # Sidebar for global settings
    st.sidebar.header("Global Settings")
    
    # Asset selection
    st.sidebar.subheader("Select Assets")
    
    # Data availability info
    st.sidebar.markdown("""
    **Data Availability:**
    - **SPY**: 1993-Present
    - **AAPL/MSFT/AMD**: 1980-Present
    - **AMZN/NVDA**: 1997-Present
    - **GOOG/GLD/TLT**: 2002-Present
    - **META**: 2012-Present
    - **TSLA**: 2010-Present
    - **Crypto**: 2014-Present
    """)
    
    selected_tickers = st.sidebar.multiselect(
        "Select Assets",
        options=AVAILABLE_TICKERS,
        default=['AAPL', 'MSFT', 'GOOG', 'SPY']
    )
    

    
    # Auto-adjust date range based on selected assets
    if selected_tickers:
        earliest_dates = []
        for ticker in selected_tickers:
            if ticker in ASSET_DATA_AVAILABILITY:
                earliest_dates.append(pd.to_datetime(ASSET_DATA_AVAILABILITY[ticker]))
        
        if earliest_dates:
            earliest_available = max(earliest_dates)  # Use the latest earliest date
            min_start_date = earliest_available
        else:
            min_start_date = pd.to_datetime('1980-01-01')
    else:
        min_start_date = pd.to_datetime('1980-01-01')
    
    # Date range selection
    st.sidebar.subheader("Date Range")
    
    start_date = st.sidebar.date_input(
        "Start Date",
        value=min_start_date,
        min_value=min_start_date,
        max_value=pd.Timestamp.now().date()
    )
    
    end_date = st.sidebar.date_input(
        "End Date",
        value=pd.Timestamp.now().date(),
        min_value=min_start_date,
        max_value=pd.Timestamp.now().date()
    )
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Home", "Portfolio Builder", "Optimization", 
        "Risk Analysis", "Correlation"
    ])
    
    # Tab 1: Home
    with tab1:
        st.header("Welcome to Interactive Portfolio Builder")
        
        st.markdown("""
        ### What You Can Do
        
        **Portfolio Builder**: Manually build and analyze your portfolio with custom asset allocations. Set your own weights and see how your choices perform over time.
        
        **Optimization**: Find the optimal portfolio weights using modern portfolio theory. Choose from multiple optimization goals including Maximum Sharpe Ratio, Minimum Volatility, Maximum Return, or custom return targets.
        
        **Risk Analysis**: Analyze portfolio risk through various metrics like Value at Risk (VaR), Maximum Drawdown, and stress testing under historical crisis scenarios including the 2008 Financial Crisis, 2020 COVID-19 Crash, and 2023 Banking Crisis.
        
        **Correlation Analysis**: View correlations between different assets to understand diversification benefits and identify potential concentration risks in your portfolio.
        
        ### Key Features
        
        - **Historical Data**: Access comprehensive historical data from 1980 onwards for most assets
        - **Real-time Calculations**: Dynamic portfolio metrics including returns, volatility, Sharpe ratio, and drawdowns
        - **Interactive Charts**: Beautiful Plotly visualizations for portfolio performance, efficient frontiers, and correlation matrices
        - **Multiple Optimization Strategies**: Choose from various optimization objectives to match your investment goals
        - **Risk Management Tools**: Comprehensive risk analysis with stress testing capabilities
        
        ### Quick Start Guide
        
        1. **Select Assets**: Choose from the sidebar - stocks, ETFs, bonds, commodities, and crypto
        2. **Set Date Range**: Pick your analysis period (data available from 1980 to present)
        3. **Build Portfolio**: Either manually set weights or use optimization to find optimal allocations
        4. **Analyze Results**: Review performance metrics, risk measures, and visualizations
        
        ### Pro Tips
        
        - Start with a few diverse assets and gradually add more to build complexity
        - Use the optimization tab to find optimal weights, then apply them to your portfolio builder
        - Monitor correlations to ensure proper diversification - aim for lower average correlations
        - Consider different time periods for analysis to understand how your portfolio performs in various market conditions
        - Use the risk analysis tab to stress test your portfolio against historical crisis scenarios
        
        ### About This Tool
        
        This application combines modern portfolio theory with practical investment analysis tools. It uses the PyPortfolioOpt library for optimization algorithms and Yahoo Finance for real-time market data. Whether you're a beginner investor learning about portfolio construction or an experienced analyst looking for advanced optimization capabilities, this tool provides the insights you need to make informed investment decisions.
        """)
    
    # Check if assets are selected
    if not selected_tickers:
        st.warning("Please select at least one asset in the sidebar to continue.")
        return
    
    # Check if selected date range is appropriate for selected assets
    if selected_tickers:
        earliest_dates = []
        for ticker in selected_tickers:
            if ticker in ASSET_DATA_AVAILABILITY:
                earliest_dates.append(pd.to_datetime(ASSET_DATA_AVAILABILITY[ticker]))
        
        if earliest_dates:
            earliest_available = max(earliest_dates)
            # Convert start_date to pd.Timestamp for comparison
            start_date_ts = pd.to_datetime(start_date)
            if start_date_ts < earliest_available:
                st.warning(f"**Date Range Warning**: You've selected a start date ({start_date.strftime('%Y-%m-%d')}) that's earlier than the earliest available data ({earliest_available.strftime('%Y-%m-%d')}) for your selected assets.")
                st.info("**Recommendation**: Consider adjusting your start date or selecting assets with longer historical data.")
    
    # Fetch data
    data = fetch_stock_data(selected_tickers, start_date, end_date)
    if data is None or data.empty:
        st.error("Unable to fetch data. Please check your selections.")
        return
    
    # Update selected_tickers to only include successful ones
    successful_tickers = list(data.columns)
    if len(successful_tickers) != len(selected_tickers):
        st.info(f"Successfully loaded {len(successful_tickers)} out of {len(successful_tickers)} selected assets")
        selected_tickers = successful_tickers
    
    # Show data range info
    
    
    returns = calculate_returns(data)
    
    # Tab 2: Portfolio Builder
    with tab2:
        st.header("Portfolio Builder")
        st.markdown("Manually build and analyze your portfolio")
        
        # Weight sliders
        st.subheader("Asset Weights")
        
        # Check if optimized weights are available
        if hasattr(st.session_state, 'optimized_weights') and st.session_state.optimized_weights:
            # Show optimization info
            opt_type = st.session_state.optimization_type if hasattr(st.session_state, 'optimization_type') else "Optimization"
            st.success(f"**{opt_type} Weights Applied**: These weights were optimized for your selected objective.")
            
            # Show optimized weights summary
            opt_weights = st.session_state.optimized_weights
            weight_summary = ", ".join([f"{k}: {v:.1%}" for k, v in opt_weights.items()])
            st.info(f"**Optimized Allocation**: {weight_summary}")
            
        weights = {}
        
        col1, col2 = st.columns(2)
        for i, ticker in enumerate(selected_tickers):
            with col1 if i % 2 == 0 else col2:
                # Use optimized weights if available, otherwise equal weights
                default_weight = 1.0/len(selected_tickers)
                if hasattr(st.session_state, 'optimized_weights') and st.session_state.optimized_weights:
                    if ticker in st.session_state.optimized_weights:
                        default_weight = st.session_state.optimized_weights[ticker]
                
                weights[ticker] = st.slider(
                    f"{ticker} Weight",
                    min_value=0.0,
                    max_value=1.0,
                    value=default_weight,
                    step=0.01,
                    help=f"Allocation weight for {ticker}"
                )
        
        # Portfolio action buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Update Portfolio"):
                st.session_state.portfolio_updated = True
        
        with col2:
            # Reset to equal weights button (only show if optimized weights are available)
            if hasattr(st.session_state, 'optimized_weights') and st.session_state.optimized_weights:
                if st.button("Reset to Equal Weights"):
                    st.session_state.optimized_weights = None
                    st.session_state.optimization_type = None
                    st.rerun()
        
        # Portfolio content (outside of columns to avoid width constraints)
        if st.session_state.get('portfolio_updated', False):
            # Check if weights are valid before calculating
            if not weights or sum(weights.values()) == 0:
                st.warning("Please set valid weights for your assets before analyzing the portfolio.")
            else:
                # Calculate portfolio returns
                portfolio_returns = calculate_portfolio_returns(returns, weights)
                portfolio_metrics = calculate_portfolio_metrics(portfolio_returns)
                
                # Display metrics
                st.subheader("Portfolio Performance Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Annual Return",
                        f"{portfolio_metrics['Annual Return']:.2%}",
                        help="Annualized portfolio return"
                    )
                
                with col2:
                    st.metric(
                        "Annual Volatility",
                        f"{portfolio_metrics['Annual Volatility']:.2%}",
                        help="Annualized portfolio volatility"
                    )
                
                with col3:
                    st.metric(
                        "Sharpe Ratio",
                        f"{portfolio_metrics['Sharpe Ratio']:.2f}",
                        help="Risk-adjusted return measure"
                    )
                
                with col4:
                    st.metric(
                        "Max Drawdown",
                        f"{portfolio_metrics['Max Drawdown']:.2%}",
                        help="Maximum historical loss from peak"
                    )
                
                # Portfolio growth chart
                st.subheader("Portfolio Growth")
                cumulative_returns = (1 + portfolio_returns).cumprod()
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=cumulative_returns.index,
                    y=cumulative_returns.values,
                    mode='lines',
                    name='Portfolio',
                    line=dict(color='#1f77b4', width=2)
                ))
                
                fig.update_layout(
                    title="Portfolio Cumulative Returns",
                    xaxis_title="Date",
                    yaxis_title="Cumulative Return",
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Individual asset performance
                st.subheader("Individual Asset Performance")
                asset_cumulative = (1 + returns).cumprod()
                
                fig2 = go.Figure()
                for ticker in selected_tickers:
                    fig2.add_trace(go.Scatter(
                        x=asset_cumulative.index,
                        y=asset_cumulative[ticker],
                        mode='lines',
                        name=ticker
                    ))
                
                fig2.update_layout(
                    title="Individual Asset Performance",
                    xaxis_title="Date",
                    yaxis_title="Cumulative Return",
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig2, use_container_width=True)
                
                # Asset Price History
                st.subheader("Asset Price History")
                
                # Create line chart showing historical prices for each asset
                fig3 = go.Figure()
                
                for ticker in selected_tickers:
                    if ticker in data.columns:
                        fig3.add_trace(go.Scatter(
                            x=data.index,
                            y=data[ticker],
                            mode='lines',
                            name=ticker,
                            line=dict(width=2)
                        ))
                
                fig3.update_layout(
                    title="Historical Asset Prices",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    height=400,
                    hovermode='x unified',
                    showlegend=True
                )
                
                st.plotly_chart(fig3, use_container_width=True)
    
    # Tab 3: Optimization
    with tab3:
        st.header("Portfolio Optimization")
        st.markdown("Find the optimal portfolio weights using modern portfolio theory")
        
        # Navigation tip
        st.info("**Pro Tip**: After optimization, use the 'Apply to Portfolio Builder' button to automatically apply the optimal weights to the Portfolio Builder tab!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            optimization_goal = st.selectbox(
                "Optimization Goal",
                ["Max Sharpe", "Min Volatility", "Max Return", "Custom Return Target"],
                help="Choose your optimization objective"
            )
        
        with col2:
            target_return = None
            if optimization_goal == "Custom Return Target":
                target_return = st.number_input(
                    "Target Annual Return",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.10,
                    step=0.01,
                    format="%.2f",
                    help="Target annual return for optimization"
                )
        
        if st.button("Optimize Portfolio"):
            # Perform optimization
            optimization_result = optimize_portfolio(returns, optimization_goal, target_return)
            
            if optimization_result:
                optimal_weights = optimization_result['weights']
                optimal_metrics = optimization_result['metrics']
                optimal_returns = optimization_result['returns']
                
                # Display optimal weights
                st.subheader("Optimal Asset Weights")
                
                weight_df = pd.DataFrame(list(optimal_weights.items()), columns=['Asset', 'Weight'])
                weight_df['Weight'] = weight_df['Weight'].apply(lambda x: f"{x:.1%}")
                
                st.dataframe(weight_df, use_container_width=True)
                
                # Store optimized weights in session state for Portfolio Builder
                st.session_state.optimized_weights = optimal_weights
                st.session_state.optimization_type = optimization_goal
                
                # Apply to Portfolio Builder button
                if st.button("Apply to Portfolio Builder", type="primary"):
                    st.success("**Weights applied!** Switch to Portfolio Builder tab to see the optimized portfolio.")
                    st.info("**Tip**: The Portfolio Builder tab now shows your optimized weights. You can further adjust them if needed.")
                
                # Display optimal metrics
                st.subheader("Optimal Portfolio Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Annual Return",
                        f"{optimal_metrics['Annual Return']:.2%}"
                    )
                
                with col2:
                    st.metric(
                        "Annual Volatility",
                        f"{optimal_metrics['Annual Volatility']:.2%}"
                    )
                
                with col3:
                    st.metric(
                        "Sharpe Ratio",
                        f"{optimal_metrics['Sharpe Ratio']:.2f}"
                    )
                
                with col4:
                    st.metric(
                        "Max Drawdown",
                        f"{optimal_metrics['Max Drawdown']:.2%}"
                    )
                
                # Efficient frontier
                st.subheader("Efficient Frontier")
                
                try:
                    # Use the same robust cleaning for efficient frontier
                    clean_returns = returns.copy()
                    clean_returns = clean_returns.replace([np.inf, -np.inf], np.nan)
                    clean_returns = clean_returns.dropna()
                    clean_returns = clean_returns.clip(-0.5, 0.5)  # Clip returns to ±50%
                    
                    if clean_returns.empty:
                        st.error("No valid data available for efficient frontier.")
                        return
                    
                    # Check if we have sufficient data for reliable calculations
                    if len(clean_returns) < 30:
                        pass
                    
                    # Generate efficient frontier points using sample covariance
                    mu = clean_returns.mean() * 252  # Annualize manually
                    S = clean_returns.cov() * 252  # Annualize
                    
                    # Check for valid covariance matrix
                    if S.isnull().any().any() or S.isin([np.inf, -np.inf]).any().any():
                        st.error("Invalid covariance matrix detected. Please try different assets or date range.")
                        return
                    
                    # Check for positive definite covariance matrix
                    try:
                        np.linalg.cholesky(S)
                    except np.linalg.LinAlgError:
                        pass
                    
                    # Additional validation for cloud environment
                    if mu.isnull().any() or mu.isin([np.inf, -np.inf]).any():
                        st.error("Invalid expected returns detected. Please try different assets or date range.")
                        return
                    
                    # Check for reasonable values
                    if (mu.abs() > 2).any():  # Returns > 200% annually are suspicious
                        pass
                    
                    # Additional validation for cloud environment
                    if S.shape[0] != S.shape[1]:
                        st.error("Covariance matrix is not square. This indicates a data issue.")
                        return
                    
                    if S.shape[0] < 2:
                        st.error("Insufficient assets for portfolio optimization. Need at least 2 assets.")
                        return
                    
                    # Additional validation for cloud environment
                    if clean_returns.shape[1] < 2:
                        st.error("Insufficient assets for portfolio optimization. Need at least 2 assets.")
                        return
                    
                    # Check for sufficient data points per asset
                    min_data_points = 20  # Minimum data points needed per asset
                    if len(clean_returns) < min_data_points:
                        pass
                    
                    ef = EfficientFrontier(mu, S)
                    
                    # Initialize variables for efficient frontier
                    ef_vols = []
                    ef_rets = []
                    
                    # Generate efficient frontier points
                    # Get min and max volatility for the frontier
                    try:
                        # Try to calculate minimum volatility portfolio
                        try:
                            min_vol = ef.min_volatility()
                            min_vol_ret = ef.portfolio_performance()[0]
                            min_vol_vol = ef.portfolio_performance()[1]
                        except Exception as min_vol_error:
                            # Use fallback values
                            min_vol_ret = mu.min()
                            min_vol_vol = np.sqrt(S.min().min())
                        
                        # Get max return portfolio (100% allocation to highest return asset)
                        try:
                            max_ret_asset = mu.idxmax()
                            max_ret = mu[max_ret_asset]
                            max_ret_vol = np.sqrt(S.loc[max_ret_asset, max_ret_asset])
                        except Exception as max_ret_error:
                            # Use fallback values
                            max_ret = mu.max()
                            max_ret_vol = np.sqrt(S.max().max())
                        
                        # Generate points along the frontier with more conservative approach
                        # Use fewer points for cloud environment to avoid memory issues
                        num_points = min(30, max(10, int((max_ret - min_vol_ret) * 100)))  # Adaptive number of points
                        target_returns = np.linspace(min_vol_ret, max_ret, num_points)
                        ef_points = []
                        
                        for i, target_ret in enumerate(target_returns):
                            try:
                                ef.efficient_return(target_ret)
                                vol = ef.portfolio_performance()[1]
                                # Check for valid numerical values
                                if np.isfinite(vol) and np.isfinite(target_ret) and vol > 0:
                                    # Additional validation for cloud environment
                                    if 0 < vol < 10 and -2 < target_ret < 5:  # Reasonable bounds
                                        ef_points.append((target_ret, vol))
                            except Exception as e:
                                # Skip this point and continue
                                continue
                        
                        # Extract volumes and returns for plotting
                        if ef_points and len(ef_points) >= 2:
                            ef_vols = [point[1] for point in ef_points]
                            ef_rets = [point[0] for point in ef_points]
                        elif ef_points and len(ef_points) == 1:
                            ef_vols = [point[1] for point in ef_points]
                            ef_rets = [point[0] for point in ef_points]
                        else:
                            # Fallback: try to generate a simple frontier with just min vol and max return
                            try:
                                ef_vols = [min_vol_vol, max_ret_vol]
                                ef_rets = [min_vol_ret, max_ret]
                            except Exception as fallback_error:
                                pass
                        
                        # Set default values if frontier generation fails
                        if not ef_vols or not ef_rets:
                            ef_vols = []
                            ef_rets = []
                    
                    except Exception as e:
                        # Set default values if frontier generation fails
                        ef_vols = []
                        ef_rets = []
                    
                    fig = go.Figure()
                    

                    
                    # Plot efficient frontier
                    if ef_vols and ef_rets and len(ef_vols) >= 2 and len(ef_rets) >= 2:
                        # Ensure data is properly formatted for Plotly
                        try:
                            # Convert to numpy arrays and check for valid data
                            ef_vols_array = np.array(ef_vols, dtype=float)
                            ef_rets_array = np.array(ef_rets, dtype=float)
                            
                            # Remove any invalid values
                            valid_mask = np.isfinite(ef_vols_array) & np.isfinite(ef_rets_array) & (ef_vols_array > 0)
                            if np.sum(valid_mask) >= 2:
                                ef_vols_clean = ef_vols_array[valid_mask]
                                ef_rets_clean = ef_rets_array[valid_mask]
                                
                                # Additional data validation for cloud environment
                                if len(ef_vols_clean) != len(ef_rets_clean):
                                    st.error("Mismatch in volatility and return array lengths")
                                    return
                                
                                # Check for reasonable value ranges
                                if np.any(ef_vols_clean > 10) or np.any(ef_rets_clean > 5):
                                    ef_vols_clean = np.clip(ef_vols_clean, 0, 10)
                                    ef_rets_clean = np.clip(ef_rets_clean, -2, 5)
                                
                                # Ensure data is sorted for proper line plotting
                                sort_idx = np.argsort(ef_vols_clean)
                                ef_vols_sorted = ef_vols_clean[sort_idx]
                                ef_rets_sorted = ef_rets_clean[sort_idx]
                                
                                # Limit number of points for cloud rendering to avoid memory issues
                                if len(ef_vols_sorted) > 20:
                                    step = len(ef_vols_sorted) // 20
                                    ef_vols_limited = ef_vols_sorted[::step]
                                    ef_rets_limited = ef_rets_sorted[::step]
                                else:
                                    ef_vols_limited = ef_vols_sorted
                                    ef_rets_limited = ef_rets_sorted
                                
                                fig.add_trace(go.Scatter(
                                    x=ef_vols_limited,
                                    y=ef_rets_limited,
                                    mode='lines',
                                    name='Efficient Frontier',
                                    line=dict(color='#1f77b4', width=2)
                                ))
                            else:
                                pass
                        except Exception as plot_error:
                            st.error(f"Error plotting efficient frontier: {plot_error}")
                    else:
                        pass
                    
                    # Plot individual assets
                    for ticker in selected_tickers:
                        if ticker in mu.index:
                            asset_return = mu[ticker]
                            asset_vol = np.sqrt(S.loc[ticker, ticker])
                            fig.add_trace(go.Scatter(
                                x=[asset_vol],
                                y=[asset_return],
                                mode='markers+text',
                                name=ticker,
                                text=[ticker],
                                textposition="top center",
                                marker=dict(size=10)
                            ))
                    
                    # Plot optimal portfolio
                    optimal_vol = optimal_metrics['Annual Volatility']
                    optimal_ret = optimal_metrics['Annual Return']
                    fig.add_trace(go.Scatter(
                        x=[optimal_vol],
                        y=[optimal_ret],
                        mode='markers',
                        name='Optimal Portfolio',
                        marker=dict(size=15, color='red', symbol='star')
                    ))
                    
                    # Add free return line tangent to Sharpe ratio point
                    if optimization_goal == "Max Sharpe":
                        # Calculate Sharpe ratio portfolio for tangent line
                        try:
                            ef_sharpe = EfficientFrontier(mu, S)
                            ef_sharpe.max_sharpe()
                            sharpe_weights = ef_sharpe.clean_weights()
                            
                            # Calculate Sharpe portfolio metrics
                            sharpe_returns = calculate_portfolio_returns(clean_returns, sharpe_weights)
                            sharpe_metrics = calculate_portfolio_metrics(sharpe_returns)
                            sharpe_vol = sharpe_metrics['Annual Volatility']
                            sharpe_ret = sharpe_metrics['Annual Return']
                            
                            # Risk-free rate (using 3-month Treasury yield as approximation)
                            risk_free_rate = 0.02  # 2% annual rate
                            
                            # Calculate tangent line points
                            # Line goes from (0, risk_free_rate) to (sharpe_vol, sharpe_ret)
                            # Extend line to x-axis limit for better visualization
                            if ef_vols:
                                max_vol = max(ef_vols)
                            else:
                                max_vol = sharpe_vol * 1.5
                            
                            # Calculate slope of tangent line
                            slope = (sharpe_ret - risk_free_rate) / sharpe_vol
                            
                            # Generate points for tangent line
                            x_tangent = [0, max_vol]
                            y_tangent = [risk_free_rate, risk_free_rate + slope * max_vol]
                            
                            # Plot tangent line
                            fig.add_trace(go.Scatter(
                                x=x_tangent,
                                y=y_tangent,
                                mode='lines',
                                name='Capital Allocation Line (Tangent)',
                                line=dict(color='green', width=2, dash='dash'),
                                showlegend=True
                            ))
                            
                            # Plot Sharpe ratio portfolio point
                            fig.add_trace(go.Scatter(
                                x=[sharpe_vol],
                                y=[sharpe_ret],
                                mode='markers',
                                name='Max Sharpe Portfolio',
                                marker=dict(size=12, color='green', symbol='diamond'),
                                showlegend=True
                            ))
                            
                            # Add risk-free rate point
                            fig.add_trace(go.Scatter(
                                x=[0],
                                y=[risk_free_rate],
                                mode='markers',
                                name='Risk-Free Rate',
                                marker=dict(size=10, color='orange', symbol='circle'),
                                showlegend=True
                            ))
                            
                        except Exception as e:
                            pass
                    
                    fig.update_layout(
                        title="Efficient Frontier",
                        xaxis_title="Annual Volatility",
                        yaxis_title="Annual Return",
                        height=600,
                        # Add additional layout options for better cloud rendering
                        margin=dict(l=50, r=50, t=80, b=50),
                        showlegend=True,
                        legend=dict(x=0.02, y=0.98),
                        # Ensure proper axis configuration
                        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
                        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray')
                    )
                    
                    # Render the plot directly
                    try:
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as plot_render_error:
                        st.error(f"Error rendering plot: {plot_render_error}")
                        # Simple fallback: show data as text
                        if ef_vols and ef_rets and len(ef_vols) >= 2:
                            st.write("Efficient Frontier Data:")
                            for i, (vol, ret) in enumerate(zip(ef_vols, ef_rets)):
                                st.write(f"Point {i+1}: Volatility = {vol:.4f}, Return = {ret:.4f}")
                    
                except Exception as e:
                    st.error(f"Error generating efficient frontier: {e}")
    
    # Tab 4: Risk Analysis
    with tab4:
        st.header("Risk Analysis")
        st.markdown("Analyze portfolio risk and stress test scenarios")
        
        col1, col2 = st.columns(2)
        
        with col1:
            scenario = st.selectbox(
                "Stress Test Scenario",
                ["2008 Financial Crisis", "2020 COVID Crash", "Custom Period"],
                help="Choose a historical crisis period or custom range"
            )
        
        with col2:
            lookback_days = st.slider(
                "Volatility Lookback (days)",
                min_value=20,
                max_value=252,
                value=60,
                help="Period for volatility calculation"
            )
        
        # Custom period date selection
        custom_start = None
        custom_end = None
        if scenario == "Custom Period":
            st.subheader("Custom Period Selection")
            col1, col2 = st.columns(2)
            
            with col1:
                custom_start = st.date_input(
                    "Custom Start Date",
                    value=pd.to_datetime('2008-09-01'),
                    min_value=pd.to_datetime('1980-01-01'),
                    max_value=pd.Timestamp.now().date(),
                    help="Select start date for custom stress test period"
                )
            
            with col2:
                custom_end = st.date_input(
                    "Custom End Date",
                    value=pd.to_datetime('2009-03-01'),
                    min_value=pd.to_datetime('1980-01-01'),
                    max_value=pd.Timestamp.now().date(),
                    help="Select end date for custom stress test period"
                )
        
        # Define crisis periods
        crisis_periods = {
            "1987 Black Monday": (pd.to_datetime("1987-10-01"), pd.to_datetime("1987-12-01")),
            "2000 Dot-Com Bubble": (pd.to_datetime("2000-03-01"), pd.to_datetime("2002-10-01")),
            "2008 Financial Crisis": (pd.to_datetime("2008-09-01"), pd.to_datetime("2009-03-01")),
            "2011 Euro Crisis": (pd.to_datetime("2011-07-01"), pd.to_datetime("2011-10-01")),
            "2018 Q4 Selloff": (pd.to_datetime("2018-10-01"), pd.to_datetime("2018-12-01")),
            "2020 COVID Crash": (pd.to_datetime("2020-02-01"), pd.to_datetime("2020-04-01")),
            "2022 Inflation Crisis": (pd.to_datetime("2022-01-01"), pd.to_datetime("2022-10-01")),
            "2023 Banking Crisis": (pd.to_datetime("2023-03-01"), pd.to_datetime("2023-05-01"))
        }
        
        # Handle custom period
        if scenario == "Custom Period" and custom_start and custom_end:
            crisis_start, crisis_end = custom_start, custom_end
            crisis_data = fetch_stock_data(selected_tickers, crisis_start, crisis_end)
        elif scenario in crisis_periods:
            crisis_start, crisis_end = crisis_periods[scenario]
            crisis_data = fetch_stock_data(selected_tickers, crisis_start, crisis_end)
        
        # Perform analysis for any scenario (including custom period)
        if crisis_data is not None:
            crisis_returns = calculate_returns(crisis_data)
            
            # Calculate equal-weighted portfolio for crisis analysis
            equal_weights = {ticker: 1.0/len(selected_tickers) for ticker in selected_tickers}
            crisis_portfolio_returns = calculate_portfolio_returns(crisis_returns, equal_weights)
            crisis_metrics = calculate_portfolio_metrics(crisis_portfolio_returns)
            
            st.subheader(f"Performance During {scenario}")
            
            # Crisis metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Crisis Return",
                    f"{crisis_metrics['Annual Return']:.2%}"
                )
            
            with col2:
                st.metric(
                    "Crisis Volatility",
                    f"{crisis_metrics['Annual Volatility']:.2%}"
                )
            
            with col3:
                st.metric(
                    "Crisis Sharpe",
                    f"{crisis_metrics['Sharpe Ratio']:.2f}"
                )
            
            with col4:
                st.metric(
                    "Crisis Max Drawdown",
                    f"{crisis_metrics['Max Drawdown']:.2%}"
                )
            
            # Crisis drawdown chart
            st.subheader("Crisis Drawdown Analysis")
            crisis_drawdown = calculate_drawdown(crisis_portfolio_returns)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=crisis_drawdown.index,
                y=crisis_drawdown.values * 100,
                mode='lines',
                fill='tonexty',
                name='Drawdown %',
                line=dict(color='red')
            ))
            
            fig.update_layout(
                title=f"Portfolio Drawdown During {scenario}",
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Rolling volatility analysis
        st.subheader("Rolling Volatility Analysis")
        
        # Calculate rolling volatility for portfolio
        equal_weights = {ticker: 1.0/len(selected_tickers) for ticker in selected_tickers}
        portfolio_returns = calculate_portfolio_returns(returns, equal_weights)
        rolling_vol = portfolio_returns.rolling(lookback_days).std() * np.sqrt(252)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rolling_vol.index,
            y=rolling_vol.values * 100,
            mode='lines',
            name='Rolling Volatility',
            line=dict(color='orange')
        ))
        
        fig.update_layout(
            title=f"Rolling {lookback_days}-Day Volatility",
            xaxis_title="Date",
            yaxis_title="Volatility (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 5: Correlation Analysis
    with tab5:
        st.header("Correlation Analysis")
        st.markdown("View correlations between different assets")
        
        # Correlation heatmap
        st.subheader("Asset Correlation Matrix")
        corr_fig = create_correlation_heatmap(returns)
        st.plotly_chart(corr_fig, use_container_width=True)
        
        # Calculate correlation matrix for diversification analysis
        corr_matrix = returns.corr()
        
        # Diversification score
        st.subheader("Diversification Analysis")
        
        # Calculate average correlation
        avg_correlation = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
        
        # Calculate diversification score (lower correlation = higher diversification)
        diversification_score = 1 - abs(avg_correlation)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Average Correlation",
                f"{avg_correlation:.3f}",
                help="Average correlation between all asset pairs"
            )
        
        with col2:
            st.metric(
                "Diversification Score",
                f"{diversification_score:.3f}",
                help="Higher score indicates better diversification"
            )
        
        # Rolling correlation analysis
        st.subheader("Rolling Correlation Analysis")
        
        # Calculate rolling correlation between first two assets
        if len(selected_tickers) >= 2:
            asset1, asset2 = selected_tickers[0], selected_tickers[1]
            rolling_corr = returns[asset1].rolling(60).corr(returns[asset2])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=rolling_corr.index,
                y=rolling_corr.values,
                mode='lines',
                name=f'{asset1} vs {asset2}',
                line=dict(color='purple')
            ))
            
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            
            fig.update_layout(
                title=f"60-Day Rolling Correlation: {asset1} vs {asset2}",
                xaxis_title="Date",
                yaxis_title="Correlation",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Footer for all pages
    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; color: #888888; font-size: 12px; padding: 20px 0;">'
        'Made by: Abdul Ahad Ali Khan<br>'
        'Contact: abdulahadalikhan12@gmail.com'
        '</div>',
        unsafe_allow_html=True
    )



if __name__ == "__main__":
    main()

