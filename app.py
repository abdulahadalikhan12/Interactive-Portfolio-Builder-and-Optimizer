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
from pypfopt import EfficientFrontier, risk_models, expected_returns
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

# Pre-calculated efficient frontier data for common asset combinations
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_precalculated_efficient_frontier(selected_tickers, start_date, end_date):
    """
    Get pre-calculated efficient frontier data for common asset combinations.
    This avoids real-time computation issues on Streamlit Cloud.
    """
    # Create a unique key for the combination
    ticker_key = '_'.join(sorted(selected_tickers))
    date_key = f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
    cache_key = f"{ticker_key}_{date_key}"
    
    # Check if we have cached data
    if hasattr(st.session_state, 'ef_cache') and cache_key in st.session_state.ef_cache:
        return st.session_state.ef_cache[cache_key]
    
    return None

@st.cache_data(ttl=3600)  # Cache for 1 hour
def calculate_efficient_frontier_data(returns, selected_tickers):
    """
    Calculate efficient frontier data with robust error handling.
    This function is cached to avoid repeated calculations.
    """
    try:
        # Add timeout protection for Streamlit Cloud
        import time
        start_time = time.time()
        max_execution_time = 30  # 30 seconds max
        
        # Clean the data
        clean_returns = returns.copy()
        clean_returns = clean_returns.replace([np.inf, -np.inf], np.nan)
        clean_returns = clean_returns.dropna()
        clean_returns = clean_returns.clip(-0.5, 0.5)
        
        if clean_returns.empty or len(clean_returns) < 30:
            return None
        
        # Check timeout
        if time.time() - start_time > max_execution_time:
            st.warning("Calculation taking too long, using simplified method...")
            return None
        
        # Calculate expected returns and covariance
        mu = clean_returns.mean() * 252
        S = clean_returns.cov() * 252
        
        # Validate data
        if mu.isna().any() or S.isna().any().any():
            return None
        
        # Generate efficient frontier points
        ef = EfficientFrontier(mu, S)
        
        # Get min volatility portfolio
        try:
            ef.min_volatility()
            min_vol_ret = ef.portfolio_performance()[0]
            min_vol_vol = ef.portfolio_performance()[1]
        except:
            return None
        
        # Get max return portfolio
        max_ret_asset = mu.idxmax()
        max_ret = mu[max_ret_asset]
        max_ret_vol = np.sqrt(S.loc[max_ret_asset, max_ret_asset])
        
        # Generate frontier points (optimized for speed)
        # Use fewer points for faster computation but maintain quality
        num_points = min(20, len(clean_returns.columns) * 2)  # Reduced for Streamlit Cloud
        target_returns = np.linspace(min_vol_ret, max_ret, num_points)
        ef_points = []
        
        # Add progress tracking for better user experience
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        successful_points = 0
        for i, target_ret in enumerate(target_returns):
            # Check timeout every few iterations
            if i % 5 == 0 and time.time() - start_time > max_execution_time:
                st.warning("Calculation timeout reached, using available points...")
                break
                
            try:
                status_text.text(f"Calculating point {i+1}/{num_points}...")
                ef.efficient_return(target_ret)
                vol = ef.portfolio_performance()[1]
                ef_points.append((target_ret, vol))
                successful_points += 1
                progress_bar.progress((i + 1) / num_points)
            except Exception as e:
                # Log the specific error for debugging
                st.debug(f"Failed at point {i+1}: {e}")
                continue
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Check if we got enough points
        if successful_points < 2:
            st.warning(f"Only {successful_points} frontier points generated. This may affect graph quality.")
        
        # Calculate individual asset points
        asset_points = []
        for ticker in selected_tickers:
            if ticker in mu.index:
                asset_return = mu[ticker]
                asset_vol = np.sqrt(S.loc[ticker, ticker])
                asset_points.append({
                    'ticker': ticker,
                    'return': asset_return,
                    'volatility': asset_vol
                })
        
        # Calculate Sharpe ratio portfolio
        sharpe_data = None
        try:
            ef_sharpe = EfficientFrontier(mu, S)
            ef_sharpe.max_sharpe()
            sharpe_weights = ef_sharpe.clean_weights()
            sharpe_returns = calculate_portfolio_returns(clean_returns, sharpe_weights)
            sharpe_metrics = calculate_portfolio_metrics(sharpe_returns)
            sharpe_data = {
                'volatility': sharpe_metrics['Annual Volatility'],
                'return': sharpe_metrics['Annual Return'],
                'weights': sharpe_weights
            }
        except:
            pass
        
        return {
            'ef_points': ef_points,
            'asset_points': asset_points,
            'sharpe_data': sharpe_data,
            'min_vol_ret': min_vol_ret,
            'min_vol_vol': min_vol_vol,
            'max_ret': max_ret,
            'max_ret_vol': max_ret_vol
        }
        
    except Exception as e:
        # Fallback: Try simpler calculation method
        try:
            st.warning("Full optimization failed, trying simplified method...")
            
            # Simple two-point frontier: min vol and max return
            clean_returns = returns.copy()
            clean_returns = clean_returns.replace([np.inf, -np.inf], np.nan)
            clean_returns = clean_returns.dropna()
            clean_returns = clean_returns.clip(-0.5, 0.5)
            
            if clean_returns.empty or len(clean_returns) < 30:
                return None
            
            mu = clean_returns.mean() * 252
            S = clean_returns.cov() * 252
            
            # Simple min vol (equal weight)
            n_assets = len(clean_returns.columns)
            equal_weights = {asset: 1.0/n_assets for asset in clean_returns.columns}
            equal_returns = calculate_portfolio_returns(clean_returns, equal_weights)
            equal_metrics = calculate_portfolio_metrics(equal_returns)
            
            # Max return (100% to highest return asset)
            max_ret_asset = mu.idxmax()
            max_ret = mu[max_ret_asset]
            max_ret_vol = np.sqrt(S.loc[max_ret_asset, max_ret_asset])
            
            # Create simple frontier with just two points
            ef_points = [
                (equal_metrics['Annual Return'], equal_metrics['Annual Volatility']),
                (max_ret, max_ret_vol)
            ]
            
            # Individual asset points
            asset_points = []
            for ticker in selected_tickers:
                if ticker in mu.index:
                    asset_return = mu[ticker]
                    asset_vol = np.sqrt(S.loc[ticker, ticker])
                    asset_points.append({
                        'ticker': ticker,
                        'return': asset_return,
                        'volatility': asset_vol
                    })
            
            return {
                'ef_points': ef_points,
                'asset_points': asset_points,
                'sharpe_data': None,
                'min_vol_ret': equal_metrics['Annual Return'],
                'min_vol_vol': equal_metrics['Annual Volatility'],
                'max_ret': max_ret,
                'max_ret_vol': max_ret_vol
            }
            
        except Exception as fallback_error:
            st.error(f"Both optimization methods failed: {fallback_error}")
            return None

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
        
        # Download data
        data = yf.download(tickers, start=start_str, end=end_str, progress=False)
        
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
                
                # Show loading message
                with st.spinner("Generating efficient frontier... This may take a moment for complex portfolios."):
                    try:
                        # Use pre-calculated efficient frontier data
                        st.info(f"Calculating efficient frontier for {len(selected_tickers)} assets...")
                        ef_data = calculate_efficient_frontier_data(returns, selected_tickers)
                        
                        if ef_data is None:
                            st.error("Could not generate efficient frontier data. This may happen with limited data or extreme market conditions.")
                            st.info("Try selecting different assets or a longer time period.")
                            return
                        
                        # Debug information
                        st.success(f"✅ Efficient frontier data generated successfully!")
                        st.info(f"📊 Frontier points: {len(ef_data['ef_points']) if ef_data['ef_points'] else 0}")
                        st.info(f"🏢 Asset points: {len(ef_data['asset_points'])}")
                        
                        # Create the plot
                        fig = go.Figure()
                        
                        # Initialize variables for tangent line calculation
                        ef_vols = []
                        ef_rets = []
                        
                        # Plot efficient frontier
                        if ef_data['ef_points'] and len(ef_data['ef_points']) >= 2:
                            ef_vols = [point[1] for point in ef_data['ef_points']]
                            ef_rets = [point[0] for point in ef_data['ef_points']]
                            
                            # If we have enough points, plot as line
                            if len(ef_vols) >= 3:
                                fig.add_trace(go.Scatter(
                                    x=ef_vols,
                                    y=ef_rets,
                                    mode='lines',
                                    name='Efficient Frontier',
                                    line=dict(color='#1f77b4', width=2)
                                ))
                            else:
                                # If few points, plot as markers
                                fig.add_trace(go.Scatter(
                                    x=ef_vols,
                                    y=ef_rets,
                                    mode='markers',
                                    name='Efficient Frontier Points',
                                    marker=dict(color='#1f77b4', size=8)
                                ))
                        else:
                            st.warning("Could not generate efficient frontier points. This may happen with limited data or extreme market conditions.")
                            # Create a simple line from min vol to max return
                            if ef_data['min_vol_vol'] and ef_data['max_ret_vol']:
                                fig.add_trace(go.Scatter(
                                    x=[ef_data['min_vol_vol'], ef_data['max_ret_vol']],
                                    y=[ef_data['min_vol_ret'], ef_data['max_ret']],
                                    mode='lines',
                                    name='Simple Frontier',
                                    line=dict(color='#1f77b4', width=2, dash='dash')
                                ))
                        
                        # Plot individual assets
                        for asset in ef_data['asset_points']:
                            fig.add_trace(go.Scatter(
                                x=[asset['volatility']],
                                y=[asset['return']],
                                mode='markers+text',
                                name=asset['ticker'],
                                text=[asset['ticker']],
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
                        if optimization_goal == "Max Sharpe" and ef_data['sharpe_data']:
                            sharpe_vol = ef_data['sharpe_data']['volatility']
                            sharpe_ret = ef_data['sharpe_data']['return']
                            
                            # Risk-free rate
                            risk_free_rate = 0.02
                            
                            # Calculate tangent line points
                            if ef_vols and len(ef_vols) > 0:
                                max_vol = max(ef_vols)
                            else:
                                max_vol = sharpe_vol * 1.5
                            
                            # Calculate slope of tangent line (avoid division by zero)
                            if sharpe_vol > 0:
                                slope = (sharpe_ret - risk_free_rate) / sharpe_vol
                                
                                # Generate points for tangent line
                                x_tangent = [0, max_vol]
                                y_tangent = [risk_free_rate, risk_free_rate + slope * max_vol]
                            else:
                                # Fallback if sharpe_vol is zero or negative
                                x_tangent = [0, max_vol]
                                y_tangent = [risk_free_rate, risk_free_rate]
                            
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
                        
                        fig.update_layout(
                            title="Efficient Frontier",
                            xaxis_title="Annual Volatility",
                            yaxis_title="Annual Return",
                            height=600
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Store in cache for future use
                        if not hasattr(st.session_state, 'ef_cache'):
                            st.session_state.ef_cache = {}
                        
                        ticker_key = '_'.join(sorted(selected_tickers))
                        date_key = f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
                        cache_key = f"{ticker_key}_{date_key}"
                        st.session_state.ef_cache[cache_key] = ef_data
                        
                    except Exception as e:
                        st.error(f"Error generating efficient frontier: {e}")
                        st.info("This may happen with limited data or extreme market conditions. Try selecting different assets or a longer time period.")
                        
                        # Show debug information
                        st.error("Debug Information:")
                        st.error(f"Selected tickers: {selected_tickers}")
                        st.error(f"Returns shape: {returns.shape}")
                        st.error(f"Date range: {start_date} to {end_date}")
                        
                        # Try to show a simple plot anyway
                        try:
                            fig = go.Figure()
                            fig.add_annotation(
                                text="Efficient Frontier could not be generated",
                                xref="paper", yref="paper",
                                x=0.5, y=0.5, showarrow=False
                            )
                            fig.update_layout(
                                title="Efficient Frontier (Error)",
                                xaxis_title="Annual Volatility",
                                yaxis_title="Annual Return",
                                height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        except:
                            st.error("Could not display any chart.")
    
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

