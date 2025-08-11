# Interactive Portfolio Analysis & Optimization Dashboard

A comprehensive, interactive web application for building, analyzing, and optimizing investment portfolios using modern portfolio theory and real-time market data.

## 🚀 Features

### 📊 Portfolio Builder
- **Manual Portfolio Construction**: Set custom asset weights and allocations
- **Real-time Performance Metrics**: Track returns, volatility, Sharpe ratio, and maximum drawdown
- **Interactive Charts**: Beautiful Plotly visualizations for portfolio growth and individual asset performance
- **Historical Price Analysis**: View asset price history and performance over time

### 🎯 Portfolio Optimization
- **Multiple Optimization Strategies**:
  - Maximum Sharpe Ratio (optimal risk-adjusted returns)
  - Minimum Volatility (lowest risk portfolio)
  - Maximum Return (highest expected returns)
  - Custom Return Target (user-defined return objectives)
- **Efficient Frontier Visualization**: Interactive charts showing optimal portfolio combinations
- **Capital Allocation Line**: Tangent line analysis for optimal risk-return trade-offs
- **One-Click Application**: Apply optimized weights directly to portfolio builder

### 🛡️ Risk Analysis
- **Stress Testing**: Analyze portfolio performance during historical crisis periods:
  - 1987 Black Monday
  - 2000 Dot-Com Bubble
  - 2008 Financial Crisis
  - 2011 Euro Crisis
  - 2018 Q4 Selloff
  - 2020 COVID-19 Crash
  - 2022 Inflation Crisis
  - 2023 Banking Crisis
- **Custom Period Analysis**: Define your own stress test periods
- **Rolling Volatility**: Dynamic volatility analysis with customizable lookback periods
- **Drawdown Analysis**: Comprehensive loss analysis during crisis scenarios

### 🔗 Correlation Analysis
- **Asset Correlation Matrix**: Interactive heatmap showing relationships between assets
- **Diversification Scoring**: Quantitative measures of portfolio diversification
- **Rolling Correlation**: Time-varying correlation analysis between asset pairs
- **Diversification Insights**: Identify concentration risks and diversification opportunities

## 🛠️ Technology Stack

- **Frontend**: Streamlit (Python web framework)
- **Data Processing**: Pandas, NumPy
- **Market Data**: Yahoo Finance (yfinance)
- **Visualization**: Plotly, Seaborn, Matplotlib
- **Portfolio Optimization**: PyPortfolioOpt (Modern Portfolio Theory implementation)

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "Interactive Portfolio Analysis & Optimization Dashboard"
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:8501` to access the dashboard

## 🎮 Usage Guide

### Getting Started

1. **Select Assets**: Choose from the sidebar - stocks, ETFs, bonds, commodities, and crypto
2. **Set Date Range**: Pick your analysis period (data available from 1980 to present)
3. **Build Portfolio**: Either manually set weights or use optimization to find optimal allocations
4. **Analyze Results**: Review performance metrics, risk measures, and visualizations

### Available Assets

#### Major Tech Stocks
- **AAPL** (Apple) - Available from 1980
- **MSFT** (Microsoft) - Available from 1986
- **GOOG** (Alphabet) - Available from 2004
- **AMZN** (Amazon) - Available from 1997
- **NVDA** (NVIDIA) - Available from 1999
- **META** (Meta) - Available from 2012
- **TSLA** (Tesla) - Available from 2010
- **AMD** (Advanced Micro Devices) - Available from 1980
- **CRM** (Salesforce) - Available from 2004
- **NFLX** (Netflix) - Available from 2002

#### ETFs & Bonds
- **SPY** (S&P 500 ETF) - Available from 1993
- **GLD** (Gold ETF) - Available from 2004
- **TLT** (20+ Year Treasury Bond ETF) - Available from 2002
- **VNQ** (Real Estate ETF) - Available from 2004

#### Cryptocurrencies
- **BTC-USD** (Bitcoin) - Available from 2014
- **ETH-USD** (Ethereum) - Available from 2017

### Pro Tips

- **Start Simple**: Begin with a few diverse assets and gradually add complexity
- **Use Optimization**: Find optimal weights first, then apply them to your portfolio builder
- **Monitor Correlations**: Aim for lower average correlations for better diversification
- **Test Different Periods**: Analyze performance across various market conditions
- **Stress Test**: Use the risk analysis tab to understand crisis performance

## 📊 Key Metrics Explained

### Performance Metrics
- **Annual Return**: Total return over a year, annualized
- **Annual Volatility**: Standard deviation of returns, annualized
- **Sharpe Ratio**: Risk-adjusted return measure (higher is better)
- **Maximum Drawdown**: Largest peak-to-trough decline

### Risk Metrics
- **Value at Risk (VaR)**: Potential loss at a given confidence level
- **Rolling Volatility**: Time-varying risk measure
- **Crisis Performance**: How portfolio performs during market stress

### Diversification Metrics
- **Average Correlation**: Mean correlation between all asset pairs
- **Diversification Score**: Higher score indicates better diversification

## 🔧 Configuration

### Customization Options
- **Risk-Free Rate**: Adjustable for Sharpe ratio calculations (default: 2%)
- **Volatility Lookback**: Customizable rolling volatility periods (20-252 days)
- **Custom Stress Test Periods**: Define your own crisis analysis windows
- **Optimization Constraints**: Set return targets and risk limits

### Data Settings
- **Date Range**: Flexible historical analysis from 1980 to present
- **Asset Selection**: Mix and match from 20+ available assets
- **Data Quality**: Automatic handling of missing data and failed tickers

## 🚨 Troubleshooting

### Common Issues

1. **Data Fetching Errors**
   - Check internet connection
   - Verify ticker symbols are correct
   - Ensure date range is within asset availability

2. **Optimization Failures**
   - Ensure sufficient historical data (minimum 30 days)
   - Check for extreme return values
   - Try different optimization objectives

3. **Performance Issues**
   - Reduce number of selected assets
   - Shorten analysis period
   - Clear browser cache

### Error Messages
- **"No valid data available"**: Check asset selection and date range
- **"Insufficient data"**: Extend analysis period or select different assets
- **"Optimization error"**: Try alternative optimization goals

## 📈 Use Cases

### Individual Investors
- Portfolio construction and rebalancing
- Risk assessment and management
- Performance tracking and analysis

### Financial Analysts
- Asset allocation research
- Risk modeling and stress testing
- Correlation analysis and diversification studies

### Educational Purposes
- Learning modern portfolio theory
- Understanding risk-return relationships
- Practicing portfolio optimization techniques

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Abdul Ahad Ali Khan**
- Email: abdulahadalikhan12@gmail.com
- GitHub: [Your GitHub Profile]

## 🙏 Acknowledgments

- **PyPortfolioOpt**: For portfolio optimization algorithms
- **Yahoo Finance**: For real-time market data
- **Streamlit**: For the web application framework
- **Plotly**: For interactive visualizations

## 📞 Support

If you encounter any issues or have questions:
1. Check the troubleshooting section above
2. Review the error messages in the application
3. Contact the author at abdulahadalikhan12@gmail.com

---

**Disclaimer**: This tool is for educational and informational purposes only. It does not constitute financial advice. Always consult with qualified financial professionals before making investment decisions. Past performance does not guarantee future results.
