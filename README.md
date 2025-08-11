# Interactive Portfolio Analysis & Optimization Dashboard

## Professional Portfolio Analysis & Optimization Tool

A comprehensive, interactive web application for building, analyzing, and optimizing investment portfolios using modern portfolio theory and real-time market data.

## Features

### Core Functionality
- **Portfolio Builder**: Manually construct portfolios with interactive weight sliders
- **Portfolio Optimization**: Advanced optimization using Modern Portfolio Theory (MPT)
- **Risk Analysis**: Comprehensive risk assessment with stress testing
- **Correlation Analysis**: Asset correlation visualization and diversification insights
- **Real-time Data**: Live market data from Yahoo Finance

### Advanced Capabilities
- **Multiple Optimization Goals**: Maximize Sharpe ratio, minimize volatility, maximize return, or target specific returns
- **Efficient Frontier Visualization**: Interactive charts showing optimal risk-return combinations
- **Capital Allocation Line**: Tangent line analysis for optimal risk-free asset combination
- **Historical Crisis Testing**: Stress test portfolios against major market events
- **Rolling Analysis**: Dynamic volatility and correlation analysis over time

### Data Coverage
- **Stocks**: Major tech companies (AAPL, MSFT, GOOG, AMZN, NVDA, META, TSLA, AMD, CRM, NFLX)
- **ETFs**: SPY, GLD, TLT, VNQ
- **Cryptocurrencies**: Bitcoin (BTC-USD), Ethereum (ETH-USD)
- **Historical Data**: Available from 1980 to present for comprehensive analysis

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Steps
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Interactive-Portfolio-Analysis-Optimization-Dashboard
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```

3. Activate the virtual environment:
   - **Windows**: `.venv\Scripts\activate`
   - **macOS/Linux**: `source .venv/bin/activate`

4. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

5. Run the application:
   ```bash
   streamlit run app.py
   ```

## How to Use

### Getting Started
1. **Select Assets**: Choose from the curated list of stocks, ETFs, and cryptocurrencies in the sidebar
2. **Set Date Range**: Pick your analysis period (data available from 1980 to present)
3. **Build Portfolio**: Use the Portfolio Builder tab to set your asset weights
4. **Optimize**: Use the Optimization tab to find optimal allocations based on your goals
5. **Analyze Risk**: Test your portfolio against historical crisis scenarios

### Portfolio Builder Tab
- **Weight Sliders**: Adjust allocation percentages for each asset
- **Performance Metrics**: View annual return, volatility, Sharpe ratio, and maximum drawdown
- **Interactive Charts**: Portfolio growth and individual asset performance visualization
- **Optimized Weights**: Apply weights from the optimization tab directly

### Optimization Tab
- **Multiple Goals**: Choose between maximizing Sharpe ratio, minimizing volatility, maximizing return, or targeting specific returns
- **Efficient Frontier**: Visual representation of optimal risk-return combinations
- **Capital Allocation Line**: Tangent line analysis for optimal risk-free asset combination
- **Weight Application**: One-click application of optimized weights to the Portfolio Builder

### Risk Analysis Tab
- **Stress Testing**: Test portfolios against historical crisis periods
- **Custom Periods**: Define your own stress test scenarios
- **Crisis Metrics**: Performance analysis during market downturns
- **Rolling Volatility**: Dynamic risk assessment over time

### Correlation Analysis Tab
- **Correlation Matrix**: Heatmap visualization of asset relationships
- **Diversification Score**: Quantitative measure of portfolio diversification
- **Rolling Correlations**: Dynamic correlation analysis over time


## Technical Details

### Built With
- **Streamlit**: Interactive web application framework
- **PyPortfolioOpt**: Modern portfolio theory implementation
- **Yahoo Finance**: Real-time market data
- **Plotly**: Interactive financial charts
- **Pandas & NumPy**: Data manipulation and numerical computations

### Key Algorithms
- **Modern Portfolio Theory (MPT)**: Risk-return optimization framework
- **Efficient Frontier**: Optimal portfolio combinations
- **Sharpe Ratio**: Risk-adjusted return measurement
- **Correlation Analysis**: Diversification assessment
- **Stress Testing**: Historical crisis scenario analysis

### Data Processing
- **Real-time Fetching**: Live market data updates
- **Data Cleaning**: Handling of missing values and outliers
- **Return Calculation**: Log returns for accurate financial analysis
- **Risk Metrics**: Professional-grade portfolio risk assessment

## Important Notes

### Data Limitations
- Historical data availability varies by asset
- Some assets have limited data before certain dates
- Data quality depends on Yahoo Finance availability

### Optimization Considerations
- Past performance doesn't guarantee future results
- Optimization results are based on historical data
- Consider transaction costs and tax implications
- Regular rebalancing may be necessary

### Risk Warnings
- Investment involves risk of loss
- This tool is for educational and analysis purposes
- Consult with financial advisors for investment decisions
- Diversification doesn't eliminate investment risk

## Advanced Features

### Efficient Frontier Analysis
- **Frontier Generation**: 100-point efficient frontier calculation
- **Asset Positioning**: Individual asset placement on risk-return spectrum
- **Optimal Portfolio**: Highlighted optimal portfolio point
- **Tangent Line**: Capital allocation line for risk-free asset combination

### Stress Testing Scenarios
- **1987 Black Monday**: October-December 1987 market crash
- **2000 Dot-Com Bubble**: March 2000-October 2002 tech bubble burst
- **2008 Financial Crisis**: September 2008-March 2009 global crisis
- **2011 Euro Crisis**: July-October 2011 European debt crisis
- **2018 Q4 Selloff**: October-December 2018 market decline
- **2020 COVID Crash**: February-April 2020 pandemic crash
- **2022 Inflation Crisis**: January-October 2022 inflation period
- **2023 Banking Crisis**: March-May 2023 regional banking crisis

### Custom Analysis
- **Custom Date Ranges**: User-defined stress test periods
- **Rolling Windows**: Configurable time periods for analysis
- **Correlation Thresholds**: Customizable diversification metrics
- **Performance Benchmarks**: Comparative analysis capabilities

## Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### Code Standards
- Follow PEP 8 style guidelines
- Add docstrings for all functions
- Include type hints where appropriate
- Write comprehensive tests

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

**Built with dedication using Streamlit, PyPortfolioOpt, and Yahoo Finance**

### Data Sources
- **Yahoo Finance**: Real-time market data and historical prices
- **Financial Libraries**: PyPortfolioOpt for portfolio optimization algorithms

### Educational Resources
- Modern Portfolio Theory principles
- Financial risk management concepts
- Investment diversification strategies

---

*This tool is designed for educational and analysis purposes. Always consult with qualified financial professionals before making investment decisions.*
