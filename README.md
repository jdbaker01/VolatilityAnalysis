# Stock Volatility Analysis

A Streamlit application for analyzing stock volatility through daily percentage returns and annualized volatility calculations.

## Features

- Calculate and display daily percentage returns
- Calculate and display daily annualized volatility
- Configurable lookback window for volatility calculations
- Interactive data visualization
- Percentage formatting for all metrics
- Default 3-year analysis period
- Streamlined single-column layout

## Setup

1. Create and activate a virtual environment:
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Unix/macOS
# venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip3 install -r requirements.txt
```

## Usage

1. Run the application:
```bash
streamlit run src/app.py
```

2. In the application:
   - Enter a stock symbol
   - Select start and end dates
   - Specify the lookback window for volatility calculation
   - View the results in both table and chart format

## Technical Details

### Dependencies

- streamlit>=1.25.0: UI framework
- pandas>=2.0.0: Data manipulation
- yfinance>=0.2.0: Stock data retrieval
- numpy>=1.24.0: Numerical computations
- plotly>=5.15.0: Interactive charts

### Project Structure

```
.
├── src/
│   ├── app.py           # Main Streamlit interface
│   ├── calculations.py  # Returns and volatility computations
│   └── data_handler.py  # Stock data retrieval and processing
├── requirements.txt     # Project dependencies
└── README.md           # This file
```

### Features

1. Data Processing
   - Stock data retrieval with error handling
   - Daily returns calculation
   - Cumulative returns calculation
   - Volatility computation with configurable window
   - Data validation and cleaning

2. User Interface
   - Simple input interface for stock symbol and dates
   - Clear data visualization
   - Interactive charts
   - Tabular data display
   - Error messaging

3. Error Handling
   - Invalid stock symbols
   - Network issues
   - Date range validation
   - Calculation errors

## Development

### Environment Requirements

- Python 3.x
- Virtual environment for dependency isolation
- VS Code (recommended IDE)

### Running Tests

The project uses pytest for testing. To run the tests:

1. Install test dependencies:
```bash
pip3 install pytest pytest-cov
```

2. Run all tests:
```bash
python3 -m pytest
```

3. Run tests with coverage report:
```bash
python3 -m pytest --cov=src --cov-report=term-missing
```

4. Run tests verbosely:
```bash
python3 -m pytest -v
```

Test Categories:
- Cache Manager: Data storage, retrieval, and validation
- Calculations: Returns, volatility, and portfolio metrics
- Correlation: Matrix calculation and formatting
- Integration: Multi-component functionality

The test suite provides 100% code coverage and validates all core functionality.

### Performance Considerations

- Yahoo Finance API rate limits
- Historical data availability
- Market hours consideration
- Efficient data caching
- Optimized calculations for large datasets
- Memory management for data processing
