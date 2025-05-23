import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import pytz
try:
    # Try relative imports first (for local development)
    from .logger import logger
    from .data_handler import get_multiple_stocks_data
    from .calculations import (
        calculate_daily_returns,
        calculate_cumulative_returns,
        calculate_volatility,
        calculate_portfolio_returns,
        calculate_portfolio_volatility
    )
    from .correlation import calculate_correlation_matrix, format_correlation_matrix
except ImportError:
    # Fall back to absolute imports (for Streamlit Cloud)
    from src.logger import logger
    from src.data_handler import get_multiple_stocks_data
    from src.calculations import (
        calculate_daily_returns,
        calculate_cumulative_returns,
        calculate_volatility,
        calculate_portfolio_returns,
        calculate_portfolio_volatility
    )
    from src.correlation import calculate_correlation_matrix, format_correlation_matrix

def main():
    st.title("Stock Volatility Analysis")
    
    # Input section
    st.sidebar.header("Input Parameters")
    symbols_input = st.sidebar.text_input(
        "Stock Symbols (comma-separated)", 
        "AAPL,MSFT,GOOGL",
        help="Enter stock symbols separated by commas (e.g., AAPL,MSFT,GOOGL)"
    )
    
    # Default dates
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=3*365)  # 3 years ago
    
    start_date = st.sidebar.date_input("Start Date", value=start_date)
    end_date = st.sidebar.date_input("End Date", value=end_date)
    lookback_window = st.sidebar.slider(
        "Volatility Lookback Window (Trading Days)", 
        min_value=5, 
        max_value=252, 
        value=21,
        help="Number of trading days to use for volatility calculation"
    )
    
    if st.sidebar.button("Analyze"):
        try:
            try:
                # Parse symbols
                symbols = [s.strip() for s in symbols_input.split(',') if s.strip()]
                if not symbols:
                    logger.warning("No stock symbols provided")
                    raise ValueError("Please enter at least one stock symbol")
                logger.info(f"Analyzing symbols: {', '.join(symbols)}")
                
                # Convert dates to datetime with UTC timezone
                start_datetime = pytz.UTC.localize(datetime.combine(start_date, datetime.min.time()))
                end_datetime = pytz.UTC.localize(datetime.combine(end_date, datetime.min.time()))
                
                # Add one day to end_datetime to include the end date in the results
                end_datetime = end_datetime + timedelta(days=1)
                
                with st.spinner('Fetching stock data...'):
                    logger.info(f"Fetching data from {start_datetime} to {end_datetime}")
                    # Get stock data for all symbols
                    stock_data = get_multiple_stocks_data(symbols, start_datetime, end_datetime)
                    
                logger.info(f"Successfully retrieved data for {len(stock_data)} symbol(s)")
                st.success(f"Successfully retrieved data for {len(stock_data)} symbol(s)")
            except ValueError as e:
                logger.error(f"Error during data retrieval: {str(e)}")
                st.error(str(e))
                return
            
            # Calculate individual stock metrics
            stock_returns = {}
            stock_cumulative = {}
            stock_volatility = {}
            
            for symbol, df in stock_data.items():
                returns = calculate_daily_returns(df)
                stock_returns[symbol] = returns
                stock_cumulative[symbol] = calculate_cumulative_returns(returns) * 100  # Convert to percentage
                stock_volatility[symbol] = calculate_volatility(returns, window=lookback_window) * 100  # Convert to percentage
            
            # Calculate portfolio metrics
            portfolio_returns = calculate_portfolio_returns(stock_returns)
            portfolio_cumulative = calculate_cumulative_returns(portfolio_returns) * 100
            portfolio_volatility = calculate_portfolio_volatility(stock_returns, window=lookback_window) * 100
            
            # Display results in tabs
            st.header("Analysis Results")
            tabs = st.tabs(symbols + ["Portfolio", "Correlation"])
            
            for i, symbol in enumerate(symbols):
                with tabs[i]:
                    # Time period charts for cumulative returns in 2x2 grid
                    st.subheader(f"{symbol} Cumulative Returns (%)")
                    
                    periods = {
                        "1 Week": 7,
                        "1 Month": 30,
                        "3 Months": 90,
                        "1 Year": 365
                    }
                    
                    col1, col2 = st.columns(2)
                    cols = [col1, col2, col1, col2]  # Reuse columns for 2x2 grid
                    
                    for i, (period_name, days) in enumerate(periods.items()):
                        period_start = stock_cumulative[symbol].index[-1] - pd.Timedelta(days=days)
                        period_data = stock_cumulative[symbol][stock_cumulative[symbol].index >= period_start]
                        
                        # Calculate returns relative to start (starting from 0)
                        period_data = period_data - period_data.iloc[0]
                        
                        with cols[i]:
                            fig = px.line(
                                period_data,
                                title=f"{period_name}",
                                labels={"value": "Return (%)", "date": "Date"},
                                height=250  # Smaller height for grid layout
                            )
                            fig.update_layout(
                                showlegend=False,
                                margin=dict(l=40, r=20, t=30, b=20)  # Compact margins
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    st.subheader(f"{symbol} Annualized Volatility (%) - {lookback_window} Day Window")
                    st.line_chart(stock_volatility[symbol])
                    
                    st.subheader(f"{symbol} Data Table")
                    results_df = pd.DataFrame({
                        'Cumulative Returns (%)': stock_cumulative[symbol],
                        'Annualized Volatility (%)': stock_volatility[symbol]
                    })
                    st.dataframe(results_df.round(2))
            
            # Display portfolio results
            with tabs[-2]:
                st.subheader("Portfolio Cumulative Returns (%)")
                st.line_chart(portfolio_cumulative)
                
                st.subheader(f"Portfolio Annualized Volatility (%) - {lookback_window} Day Window")
                st.line_chart(portfolio_volatility)
                
                st.subheader("Portfolio Data Table")
                portfolio_df = pd.DataFrame({
                    'Cumulative Returns (%)': portfolio_cumulative,
                    'Annualized Volatility (%)': portfolio_volatility
                })
                st.dataframe(portfolio_df.round(2))
                
            # Display correlation matrix
            with tabs[-1]:
                st.subheader("Correlation Matrix")
                
                # Calculate correlation matrix
                correlation_matrix = calculate_correlation_matrix(stock_returns)
                
                # Create heatmap using plotly
                fig = px.imshow(
                    correlation_matrix,
                    labels=dict(color="Correlation"),
                    color_continuous_scale="RdBu",
                    aspect="auto"
                )
                fig.update_layout(
                    title="Stock Returns Correlation Heatmap",
                    xaxis_title="Stock Symbol",
                    yaxis_title="Stock Symbol"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Display formatted correlation matrix
                st.subheader("Correlation Matrix Table")
                formatted_matrix = format_correlation_matrix(correlation_matrix)
                st.dataframe(formatted_matrix)
            
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}", exc_info=True)
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
