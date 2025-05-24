import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
import pytz
import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import after path setup
from src.logger import logger
from src.data_handler import get_multiple_stocks_data
from src.calculations import (
    calculate_daily_returns,
    calculate_cumulative_returns,
    calculate_volatility,
    calculate_portfolio_returns,
    calculate_portfolio_volatility,
    calculate_rolling_var,
    calculate_covariance_matrix
)
from src.correlation import calculate_correlation_matrix, format_correlation_matrix

def main():
    st.title("Stock Volatility Analysis")
    
    # Input section
    st.sidebar.header("Input Parameters")
    symbols_input = st.sidebar.text_input(
        "Stock Symbols (comma-separated)", 
        "SPY,IEF,GSG",
        help="Enter stock symbols separated by commas (e.g., SPY,IEF,GSG)"
    )
    
    # Parse symbols
    symbols = [s.strip() for s in symbols_input.split(',') if s.strip()]
    if not symbols:
        st.error("Please enter at least one stock symbol")
        return
    
    # Initialize portfolio weights if not exists
    if 'portfolio_weights' not in st.session_state:
        st.session_state.portfolio_weights = {symbol: 1.0/len(symbols) for symbol in symbols}
    
    # Add portfolio weights section to sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Portfolio Weights")
    st.sidebar.write("Enter weights (%) for each asset:")
    
    # Create a form for weights
    with st.sidebar.form(key=f"portfolio_weights_{','.join(symbols)}"):
        total_weight = 0
        new_weights = {}
        
        # Create weight inputs
        for symbol in symbols:
            weight = st.number_input(
                f"{symbol} Weight (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(st.session_state.portfolio_weights[symbol] * 100),
                step=0.01,
                format="%.2f",
                key=f"weight_{symbol}_{','.join(symbols)}",
                help="Enter a number between 0 and 100"
            )
            new_weights[symbol] = weight
            total_weight += weight
        
        st.write(f"Total: {total_weight:.1f}%")
        
        # Submit button for the form
        submitted = st.form_submit_button("Apply Weights")
        if submitted:
            if np.isclose(total_weight, 100.0, rtol=1e-5):
                # Convert percentages to decimals and update session state
                st.session_state.portfolio_weights = {
                    symbol: weight / 100 
                    for symbol, weight in new_weights.items()
                }
                st.success("Weights updated. Click 'Analyze' to recalculate portfolio metrics.")
            else:
                st.warning("⚠️ Weights must sum to 100%")
    
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
            
            # Initialize portfolio weights
            if 'portfolio_weights' not in st.session_state:
                st.session_state.portfolio_weights = {symbol: 1.0/len(symbols) for symbol in symbols}
            
            # Calculate portfolio metrics with current weights
            portfolio_returns = calculate_portfolio_returns(stock_returns, st.session_state.portfolio_weights)
            portfolio_cumulative = calculate_cumulative_returns(portfolio_returns) * 100
            portfolio_volatility = calculate_portfolio_volatility(stock_returns, window=lookback_window, weights=st.session_state.portfolio_weights) * 100
            
            # Display results in tabs
            st.header("Analysis Results")
            tabs = st.tabs(symbols + ["Portfolio", "Correlation", "Value-at-Risk", "Covariance", "Portfolio Composition"])
            
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
            with tabs[len(symbols)]:
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
            with tabs[len(symbols) + 1]:
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
                
            # Display Value-at-Risk analysis
            with tabs[len(symbols) + 2]:
                st.subheader("Value-at-Risk (VaR) Analysis")
                
                # Add VaR confidence level selector
                confidence_level = st.selectbox(
                    "Confidence Level",
                    [0.90, 0.95, 0.99],
                    index=1,
                    format_func=lambda x: f"{int(x*100)}%",
                    help="Probability level for VaR calculation"
                )
                
                # Calculate VaR for each asset and portfolio
                var_results = {}
                
                for symbol, returns in stock_returns.items():
                    var = calculate_rolling_var(
                        returns,
                        window=lookback_window,
                        confidence_level=confidence_level,
                        method="historical"
                    ) * 100  # Convert to percentage
                    var_results[symbol] = var
                
                # Calculate portfolio VaR
                portfolio_var = calculate_rolling_var(
                    portfolio_returns,
                    window=lookback_window,
                    confidence_level=confidence_level,
                    method="historical"
                ) * 100  # Convert to percentage
                
                # Display Portfolio VaR first
                st.subheader("Portfolio Daily VaR (%)")
                st.line_chart(portfolio_var)
                
                # Calculate and display average VaR values
                st.subheader("Portfolio VaR Averages")
                periods = {
                    "20 Days": 20,
                    "90 Days": 90,
                    "200 Days": 200
                }
                
                avg_vars = {}
                for period_name, days in periods.items():
                    if len(portfolio_var) >= days:
                        avg_var = portfolio_var.tail(days).mean()
                        avg_vars[period_name] = avg_var
                
                avg_var_df = pd.DataFrame({
                    f"{int(confidence_level*100)}% Daily VaR Average (%)": avg_vars
                }).round(2)
                st.dataframe(avg_var_df)
                
                # Display current VaR values
                st.subheader("Current VaR Values")
                current_vars = {"Portfolio": portfolio_var.iloc[-1]}
                for symbol, var_series in var_results.items():
                    current_vars[symbol] = var_series.iloc[-1]
                
                var_df = pd.DataFrame({
                    f"{int(confidence_level*100)}% Daily VaR (%)": current_vars
                }).round(2)
                st.dataframe(var_df)
                
                # Individual assets VaR
                st.subheader("Individual Assets Daily VaR (%)")
                for symbol in stock_returns.keys():
                    st.subheader(f"{symbol}")
                    st.line_chart(var_results[symbol])
                
            # Display covariance matrix
            with tabs[len(symbols) + 3]:
                st.subheader("Covariance Matrix")
                
                # Calculate covariance matrix
                covariance_matrix = calculate_covariance_matrix(stock_returns)
                
                # Create heatmap using plotly
                fig = px.imshow(
                    covariance_matrix,
                    labels=dict(color="Covariance"),
                    color_continuous_scale="RdBu",
                    aspect="auto"
                )
                fig.update_layout(
                    title="Returns Covariance Heatmap",
                    xaxis_title="Stock Symbol",
                    yaxis_title="Stock Symbol"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Display covariance matrix table
                st.subheader("Covariance Matrix Table (%)")
                # Convert to percentage and round to 4 decimal places
                covariance_matrix_pct = (covariance_matrix * 100).round(4)
                st.dataframe(covariance_matrix_pct)
                
                # Update heatmap to show percentages
                fig = px.imshow(
                    covariance_matrix_pct,
                    labels=dict(color="Covariance (%)"),
                    color_continuous_scale="RdBu",
                    aspect="auto"
                )
                fig.update_layout(
                    title="Returns Covariance Heatmap (%)",
                    xaxis_title="Stock Symbol",
                    yaxis_title="Stock Symbol"
                )
                
            # Display Portfolio Composition tab
            with tabs[len(symbols) + 4]:
                st.subheader("Portfolio Composition")
                st.write("Portfolio weights can be adjusted in the sidebar. Current weights:")
                
                # Display current weights in a table
                weights_df = pd.DataFrame({
                    'Symbol': list(st.session_state.portfolio_weights.keys()),
                    'Weight (%)': [f"{w * 100:.2f}" for w in st.session_state.portfolio_weights.values()]
                })
                st.dataframe(weights_df)
    

if __name__ == "__main__":
    main()
