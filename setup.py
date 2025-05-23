from setuptools import setup, find_packages

setup(
    name="volatility_analysis",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "pandas",
        "numpy",
        "plotly",
        "yfinance",
        "pytz"
    ],
)
