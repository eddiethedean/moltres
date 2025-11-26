# Moltres Demo Notebooks

This directory contains demonstration notebooks showcasing Moltres capabilities with realistic, end-to-end workflows.

## Available Notebooks

### `ecommerce_analytics_demo.ipynb`

A comprehensive e-commerce analytics demo that demonstrates:

- **Database Setup**: Creating tables with proper schemas using DuckDB
- **Data Generation**: Realistic sample data for customers, products, orders, and order items
- **Basic Queries**: Filtering, selecting, and aggregating data
- **Joins**: Combining data from multiple tables
- **Revenue Analysis**: Calculating totals, revenue by category, and top-selling products
- **Customer Analytics**: Customer Lifetime Value (CLV), segmentation, and behavior analysis
- **Time Series Analysis**: Monthly revenue trends and day-of-week patterns
- **Profitability Analysis**: Margin calculations and profit analysis
- **Geographic Analysis**: Sales by location (state-level)
- **Advanced Analytics**: Basket analysis for product recommendations
- **Data Export**: Exporting results to CSV
- **Visualization**: Creating charts and dashboards (requires pandas/matplotlib)

## Requirements

To run the notebooks, you'll need:

```bash
# Core dependencies
pip install moltres

# For the e-commerce demo
pip install duckdb-engine

# Optional: For visualization
pip install pandas matplotlib seaborn
```

## Running the Notebooks

1. Start Jupyter:
   ```bash
   jupyter notebook
   ```

2. Navigate to the `notebooks/` directory

3. Open and run `ecommerce_analytics_demo.ipynb`

## Output Files

The notebooks may generate output files (CSV exports, visualizations) in this directory. These are ignored by git (see `.gitignore`).

## Notes

- The e-commerce demo uses DuckDB for in-memory analytics, making it fast and perfect for demonstrations
- All data is generated programmatically, so results will vary slightly each run (but are seeded for reproducibility)
- The demo showcases real-world analytics patterns commonly used in business intelligence and data science

