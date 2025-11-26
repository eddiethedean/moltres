# Jupyter Kernel Setup for Moltres Notebooks

A Jupyter kernel named **"Python (Moltres Demo)"** has been created and configured with all necessary dependencies.

## Kernel Information

- **Kernel Name**: `moltres-demo`
- **Display Name**: `Python (Moltres Demo)`
- **Location**: `~/Library/Jupyter/kernels/moltres-demo/`

## Installed Dependencies

The kernel includes all packages needed to run the e-commerce analytics demo:

- ✅ **moltres** (0.16.0) - Installed in editable mode
- ✅ **duckdb-engine** - For DuckDB database connections
- ✅ **pandas** - For data manipulation and visualization
- ✅ **matplotlib** - For plotting charts
- ✅ **seaborn** - For enhanced visualizations
- ✅ **jupyter** - Jupyter notebook environment
- ✅ **ipykernel** - Jupyter kernel support

## Using the Kernel

1. **Start Jupyter**:
   ```bash
   jupyter notebook
   # or
   jupyter lab
   ```

2. **Open the notebook**: `notebooks/ecommerce_analytics_demo.ipynb`

3. **Select the kernel**: 
   - Click on "Kernel" → "Change Kernel" → "Python (Moltres Demo)"
   - Or use the kernel selector in the top-right corner

## Verifying the Setup

You can verify all dependencies are available by running:

```python
import moltres
import duckdb_engine
import pandas
import matplotlib
import seaborn

print(f"Moltres version: {moltres.__version__}")
print("✅ All dependencies are available!")
```

## Troubleshooting

If you encounter issues:

1. **Kernel not appearing**: 
   ```bash
   jupyter kernelspec list
   ```
   Should show `moltres-demo` in the list.

2. **Import errors**: Make sure moltres is installed in editable mode:
   ```bash
   pip install -e .
   ```

3. **Missing dependencies**: Reinstall all dependencies:
   ```bash
   pip install ipykernel jupyter duckdb-engine pandas matplotlib seaborn
   pip install -e .
   ```

## Updating the Kernel

If you need to update the kernel or reinstall dependencies:

```bash
# Reinstall dependencies
pip install --upgrade ipykernel jupyter duckdb-engine pandas matplotlib seaborn
pip install -e .

# Reinstall the kernel
python -m ipykernel install --user --name moltres-demo --display-name "Python (Moltres Demo)" --force
```

