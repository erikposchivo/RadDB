# RadDB Examples

This directory contains example scripts demonstrating how to use the RadDB package.

## Examples

### 1. basic_usage.py

**For beginners** - A straightforward example showing the essential workflow:

- Initialize RadDB
- Process and store radar data
- Load and visualize results

Start with this example if you're new to RadDB. It uses minimal settings and clearly explains each step.

### 2. advanced_usage.py

**For advanced users** - A comprehensive example showing:

- LUT (Look-Up Table) generation
- Processing with all optional features (HZT, visibility correction, hydrometeor classification)
- Multiple visualization types (PPI, RHI, volume panels, classification plots)
- Low-level API for custom workflows
- DataFrame operations and spatial data analysis

Use this example when you need more control or want to explore advanced features.

## Running the Examples

1. **Configure paths**: Edit the configuration section in each example to match your system:
   - `BASE_PATH` - Output directory for processed data
   - `RAW_DATA_DIR` - Input directory with raw radar files
   - `RADAR_NAME` - Radar identifier (e.g., "A", "MLA", "D", etc.)

2. **Run the example**:
   ```bash
   python basic_usage.py
   ```
   or
   ```bash
   python advanced_usage.py
   ```

## Notes

- **Radar name format**: Both "MLA" and "A" formats are supported and automatically normalized
- **Optional features**: Disable features (like HZT, visibility) if you don't have the required data files
- **Data requirements**: You need METRANET format radar files to run these examples
- **LUT generation**: Run the LUT generation step (in advanced_usage.py) once per radar before processing data

## Troubleshooting

If you encounter errors:

1. Check that all paths in the configuration section are correct
2. Verify that radar data files exist for the specified time range
3. Make sure optional features are disabled if you don't have the required auxiliary data
4. Check the verbose output for detailed error messages

For more help, see the main package documentation.
