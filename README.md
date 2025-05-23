# ProcessWallPressureSignal

**ProcessWallPressureSignal** is a Python toolkit for processing experimental **wall pressure signal** data. It helps researchers and engineers analyse pressure measurements recorded on walls (e.g., wind tunnel models or pipe walls) with minimal coding required. The repository is designed to be accessible for experimentalists who may not be highly familiar with programming, providing an easy way to load data, perform common analyses, and obtain results.

## Introduction

Wall pressure signals are time-series data representing pressure fluctuations on a surface during an experiment. This tool streamlines the analysis of such data by automating tasks like data loading, filtering, and basic statistical or spectral analysis. The **purpose of this repository** is to enable you to go from raw pressure data to meaningful results with just a few simple steps. By following this guide, you can install the module, prepare your input files, and run the analysis without needing extensive coding knowledge.

## Installation

Before using the toolkit, you need to install it and its dependencies. Make sure you have **Python 3** (and the package manager **pip**) available on your system. You can install the `ProcessWallPressureSignal` package directly from the source. There are two main ways to do this:

- **Clone or Download the Repository:** If you have Git installed, you can clone the repository using:  
  ```bash
  git clone https://github.com/J-Massey/ProcessWallPressureSignal.git
  ```
  Alternatively, download the repository as a ZIP file from the project page (e.g., via the "Code" > "Download ZIP" option) and unzip it on your computer.

- **Install via pip:** Once you have the code, navigate to the repository folder in your terminal or command prompt. Then run:  
  ```bash
  pip install .
  ```  
  This will install the package and all required dependencies. If you want to install in **editable** (development) mode so that changes to your code take effect without reinstallation, use:
  ```bash
  pip install -e .
  ```

## Input Data Requirements

The analysis expects your experimental data to be provided as **MATLAB `.mat` files** containing the wall pressure signal data. Typically, a `.mat` file might include arrays or variables for time and pressure (or pressure fluctuations). Before running the analysis, you will need to tell the program where to find your data file(s).

**Preparing your data:** Make sure you have your wall pressure data saved in a `.mat` file (or multiple files, if you have several runs). Take note of the file path to this data on your computer. If your data is currently in another format (CSV, text, etc.), you may need to convert it to `.mat` or adapt the code (this toolkit is primarily set up for `.mat` files).

**Configuring the file paths:** Open the script `src/run.py` in a text editor.
At the top of the file you will find two variables defining the locations of
the wall-pressure and free-stream pressure data:

```python
# Inside src/run.py
WALL_PRESSURE_MAT = "data/wallpressure_booman_Pa.mat"
FREESTREAM_PRESSURE_MAT = "data/booman_wallpressure_fspressure_650sec_40khz.mat"
```

Replace these with the paths to your own `.mat` files, for example:

```python
WALL_PRESSURE_MAT = "C:/Users/YourName/Experiment/wall.mat"
FREESTREAM_PRESSURE_MAT = "C:/Users/YourName/Experiment/free_stream.mat"
```

## Usage

Once you have installed the package and set up your input data paths in `src/run.py`, you can run the analysis. The usage steps are outlined below:

1. **Install the package** â€“ (If you haven't done this yet, see the [Installation](#installation) section above to set up `ProcessWallPressureSignal` using pip.)
2. **Prepare your input data** â€“ Ensure you have your wall pressure data available in a `.mat` file. For example, `my_experiment.mat` containing the pressure time series from your test. Place it in a convenient location or within the repository folder (or note its full file path).
3. **Update the `src/run.py` file** â€“ Edit the script to point to your `.mat` files, as described above.
4. **Run the analysis script** â€“ In the terminal:
   ```bash
   python src/run.py
   ```
5. **View the results** â€“ Check console output and any saved figures or processed data files in the repository directory.

## Example Scenario

Suppose you conducted an experiment and recorded wall pressure fluctuations, saving the data in files `pressure_test.mat` and `pressure_test_fs.mat`. To analyze this data with ProcessWallPressureSignal:

- After installation, open `src/run.py` and set:
  ```python
  WALL_PRESSURE_MAT = "pressure_test.mat"
  FREESTREAM_PRESSURE_MAT = "pressure_test_fs.mat"
  ```
- Run:
  ```bash
  python src/run.py
  ```
- Review the output (e.g., `spectrum.png` or `pressure_test_processed.mat`) saved in the repository folder.

## Technical Details (Beginner-Friendly)

- **Language and Libraries:** Written in Python, using libraries such as **NumPy**, **SciPy** (for .mat file handling), **Matplotlib** (for plotting), and other dependencies installed automatically via pip.
- **Editable vs Installed Usage:** Editable mode (`pip install -e .`) lets you modify code and see changes immediately. A standard install also allows running `src/run.py` directly.
- **Modifying the Analysis:** You can customize `src/run.py` or module code to adjust filters, add calculations, or change output formats.

## Troubleshooting

- **Module not found:** Ensure you ran `pip install .` in the repository root.
- **pip/python not recognized:** Use `python3`/`pip3` or add Python to your PATH.
- **.mat load errors:** Verify your file path and MATLAB version; SciPy supports most standard formats.
- **No output:** Check that `src/run.py` points to the correct path and prints status messages.

## Conclusion

**ProcessWallPressureSignal** simplifies the workflow of analyzing wall pressure data from experiments. By providing a ready-made script and modular functions, it allows you to focus on experimental results rather than coding. If you encounter issues or have suggestions, please open an issue or pull request. Happy data processing!
