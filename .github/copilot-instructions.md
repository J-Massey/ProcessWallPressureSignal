# GitHub Copilot Instructions

This document provides guidance for AI agents to effectively contribute to the `ProcessWallPressureSignal` codebase.

## 1. Project Overview

This is a Python-based toolkit for processing experimental wall pressure signals. The primary goal is to analyze time-series data from `.mat` files, apply various signal processing techniques to remove noise and artifacts, and produce a corrected wall pressure spectrum. The project is heavily based on `torch` for numerical computations and `scipy` for signal processing tasks.

The main entry point for the processing pipeline is `src/run.py`.

## 2. Core Architecture

The architecture revolves around the `WallPressureProcessor` class, located in `src/processor.py`. This class encapsulates the entire processing pipeline.

### Key Modules:

-   `src/run.py`: The main script that orchestrates the analysis. It defines physical parameters, file paths, and calls the processing steps.
-   `src/processor.py`: Contains the central `WallPressureProcessor` class. This is where the main logic for data loading, filtering, and analysis resides.
-   `src/processing.py`: Provides lower-level signal processing functions, such as calculating duct modes, applying notch filters, and computing power spectral densities (PSDs). Many of these functions are implemented in PyTorch.
-   `src/i_o.py`: Handles loading data from `.mat` files.
-   `src/plotting.py`: Contains functions for generating plots of the results.
-   `setup.py`: Defines project dependencies. The project is installable as a package.

### Data Flow and Processing Pipeline:

The typical processing sequence, as implemented in `WallPressureProcessor`, is as follows:

1.  **Load Data**: Wall pressure (`p_w`) and free-stream pressure (`p_fs`) signals are loaded from `.mat` files using `load_test()` or `load_data()`.
2.  **Duct Mode Calculation**: Theoretical acoustic duct mode frequencies are calculated via `compute_duct_modes()`. These are frequencies that may contaminate the signal.
3.  **Notch Filtering**: The `notch_filter()` method removes the calculated duct mode frequencies from the pressure signals.
4.  **Phase Matching**: The `phase_match()` method aligns the phase of the wall pressure signal with the free-stream signal. This is a prerequisite for effective noise cancellation.
5.  **Noise Rejection**: The `reject_free_stream_noise()` method applies a Wiener filter to subtract the free-stream noise from the wall pressure signal, resulting in a cleaned signal.
6.  **Transfer Function**: `compute_transfer_function()` is used to create a transfer function from a known reference spectrum. This function is then used to correct the measured wall pressure spectrum to account for sensor and setup effects.
7.  **Final Spectrum**: The transfer function is applied to get the final, corrected wall pressure spectrum.

## 3. Developer Workflow

### Running the Analysis

The main analysis is run from the root of the repository:

```bash
python src/run.py
```

Input data files and physical parameters are configured at the top of `src/run.py`. When modifying the workflow, this is the primary file to edit.

### Dependencies

The project uses `pip` for dependency management. To install the necessary packages, run:

```bash
pip install .
```

Or for development mode:

```bash
pip install -e .
```

The dependencies are listed in `setup.py` and include `torch`, `numpy`, `scipy`, `matplotlib`, and `scienceplots`.

### Testing

Unit tests are located in `tests/test_processor.py`. They use Python's `unittest` framework. To run the tests:

```bash
python -m unittest tests/test_processor.py
```

The tests mock several dependencies and file loading operations to isolate the functionality of the `WallPressureProcessor` class. When adding new functionality to the processor, corresponding tests should be added here.

## 4. Code Conventions

-   **PyTorch-centric**: The core processing logic in `processor.py` and `processing.py` is written using the `torch` library for performance and potential GPU acceleration. When adding new numerical algorithms, prefer a `torch`-based implementation.
-   **Configuration in `run.py`**: The main script `src/run.py` is used for high-level configuration (file paths, physical constants). Avoid hard-coding these values deep inside the processing classes.
-   **Class-based Pipeline**: The main processing logic is encapsulated within the `WallPressureProcessor` class. New processing steps should be added as methods to this class to maintain a clear and sequential pipeline.
-   **Modularity**: Utility functions (e.g., for I/O, plotting, or specific calculations) are kept in separate modules (`i_o.py`, `plotting.py`, `processing.py`).
