BilgeBot — ALL-IN Final (Master-Class Edition)

A professional-grade, multi-process Bilge bot powered by a hybrid MCTS-deep learning AI,
with a high-performance vision pipeline and a comprehensive data collection suite.
---

Table of Contents
1. Quick Start
2. Directory Structure
3. Core Features
4. Installation
5. Configuration (`config.json`)
6. Calibration Guide
7. Troubleshooting
8. Advanced Usage

---

1. Quick Start

This section will get the bot up and running with a basic configuration.
PowerShell:
  # The first time you run this, you must bypass the execution policy.
Set-ExecutionPolicy Bypass -Scope Process -Force
  
  # Create a virtual environment to manage dependencies.
py -3 -m venv .venv;
  
  # Activate the virtual environment.
. .\.venv\Scripts\Activate.ps1
  
  # Upgrade pip and install all required libraries.
python -m pip install --upgrade pip
pip install -r requirements.txt
  
  # Run the main bot application.
python .\main.py
  
2. Directory Structure

The Master-Class Edition is meticulously organized into logical folders for a clean, maintainable codebase.
* `ai/`: Contains all the core AI components, including the MCTS engine, scoring logic, and the Numba-optimized game engine.
* `control/`: Houses the low-level Windows input controller for human-like mouse movements and clicks.
* `core/`: Contains core utilities, including the robust configuration manager and the new model management system.
* `dataset/`: Storage for all collected and augmented data, vital for training new models.
* `exporters/`: Scripts for optimizing and exporting trained models to performant formats like ONNX.
* `labeling/`: Tools for collecting and annotating gameplay data for supervised learning.
* `models/`: The central repository for all trained AI models (`.pt`, `.onnx`).
* `policy/`: Defines the neural network architecture and inference logic for the hybrid AI engine.
* `tools/`: Command-line tools for testing and debugging, such as the click accuracy tester.
* `training/`: Scripts for training the AI models from scratch using collected data.
* `ui/`: Contains the graphical user interface for the bot's control panel and calibration suite.
* `utils/`: General-purpose utilities, including the high-performance screen capture engine.
* `vision/`: The entire vision pipeline, from board detection and OCR to gem perception and water level analysis.
* `win/`: Windows-specific API helpers for robust interaction with the operating system.

3. Core Features

The Master-Class Edition is a complete re-engineering of the original bot, focusing on performance, stability, and intelligence.
* **Multi-Process Architecture**: Isolates the `tkinter` UI  from the core logic, ensuring a smooth, non-blocking user experience.
* **Hybrid AI Engine**: Combines a fast, deep learning-guided MCTS (Monte Carlo Tree Search) with a robust scoring engine to find superhuman moves.
* **High-Performance Vision**: Uses a multi-tiered vision pipeline with hardware-accelerated deep learning (YOLO) and Numba-optimized fallbacks for lightning-fast board and gem detection.
* **Low-Latency Input**: Employs a master-class input engine with low-level Windows API calls to ensure precise, human-like clicks and swaps.
* **Automated Calibration**: Features a comprehensive, one-click calibration suite for setting up all game regions.
* **Data-Driven Development**: Includes a full suite of tools for collecting human and bot gameplay data, enabling continuous improvement of the AI models.

4. Installation

The primary dependency for this project is a Python 3.8+ environment.
* **GPU Drivers**: Ensure you have the latest drivers for your AMD Radeon RX 570 GPU. The bot will automatically use `torch-directml` for hardware acceleration.
* **Virtual Environment**: Always run the bot in a virtual environment to prevent dependency conflicts[cite: 38, 41].
    `py -3 -m venv .venv`
* **Dependencies**: Install the required libraries using the `requirements.txt` file[cite: 42].
    `pip install -r requirements.txt`

5. Configuration (`config.json`)

The bot's behavior is controlled by the `config.json` file. The `config_io.py` module handles its loading, validation, and migration.
* `version`: The version of the configuration file.
* `click`: Defines mouse click behavior, including offsets and scaling to account for different screen resolutions.
* `window`: Specifies the target window to bind to, using the title and window handle (`hwnd`).
* `runtime`: General bot settings, such as `fps`, `background_sims` for the simulator, and whether to `auto_play`.
* `roi`: The regions of interest (ROIs) for the vision pipeline, including the board, HUD, and other panels.
* `paths`: File paths for outputs and models.

6. Calibration Guide

The Master-Class Edition features a dedicated calibration tool for a fast and accurate setup.
1.  **Launch the Calibration UI**: Run `python .\ui\calibration_ui.py` to open the standalone calibration suite.
2.  **Bind to the Client**: In the main bot UI, enter a part of your Puzzle Pirates window title (e.g., "Puzzle Pirates") and click "Bind Client".
3.  **Auto-Detect**: In the calibration suite, click "Auto-Detect All". The bot will search for all key regions.
4.  **Review and Save**: The live video feed will show overlays of the detected regions. If they look correct, click "Save & Exit" to save the configuration.
5.  **Test Clicks**: Use the "Test Clicks" button on the main bot UI to confirm that the bot can accurately click on tiles.

7. Troubleshooting

* **Bot is not moving**: Ensure your calibration is correct and that the "Auto Play" option is enabled in the configuration[cite: 49].
* **A module has crashed**: This bot uses a multi-process architecture. Check the main console window for a "Crashed" status message. The bot's core logic is run in a separate process, so if it crashes, the UI will remain responsive. Check the log files in the `dataset/telemetry` folder for a detailed stack trace.
* **Calibration is inaccurate**: Check that your game client is at its default zoom level. If the auto-detection fails, try manually calibrating by defining the `roi` coordinates in `config.json`.
* **Performance is slow**: Ensure the bot is using your GPU by checking the logs for a message about "DirectML acceleration"[cite: 54, 55]. If not, ensure your GPU drivers are up to date and that the correct version of `torch-directml` is installed.

8. Advanced Usage

* **Data Collection**: Use the `labeling/recorder.py` and `labeling/annotator.py` scripts to collect new gameplay data for training[cite: 56].
* **Training**: Use the `training/train_yolo.py` and `training/train_policy.py` scripts to train new AI models[cite: 57].
* **Exporting**: Use the `exporters/yolo_export.py` script to export your trained models for optimal performance on your hardware[cite: 58].