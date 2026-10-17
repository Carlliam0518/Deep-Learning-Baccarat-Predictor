# Deep-Learning-Baccarat-Predictor

Here is your updated, complete **`README.md`** file. It now includes step-by-step instructions for setting up a Python virtual environment (`venv`) and configuring it inside Visual Studio Code (VS Code) so all your packages run smoothly.

---

```markdown
# Baccarat AI Companion: Vision + 8-Deck Shoe Counting Engine

An end-to-end, modular deep learning computer vision and probability analytics system built with **PyTorch**, **OpenCV**, and **Tailwind CSS**. This project automatically captures live online Baccarat gameplay from your monitor screen, recognizes dealt playing cards in real time using fine-tuned transfer learning models (`ResNet50` and `MobileNetV3`), updates an 8-deck (416-card) shoe inventory, and computes live Monte Carlo Expected Value (EV) odds.

---

## Key Features

* **Modular Deep Learning Architecture:** Built across clean, reusable Python modules (`config.py`, `utils.py`, `data_loader.py`, `model_builder.py`, `trainer.py`, `evaluator.py`, `visualizer.py`).
* **Playing Card Image Classification:** Fine-tunes pre-trained CNN backbones on the 53-class playing cards dataset via KaggleHub.
* **Interactive Screen ROI Calibration (`calibrate_rois.py`):** An OpenCV click-and-drag tool that lets you map out exact screen coordinates for Player and Banker card slots.
* **Live Screen-Capture Vision Engine (`live_capture.py`):** Utilizes `mss` and `OpenCV` for low-latency screen monitoring, automatically filtering duplicate frames to count each physical card exactly once per hand.
* **8-Deck Shoe & Probability Engine (`baccarat_engine.py`):** Manages a 416-card shoe, applies exact Punto Banco third-card drawing rules, and calculates real-time win probabilities and house edge (EV).
* **Real-Time Web Dashboard (`baccarat_dashboard.html`):** A sleek, single-file Tailwind CSS dashboard that polls local JSON data to display live shoe inventories, recent card history, and optimal betting edges.

---

## Project Directory Structure

```text
baccarat-ai-companion/
│
├── config.py             # Global constants, paths, and dynamic ROI loader
├── utils.py              # Shared logging, seeding, and table display helpers
├── data_loader.py        # Kaggle download, stratified splitting, and DataLoaders
├── model_builder.py      # ResNet50 & MobileNetV3 initialization and parameter counting
├── trainer.py            # Training loops, learning rate scheduling, and checkpointing
├── evaluator.py          # Scikit-learn test metrics (Accuracy, F1, ROC-AUC, Confusion Matrix)
├── visualizer.py         # Matplotlib learning curves, distributions, and confusion matrices
├── main.py               # Master script executing Sections 1 through 6
├── calibrate_rois.py     # Interactive screen coordinate calibration tool
├── live_capture.py       # Real-time screen capture and shoe tracker coordinator
└── baccarat_dashboard.html # Single-file Tailwind CSS web UI for live odds

```

---

## Installation & Setup in VS Code (Virtual Environment)

To keep your global Python environment clean and ensure all PyTorch and OpenCV packages run smoothly, follow these steps to set up a virtual environment (`venv`) inside Visual Studio Code.

### Step 1: Open Project in VS Code

Open your project folder (`baccarat-ai-companion`) in VS Code, and open the integrated terminal (`Ctrl + ~` on Windows/Linux or `Cmd + ~` on macOS).

### Step 2: Create the Virtual Environment

Run the following command in your terminal to create a virtual environment named `venv`:

```bash
python -m venv venv

```

### Step 3: Activate the Virtual Environment

Activate the environment depending on your operating system:

* **Windows (Command Prompt / PowerShell):**
```bash
venv\Scripts\activate

```


* **macOS / Linux:**
```bash
source venv/bin/activate

```



*(Once activated, you will see `(venv)` appear at the beginning of your terminal line).*

### Step 4: Select the Interpreter in VS Code

1. Open the VS Code Command Palette by pressing `Ctrl + Shift + P` (or `Cmd + Shift + P` on macOS).
2. Type and select **`Python: Select Interpreter`**.
3. Choose the interpreter path pointing to your virtual environment (e.g., `./venv/Scripts/python.exe` or `./venv/bin/python`).

### Step 5: Install Required Packages

With your virtual environment active, upgrade pip and install all project dependencies:

```bash
python -m pip install --upgrade pip
pip install torch torchvision numpy pandas matplotlib scikit-learn pillow ipython kagglehub mss opencv-python

```

---

## Step-by-Step Usage Guide

### Step 1: Train the Vision Models

Run the master training script. This will download the playing card dataset via KaggleHub, train both `ResNet50` and `MobileNetV3`, evaluate their performance on the hold-out test set, and save the best weights (`.pth`) and class mappings to the `checkpoints/` folder.

```bash
python main.py

```

### Step 2: Calibrate Your Screen Coordinates

Open your Baccarat game in your browser so the card dealing area is visible on your monitor. Run the calibration script:

```bash
python calibrate_rois.py

```

* You will have 3 seconds to switch to your browser.
* Click and drag a bounding box around **Player Card 1**, then press **`ENTER`** or **`SPACE`**.
* Repeat the click-and-drag process for all 6 card slots (`Player_Card_1` to `Banker_Card_3`).
* This automatically generates and saves `checkpoints/card_rois.json`.

### Step 3: Run the Live Screen Capture & Probability Engine

With your browser game open, start the live monitoring script:

```bash
python live_capture.py

```

The script will scan your calibrated screen regions every second, detect newly dealt cards, deduct them from the 416-card shoe, compute live Monte Carlo probabilities, and print the optimal mathematical recommendation directly in your terminal.

### Step 4: Open the Live Web Dashboard

To view your live shoe stats and probabilities in a modern visual interface:

1. Start a local Python web server in your project folder:
```bash
python -m http.server 8000

```


2. Open your browser and navigate to:
**`http://localhost:8000/baccarat_dashboard.html`**

---

## Disclaimer

*This project is for educational, research, and software engineering demonstration purposes only. Automated computer vision and card-counting engines do not guarantee wins in casino games due to house edges and inherent probabilistic variance. Please gamble responsibly.*

```

<FollowUp label="Would you like to add instructions on how to push this repository to GitHub via the terminal?" query="Show me the git commands to initialize my repository and push this project to GitHub."/>

```
