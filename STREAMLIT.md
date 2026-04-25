## Getting Started

Follow these steps to set up and run the Streamlit app on your local machine.

### Prerequisites
Ensure you have **Python** installed on your system. You can check by running `python --version` in your terminal.

### 1. Set Up a Virtual Environment (Recommended)
Creating a local environment keeps your dependencies organized and prevents conflicts.

```bash
# Create the environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate
```

### 2. Install Dependencies
Install the required libraries using the provided requirements file:

```bash
pip install -r requirements.txt
```

*The `requirements.txt` includes: streamlit, torch, torchvision, torchaudio, transformers, and numpy.*

### 3. Run the App
Launch the application with the following command:

```bash
streamlit run app.py
```

**Note:** Please wait for the models to load after the command is executed.
