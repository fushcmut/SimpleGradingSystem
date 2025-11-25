# Automated Grading System (AGS)
A minimalist web application built with Flask and Python that automatically grades multiple-choice answer files based on a teacher’s key. It supports multiple input data formats (Mode 1, 2, 3) and generates detailed score reports along with mistake-statistics plots.

---

## Prerequisites
- Python 3.8+ installed on your system.

---

## Setup and Run Instructions

### Step 1: Prepare the Project
Ensure the following items are in the same directory: `app.py`, `GS.py`, `requirements.txt`, and the `templates/` folder containing `index.html`.

### Step 2: Create and Activate a Virtual Environment
Create the venv: 
```
python -m venv venv
```

Windows: 
```
venv\Scripts\activate
```

macOS/Linux: 
```
source venv/bin/activate
```

### Step 3: Install Dependencies
Install required libraries: 
```
pip install -r requirements.txt
```

### Step 4: Run the Application
Start the server: 
```
python app.py
```

### Step 5: Access the App
Open your browser and go to `http://127.0.0.1:5000/` (or the URL displayed in the terminal).

---

## Additional Notes
- The file `generate.py` is included to help generate sample input files for testing the web application's functionality.  
- This project is intended as a lightweight, easy-to-run grading tool suitable for practice tests and assignments.
