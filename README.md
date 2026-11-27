# Student-Exam-Score-Predictor
🎓 Student Exam Score Predictor

A machine learning web app that predicts a student's exam score (0–100) from their study habits and lifestyle, then gives personalised feedback and recommendations. Built with scikit-learn and Streamlit.

📌 Overview

The model is trained on the Student Habits vs Academic Performance dataset (1,000 students, 16 columns). Three algorithms were compared with 5-fold GridSearchCV, and the best performer was saved as best_model.pkl and loaded by the Streamlit interface.

🚀 Features
Predicts exam score using a trained regression model
Interactive Streamlit UI with three tabs: Predictor, Insights, About
Gauge chart (Plotly) showing the predicted score
Personalised insights and recommendations based on score band
Optional comparison against your previous exam score
Input validation with warnings (e.g. very low sleep, unrealistic study hours)
📊 Input Features
Feature	Description
Study hours	Hours studied per day
Attendance	Class attendance (%)
Sleep hours	Average hours of sleep per night
Mental health	Self-rating from 1 (poor) to 10 (excellent)
Part-time job	Yes / No
Previous exam score	Optional, used only for comparison
📈 Output
Predicted exam score (0–100)
Gauge chart visualisation
Personalised insights and recommendations
Immediate action list for low predicted scores
🤖 Model Results

Models were evaluated on a 20% hold-out test set (200 rows):

Model	Best Params	RMSE	R²
Linear Regression ✅	–	6.89	0.828
Random Forest	max_depth=5, n_estimators=50	7.46	0.798
Decision Tree	max_depth=5, min_samples_split=5	8.40	0.744

Linear Regression had the lowest RMSE, so it was refit on the full dataset and exported to best_model.pkl.

🛠 Technologies Used
Python
Streamlit
scikit-learn
Pandas & NumPy
Plotly
Matplotlib & Seaborn (EDA)
Joblib
📁 Project Structure
├── ScorePrediction.py                 # Streamlit user interface
├── StudentScoreExamPredictor.ipynb    # Data exploration + model training
├── best_model.pkl                     # Trained model (loaded by the app)
├── student_habits_performance.csv     # Dataset
├── requirements.txt                   # Python dependencies
├── Sample Data.txt                    # Sample inputs and expected UI output
└── README.md
▶️ How to Run
Clone the repository
bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
(Optional) Create a virtual environment
bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux
Install dependencies
bash
pip install -r requirements.txt
Run the app
bash
streamlit run ScorePrediction.py

The app opens in your browser at http://localhost:8501.

To retrain the model, open StudentScoreExamPredictor.ipynb in Jupyter and run all cells. This regenerates best_model.pkl.

🧪 Sample Inputs

See Sample Data.txt for example inputs and the output the app should produce.

⚠️ Notes
best_model.pkl must be in the same directory as ScorePrediction.py.
The model uses only 5 features, so predictions are estimates with a typical error of roughly ±7 points (RMSE ≈ 6.9).
Predictions are clipped to the 0–100 range.
This tool is for educational purposes and is not a substitute for academic advice.
👨‍💻 Author

Kirtanan Rethi (I23024997)
