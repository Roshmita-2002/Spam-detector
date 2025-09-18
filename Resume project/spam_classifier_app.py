from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

app = Flask(__name__)

# -----------------------------
# 1. Create sample dataset (50 entries)
# -----------------------------
texts = [
    "Win a free iPhone now", "Limited time offer click here", "Congratulations you won cash prize",
    "Get cheap loans today", "Exclusive deal for you", "Claim your lottery reward",
    "You have been selected for prize", "Earn money fast online", "Win jackpot instantly",
    "Act now to get free gift", "Meeting at 10 am tomorrow", "Project submission due today",
    "Lunch at 1 pm", "Call me when you reach", "Let's catch up this weekend",
    "Don't forget the assignment", "Happy birthday!", "Team meeting rescheduled",
    "Please review the document", "Let's go for dinner", "Your OTP is 4567", "Invoice attached",
    "Payment received successfully", "Schedule interview for Monday", "Join Zoom call at 3pm",
    "Important notice regarding exams", "Class is postponed today", "Notes for the lecture attached",
    "Can we meet tomorrow?", "Submit report by tonight", "Special discount only for you",
    "100% guaranteed returns", "Cheap medicines online", "Get rich quick scheme",
    "Buy followers instantly", "Limited seats only hurry", "Lottery ticket confirmation",
    "Earn $1000 per day", "Double your money fast", "Free vacation trip now"
]

labels = [
    1,1,1,1,1,1,1,1,1,1,   # 10 spam
    0,0,0,0,0,0,0,0,0,0,   # 10 not spam
    0,0,0,0,0,0,0,0,0,0,   # 10 not spam
    1,1,1,1,1,1,1,1,1,1    # 10 spam
]  

# -----------------------------
# 2. Train Model
# -----------------------------
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
y = np.array(labels)

model = MultinomialNB()
model.fit(X, y)

# Evaluate (just to show performance in console)
y_pred = model.predict(X)
print("✅ Accuracy:", accuracy_score(y, y_pred))
print("✅ Confusion Matrix:\n", confusion_matrix(y, y_pred))

# -----------------------------
# 3. Flask Routes
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    user_text = request.form["message"]
    vect = vectorizer.transform([user_text])
    prediction = model.predict(vect)[0]
    result = "Spam" if prediction == 1 else "Not Spam"
    return render_template("index.html", prediction=result, user_input=user_text)

if __name__ == "__main__":
    app.run(debug=True)
