"""Simple spam classifier using TF-IDF + Naive Bayes (scikit-learn)."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Small built-in dataset: (message, label) where 1 = spam, 0 = not spam
DATA = [
    ("Congratulations! You won a free iPhone. Click here to claim", 1),
    ("URGENT: your account is suspended, verify your password now", 1),
    ("Win cash prizes today, limited time offer", 1),
    ("Get cheap loans with no credit check, apply now", 1),
    ("You have been selected for a free vacation, reply YES", 1),
    ("Exclusive deal: buy one get three free, act fast", 1),
    ("Claim your lottery reward now, send bank details", 1),
    ("Free gift card waiting for you, click the link", 1),
    ("Lose 10 kg in a week with this miracle pill", 1),
    ("Earn money from home, no experience needed", 1),
    ("Hey, are we still meeting for lunch tomorrow?", 0),
    ("Please find the attached report for the Monday meeting", 0),
    ("Can you send me the notes from today's lecture?", 0),
    ("Happy birthday! Hope you have a great day", 0),
    ("Reminder: project submission is due on Friday", 0),
    ("Let's study together for the SQL exam this weekend", 0),
    ("Thanks for helping me with the Python assignment", 0),
    ("Dinner is ready, come downstairs", 0),
    ("The interview is scheduled for 10 AM on Thursday", 0),
    ("I pushed the latest code to the repo, please review", 0),
    ("Free entry in a weekly contest, text WIN to 80085 now", 1),
    ("Your prize is waiting, claim free cash today", 1),
    ("Limited offer: free trial, click here to subscribe now", 1),
    ("Final notice: you won a free cruise, call now", 1),
    ("Cheap meds online, no prescription, order now", 1),
    ("Act now and get free money, click this link", 1),
    ("Are you free to join the group call at 5 PM?", 0),
    ("I will send the assignment files by tonight", 0),
    ("Mom asked if you are coming home this weekend", 0),
    ("Great work on the dashboard, the charts look clear", 0),
    ("Please review my pull request when you get time", 0),
    ("What time does the placement session start tomorrow?", 0),
]

def train():
    texts, labels = zip(*DATA)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )
    model = make_pipeline(TfidfVectorizer(stop_words="english"), MultinomialNB())
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"Test accuracy: {accuracy_score(y_test, preds):.2f}")
    print(classification_report(y_test, preds, target_names=["not spam", "spam"], zero_division=0))
    # Retrain on all data for the best final model
    model.fit(texts, labels)
    return model

def main():
    model = train()
    print("Type a message to classify (or 'quit' to exit).")
    while True:
        msg = input("> ").strip()
        if msg.lower() in {"quit", "exit", ""}:
            break
        prob = model.predict_proba([msg])[0][1]
        print(f"{'SPAM' if prob >= 0.5 else 'NOT SPAM'} (spam probability: {prob:.2f})")

if __name__ == "__main__":
    main()
