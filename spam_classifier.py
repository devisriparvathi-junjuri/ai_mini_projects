"""Simple spam classifier using TF-IDF + Naive Bayes (scikit-learn)."""
import argparse
import csv
import sys

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

# Column names accepted (case-insensitive) when a data file has a header row
LABEL_COLUMNS = {"label", "class", "category", "v1"}
TEXT_COLUMNS = {"text", "message", "sms", "v2"}
# Label values accepted in data files -> 1 = spam, 0 = not spam
LABEL_VALUES = {"spam": 1, "1": 1, "ham": 0, "not spam": 0, "0": 0}
# Each class needs a few examples so it can appear in both train and test splits
MIN_PER_CLASS = 5

def load_dataset(path):
    """Load (message, label) pairs from a CSV or TSV file.

    Supported layouts:
      - a header row with a label column and a text column
        (names listed in LABEL_COLUMNS / TEXT_COLUMNS)
      - no header, two columns: label then text (e.g. the SMS Spam Collection)
    Labels can be spam/ham or 1/0. Invalid rows are skipped with a warning.
    """
    try:
        with open(path, newline="", encoding="utf-8-sig", errors="replace") as f:
            delimiter = "\t" if "\t" in f.readline() else ","
            f.seek(0)
            if delimiter == "\t":
                # SMS text often contains quote marks, so don't treat them specially
                reader = csv.reader(f, delimiter=delimiter, quoting=csv.QUOTE_NONE)
            else:
                reader = csv.reader(f, delimiter=delimiter)
            rows = [r for r in reader if any(cell.strip() for cell in r)]
    except (OSError, csv.Error) as e:
        raise ValueError(f"Could not read '{path}': {e}")

    if not rows:
        raise ValueError(f"'{path}' is empty.")

    header = [cell.strip().lower() for cell in rows[0]]
    label_idx = next((i for i, c in enumerate(header) if c in LABEL_COLUMNS), None)
    text_idx = next((i for i, c in enumerate(header) if c in TEXT_COLUMNS), None)
    if label_idx is not None and text_idx is not None:
        rows = rows[1:]  # first row was a header
    elif len(header) >= 2 and header[0] in LABEL_VALUES:
        label_idx, text_idx = 0, 1  # no header: label, text
    else:
        raise ValueError(
            "Could not find the label and text columns. Use a header row with "
            "'label' and 'text' columns, or two columns in the order: label, text."
        )

    data, skipped = [], 0
    for row in rows:
        if len(row) <= max(label_idx, text_idx):
            skipped += 1
            continue
        label = LABEL_VALUES.get(row[label_idx].strip().lower())
        text = row[text_idx].strip()
        if label is None or not text:
            skipped += 1
            continue
        data.append((text, label))

    if skipped:
        print(f"Warning: skipped {skipped} invalid row(s) in '{path}'.")
    return data

def validate_dataset(data):
    """Raise ValueError if the data can't be used to train and evaluate a model."""
    counts = {0: 0, 1: 0}
    for _, label in data:
        counts[label] += 1
    for label, name in ((0, "not spam"), (1, "spam")):
        if counts[label] < MIN_PER_CLASS:
            raise ValueError(
                f"Need at least {MIN_PER_CLASS} '{name}' examples, found {counts[label]}."
            )
    print(f"Training on {len(data)} messages ({counts[1]} spam, {counts[0]} not spam)")

def train(data=DATA):
    validate_dataset(data)
    texts, labels = zip(*data)
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

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a spam classifier and classify messages interactively."
    )
    parser.add_argument(
        "--data",
        metavar="FILE",
        help="CSV/TSV file of labelled messages (default: built-in sample data)",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    try:
        data = load_dataset(args.data) if args.data else DATA
        model = train(data)
    except ValueError as e:
        sys.exit(f"Error: {e}")

    print("Type a message to classify (or 'quit' to exit).")
    while True:
        try:
            msg = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if msg.lower() in {"quit", "exit", ""}:
            break
        prob = model.predict_proba([msg])[0][1]
        print(f"{'SPAM' if prob >= 0.5 else 'NOT SPAM'} (spam probability: {prob:.2f})")

if __name__ == "__main__":
    main()
