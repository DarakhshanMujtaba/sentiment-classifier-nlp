# Task 2: Multi-Class Sentiment Classifier (code as run in Google Colab, cell by cell)

# ---- Step 1: load the data
import pandas as pd, urllib.request
base = "https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment/"
def load(split):
    text = urllib.request.urlopen(base + split + "_text.txt").read().decode().splitlines()
    labels = urllib.request.urlopen(base + split + "_labels.txt").read().decode().split()
    return pd.DataFrame({"text": text, "label": [int(x) for x in labels]})
train, test = load("train"), load("test")
print(len(train), len(test))

# ---- Step 2: clean the text (stop-word removal + lemmatization)
import re, nltk
nltk.download("stopwords"); nltk.download("wordnet"); nltk.download("omw-1.4")
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

keep = {"no", "not", "nor", "never"}
stop = set(stopwords.words("english")) - keep
lemma = WordNetLemmatizer()

def clean(text):
    text = re.sub(r"\\u2019", "'", text).lower()       # fix escaped apostrophes
    text = re.sub(r"http\S+", " url ", text)          # links
    text = re.sub(r"@\w+", " user ", text)            # mentions
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", text)   # split into words
    words = [w for w in words if w not in stop and len(w) > 1]
    words = [lemma.lemmatize(lemma.lemmatize(w, "v"), "n") for w in words]
    return " ".join(words)

train["clean"] = train["text"].apply(clean)
test["clean"] = test["text"].apply(clean)

# ---- Step 3: TF-IDF features
from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True)
X_train = vec.fit_transform(train["clean"])
X_test = vec.transform(test["clean"])
y_train, y_test = train["label"], test["label"]
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# ---- Step 4: train and compare three models
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score

models = {
    "Naive Bayes": MultinomialNB(alpha=0.1),
    "Logistic Regression": LogisticRegression(C=2, max_iter=2000, class_weight="balanced"),
    "Linear SVM": LinearSVC(C=0.05, class_weight="balanced"),
}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"{name:20s} accuracy={accuracy_score(y_test, pred):.3f}  macro-F1={f1_score(y_test, pred, average='macro'):.3f}")

# ---- Step 5: evaluate the best model
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

best = models["Linear SVM"]
pred = best.predict(X_test)
labels = ["negative", "neutral", "positive"]
print(classification_report(y_test, pred, target_names=labels, digits=3))

ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=labels, cmap="Blues")
plt.title("Confusion matrix: Linear SVM")
plt.savefig("confusion_matrix.png", dpi=200, bbox_inches="tight")
plt.show()

mine = ["I love this internship!", "The class starts at 9am.", "This app keeps crashing, so annoying."]
print(best.predict(vec.transform([clean(s) for s in mine])))
