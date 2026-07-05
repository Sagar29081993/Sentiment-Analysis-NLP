import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

vocab_size = 10000
max_length = 200

(_, _), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
x_test = pad_sequences(x_test, maxlen=max_length)

model = tf.keras.models.load_model("models/sentiment_model.keras")

predictions = model.predict(x_test)
y_pred = (predictions > 0.5).astype("int32")

report = classification_report(y_test, y_pred)

with open("results/classification_report.txt", "w") as f:
    f.write(report)

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("results/confusion_matrix.png")
