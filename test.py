import pandas as pd
from tensorflow.keras.models import load_model

from config import test_file, model_file, preprocessDataFrame, oneHotEncodeY

# Prepare data
df = pd.read_csv(test_file)
X_test, Y_test = preprocessDataFrame(df)

# Load model
model = load_model(model_file)

# Test model
loss, accuracy = model.evaluate(X_test, oneHotEncodeY(Y_test))
print('Loss', loss)
print('Accuracy', accuracy)