import pandas as pd
import matplotlib.pyplot as plt

from config import epochs, batch_size, train_file, model_file, preprocessDataFrame, oneHotEncodeY
from model import createModel

# Prepare data
df = pd.read_csv(train_file)
X_train, Y_train = preprocessDataFrame(df)

# Fit model
model = createModel()
r = model.fit(X_train, oneHotEncodeY(Y_train), epochs = epochs, batch_size = batch_size)

# Save weights
model.save(model_file)

# Plot results
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['acc'], label='acc')
plt.legend()
plt.show()