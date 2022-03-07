import fasttext

# Train model
model = fasttext.train_supervised(input="/workspace/datasets/categories/train")

# Test single prediction
print("phone predict", model.predict("phone"))

# Evaluate on test data
model.test("/workspace/datasets/categories/train.fasttext")

# Retrain with 25 epochs, bigrams, and learning rate of 1.0 and evaluate again
model = fasttext.train_supervised(input="/workspace/datasets/categories/train", lr=1.0, epoch=25, wordNgrams=2)
model.test("/workspace/datasets/categories/test")
