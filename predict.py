import preprocess
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import sys

token2id_path = sys.argv[2]  # 'token2id.pkl'
model_path = sys.argv[3]  # 'model.h5'

with open(token2id_path, 'rb') as f:
    token2id = pickle.load(f)
model = load_model(model_path)
model.summary()

while True:
    text = input("Input your comment: ")
    processed_tokens = preprocess.preprocess_input(text)
    id_vector = preprocess.get_id_vector(processed_tokens, token2id)
    X = pad_sequences([id_vector], 500)
    result = model.predict(X)[0]
    print("Prediction: " + str(result))
