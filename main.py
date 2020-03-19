import load_data
import preprocess
import train
from os import path
import pickle
import sys

data_path = sys.argv[1]
token2id_path = sys.argv[2]  # 'token2id.pkl'
model_path = sys.argv[3]  # 'model.h5'

(comments, sentinents) = load_data.load_dataset(data_path, 50000)
processed_comments = [preprocess.preprocess_input(item) for item in comments]

token2id = None
if (path.exists(token2id_path)):
    with open(token2id_path, 'rb') as f:
        token2id = pickle.load(f)
else:
    token2id = preprocess.get_token_to_id_dict(processed_comments)
    with open(token2id_path, 'wb') as f:
        pickle.dump(token2id, f)

X = [preprocess.get_id_vector(item, token2id) for item in processed_comments]
y = [1 if item == 'positive' else 0 for item in sentinents]

model = train.train_model(X, y, model_path, 500,
                          len(token2id) + 1, 128, 0.5, 256)
