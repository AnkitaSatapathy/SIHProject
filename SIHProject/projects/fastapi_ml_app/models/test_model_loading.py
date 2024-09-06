import pickle

with open('model/sarimax_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

print("Model loaded successfully.")
