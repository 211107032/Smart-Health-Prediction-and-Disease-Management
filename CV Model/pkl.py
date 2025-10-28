import pickle
import joblib

from SKD.evaluate import model

# Save the trained model in HDF5 format
model.save("skin_disease_model.h5")

# Save class indices using joblib or pickle
class_indices = train_generator.class_indices
with open("class_indices.pkl", "wb") as f:
    pickle.dump(class_indices, f)
