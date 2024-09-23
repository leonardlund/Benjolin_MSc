import numpy as np
import matplotlib.pyplot as plt

r"""train_loss = np.load(r"C:\Users\Leonard\Desktop\benjo\training_losses_16.npy")
val_loss = np.load(r"C:\Users\Leonard\Desktop\benjo\validation_losses_16.npy")
plt.plot(train_loss, label="Training loss")
plt.plot(val_loss, label="Validation loss")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.title("Loss during training")
plt.show()"""

dataset = np.load(r"C:\Users\Leonard\Desktop\benjo\latent_param_dataset_16.npz")
latent = np.squeeze(dataset['reduced_latent_matrix'][:, :2])
np.savetxt("dataset.csv", latent, delimiter=",")