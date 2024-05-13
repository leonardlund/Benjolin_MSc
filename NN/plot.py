import matplotlib.pyplot as plt
import random

def plot_mfcc_spectrogram(mfccs):
  plt.imshow(mfccs.T, cmap='inferno', aspect='auto', origin='lower')  # Plot using colormap
  plt.colorbar() # format='%2.f dB')  # Add colorbar
  plt.xlabel('Time')
  plt.ylabel('Mel Frequency')
  plt.title('MFCC Spectrogram')
  plt.tight_layout()
  plt.show()

def plot_mfcc_spectrograms_side_by_side(mfccs1, mfccs2):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # Create a figure with 2 subplots

    for i, (mfccs, ax) in enumerate(zip([mfccs1.T, mfccs2.T], axes)):
        ax.imshow(mfccs.T, cmap='inferno', aspect='auto', origin='lower')
        ax.set_title(f'MFCC Spectrogram {i + 1}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Mel Frequency')

    plt.tight_layout()
    plt.show()

