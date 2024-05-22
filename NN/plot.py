import matplotlib.pyplot as plt


def plot_mfcc_spectrograms_side_by_side(mfccs1, mfccs2, benjo_params, latent_space):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # Create a figure with 2 subplots

    for i, (mfccs, ax) in enumerate(zip([mfccs1.T, mfccs2.T], axes)):
        ax.imshow(mfccs.T, cmap='inferno', aspect='auto', origin='lower')
        ax.set_title(f'MFCC Spectrogram {"Original" if i==0 else "Reconstructed"}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Mel Frequency')
    plt.suptitle(f'Original and Reconstructed MFCCs for Benjolin parameters: {benjo_params}' +
                 f'\nLocation in latent space: {latent_space}')
    plt.tight_layout()
    plt.show()

