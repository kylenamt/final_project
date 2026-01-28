import numpy as np
import librosa

class Loss:
    """
    A class to compute a combination of different loss functions.
    """
    def __init__(self, loss_types=['spectral'], fft_size=2048, hop_size=512, win_size=2048):
        """
        Initializes the Loss class.

        Args:
            loss_types (list): A list of strings specifying which losses to compute. 
                               Available options: 'spectral'.
            fft_size (int): FFT size for spectral loss.
            hop_size (int): Hop size for spectral loss.
            win_size (int): Window size for spectral loss.
        """
        self.loss_types = loss_types
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_size = win_size
        
        # Dictionary to map loss type strings to their respective methods
        self.loss_functions = {
            'spectral': self._spectral_loss
        }

    def _spectral_loss(self, y_true, y_pred):
        """
        Computes the spectral loss (sum of log-magnitude differences).
        """
        true_stft = librosa.stft(y_true,
                                 n_fft=self.fft_size,
                                 hop_length=self.hop_size,
                                 win_length=self.win_size)
        
        pred_stft = librosa.stft(y_pred,
                                 n_fft=self.fft_size,
                                 hop_length=self.hop_size,
                                 win_length=self.win_size)

        true_mag = np.abs(true_stft)
        pred_mag = np.abs(pred_stft)

        # Add a small epsilon to avoid log(0)
        loss = np.mean(np.abs(np.log(true_mag + 1e-8) - np.log(pred_mag + 1e-8)))
        
        return loss

    def __call__(self, y_true, y_pred):
        """
        Computes the total loss by summing the specified individual losses.

        Args:
            y_true: The ground truth audio signal.
            y_pred: The predicted audio signal.

        Returns:
            A dictionary containing the total loss and individual loss values.
        """
        total_loss = 0.0
        loss_dict = {}

        for loss_type in self.loss_types:
            if loss_type in self.loss_functions:
                loss_value = self.loss_functions[loss_type](y_true, y_pred)
                total_loss += loss_value
                loss_dict[f'{loss_type}_loss'] = loss_value
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")
        
        loss_dict['total_loss'] = total_loss
        return loss_dict

# Example Usage:
#
# loss_computer = Loss(loss_types=['spectral'])
# y_true_audio = ... # your ground truth audio tensor
# y_pred_audio = ... # your predicted audio tensor
# losses = loss_computer(y_true_audio, y_pred_audio)
# print(losses) 
# # Output would be something like: {'spectral_loss': <tf.Tensor>, 'total_loss': <tf.Tensor>}
