import numpy as np
import glob, os
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import sklearn.preprocessing as preprocessing
import warnings

DATAPATH = os.path.expanduser("~/cmf_data/VoxCeleb/")

class VoxCeleb:
    """Spectrogram data from the Vox Celeb Dataset."""

    def __init__(self, secs, path, concat=True):
        """
        Create a VoxCeleb dataset object.
        
        Parameters
        ----------
        secs : int
            Number of seconds of the dataset to be generated.
            Multiple .wav files will be combined if necessary.

        path : string
            Path to folder containing .wav file(s)

        concat : bool
            Whether or not to concatenate multiple files. If false, 
            only the first file in the directory will be used.
        """
        audio = np.empty((1,))
        secs_loaded = 0
        files_loaded = 0
        files = glob.glob(path + "*.wav")
        for file in files:
            (sr, samples) = wavfile.read(file)
            audio = np.concatenate((audio, samples))

            # Keep track of the duration (in seconds) of our audio clip
            dur = len(samples) / sr
            secs_loaded = secs_loaded + dur
            files_loaded = files_loaded + 1
            if (secs_loaded >= secs):
                break
            if not concat:
                break
        
        # We're assuming that all files use the same sampling frequency.
        # Truncate  audio samples so that we end up with the duration specified.
        total_samples = int(round(secs * sr))
        if total_samples > len(audio):
            warnings.warn("Found fewer than %.2f seconds of audio. "
                          "Returning %.2f seconds of audio." % (secs, len(audio) / sr))     
        audio = audio[0:total_samples]

        self.audio = audio
        self.sampling_rate = sr

    def generate(self, 
                 seg_length=20e-3, 
                 overlap=0.3, 
                 normalize=True, 
                 **spectrogram_kwargs):
        """
        Generate spectrogram from collected audio samples.

        Parameters
        ----------
        seg_length : float
            Length of FFT segment (in seconds) for computing the spectrogram.
            Defaults to 20 milliseconds.

        overlap : float
            Float between 0 and 1 specifying the fraction of segments which should 
            overlap. Defaults to 0.3.

        normalize : boolean
            If true, divide each frequency bin by its standard deviation.

        spectrogram_kwargs : keyword arguments
            Optional arguments to scipy.signal.spectrogram(). Note that these keyword arguments
            should not contain noverlap, nseg, fs, or return_onesided,
            as these are specified directly as parameters to generate().

        Returns
        -------
        S : array
            Spectrogram
        audio : array
            Raw audio samples
        """
        nperseg = round(int(seg_length * self.sampling_rate))
        noverlap = round(int(nperseg * overlap))

        spectrogram_kwargs['fs'] = self.sampling_rate
        spectrogram_kwargs['nperseg'] = nperseg
        spectrogram_kwargs['noverlap'] = noverlap

        (_, _, S) = signal.spectrogram(self.audio, **spectrogram_kwargs)

        if normalize:
            scaler = preprocessing.StandardScaler(with_mean=False)
            S = scaler.fit_transform(S.T).T

        return (S, self.audio)
