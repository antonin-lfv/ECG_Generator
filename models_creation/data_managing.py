import pickle
import scipy
from plotly.offline import plot
from plotly.subplots import make_subplots
import torch
from torch.utils.data import Dataset
from config import Config
import numpy as np


class ECGDatasetFiltered(Dataset):
    def __init__(self, train_data, ECG_class, max_samples=300):
        self.signals = []
        self.categories = []
        self.max_samples = max_samples
        self.ECG_class = ECG_class
        self._build_dataset(train_data)

    def _build_dataset(self, train_data):
        sample_count = 0
        for rec in train_data:
            record = Record_managment(rec)
            t = record.iter_ECG()
            try:
                while sample_count < self.max_samples:
                    _, categ, signal, _ = next(t)
                    if categ == self.ECG_class:
                        self.signals.append(signal)
                        self.categories.append(categ)
                        sample_count += 1
            except StopIteration:
                continue

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return torch.tensor(self.signals[idx], dtype=torch.float), self.categories[idx]   


class ECGDataset:
    def __init__(self):
        """
        Initializes the ECGDataset class. Load the dataset with all classes
        """

        with open(f"{Config.PATH}/{Config.ALL_NAME_FILE}", "rb") as f:
            self.train_data, self.test_data = pickle.load(f)

        # Search for the max and min values of the ECGs
        self.max_value = -np.inf
        self.min_value = np.inf
        for record in self.train_data + self.test_data:
            ecg = record["signal"]
            self.max_value = max(self.max_value, np.max(ecg))
            self.min_value = min(self.min_value, np.min(ecg))

    def get_data(self):
        """
        Returns the training and testing data.

        Returns:
            Tuple containing the training and testing data.
        """
        return self.train_data, self.test_data

    def get_max_value(self):
        """
        Returns the maximum value of the ECG signals.

        Returns:
            Float representing the maximum value.
        """
        return self.max_value

    def get_min_value(self):
        """
        Returns the minimum value of the ECG signals.

        Returns:
            Float representing the minimum value.
        """
        return self.min_value


class Record_managment:
    """
    Methods of Record_managment:
    - iter_ECG: iterate over the ECGs of a record of the dataset
    - index_significant_points_detection: detect the indexes of the significant points of the ECG
    - plot_record: plot the record

    This class is used to iterate over the ECGs of a record of the dataset.

    Example:
    >>> ECG = ECGDataset()
    >>> train_data, test_data = ECG.get_data()
    >>> # Iter through the ECGs of a record
    >>> record = Record_managment(train_data[7])
    >>> t = record.iter_ECG()
    >>> ind, categ, signal, other_info = t.__next__()
    >>> plot_output_iter_ECG(ind, categ, signal)
    """

    def __init__(self, Record: dict):
        """
        Dict of the record, keys:
        - "signal" (np.ndarray): name of the record
        - "r_peaks" (np.ndarray): indexes of the R peaks
        - "categories" (list): categories of the ECG related to the R peaks
        - "record" (str): name of the record

        Args:
            Record (dict): Dict of the record
        """
        assert Record is not None
        self.signal = Record["signal"]
        self.r_peaks = Record["r_peaks"]
        self.categories = Record["categories"]
        self.record = Record["record"]
        self.categories_ECG = Config.ALL_categories

    def iter_ECG(self):
        """
        Iterate over both r_peaks and categories, at each iteration we compute a list with Config.BEFORE_R_PEAK
        and Config.AFTER_R_PEAK indexes of the signal arround the R peak.
        With this list we can compute the peaks of the ECG with the index_significant_points_detection method.
        Use yield to return the list of index_significant_points and the category of the ECG.

        Use plot_output_iter_ECG(args) to plot the ECG with the peaks.

        Return:
        - index_significant_points (list): list of indexes of the peaks of the ECG
        - category (str): category of the ECG
        - signal (np.ndarray): amplitude of the ECG
        - dict with other information:
            {
                "previous R peak": index of the previous R peak,
                "next R peak": index of the next R peak,
                "ST segment": length of the ST segment,
                "QT interval": length of the QT interval,
            }
        """
        for r_peak, category in zip(self.r_peaks, self.categories):
            # Compute the indexes of the signal arround the R peak
            indexes = np.arange(r_peak - Config.BEFORE_R_PEAK, min(r_peak + Config.AFTER_R_PEAK, len(self.signal)))
            # Get the signal arround the R peak (amplitude of the ECG)
            signal = self.signal[indexes]
            # Compute the peaks and valleys of the ECG
            index_significant_points, s_t_segment, q_t_interval, p_r_interval, p_wave_amplitude, q_peak_amplitude, \
                r_peak_amplitude, s_peak_amplitude, t_wave_amplitude, p_wave_index, q_peak_index, r_peak_index, \
                s_peak_index, t_wave_index = \
                self.index_significant_points_detection(signal, Config.BEFORE_R_PEAK)
            # for the first R peak we take as previous distance R peak the distance
            # between the first and the second R peak
            # for the last R peak we take as next distance R peak the distance
            # between the last and the second last R peak
            if r_peak == self.r_peaks[0]:
                other_informations = {
                    "previous R peak": self.r_peaks[1] - self.r_peaks[0],
                    "next R peak": self.r_peaks[1] - self.r_peaks[0],
                    "ST segment": s_t_segment,
                    "QT interval": q_t_interval,
                    "PR interval": p_r_interval,
                    "P wave amplitude": p_wave_amplitude,
                    "Q peak amplitude": q_peak_amplitude,
                    "R peak amplitude": r_peak_amplitude,
                    "S peak amplitude": s_peak_amplitude,
                    "T wave amplitude": t_wave_amplitude,
                    "P wave index": p_wave_index,
                    "Q peak index": q_peak_index,
                    "R peak index": r_peak_index,
                    "S peak index": s_peak_index,
                    "T wave index": t_wave_index,
                }
            elif r_peak == self.r_peaks[-1]:
                other_informations = {
                    "previous R peak": self.r_peaks[-1] - self.r_peaks[-2],
                    "next R peak": self.r_peaks[-1] - self.r_peaks[-2],
                    "ST segment": s_t_segment,
                    "QT interval": q_t_interval,
                    "PR interval": p_r_interval,
                    "P wave amplitude": p_wave_amplitude,
                    "Q peak amplitude": q_peak_amplitude,
                    "R peak amplitude": r_peak_amplitude,
                    "S peak amplitude": s_peak_amplitude,
                    "T wave amplitude": t_wave_amplitude,
                    "P wave index": p_wave_index,
                    "Q peak index": q_peak_index,
                    "R peak index": r_peak_index,
                    "S peak index": s_peak_index,
                    "T wave index": t_wave_index,
                }
            else:
                other_informations = {
                    "previous R peak": r_peak - self.r_peaks[np.where(self.r_peaks == r_peak)[0][0] - 1],
                    "next R peak": self.r_peaks[np.where(self.r_peaks == r_peak)[0][0] + 1] - r_peak,
                    "ST segment": s_t_segment,
                    "QT interval": q_t_interval,
                    "PR interval": p_r_interval,
                    "P wave amplitude": p_wave_amplitude,
                    "Q peak amplitude": q_peak_amplitude,
                    "R peak amplitude": r_peak_amplitude,
                    "S peak amplitude": s_peak_amplitude,
                    "T wave amplitude": t_wave_amplitude,
                    "P wave index": p_wave_index,
                    "Q peak index": q_peak_index,
                    "R peak index": r_peak_index,
                    "S peak index": s_peak_index,
                    "T wave index": t_wave_index,
                }
            yield index_significant_points, category, signal, other_informations

    @staticmethod
    def index_significant_points_detection(signal: np.ndarray, r_peak_index: int,
                                           number_inter_points: int = Config.INTER_POINTS_NUMBER):
        """
        Compute the indexes of peaks of the signal.
        Peaks P, Q, R, S and T are detected. And we add number_inter_points points between each pair of peaks.

        Parameters:
        - signal (np.ndarray): signal of the ECG (amplitude)
        - r_peak (int): index of the R peak
        - number_inter_points (int): number of points between each pair of peaks

        Return:
        - index_significant_points (list): all the peaks and valleys of the ECG (sorted indexes)
        """
        assert signal.ndim == 1, "ecg_signal must be a 1D array"

        # We search the point Q 30 indexes before the R peak
        q_peak_index = max(r_peak_index - 30, 0)
        q_peak_index = np.argmin(signal[q_peak_index:r_peak_index - 1]) + q_peak_index

        # We search the point S 30 indexes after the R peak
        s_peak_index = min(r_peak_index + 30, len(signal) - 1)
        s_peak_index = np.argmin(signal[r_peak_index + 1:s_peak_index]) + r_peak_index

        # We search the point P 80 indexes before the Q peak
        p_wave_start = max(q_peak_index - 80, 0)
        p_wave_index = np.argmax(signal[p_wave_start:q_peak_index - 1]) + p_wave_start

        # We search the point T 80 indexes after the S peak
        t_wave_start = min(s_peak_index + 80, len(signal) - 1)
        t_wave_index = np.argmax(signal[s_peak_index + 1:t_wave_start]) + s_peak_index

        # compute significant distances
        s_t_segment = t_wave_index - s_peak_index
        q_t_interval = t_wave_index - q_peak_index
        p_r_interval = r_peak_index - p_wave_index

        # compute amplitude of P, Q, R, S and T
        p_wave_amplitude = signal[p_wave_index]
        q_peak_amplitude = signal[q_peak_index]
        r_peak_amplitude = signal[r_peak_index]
        s_peak_amplitude = signal[s_peak_index]
        t_wave_amplitude = signal[t_wave_index]

        # Add 3 points between the P wave and the Q peak
        p_q_points = np.linspace(p_wave_index, q_peak_index, number_inter_points + 2, dtype=int)

        # Add 3 points between the Q peak and the R peak
        q_r_points = np.linspace(q_peak_index, r_peak_index, number_inter_points + 2, dtype=int)

        # Add 3 points between the R peak and the S peak
        r_s_points = np.linspace(r_peak_index, s_peak_index, number_inter_points + 2, dtype=int)

        # Add 3 points between the S peak and the T wave
        s_t_points = np.linspace(s_peak_index, t_wave_index, number_inter_points + 2, dtype=int)

        # Concatenate all the points
        index_significant_points = np.concatenate((p_q_points, q_r_points, r_s_points, s_t_points))

        # Convert to list
        index_significant_points = list(index_significant_points)

        # Avoid duplicates
        index_significant_points = list(set(index_significant_points))

        # Sort the list
        index_significant_points.sort()

        return index_significant_points, s_t_segment, q_t_interval, p_r_interval, p_wave_amplitude, q_peak_amplitude, \
            r_peak_amplitude, s_peak_amplitude, t_wave_amplitude, p_wave_index, q_peak_index, r_peak_index, \
            s_peak_index, t_wave_index

    def plot_record(self):
        """
        Plot the ECG with the peaks
        """
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.arange(len(self.signal)), y=self.signal, mode="lines",
                                 name="Filtered and normalized signal"))
        fig.add_trace(go.Scatter(x=self.r_peaks, y=self.signal[self.r_peaks], mode="markers",
                                 name="R peak",
                                 text=self.categories))
        fig.update_layout(title=f"Record: {self.record} - Signal and R peak",
                          xaxis_title="time", yaxis_title="Normalized amplitude")
        fig.update_layout(template="plotly_white")
        plot(fig)
