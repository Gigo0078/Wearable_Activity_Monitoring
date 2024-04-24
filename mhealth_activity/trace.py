from typing import Dict, List, Tuple

import numpy as np
from scipy import interpolate


class Trace:
    """
    A class representing a collection of sensor/data readings.

    Attributes:
        title (str): Title for this collection of data.
        values (numpy.ndarray): An array of sensor/data values in chronological order, representing the measured values over the duration of the recording.
        raw_timestamps (List[Tuple[int, int]]): A list of tuples, each containing a data index and its corresponding timestamp in milliseconds. This list tracks every packet received, including those without new data. Data index may be repeated (last entry counts); timestamp in milliseconds.
        total_time (float): Time in seconds between start and end of recording (i.e., length of recording).
        timestamps (numpy.ndarray): Timestamp offset for each sensor/data value (since start of recording in seconds) obtained by linearly interpolating between the start and end of the recording, interpolates also over sampling gaps (i.e., timestamps are potentially inaccurate in case a sensor fails to report values for longer periods of time).
        samplerate (float): The mean sample rate calculated over the entire duration of the recording, expressed in samples per second (samples/s).
        _update_timestamps (List[float]): Timestamps for values with consecutive duplicate entries removed.
        _update_values (List[float]): Sensor values with consecutive duplicate entries removed.
        _update_idxs (List[int]): List indices for values with consecutive duplicate entries removed.
        _max_update_gap (float): Maximum duration in seconds for which the sensor value does not change.
        _int_timestamps (numpy.ndarray): Interpolated timestamp offset for each sensor value (since start of recording in seconds) obtained by interpolating between raw timestamps, should be correct also in case a sensor stops reporting values over a segment of the recording but individual sampling intervals might vary.

    Methods:
        update_timestamps(self) -> List[float]:
            Retrieves the list of updated timestamps, ensuring that only timestamps corresponding to actual data changes are included.        

        update_values(self) -> List[float]:
            Retrieves the list of sensor values with consecutive duplicates removed.      

        update_idxs(self) -> List[int]:
            Retrieves the indices at which actual updates to sensor values occur.       

        max_update_gap(self) -> float:
            Calculates and returns the maximum gap between data updates.    

        int_timestamps(self) -> np.ndarray:
            Provides an array of interpolated timestamps for each sensor value.   

        to_dict(self) -> Dict[str, any]:
            Converts the Trace instance into a dictionary format.   

        from_dict(cls, data_dict: Dict[str, any]) -> 'Trace':
            A class method to instantiate a `Trace` object from a dictionary representation.

        from_modified(cls, title: str, values: np.ndarray, timestamps: np.ndarray, time_offset: float = 0) -> 'Trace':
            Creates an instance from modified inputs, with timestamps provided as a numpy.ndarray.
    """

    def __init__(self, title: str, values: List[float], raw_timestamps: List[Tuple[int, int]]):
        self.title: str = title
        self.values: np.ndarray = np.array(values)
        self.raw_timestamps: List[Tuple[int, int]] = raw_timestamps
        self.total_time: float = (self.raw_timestamps[-1][1] - self.raw_timestamps[0][1]) / 1000 
        self.timestamps: np.ndarray = np.linspace(0, self.total_time, num=len(values))
        self.samplerate: float = len(self.values) / self.total_time
        self._update_timestamps: List[float] = None
        self._update_values: List[float] = None
        self._update_idxs: List[int] = None
        self._max_update_gap: float = None
        self._int_timestamps: np.ndarray = None        

    def to_dict(self) -> Dict[str, any]:
        return {
            "title": self.title,
            "values": self.values.tolist(),
            "raw_timestamps": self.raw_timestamps,
        }

    @classmethod
    def from_dict(cls, data_dict: Dict[str, any]) -> 'Trace':
        instance = cls(data_dict["title"], np.array(data_dict["values"]), data_dict["raw_timestamps"])
        return instance
    
    @classmethod
    def from_modified(cls, title: str, values: np.ndarray, timestamps: np.ndarray, time_offset: float = 0) -> 'Trace':
        indices = np.arange(len(timestamps))
        # weird behavior here: astype(int) does not work, sticking with list comprehension and back to numpy array
        # converted_timestamps = np.array([int(x) for x in ((timestamps * 1000)+time_offset)])
        converted_timestamps = ((timestamps * 1000)+time_offset).astype(np.int64)
        raw_timestamps = list(zip(indices, converted_timestamps))
        values = values.tolist()
        return cls(title, values, raw_timestamps)

    @property
    def update_timestamps(self) -> List[float]:
        if self._update_timestamps is None:
            self._update_timestamps = [self.int_timestamps[idx] for idx in self.update_idxs]
        return self._update_timestamps

    @property
    def update_values(self) -> List[float]:
        if self._update_values is None:
            self._update_values = [self.values[idx] for idx in self.update_idxs]
        return self._update_values

    @property
    def update_idxs(self) -> List[int]:
        if self._update_idxs is None:
            try:
                vals = self.values.astype(float)
                val_diffs = np.diff(vals, prepend=vals[0]-1)
            except ValueError:
                vals = np.concatenate(([object()], self.values))
                val_diffs = vals[1:] != vals[:-1]
            self._update_idxs = np.where(val_diffs != 0)[0].tolist()
        return self._update_idxs

    @property
    def max_update_gap(self) -> float:
        if self._max_update_gap is None:
            timestamps = self.update_timestamps.copy()
            if self.update_timestamps[0] > 0: # if first sensor value was captured after start of recording
                timestamps = [0,] + timestamps
            if self.update_timestamps[-1] < self.total_time: # if last sensor update happened before end of recording
                timestamps.append(self.total_time)
            self._max_update_gap = max(np.diff(timestamps)) if len(timestamps) > 1 else np.inf
        return self._max_update_gap

    @property
    def int_timestamps(self) -> np.ndarray:
        if self._int_timestamps is None:
            idxs = []
            ts = []
            assert len(self.raw_timestamps)>1
            for rts, next_rts in zip(self.raw_timestamps[:-1], self.raw_timestamps[1:]):
                if rts[0] != next_rts[0]:
                    idxs.append(rts[0])
                    ts.append(rts[1])

            if len(self.values) > self.raw_timestamps[-1][0]:
                idxs.append(self.raw_timestamps[-1][0])
                ts.append(self.raw_timestamps[-1][1])

            self._int_timestamps = (interpolate.interp1d(idxs, ts, fill_value="extrapolate")(np.arange(len(self.values))) - self.raw_timestamps[0][1]) / 1000
        return self._int_timestamps

    def __repr__(self) -> str:
        return f"Trace(title='{self.title}', total_time={self.total_time:.2f}, samplerate={self.samplerate:.2f}, max_update_gap={self.max_update_gap:.2f})"
