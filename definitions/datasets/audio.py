# -*- coding: utf-8 -*-

"""
Audio feature extraction routines.

Author: Jan Schlüter
"""

import subprocess
import wave

import numpy as np
try:
    from pyfftw.builders import rfft as rfft_builder
    from pyfftw import empty_aligned
except ImportError:
    def rfft_builder(*args, **kwargs):
        if samples.dtype == np.float32:
            return lambda *a, **kw: np.fft.rfft(*a, **kw).astype(np.complex64)
        else:
            return np.fft.rfft
    empty_aligned = np.empty


def read_ffmpeg(infile, sample_rate, dtype=np.float32, cmd='ffmpeg'):
    """
    Decodes a given audio file using ffmpeg, resampled to a given sample rate,
    downmixed to mono, and converted to a given dtype. Returns a numpy array.
    """
    ffmpeg_dtype = {'float32': 'f32le', 'int16': 's16le'}[np.dtype(dtype).name]
    call = [cmd, "-v", "quiet", "-i", infile, "-f", ffmpeg_dtype,
            "-ar", str(sample_rate), "-ac", "1", "pipe:1"]
    samples = subprocess.check_output(call)
    return np.frombuffer(samples, dtype=dtype)


def spectrogram(samples, sample_rate, frame_len, fps, batch=48, dtype=None,
                bins=None, plans=None):
    """
    Computes a magnitude spectrogram for a given vector of samples at a given
    sample rate (in Hz), frame length (in samples) and frame rate (in Hz).
    Allows to transform multiple frames at once for improved performance (with
    a default value of 48, more is not always better). Returns a numpy array.
    Allows to return a limited number of bins only, with improved performance
    over discarding them afterwards. Optionally accepts a set of precomputed
    plans created with spectrogram_plans(), required when multi-threading.
    """
    if dtype is None:
        dtype = samples.dtype
    if bins is None:
        bins = frame_len // 2 + 1
    if len(samples) < frame_len:
        return np.empty((0, bins), dtype=dtype)
    if plans is None:
        plans = spectrogram_plans(frame_len, batch, dtype)
    rfft1, rfft, win = plans
    hopsize = int(sample_rate // fps)
    num_frames = (len(samples) - frame_len) // hopsize + 1
    nabs = np.abs
    naa = np.asanyarray
    if batch > 1 and num_frames >= batch and samples.flags.c_contiguous:
        frames = np.lib.stride_tricks.as_strided(
                samples, shape=(num_frames, frame_len),
                strides=(samples.strides[0] * hopsize, samples.strides[0]))
        spect = [nabs(rfft(naa(frames[pos:pos + batch:], dtype) * win)[:, :bins])
                 for pos in range(0, num_frames - batch + 1, batch)]
        samples = samples[(num_frames // batch * batch) * hopsize::]
        num_frames = num_frames % batch
    else:
        spect = []
    if num_frames:
        spect.append(np.vstack(
                nabs(rfft1(naa(samples[pos:pos + frame_len:],
                               dtype) * win)[:bins:])
                for pos in range(0, len(samples) - frame_len + 1, hopsize)))
    return np.vstack(spect) if len(spect) > 1 else spect[0]


def spectrogram_plans(frame_len, batch=48, dtype=np.float32):
    """
    Precompute plans for spectrogram(), for a given frame length, batch size
    and dtype. Returns two plans (single spectrum and batch), and a window.
    """
    input_array = empty_aligned((batch, frame_len), dtype=dtype)
    win = np.hanning(frame_len).astype(dtype)
    return (rfft_builder(input_array[0]), rfft_builder(input_array), win)


class WavFile(object):
    """
    Encapsulates a RIFF wave file providing memmapped access to its samples.
    If `sample_rate`, `channels` or `width` are given, a RuntimeError is raised
    if it does not match the file's format.
    """
    def __init__(self, filename, sample_rate=None, channels=None, width=None):
        self.filename = filename
        with open(filename, 'rb') as f:
            try:
                w = wave.open(f, 'r')
            except Exception as e:
                raise RuntimeError("could not read %s: %r" % (filename, e))
            self.sample_rate = w.getframerate()
            self.channels = w.getnchannels()
            self.width = w.getsampwidth()
            self.expected_length = w.getnframes()
            self.offset = w._data_chunk.offset + 8  # start of samples in file
            # to not bark on truncated files, we cross-check the file size
            f.seek(0, 2)
            self.length = min(
                    self.expected_length,
                    (f.tell() - self.offset) // (self.width * self.channels))
        if (sample_rate is not None) and (sample_rate != self.sample_rate):
            raise RuntimeError("%s has sample rate %d Hz, wanted %d" %
                               (self.sample_rate, sample_rate))
        if (channels is not None) and (channels != self.channels):
            raise RuntimeError("%s has %d channel(s), wanted %d" %
                               (self.channels, channels))
        if (width is not None) and (width != self.width):
            raise RuntimeError("%s has sample width %d byte(s), wanted %d" %
                               (self.width, width))

    @property
    def shape(self):
        return (self.length, self.channels)

    @property
    def dtype(self):
        return {1: np.int8, 2: np.int16, 3: np.int32}[self.width]

    @property
    def samples(self):
        """
        Read-only access of the samples as a memory-mapped numpy array,
        except for 24-bit samples, in which case they will be read into
        memory.
        """
        if self.width in (1, 2):
            return np.memmap(self.filename, self.dtype, mode='r',
                             offset=self.offset, shape=self.shape)
        elif self.width == 3:
            # numpy cannot read 24-bit ints. we will go one byte back
            # and interpret the data as overlapping 32-bit ints, then
            # mask out the overlapped false bits.
            with open(self.filename, 'rb') as f:
                f.seek(self.offset - 1)
                data = np.fromfile(f, np.int8,
                                   count=(np.prod(self.shape) * 3 + 1))
            data = data[:len(data) // 4 * 4].view(np.int32)
            data = np.lib.stride_tricks.as_strided(
                    data, shape=(np.prod(self.shape),), strides=(3,))
            # note: instead of masking, we could shift (data >> 8), but
            # we want the data range to match the data type.
            data = data & np.int32(0xffffff00)
            return data.reshape(self.shape)

    def __len__(self):
        return self.length

    def __array__(self, *args):
        return np.asanyarray(self.samples, *args)

    def __getitem__(self, obj):
        # TODO: for 24-bit wavs, only convert the required excerpt
        return self.samples[obj]


def get_sample_rate(wavfile):
    """
    Finds and returns the sample rate of the given .wav file.
    """
    return WavFile(wavfile).sample_rate


def get_samples(wavfile):
    """
    Returns the samples of the given .wav file as a numpy array.
    If possible, the returned array is a memory map of the file.
    """
    return WavFile(wavfile).samples


def to_float(data, out=None):
    """
    Converts integer samples to float samples, dividing by the datatype's
    maximum on the way. If the data is in floating point already, just
    converts to float32 if needed.
    """
    data = np.asanyarray(data)
    if np.issubdtype(data.dtype, np.floating):
        if out is None:
            return np.asarray(data, dtype=np.float32)
        else:
            out[:] = data
    else:
        return np.divide(data, np.iinfo(data.dtype).max, dtype=np.float32,
                         out=out)


def normalize(data, low=-1, high=1):
    """
    Normalizes audio waveform to the specified range.
    """
    data -= np.min(data)
    data /= np.max(data)
    data *= (high-low)
    data += low
    return data
