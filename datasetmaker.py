import h5py
import numpy as np
import math
import python_speech_features
# import array
import glob
# import re
from pydub import AudioSegment
# import soundfile as sf
import webrtcvad
import torch
import matplotlib.pylab as plt
import os
from pytorch_webrtcvad import VAD
# import torch.nn as nn


# FileManager Class
class FileManager:

    def __init__(self, name, directory):
        self.name = name
        try:
            self.data = h5py.File(DATA_FOLDER + '/' + name + '.hdf5', 'a')
        except FileNotFoundError as e:
            print(f"Error: {e}. Could not find the HDF5 file.")
        self.data = h5py.File(DATA_FOLDER + '/' + name + '.hdf5', 'a')

        if 'files' not in self.data:
            files = glob.glob(directory + '/**/*.wav', recursive=True)
            files.extend(glob.glob(directory + '/**/*.flac', recursive=True))
            files = [f for f in files]

            dt = h5py.special_dtype(vlen=str)
            self.data.create_dataset('files', (len(files),), dtype=dt)

            for i, f in enumerate(files):
                self.data['files'][i] = f

    def get_track_count(self):
        return len(self.data['files'])

    def prepare_files(self, normalize=True):
        if not OBJ_PREPARE_AUDIO:
            print(f'Skipping check for {self.name}.')
            return

        print('Found {0} tracks to check.'.format(self.get_track_count()))
        progress = 1

        if 'raw' not in self.data:
            dt = h5py.special_dtype(vlen=np.dtype(np.int16))
            self.data.create_dataset('raw', (self.get_track_count(),), dtype=dt)# noqa

        for i, file in enumerate(self.data['files']):
            print('Processing {0} of {1}'.format(progress, self.get_track_count()), end='\r', flush=True)# noqa
            progress += 1

            if len(self.data['raw'][i]) > 0:
                continue

            track = (AudioSegment.from_file(file.decode("utf-8"))
                     .set_frame_rate(SAMPLE_RATE)
                     .set_sample_width(SAMPLE_WIDTH)
                     .set_channels(SAMPLE_CHANNELS))

            if normalize:
                track = track.apply_gain(-track.max_dBFS)

            self.data['raw'][i] = np.array(track.get_array_of_samples(), dtype=np.int16)# noqa

        self.data.flush()
        print('\nDone!')

    def collect_frames(self):
        if 'frames' in self.data:
            print('Frame merging already done. Skipping.')
            return

        if 'raw' not in self.data:
            print('Could not find raw data!')
            return

        frame_count = 0
        progress = 1

        for raw in self.data['raw']:
            frame_count += int((len(raw) + (FRAME_SIZE - (len(raw) % FRAME_SIZE))) / FRAME_SIZE)# noqa
            print('Counting frames ({0} of {1})'.format(progress, self.get_track_count()), end='\r', flush=True)# noqa
            progress += 1

        dt = np.dtype(np.int16)
        self.data.create_dataset('frames', (frame_count, FRAME_SIZE), dtype=dt)

        progress = 0
        buffer = np.array([])
        buffer_limit = FRAME_SIZE * 4096

        for raw in self.data['raw']:
            raw = np.concatenate((raw, np.zeros(FRAME_SIZE - (len(raw) % FRAME_SIZE))))# noqa
            buffer = np.concatenate((buffer, raw))

            if len(buffer) < buffer_limit and progress + (len(buffer) / FRAME_SIZE) < frame_count:# noqa
                continue

            frames = np.array(np.split(buffer, len(buffer) / FRAME_SIZE))
            buffer = np.array([])

            self.data['frames'][progress: progress + len(frames)] = frames

            progress += len(frames)
            print('Merging frames ({0} of {1})'.format(progress, frame_count), end='\r', flush=True) # noqa

        self.data.flush()
        print('\nDone!')

    def label_frames(self):
        if 'labels' in self.data:
            print('Frame labelling already done. Skipping.')
            return

        if 'frames' not in self.data:
            print('Could not find any frames!')
            return

        vad = VAD()
        frame_count = len(self.data['frames'])
        progress = 0
        batch_size = 65536

        dt = np.dtype(np.uint8)
        self.data.create_dataset('labels', (frame_count,), dtype=dt)

        for pos in range(0, frame_count, batch_size):
            frames = self.data['frames'][pos: pos + batch_size]
            labels = [1 if vad.is_speech(torch.from_numpy(f).int().numpy(), sample_rate=SAMPLE_RATE) else 0 for f in frames]
            self.data['labels'][pos: pos + batch_size] = np.array(labels)

            progress += len(labels)
            print('Labelling frames ({0} of {1})'.format(progress, frame_count), end='\r', flush=True)

        self.data.flush()
        print('\nDone!')


# Visualization Class
class Vis:
    @staticmethod
    def _norm_raw(raw):
        return raw / np.max(np.abs(raw), axis=0)

    @staticmethod
    def _time_axis(raw, labels):
        time = np.linspace(0, len(raw) / SAMPLE_RATE, num=len(raw))
        time_labels = np.linspace(0, len(raw) / SAMPLE_RATE, num=len(labels))
        return time, time_labels

    @staticmethod
    def _plot_waveform(frames, labels, title='Sample'):
        raw = Vis._norm_raw(frames.flatten())
        time, time_labels = Vis._time_axis(raw, labels)

        plt.figure(1, figsize=(16, 3))
        plt.title(title)
        plt.plot(time, raw)
        plt.plot(time_labels, labels - 0.5)
        plt.show()

    @staticmethod
    def plot_sample(frames, labels, title='Sample', show_distribution=True):
        Vis._plot_waveform(frames, labels, title)

        if show_distribution:
            voice = (labels.tolist().count(1) * 100) / len(labels)
            silence = (labels.tolist().count(0) * 100) / len(labels)
            print('{0:.0f} % voice {1:.0f} % silence'.format(voice, silence))

    @staticmethod
    def plot_sample_webrtc(frames, sensitivity=0):
        vad = webrtcvad.Vad(sensitivity)
        labels = np.array([1 if vad.is_speech(f.tobytes(), sample_rate=SAMPLE_RATE) else 0 for f in frames])# noqa
        Vis._plot_waveform(frames, labels, title='Sample (WebRTC)')


# DataGenerator Class
class DataGenerator:
    def __init__(self, data, size_limit=0):
        self.data = data
        self.size = size_limit if size_limit > 0 else len(data['labels'])
        self.data_mode = 0

    def set_noise_level_db(self, level, reset_data_mode=True):
        if level not in noise_levels_db:
            raise Exception(f'Noise level "{level}" not supported! Options are: {list(noise_levels_db.keys())}')# noqa

        self.noise_level = level

        if reset_data_mode:
            if self.data_mode == 0:
                self.use_train_data()
            elif self.data_mode == 1:
                self.use_validate_data()
            elif self.data_mode == 2:
                self.use_test_data()

    def setup_generation(self, frame_count, step_size, batch_size, val_part=0.1, test_part=0.1):# noqa
        self.frame_count = frame_count
        self.step_size = step_size
        self.batch_size = batch_size

        self.train_index = 0
        self.val_index = int((1.0 - val_part - test_part) * self.size)
        self.test_index = int((1.0 - test_part) * self.size)

        self.train_size = self.val_index
        self.val_size = self.test_index - self.val_index
        self.test_size = self.size - self.test_index

    def use_train_data(self):
        n = int((self.train_size - self.frame_count) / self.step_size) + 1
        self.batch_count = int(n / self.batch_size)
        self.initial_pos = self.train_index
        self.data_mode = 0

    def use_validate_data(self):
        n = int((self.val_size - self.frame_count) / self.step_size) + 1
        self.batch_count = int(n / self.batch_size)
        self.initial_pos = self.val_index
        self.data_mode = 1

    def use_test_data(self):
        n = int((self.test_size - self.frame_count) / self.step_size) + 1
        self.batch_count = int(n / self.batch_size)
        self.initial_pos = self.test_index
        self.data_mode = 2

    def get_data(self, index_from, index_to):
        frames = self.data['frames-' + self.noise_level][index_from: index_to]
        mfcc = self.data['mfcc-' + self.noise_level][index_from: index_to]
        delta = self.data['delta-' + self.noise_level][index_from: index_to]
        labels = self.data['labels'][index_from: index_to]
        return frames, mfcc, delta, labels

    def get_batch(self, index):
        pos = self.initial_pos + (self.batch_size * index) * self.step_size
        l = self.frame_count + self.step_size * self.batch_size# noqa
        frames, mfcc, delta, labels = self.get_data(pos, pos + l)

        x, y, i = [], [], 0

        while len(y) < self.batch_size:
            X = np.hstack((mfcc[i: i + self.frame_count], delta[i: i + self.frame_count]))# noqa
            x.append(X)
            y_range = labels[i: i + self.frame_count]
            y.append(int(y_range[int(self.frame_count / 2)]))
            i += self.step_size

        return x, y

    def plot_data(self, index_from, index_to, show_track=True):
        frames, mfcc, delta, labels = self.get_data(index_from, index_to)
        Vis.plot_sample(frames, labels)
        Vis.plot_sample_webrtc(frames)
        
        
# Setting up directories
path_peer = r'/home/sasha/Документы/Кодъ'
DATA_FOLDER = r'/home/sasha/Документы/Кодъ/data'
os.chdir(path_peer)
print('Directory is set to', os.getcwd())

# Constants
SAMPLE_RATE = 16000
SAMPLE_CHANNELS = 1
SAMPLE_WIDTH = 2
SLICE_MIN_MS = 1000
SLICE_MAX_MS = 5000
FRAME_SIZE_MS = 10
SLICE_MIN = int(SLICE_MIN_MS / FRAME_SIZE_MS)
SLICE_MAX = int(SLICE_MAX_MS / FRAME_SIZE_MS)
FRAME_SIZE = int(SAMPLE_RATE * (FRAME_SIZE_MS / 1000.0))
OBJ_CUDA = torch.cuda.is_available()
OBJ_PREPARE_AUDIO = True
OBJ_TRAIN_MODELS = True

noise_levels_db = {'None': None, '-15': -15, '-5': -5}

if OBJ_CUDA:
    print('CUDA has been enabled.')
else:
    print('CUDA has been disabled.')


print('Loading files...')
speech_dataset = FileManager('speech', r'/home/sasha/Документы/Кодъ/SpeechFiles')# noqa
noise_dataset = FileManager('noise', r'/home/sasha/Документы/Кодъ/NoiseFiles')
speech_dataset.prepare_files()
noise_dataset.prepare_files()
print('Collecting frames...')
speech_dataset.collect_frames()
noise_dataset.collect_frames()
print('Labelling frames...')
speech_dataset.label_frames()

data = h5py.File(DATA_FOLDER + '/data.hdf5', 'a')
noise_levels_db = { 'None': None, '-15': -15, '-3': -3 }

mfcc_window_frame_size = 4
speech_data = speech_dataset.data
noise_data = noise_dataset.data

np.random.seed(666)


# Function to add noise to speech frames
def add_noise(speech_frames, noise_frames, align_frames, noise_level_db):
    speech_track = speech_frames.flatten()
    noise_track = noise_frames.flatten()

    def from_db(ratio_db):
        ratio = 10 ** (ratio_db / 10.) - 1e-8
        return ratio

    def coef_by_snr(src_audio, ns_audio, snr):
        src_audio = src_audio
        ns_audio = ns_audio
        try:
            target_snr_n = from_db(snr)
            ns_target_sq = np.mean(src_audio ** 2) / target_snr_n
            ns_mult = math.sqrt(ns_target_sq / np.mean(ns_audio ** 2))
            abs_max = ns_mult * np.abs(ns_audio).max()
            if abs_max > 1.:
                ns_mult /= abs_max
        except:# noqa
            ns_mult = 0
        return ns_mult

    track = np.add(speech_track, noise_track * coef_by_snr(speech_track, noise_track, noise_level_db))# noqa
    raw = track
    frames = np.array(np.split(raw, len(raw) / FRAME_SIZE))

    frames_aligned = np.concatenate((align_frames, frames))

    mfcc = python_speech_features.mfcc(frames_aligned, SAMPLE_RATE, winstep=(FRAME_SIZE_MS / 1000),# noqa
                                       winlen=mfcc_window_frame_size * (FRAME_SIZE_MS / 1000), nfft=2048)# noqa

    mfcc = mfcc[:, 1:]
    delta = python_speech_features.delta(mfcc, 2)

    return frames, mfcc, delta


if 'labels' not in data:
    print('Shuffling speech data and randomly adding 50% silence.')

    pos = 0
    l = len(speech_dataset.data['frames'])# noqa
    slices = []

    while pos + SLICE_MIN < l:
        slice_indexing = (pos, pos + np.random.randint(SLICE_MIN, SLICE_MAX + 1))# noqa
        slices.append(slice_indexing)
        pos = slice_indexing[1]

    slices[-1] = (slices[-1][0], l)
    pos = 0

    while pos + SLICE_MIN < l:
        length = np.random.randint(SLICE_MIN, SLICE_MAX + 1)
        slice_indexing = (length, length)
        slices.append(slice_indexing)
        pos += length

    total = l + pos + mfcc_window_frame_size

    np.random.shuffle(slices)

    for key in noise_levels_db:
        data.create_dataset('frames-' + key, (total, FRAME_SIZE), dtype=np.dtype(np.int16))# noqa
        data.create_dataset('mfcc-' + key, (total, 12), dtype=np.dtype(np.float32))# noqa
        data.create_dataset('delta-' + key, (total, 12), dtype=np.dtype(np.float32))# noqa

    dt = np.dtype(np.int8)
    data.create_dataset('labels', (total,), dtype=dt)

    pos = 0

    for s in slices:
        if s[0] == s[1]:
            frames = np.zeros((s[0], FRAME_SIZE))
            labels = np.zeros(s[0])
        else:
            frames = speech_data['frames'][s[0]: s[1]]
            labels = speech_data['labels'][s[0]: s[1]]

        i = np.random.randint(0, len(noise_data['frames']) - len(labels))
        noise = noise_data['frames'][i: i + len(labels)]

        for key in noise_levels_db:
            if pos == 0:
                align_frames = np.zeros((mfcc_window_frame_size - 1, FRAME_SIZE))# noqa
            else:
                align_frames = data['frames-' + key][pos - mfcc_window_frame_size + 1: pos]# noqa

            frames, mfcc, delta = add_noise(np.int16(frames), np.int16(noise),
                                            np.int16(align_frames), noise_levels_db[key])# noqa
            print(frames, shape)
            print(len(labels))
            data['frames-' + key][pos: pos + len(labels)] = frames
            data['mfcc-' + key][pos: pos + len(labels)] = mfcc
            data['delta-' + key][pos: pos + len(labels)] = delta

        data['labels'][pos: pos + len(labels)] = labels

        pos += len(labels)
        print('Generating data ({0:.2f} %)'.format((pos * 100) / total), end='\r', flush=True)# noqa

    data.flush()
    print('\nDone!')
else:
    print('Speech data already generated. Skipping')

# Test
generator = DataGenerator(data, size_limit=10000)
generator.setup_generation(frame_count=3, step_size=1, batch_size=2)
generator.set_noise_level_db('-15')
generator.use_train_data()
X, y = generator.get_batch(0)
print(f'Load a few frames into memory:\n{X[0]}\n\nCorresponding label: {y[0]}')
generator.plot_data(20000, 40000)
