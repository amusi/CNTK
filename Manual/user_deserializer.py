from __future__ import print_function
import os
import cntk
import cntk.ops
import cntk.io
import cntk.train
import pandas as pd
import random
import os
import numpy as np
import pandas as pd
import cntk.tests.test_utils
import io
import time
from cntk.io import MinibatchSource, UserChunk, UserDeserializer, StreamInformation, SequenceInformation

cntk.tests.test_utils.set_device_from_pytest_env() # (only needed for our build system)
cntk.cntk_py.set_fixed_random_seed(1) # fix the random seed so that LR examples are repeatable

import csv
filename = 'big_file.tmp'

#with open(filename, 'w', newline='') as data:
#    w = csv.writer(data, quoting=csv.QUOTE_ALL)
#    for i in range(200000):
#        w.writerow([float(i) for j in range(151)])
#        if i % 10000 == 0:
#            print('%d records generated' % i)

print("Input file is generated")

class CSVChunk(UserChunk):
    def __init__(self, chunk_id, data, stream_infos):
        super(CSVChunk, self).__init__(stream_infos)
        self._data = data
        self._num_sequences = self._data.shape[0]
        self._chunk_id = chunk_id
        self._offsets = [0]        
        for i, s in enumerate(stream_infos):
            self._offsets.append(s.sample_shape[0] + self._offsets[-1])
        self._read = False

    def sequence_infos(self):
        start = time.time()
        result = [SequenceInformation(index_in_chunk=i, number_of_samples=1, # In case sequence to sequence - should contain max across all streams.
                                    chunk_id=self._chunk_id, id=i) for i in range(self._num_sequences)]
        end = time.time()
        print("Generation took %f" % (end - start))
        return result
  
    def get_sequence(self, sequence_id):
        if not self._read:
            print('Chunk %d is being processed' % self._chunk_id)
            self._read = True   
        return [self._data.iloc[sequence_id].values[self._offsets[i]:self._offsets[i + 1]]
                for i in range(len(self._offsets) - 1)]

class CSVDeserializer(UserDeserializer):
    def __init__(self, filename, streams, chunksize = 32*1024*1024):
        super(CSVDeserializer, self).__init__()
        self._filesize = os.stat(filename).st_size
        self._chunksize = chunksize
        rest = 1 if self._filesize % self._chunksize != 0 else 0
        self._num_chunks = int(self._filesize/self._chunksize) + rest
        self._fin = open(filename, "rb")
        self._streams = [cntk.io.StreamInformation(s['name'], i, 'dense', np.float32, s['shape'])
                         for i, s in enumerate(streams)]

    # What streams we expose
    def stream_infos(self):
        return self._streams;
    
    # What chunk we have
    def chunk_infos(self):
        return [i for i in range(self._num_chunks)]

    # Should return the UserChunk, let's read the chunk here
    def get_chunk(self, chunk_id):
        print('Chunk %d requested' % chunk_id)
        endline = ord('\n')
        _64KB = 64 * 1024;
        offset = chunk_id * self._chunksize
        if offset != 0: # Need to find the beginning of the string
            while offset > 0:
                offset -= _64KB
                buf = self._fin.read(_64KB)
                index = buf.rindex(endline)
                if index != -1: # Found, breaking
                    offset += index
                    break
            if offset == 0:
                raise ValueError('A single row does not fit into the chunk, consider increasing the chunk size')

        # reading the data
        self._fin.seek(offset)
        size = (chunk_id + 1) * self._chunksize - offset
        data = self._fin.read(size)
        last_endline = data.rindex(endline)
        if last_endline == -1:
            raise ValueError('A single row does not fit into the chunk, consider increasing the chunk size')
        data = data[:last_endline + 1]
        print('Chunk %d was read' % chunk_id)
        df = pd.read_csv(io.BytesIO(data), engine='c', dtype=np.float32)
        print('Chunk %d was csved' % chunk_id)
        result = CSVChunk(chunk_id, df, self._streams)
        print('Chunk %d has been delivered, size %d ' % (chunk_id, result._num_sequences))
        return result

    # Make sure we release all resources
    def __del__(self):
        close(self._fin)

d = CSVDeserializer(filename = filename, streams=[dict(name='x', shape=(150,)), dict(name='y', shape=(1,))])
mbs = MinibatchSource([d], randomize=False, max_sweeps=1, multithreaded_deserializer=False)

total_num_samples = 0
current_num_samples = 0
start = time.time()
while True:
    mb = mbs.next_minibatch(128)
    if not mb:
        break
    current_num_samples += mb[mbs.streams.x].number_of_samples
    if current_num_samples % 1280 == 0:
        end = time.time()
        print('Number of samples per second %f' % (current_num_samples/(end-start)))
        start = time.time()
        total_num_samples += current_num_samples
        current_num_samples = 0

total_num_samples += current_num_samples
print('Total number of samples %d' % total_num_samples)

