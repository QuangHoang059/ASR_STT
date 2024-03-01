# start imports
import numpy as np
import pyaudio
import wave

'''
Class that is mainly responsible for recognizing from microphone
'''
class MicRecorder(object):
    
    '''
    Constructor
    '''
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.data = []
        self.channels = 2
        self.chunksize = 4096* 2
        self.samplerate = 41000
        self.recorded = False


    '''
    Method that starts the recording
    '''
    def start_recording(self):
        print ("Recoding started ...")

        if self.stream:
            self.stream.stop_stream()
            self.stream.close()

        # Open stream
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.samplerate,
            input=True,
            frames_per_buffer=self.chunksize,
        )

        # Collect data for this chunk
        self.data = [[] for i in range(self.channels)]
        self.frames=[]
    '''
    Method that processes the recordings from the stream and adds to the data
    '''
    def process_recording(self):
        data = self.stream.read(self.chunksize)
        self.frames.append(data)
        nums = np.fromstring(data, np.int16)
        for c in range(self.channels):
            self.data[c].extend(nums[c::self.channels])

    '''
    Method that stops the recording
    '''
    def stop_recording(self):
        self.stream.stop_stream()
        self.stream.close()
        self.stream = None
        self.recorded = True
        print ("Recording completed.")

    '''
    API method that returns the recorded data
    '''
    def get_recording(self, seconds):
        self.start_recording()

        # Record and process the data chunk by chunk
        for i in range(0, int(self.samplerate / self.chunksize * seconds)):
            self.process_recording()
        # Stop recording
        self.stop_recording()
        self.save_recorded('recording.wav')
        return self.get_recorded_data()

    '''
    Internal method for safety check
    '''
    def get_recorded_data(self):
        if not self.recorded:
            raise NoRecordingError("Recording was not complete/begun")
        return self.data

    def save_recorded(self, output_filename):
        wf = wave.open(output_filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(self.samplerate)
        wf.writeframes(b''.join(self.frames))
        wf.close()
class NoRecordingError(Exception):
    pass

