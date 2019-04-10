from __future__ import print_function

import sys
import wave

from io import StringIO

import alsaaudio
import colorama
import numpy as np

from reedsolo import RSCodec, ReedSolomonError
from termcolor import cprint
from pyfiglet import figlet_format

import sounddevice as sd

HANDSHAKE_START_HZ = 4096
HANDSHAKE_END_HZ = 6144

START_HZ = 1024
STEP_HZ = 256
BITS = 4

FEC_BYTES = 4

error_byte_chunks = []

def stereo_to_mono(input_file, output_file):
    inp = wave.open(input_file, 'r')
    params = list(inp.getparams())
    params[0] = 1 # nchannels
    params[3] = 0 # nframes

    out = wave.open(output_file, 'w')
    out.setparams(tuple(params))

    frame_rate = inp.getframerate()
    frames = inp.readframes(inp.getnframes())
    data = np.fromstring(frames, dtype=np.int16)
    left = data[0::2]
    out.writeframes(left.tostring())

    inp.close()
    out.close()

def yield_chunks(input_file, interval):
    wav = wave.open(input_file)
    frame_rate = wav.getframerate()

    chunk_size = int(round(frame_rate * interval))
    total_size = wav.getnframes()

    while True:
        chunk = wav.readframes(chunk_size)
        if len(chunk) == 0:
            return

        yield frame_rate, np.fromstring(chunk, dtype=np.int16)

def dominant(frame_rate, chunk):
    w = np.fft.fft(chunk)
    freqs = np.fft.fftfreq(len(chunk))
    peak_coeff = np.argmax(np.abs(w))
    peak_freq = freqs[peak_coeff]
    return abs(peak_freq * frame_rate) # in Hz

def match(freq1, freq2):
    return abs(freq1 - freq2) < 20

def decode_bitchunks(chunk_bits, chunks):
    out_bytes = []

    next_read_chunk = 0
    next_read_bit = 0

    byte = 0
    bits_left = 8
    while next_read_chunk < len(chunks):
        can_fill = chunk_bits - next_read_bit
        to_fill = min(bits_left, can_fill)
        offset = chunk_bits - next_read_bit - to_fill
        byte <<= to_fill
        shifted = chunks[next_read_chunk] & (((1 << to_fill) - 1) << offset)
        byte |= shifted >> offset;
        bits_left -= to_fill
        next_read_bit += to_fill
        if bits_left <= 0:

            out_bytes.append(byte)
            byte = 0
            bits_left = 8

        if next_read_bit >= chunk_bits:
            next_read_chunk += 1
            next_read_bit -= chunk_bits

    return out_bytes

def decode_file(input_file, speed):
    wav = wave.open(input_file)
    if wav.getnchannels() == 2:
        mono = StringIO()
        stereo_to_mono(input_file, mono)

        mono.seek(0)
        input_file = mono
    wav.close()

    offset = 0
    for frame_rate, chunk in yield_chunks(input_file, speed / 2):
        dom = dominant(frame_rate, chunk)
        print("{} => {}".format(offset, dom))
        offset += 1

def extract_packet(freqs):
    freqs = freqs[::2]
    bit_chunks = [int(round((f - START_HZ) / STEP_HZ)) for f in freqs]
    bit_chunks = [c for c in bit_chunks[1:] if 0 <= c < (2 ** BITS)]

    # save error bytes everytime
    global error_byte_chunks
    error_byte_chunks = bit_chunks[-8:]

    return bytearray(decode_bitchunks(BITS, bit_chunks))

def display(s):
    cprint(figlet_format(s.replace(' ', '   '), font='doom'), 'yellow')


TIME_DATA = 0.1 # 0.1 sec per data
# 0.1 sec per data.
# 2 data per 1 ASCII

# list = np.zeros(0, dtype=np.int)    # make int array
#list = []

def make_sin(step_list, interval, frame_rate=44100):

    sin = np.zeros(0)
    time_array = np.arange(0., interval, 1/frame_rate)

    # make sin from step_list
    for step in step_list:
        # make freq_hz
        freq_hz = START_HZ + STEP_HZ*step

        # make new sin array
        new_sin = np.sin(2*np.pi * freq_hz * time_array)

        # attach new sin array
        sin = np.append(sin, new_sin)


    # return completed sin list
    return sin

def playSound(sin, total_time, frame_rate=44100):
    # total_time = total sound's play time
    sd.play(sin, frame_rate)
    sd.sleep(total_time)

def extract_step(string):
    # string = inputted characters without my student number(201502111)
    list = []

    # standard to sparate from 1 char(byte), in this case, it is 16
    # a=97, 0110 / 0001,  left=0110(bin)=6(dec)=97/16, right=0001(bin)=1(dec)=97%16
    standard_data = 2**BITS

    # make step list from string
    for s in string:
        # example. 'a'=97(dec)=0110 0001(bin)
        list.append(int(ord(s)/standard_data)) # 0110
        list.append(ord(s)%standard_data) # 0001

    # attach step from error byte chunks
    for step in error_byte_chunks:
        list.append(step)

    # return completed step list
    return list

def play_linux(string, interval, frame_rate=44100):
    # interval = 0.1 , time per data

    num_word = len(string)  # 201502111
    num_data = num_word * 2  # 1 word = 2 data

    num_error_data = FEC_BYTES*2  # 4 error byte, 8 data
    total_time = (num_data + num_error_data)*interval  # time for all data

    # extract step list from string
    step_list = extract_step(string)

    # make a complete sin function from step list
    sin = make_sin(step_list, interval)

    # play sound for total_time
    playSound(sin, (int)(total_time * 1000))

def listen_linux(frame_rate=44100, interval=0.1):

    mic = alsaaudio.PCM(alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NORMAL, device="default")
    mic.setchannels(1)
    mic.setrate(44100)
    mic.setformat(alsaaudio.PCM_FORMAT_S16_LE)

    num_frames = int(round((interval / 2) * frame_rate))
    mic.setperiodsize(num_frames)
    print("start...")

    in_packet = False
    packet = []

    while True:
        l, data = mic.read()
        if not l:
            continue

        chunk = np.fromstring(data, dtype=np.int16)
        dom = dominant(frame_rate, chunk)
        #print(dom)

        if in_packet and match(dom, HANDSHAKE_END_HZ):
            byte_stream = extract_packet(packet)
            try:
                byte_stream = RSCodec(FEC_BYTES).decode(byte_stream)
                byte_stream = byte_stream.decode("utf-8")

                # change to str to use find(), replace()
                byte_stream = str(byte_stream)

                # check student number
                if(byte_stream.find("201502111") != -1):
                    # delete student number
                    byte_stream = byte_stream.replace("201502111", "")
                    display(byte_stream)
                    display("")

                    # play sound without student number
                    play_linux(byte_stream, interval)
            except ReedSolomonError as e:
                print("{}: {}".format(e, byte_stream))

            packet = []
            in_packet = False
        elif in_packet:
            packet.append(dom)
        elif match(dom, HANDSHAKE_START_HZ):
            in_packet = True

if __name__ == '__main__':
    colorama.init(strip=not sys.stdout.isatty())

    #decode_file(sys.argv[1], float(sys.argv[2]))
    listen_linux()
