# -*- coding: utf-8 -*-
import serial
import numpy as np

class Motionstim8:



    def __init__(self):

        # The Motionstim8 device has a serial port
        self.serialPort = serial.Serial()

        # The default main stimulation frequency being used is 20 Hz
        self.stimulationFrequency = 50

        # The default group stimulation frequency being used is 50 Hz
        self.groupStimulationFrequency = 50

        # Set the N factor (0 by default)
        self.nFactor = 0

        # The number of channels in the FES device
        self.nChannels = 8

        # The device channel modes (singlets)
        self.pulseModes = [0] * self.nChannels

        # The device pulse widths
        self.pulseWidths = [300] * self.nChannels

        # The device amplitudes
        self.amplitudes = [0] * self.nChannels




    def OpenSerialPort(self, comPortName):

        # configure the serial connections (the parameters differs on the device you are connecting to)
        self.serialPort.port = comPortName
        self.serialPort.baudrate = 115200
        self.serialPort.parity = serial.PARITY_NONE
        self.serialPort.stopbits = serial.STOPBITS_ONE
        self.serialPort.bytesize = serial.EIGHTBITS
        self.serialPort.timeout = 500
        self.serialPort.write_timeout = 500
        self.serialPort.xonxoff  = False
        self.serialPort.rtscts = False
        self.dsrdtr = False

        # Open the port if not already opened
        if not self.serialPort.is_open:
            self.serialPort.open()




    def CloseSerialPort(self):

        # Close the serial port if it is open
        if self.serialPort.is_open:
            self.serialPort.close()



    def WriteFES(self, bitString):

        # Write a bitstring through the open serial port
        if self.serialPort.is_open:

            # Convert the bit string in to a bytes object
            byteString = int(bitString, 2).to_bytes(len(bitString) // 8, byteorder='big')
            bytesWritten = self.serialPort.write(byteString)
            self.serialPort.flush()

        else:
            print("Error writing to FES device: serial port not opened")



    # A method to initialize the device into Channel List Mode
    def InitializeChannelListMode(self):

        # Set ts1 and ts2
        ts1 = 1000  / self.stimulationFrequency
        ts2 = 1000 / self.groupStimulationFrequency

        # Compute the group and main times
        mainTime = int((ts1 - 1.0) / 0.5)
        groupTime = int((ts2 - 1.5) / 0.5)

        # Set the active channels according to the input
        activeChannelsString = "11111111"

        # Specify which channels apply the scaler to get lower frequencies (none by default)
        lowFrequencyChannelsString = "00000000"

        # Compute checksum
        checksum = (self.nFactor + int(activeChannelsString) + int(lowFrequencyChannelsString) + groupTime + mainTime) % 8

        # Convert each parameter to its correct binary representation
        mainTimeBinary = "{0:011b}".format(mainTime)                              # binary, 11 bit width, 0 padding
        nFactorBinary = "{0:03b}".format(self.nFactor)                                 # binary, 3 bit width, 0 padding
        groupTimeBinary = "{0:05b}".format(groupTime)                             # binary, 5 bit width, 0 padding
        checksumBinary = "{0:03b}".format(checksum)                               # binary, 3 bit width, 0 padding

        # Compose the complete bistring
        bitString = "100"\
                    + checksumBinary\
                    + nFactorBinary[0:2]\
                    + "0"\
                    + nFactorBinary[2:3]\
                    + activeChannelsString[0:6]\
                    + "0"\
                    + activeChannelsString[6:8]\
                    + lowFrequencyChannelsString[0:5]\
                    + "0"\
                    + lowFrequencyChannelsString[5:8]\
                    + "00"\
                    + groupTimeBinary[0:2]\
                    + "0"\
                    + groupTimeBinary[2:5]\
                    + mainTimeBinary[0:4]\
                    + "0"\
                    + mainTimeBinary[4:11]

        # Write the bit string
        self.WriteFES(bitString)



    def UpdateChannelSettings(self, newAmplitudes):

        # Ensure that the device amplitudes are capped between the safety bounds 0 - 50 mA
        for index, amplitude in enumerate(newAmplitudes):

            if amplitude > 50:

                self.amplitudes[index] = 50

            elif amplitude < 0:

                self.amplitudes[index] = 0

            else:

                self.amplitudes[index] = newAmplitudes[index]

        # Compute checksum
        checksum = (np.sum(self.pulseModes) + np.sum(self.pulseWidths) + np.sum(self.amplitudes)) % 32

        # Construct the channel settings string
        checksumBinary = "{0:05b}".format(checksum)

        channelSettingsBinary = ""
        for amplitude, pulseWidth, pulseMode in zip(self.amplitudes, self.pulseWidths, self.pulseModes):

            pulseWidthBinary = "{0:09b}".format(pulseWidth)
            channelSettingBinary = "0"\
                                   + "{0:02b}".format(pulseMode)\
                                   + "000"\
                                   + pulseWidthBinary[0:2]\
                                   + "0"\
                                   + pulseWidthBinary[2:9]\
                                   + "0"\
                                   + "{0:07b}".format(amplitude)

            channelSettingsBinary += channelSettingBinary

        # Compose the final bitstring
        bitString = "101" + checksumBinary + channelSettingsBinary

        # Write the bit string
        self.WriteFES(bitString)


    def StopDevice(self):

        # Compose the stop command
        stopCommandBinary = "11000000"

        # Write the bit string
        self.WriteFES(stopCommandBinary)
