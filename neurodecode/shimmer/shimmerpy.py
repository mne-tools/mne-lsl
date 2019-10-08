#!/usr/bin/env python

# 20141023: rchava 

#########################################################################
#  Shimmer information
#########################################################################
# 						SAMPLE FORMAT
#	9DoF: 
#   (timestamp1, acc_x, acc_y,acc_z,ang_x,ang_y,ang_z,mag_x,mag_y,mag_z)
#   framesize = 21  
#   Packet type (1), TimeStamp (2), 3xAccel (3x2), 3xGyro (3x2), 3xMag (3x2)
#
# 						COMMANDS
# send the set sensors command
#    shim.send(struct.pack('BBB', 0x08, 0xE0, 0x00))  # all
#
# send the set sampling rate command
#    shim.send(struct.pack('BB', 0x05, 0x14))  # 51.2Hz
#    shim.send(struct.pack('BB', 0x05, 0x64))  # 10.24Hz
#
# send stop streaming command
#    shim.send(struct.pack('B', 0x20));
#########################################################################
# 					AVAILABLE SENSORS
#				"00:06:66:46:9A:67" 
#				"00:06:66:46:B6:4A"
#			    "00:06:66:46:BD:8D"
# ACC+EMG
#				"00:06:66:46:9A:1A"
#			    "00:06:66:46:BD:Bf"

#########################################################################


# node type
# 0x1 - 9DoF IMU
# 0x2 - ACC+EMG

import struct
import bluetooth


class shimmer_node(object):
    def __init__(self, addr, sock, nodetype):
        """initialization"""
        self.addr = addr
        self.sock = sock
        self.n_chan = 0
        self.fs = 0
        self.type = nodetype
        self.samplesize = 21  # constant sample size (in bytes)
        self.n_fields = 10  # constant sample size (in fields)
        self.up = 0
        if nodetype == 2:
            # Packet type (1), TimeStamp (2), 3xAccel (3x2), 1xEMG (1x2)
            self.framesize = 11
            self.senscfg_hi = 0x88
            self.senscfg_lo = 0x00
            # self.pack_str = 'HHHHH'
        else:
            # Packet type (1), TimeStamp (2), 3xAccel (3x2), 3xGyro (3x2), 3xMag (3x2)
            self.framesize = 21
            self.senscfg_hi = 0xE0
            self.senscfg_lo = 0x00
            # self.pack_str = 'HHHHHHHhhh'
        self.padding = str("0" * (self.samplesize - self.framesize))
        # self.pack_str = 'HHHHHHHhhh'
        self.pack_str = 'Hhhhhhhhhh'
        self.ack = struct.pack('B', 0xff)

    def set_type(self, value):
        self.type = value
        if self.type == 2:
            # Packet type (1), TimeStamp (2), 3xAccel (3x2), 1xEMG (1x2)
            self.framesize = 11
            self.senscfg_hi = 0x88
            self.senscfg_lo = 0x00
        # self.pack_str = 'HHHHH'
        else:
            # Packet type (1), TimeStamp (2), 3xAccel (3x2), 3xGyro (3x2), 3xMag (3x2)
            self.framesize = 21
            self.senscfg_hi = 0xE0
            self.senscfg_lo = 0x00
            # self.pack_str = 'HHHHHHHhhh'
        self.padding = str("0" * (self.samplesize - self.framesize))

    def read_inquiry(self):
        # read inquiry response
        # Packet Type | ADC Sampling rate | Accel Range | Config Byte 0 |Num Chans | Buf size | Chan1 | Chan2 | ... | ChanX
        ddata = ""
        numbytes = 0
        # receive packet type
        statussize = 6
        while numbytes < statussize:
            ddata += self.sock.recv(statussize)
            numbytes = len(ddata)

        (pckt, self.fs, acc_rng, cfg,
         self.n_chan, buf_size) = struct.unpack('B' * statussize, ddata[:statussize])

        framesize = statussize + self.n_chan
        bytestoread = framesize - numbytes
        while numbytes < framesize:
            ddata += self.sock.recv(bytestoread)
            numbytes = len(ddata)

        data = ddata[0:framesize]

        numbytes = len(data)
        print("                fs (%d)" % (self.fs) + " | n channels (%d)" % self.n_chan)
        return struct.unpack('B' * numbytes, data)

    # This implementation throws away all data received between the
    # moment we send the command (e.g.,'Stop streaming') and the ACK
    def wait_for_ack(self):
        ddata = ""
        while ddata != self.ack:
            ddata = self.sock.recv(1)
            # print "received [%s]" % struct.unpack('B', ddata)
        return

    def wait_stop_streaming(self):
        ddata = self.sock.recv(1)
        if ddata != self.ack:
            return self.ack
        else:
            numbytes = len(ddata)
            while numbytes < self.framesize:
                ddata += self.sock.recv(self.framesize)
                numbytes = len(ddata)

            data = ddata[0:self.framesize]
            data = data + self.padding
            return struct.unpack(self.pack_str, data[1:self.samplesize])

    def read_data(self):
        # read incoming data
        try:
            ddata = ""
            numbytes = 0
            while numbytes < self.framesize:
                ddata += self.sock.recv(self.framesize)
                numbytes = len(ddata)

            data = ddata[0:self.framesize]
            data = data + self.padding
        except bluetooth.btcommon.BluetoothError as e:
            # socket down
            self.up = 0
            data = str([0] * (self.samplesize + 1))  # str("0"*self.samplesize+1)
            print(("[Error] BluetoothError during read_data: {0}".format(e.strerror)))
            print("[Error] Node %s" + str(self.addr) + " down!!")
        return struct.unpack(self.pack_str, data[1:self.samplesize])
