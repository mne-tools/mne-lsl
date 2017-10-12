#!/usr/bin/env python

# 20141021: rchava
#	removed link to ROS
#   if needed install 
#	bluetooth for python: sudo aptitude install python-bluez
#	pylab: sudo apt-get install  python-numpy python-scipy python-matplotlib

#########################################################################
# TODO

# - Flush buffers before closing file (now we're throwing out the data)
# - Exception management - start only if N sensors detected
# - Plotting
# - Saving files as binary
# - Recognize different types of sensors


#########################################################################
##### Original copyright  ##### 
#  accel_gyro_mag publishes all shimmer imu data.
#  Copyright (C) 2013  Rafael Berkvens rafael.berkvens@uantwerpen.be
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.


# From original code
# http://www.loveelectronics.co.uk/Tutorials/13/tilt-compensated-compass-arduino-tutorial


from math import pi, atan2, sqrt
import sys, struct, array
import bluetooth
import select

# from numpy.ma.core import arctan2

from time import sleep
import datetime

# from  ringbuffer import *
# from numpy_ringbuffer import *

from pylab import *

import simplejson

# shimmer class
from .shimmerpy import *

# Analog plot
from .analogplot import *

#########################################################################
# Keyboard routines
#########################################################################

# getch
try:
    from msvcrt import getch
except ImportError:
    def getch():
        import sys, tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

# kbhit
try:
    from msvcrt import kbhit
except ImportError:
    def kbhit():
        """kbhit() - returns 1 if a key is ready, 0 othewise.
        kbhit always returns immediately.
        """
        import sys, select
        (read_ready, write_ready, except_ready) =\
            select.select([sys.stdin], [], [], 0.0)
        if read_ready:
            return 1
        else:
            return 0

if __name__ == '__main__':
    # Shimer MAC addresses
    shimm_addr = ["00:06:66:46:9A:67",
                  "00:06:66:46:B6:4A",
                  "00:06:66:46:BD:8D",
                  "00:06:66:46:9A:1A",
                  "00:06:66:46:BD:BF"]
    emg_addr = ["00:06:66:46:9A:1A",
                "00:06:66:46:BD:BF"]

    # Configuration parameters
    scan_flag = 1
    plot_flag = 1

    sock_port = 1
    nodes = []
    plt_axx = 500
    plt_ylim = 4000
    plt_rate = 20
    sample_idx = 0

    rng_size = 50

    # rng_acc_x=RingBuffer(50)
    # Add sample to ringbuffer
    # rng_acc_x.append(pack_0)
    # buff1 = np.zeros((n_nodes,10,rng_size),dtype=np.int)
    # buff2= np.zeros((n_nodes,10,rng_size),dtype=np.int)
    # buff_flag = 1
    # buff = [[[0 for x in range(10)] for y in range(2)] for z in range(rng_size)]
    # buff_idx = 0


    # plot parameters
    analogData = AnalogData(plt_axx)

    # Get the list of available nodes
    if scan_flag == 0:
        target_addr = shimm_addr
    else:
        try:
            target_addr = []
            print("Scanning bluetooth devices...")
            nearby_devices = bluetooth.discover_devices()
            for bdaddr in nearby_devices:
                print("      " + str(bdaddr) + " - " + bluetooth.lookup_name(bdaddr))
                if bdaddr in shimm_addr:
                    target_addr.append(bdaddr)
        except:
            print("[Error] Problem while scanning bluetooth")
            sys.exit(1)

    n_nodes = len(target_addr)
    if n_nodes > 0:
        print(("Found %d target Shimmer nodes") % (len(target_addr)))
    else:
        print("Could not find target bluetooth device nearby. Exiting")
        sys.exit(1)

    print("Configuring the nodes...")
    for node_idx, bdaddr in enumerate(target_addr):
        try:
            # Connecting to the sensors
            sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
            if bdaddr in emg_addr:
                n = shimmer_node(bdaddr, sock, 0x2)
            else:
                n = shimmer_node(bdaddr, sock, 0x1)
            nodes.append(n)

            print((bdaddr, sock_port))
            nodes[-1].sock.connect((bdaddr, sock_port))
            print("  Shimmer %d (" % (node_idx) + bluetooth.lookup_name(bdaddr) + ") [Connected]")
            # send the set sensors command
            nodes[-1].sock.send(struct.pack('BBB', 0x08, nodes[-1].senscfg_hi, nodes[-1].senscfg_lo))
            nodes[-1].wait_for_ack()

            # send the set sampling rate command
            nodes[-1].sock.send(struct.pack('BB', 0x05, 0x14))  # 51.2Hz
            nodes[-1].wait_for_ack()

            # Inquiry command
            print("  Shimmer %d (" % (node_idx) + bluetooth.lookup_name(bdaddr) + ") [Configured]")
            nodes[-1].sock.send(struct.pack('B', 0x01))
            nodes[-1].wait_for_ack()
            inq = nodes[-1].read_inquiry()
        except bluetooth.btcommon.BluetoothError as e:
            print(("BluetoothError during read_data: {0}".format(e.strerror)))
            print("Unable to connect to the nodes. Exiting")
            sys.exit(1)

    # Create file and plot
    try:
        # Create buffer
        now = datetime.datetime.now()
        logname = "./DATA/IMU_" + now.strftime("%Y%m%d%H%M") + ".log"
        print("[cnbi_shimmer] Creating file: %s" % (logname))
        outfile = open(logname, "w")
        for node_idx, shim in enumerate(nodes):
            outfile.write(str(node_idx) + ": " + str(shim.addr) + "\n")
        outfile.close

        fname = "./DATA/IMU_" + now.strftime("%Y%m%d%H%M") + ".dat"
        print("[cnbi_shimmer] Creating file: %s" % (fname))
        outfile = open(fname, "a")

        # Create plot
        if plot_flag == 1:
            analogPlot = AnalogPlot(analogData)
            plt.axis([0, plt_axx, 0, plt_ylim])
            plt.ion()
            plt.show()
    except:
        print("[Error]: Error creating file/plot!! Exiting")
        # close the socket
        print("Closing nodes")
        for node_idx, shim in enumerate(nodes):
            shim.sock.close()
            print("  Shimmer %d [Ok]" % (node_idx))
        sys.exit(1)

    print("[cnbi_shimmer] Start streaming")
    # send start streaming command
    for shim in nodes:
        shim.sock.send(struct.pack('B', 0x07))
    for node_idx, shim in enumerate(nodes):
        shim.wait_for_ack()
        shim.up = 1
        print("  Shimmer %d [Ok]" % (node_idx))

    # Main acquisition loop
    while True:
        try:
            sample = []
            for shim in nodes:
                if shim.up == 1:
                    sample.append(shim.read_data())
                else:
                    sample.append([0] * (shim.n_fields))

            simplejson.dump(sample, outfile, separators=(',', ';'))
            analogData.add([sample[0][1], sample[0][2]])
            sample_idx = sample_idx + 1

            # print sample
            # plt.title(str(sample[0][0]))

            if plot_flag == 1:
                if sample_idx % plt_rate == 0:
                    analogPlot.update(analogData)

            sleep(0.01)

        # Exit if key is pressed
        except KeyboardInterrupt:
            print("\n[cnbi_shimmer] Stopping acquisition....")
            break
        except bluetooth.btcommon.BluetoothError as e:
            print(("[Error] BluetoothError during read_data: {0}".format(e.strerror)))


            # send stop streaming command
    print("[cnbi_shimmer] Stopping streaming")
    try:
        for shim in nodes:
            shim.sock.send(struct.pack('B', 0x20));
        for node_idx, shim in enumerate(nodes):
            shim.wait_for_ack()
            print("  Shimmer %d [Ok]" % (node_idx))
    except bluetooth.btcommon.BluetoothError as e:
        print(("[Error] BluetoothError during read_data: {0}".format(e.strerror)))

    '''
      n_nodes =  len(target_addr)
      while n_nodes>0:
      sample = []
      for node_idx,shim in enumerate(nodes):
        pckt = shim.wait_stop_streaming()
        print "  Shimmer %d [waiting]" % (node_idx)
        if len(pckt) != 1:
          sample.append(pckt)
        else:
          sample.append(str("0"*(shim.samplesize)))
          nodes.remove(shim)
          n_nodes = n_nodes-1
          print "  Shimmer %d [Ok]" % (node_idx)
      simplejson.dump(sample, outfile, separators=(',',';'))
      analogData.add([sample[0][1],sample[1][1]])
      analogPlot.update(analogData)
    '''

    # Closing  file
    print("[cnbi_shimmer] Closing file: %s" % (fname))
    try:
        outfile.close
    except:
        print("      [Error] Problem closing file!")

    # close the socket
    print("[cnbi_shimmer] Closing nodes")
    for node_idx, shim in enumerate(nodes):
        shim.sock.close()
        print("  Shimmer %d [Ok]" % (node_idx))

    print("[cnbi_shimmer] Recording Finished. Press any key to exit.")
    getch()
