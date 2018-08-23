from __future__ import print_function, division

"""
Brain-Glass Interface

Controls the visual feedback on Google Glass.
When connected with USB cable, connect to the local loopback network (127.0.0.1).
adb executable must be in the system's PATH environment variable.


Kyuhwa Lee, 2018
Swiss Federal Institute of Technology (EPFL)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""

import socket
import time
import os
import sys
import pycnbi.utils.q_common as qc


class GlassControl(object):
    """
    Controls Glass UI

    Constructor:
        mock: set to False if you don't have a Glass.

    """

    def __init__(self, mock=False):
        self.BUFFER_SIZE = 1024
        self.last_dir = 'L'
        self.timer = qc.Timer(autoreset=True)
        self.mock = mock
        if self.mock:
            self.print('Using a fake, mock Glass control object.')

    def print(self, *args):
        if len(args) > 0: print('[GlassControl] ', end='')
        print(*args)

    def connect(self, ip, port):
        if self.mock: return
        self.ip = ip
        self.port = port

        # Networking via USB if IP=127.0.0.1
        if ip == '127.0.0.1':
            exe = 'adb forward tcp:%d tcp:%d' % (port, port)
            self.print(exe)
            os.system(exe)
            time.sleep(0.2)
        self.print('Connecting to %s:%d' % (ip, port))
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.ip, self.port))
        except:
            self.print('* ERROR connecting to Glass. The error was:')
            self.print(sys.exc_info()[0], sys.exc_info()[1])
            sys.exit(-1)

    def disconnect(self):
        if self.mock: return
        self.print('Disconnecting from Glass')
        self.socket.close()

    def send_byte(self, msg):
        if sys.version_info.major >= 3:
            self.socket.sendall(bytes(msg + '\n', "UTF-8"))
        else:
            self.socket.sendall(bytes(unicode(msg + '\n')))

    def send_msg(self, msg, wait=True):
        """
        Send a message to the Glass

        Glass requires some delay after when the last command was sent.
        This function will be blocked until minimum this delay is satisfied.
        Set wait=False to force sending message, but the msg is likely to be ignored.

        """
        if wait:
            # Wait only if the time hasn't passed enough
            self.timer.sleep_atleast(0.033)  # 30 Hz
        if self.mock:
            return
        try:
            self.send_byte(msg)
        except Exception as e:
            self.print('* ERROR: Glass communication failed! Attempting to reconnect again.')
            self.disconnect()
            time.sleep(2)
            # Let's try again
            self.connect(self.ip, self.port)
            try:
                self.send_byte(msg)
            except Exception as e:
                self.print('Sorry, cannot fix the problem. I give up.')
                raise Exception(e)

    # Show empty bars
    def clear(self):
        if self.mock: return
        self.send_msg('C')

    # Show empty bars
    def draw_cross(self):
        if self.mock: return
        self.clear()

    # Only one direction at a time
    def move_bar(self, new_dir, amount, overlay=False):
        if self.mock: return
        if overlay is False and self.last_dir != new_dir:
            self.send_msg('%s0' % self.last_dir)
        self.send_msg('%s%d' % (new_dir, amount))
        self.last_dir = new_dir

    # Fill screen with a solid color (None, 'R','G','B')
    def fill(self, color=None):
        if self.mock: return
        if color is None:
            self.send_msg('F0')
        elif color == 'R':
            self.send_msg('F1')
        elif color == 'G':
            self.send_msg('F2')
        elif color == 'B':
            self.send_msg('F3')
        elif color == 'K':
            self.send_msg('F4')

    def fullbar_color(self, color):
        if color not in ['R', 'G', 'B', 'Y']:
            print('**** UNSUPPORTED GLASS BAR COLOR ****')
        else:
            msg = 'B' + color[0]
            # print('*** GLASS SENDING', msg)
            self.send_msg(msg)


# Test code
if __name__ == '__main__':
    step = 5

    ui = GlassControl()
    ui.connect('127.0.0.1', 59900)
    ui.clear()

    for x in range(10):
        print('L move')
        for x in range(0, 101, step):
            ui.move_bar('L', x)
        time.sleep(0.2)

        print('R move')
        for x in range(0, 101, step):
            ui.move_bar('R', x)
        time.sleep(0.2)

        print('U move')
        for x in range(0, 101, step):
            ui.move_bar('U', x)
        time.sleep(0.2)

        print('D move')
        for x in range(0, 101, step):
            ui.move_bar('D', x)
        time.sleep(0.2)

    ui.clear()
    ui.disconnect()
