import socket, time, os
import pycnbi.utils.q_common as qc


class GlassControl():
    def __init__(self):
        self.BUFFER_SIZE = 1024
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.last_dir = 'L'
        self.timer = qc.Timer()

    def connect(self, ip, port):
        self.ip = ip
        self.port = port

        # networking via USB if IP=127.0.0.1
        if ip == '127.0.0.1':
            exe = 'adb forward tcp:%d tcp:%d' % (port, port)
            print(exe)
            os.system(exe)
            time.sleep(0.2)
        print('Connecting to %s:%d' % (ip, port))
        self.s.connect((self.ip, self.port))

    def disconnect(self):
        print('Disconnecting')
        self.s.close()

    def send_msg(self, msg):
        # self.s.sendall(bytes(msg, "UTF-8")) # for Python3
        try:
            self.s.sendall(bytes(unicode(msg)))
        except:
            print('>> ERROR: Glass communication failed! Re-initiating the connection.')
            self.disconnect()
            time.sleep(0.5)
            self.connect(self.ip, self.port)
        time.sleep(0.01)

    # clear bars in progress
    def clear(self):
        self.send_msg('L0\n')
        self.send_msg('R0\n')
        self.send_msg('U0\n')
        self.send_msg('D0\n')

    # only one direction at a time
    def move_bar(self, new_dir, amount):
        if self.last_dir != new_dir:
            self.send_msg('%s0\n' % self.last_dir)
        self.send_msg('%s%d\n' % (new_dir, amount))
        self.last_dir = new_dir


if __name__ == '__main__':
    TCP_IP = '127.0.0.1'
    TCP_PORT = 59900

    ui = GlassControl()
    ui.connect(TCP_IP, TCP_PORT)
    ui.clear()

    for x in range(10):
        print('L move')
        for x in range(0, 101, 10):
            ui.move_bar('L', x)
            time.sleep(0.03)
        print('R move')
        for x in range(0, 101, 10):
            ui.move_bar('R', x)
            time.sleep(0.03)
        print('U move')
        for x in range(0, 101, 10):
            ui.move_bar('U', x)
            time.sleep(0.03)
        print('D move')
        for x in range(0, 101, 10):
            ui.move_bar('D', x)
            time.sleep(0.03)

    ui.clear()
    ui.disconnect()
