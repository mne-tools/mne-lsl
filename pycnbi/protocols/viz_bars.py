from __future__ import print_function, division

"""
Bar visual feedback class


Kyuhwa Lee, 2015
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

import sys
import numpy as np
import time
import cv2
import pycnbi
import pycnbi.glass.bgi_client as bgi_client
import pycnbi.utils.q_common as qc
from pycnbi import logger


class BarVisual(object):
    # Default setting
    color = dict(G=(20, 140, 0), B=(210, 0, 0), R=(0, 50, 200),
        Y=(0, 215, 235), K=(0, 0, 0), W=(255, 255, 255), w=(200, 200, 200))
    barwidth = 100
    textlimit = 20  # maximum number of characters to show

    def __init__(self, use_glass=False, glass_feedback=True, pc_feedback=True, screen_pos=None, screen_size=None):
        """
        Input:
            use_glass: if False, mock Glass will be used
            glass_feedback: show feedback to the user?
            pc_feedback: show feedback on the pc screen?
            screen_pos: screen position in (x,y)
            screen_size: screen size in (x,y)
        """
        # screen size and message setting
        if screen_size is None:
            if sys.platform.startswith('win'):
                from win32api import GetSystemMetrics
                screen_width = GetSystemMetrics(0)
                screen_height = GetSystemMetrics(1)
            else:
                screen_width = 1024
                screen_height = 768
        else:
            screen_width, screen_height = screen_size

        if screen_pos is None:
            screen_x, screen_y = (0, 0)
        else:
            screen_x, screen_y = screen_pos

        self.text_x = int(screen_width / 4)
        self.text_y = int(screen_height / 2)
        self.text_size = 2
        cv2.namedWindow("img", cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow("img", screen_x, screen_y)
        cv2.setWindowProperty("img", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN);

        self.img = np.zeros((screen_height, screen_width, 3), np.uint8)
        self.glass = bgi_client.GlassControl(mock=not use_glass)
        self.glass.connect('127.0.0.1', 59900)
        self.set_glass_feedback(glass_feedback)
        self.set_pc_feedback(pc_feedback)
        self.set_cue_color(boxcol='B', crosscol='W')
        self.width = self.img.shape[1]
        self.height = self.img.shape[0]
        hw = int(self.barwidth / 2)
        self.cx = int(self.width / 2)
        self.cy = int(self.height / 2)
        self.xl1 = self.cx - hw  # 200
        self.xl2 = self.xl1 - self.barwidth  # 100
        self.xr1 = self.cx + hw  # 300
        self.xr2 = self.xr1 + self.barwidth  # 400
        self.yl1 = self.cy - hw
        self.yl2 = self.yl1 - self.barwidth
        self.yr1 = self.cy + hw
        self.yr2 = self.yr1 + self.barwidth

    def finish(self):
        cv2.destroyAllWindows()
        self.glass.disconnect()

    def set_glass_feedback(self, fb):
        self.glass_feedback = fb

    def set_pc_feedback(self, fb):
        self.pc_feedback = fb

    def set_cue_color(self, boxcol='B', crosscol='W'):
        self.boxcol = self.color[boxcol]
        self.crosscol = self.color[crosscol]

    def fill(self, fillcolor='K'):
        self.glass.fill(fillcolor)
        cv2.rectangle(self.img, (0, 0), (self.width, self.height), self.color[fillcolor], -1)

    # draw cue with custom colors
    def draw_cue(self):
        cv2.rectangle(self.img, (self.xl2, self.yl1), (self.xr2, self.yr1), self.color['w'], -1)
        cv2.rectangle(self.img, (self.xl1, self.yl2), (self.xr1, self.yr2), self.color['w'], -1)
        cv2.rectangle(self.img, (self.xl1, self.yl1), (self.xr1, self.yr1), self.boxcol, -1)

        # cross fixation point
        #cv2.rectangle( self.img, (self.cx-3,self.cy), (self.cx+3,self.cy), self.crosscol, -1 )
        #cv2.rectangle( self.img, (self.cx,self.cy-3), (self.cx,self.cy+3), self.crosscol, -1 )

        # circle fixation point
        cv2.circle(self.img, (self.cx, self.cy), 3, self.color['W'], -1)

    # paints the new bar on top of the current image
    def move(self, dir, dx, overlay=False, barcolor=None, caption='', caption_color='W'):
        if not overlay:
            self.draw_cue()

        if barcolor is None:
            if dx == self.xl2:
                c = 'G'
            else:
                c = 'R'
        else:
            c = barcolor

        self.glass.fullbar_color(c)
        color = self.color[c]

        if dir == 'L':
            if self.pc_feedback:
                cv2.rectangle(self.img, (self.xl1 - dx, self.yl1), (self.xl1, self.yr1), color, -1)
            if self.glass_feedback:
                self.glass.move_bar(dir, dx, overlay)
        elif dir == 'U':
            if self.pc_feedback:
                cv2.rectangle(self.img, (self.xl1, self.yl1 - dx), (self.xr1, self.yl1), color, -1)
            if self.glass_feedback:
                self.glass.move_bar(dir, dx, overlay)
        elif dir == 'R':
            if self.pc_feedback:
                cv2.rectangle(self.img, (self.xr1, self.yl1), (self.xr1 + dx, self.yr1), color, -1)
            if self.glass_feedback:
                self.glass.move_bar(dir, dx, overlay)
        elif dir == 'D':
            if self.pc_feedback:
                cv2.rectangle(self.img, (self.xl1, self.yr1), (self.xr1, self.yr1 + dx), color, -1)
            if self.glass_feedback:
                self.glass.move_bar(dir, dx, overlay)
        elif dir == 'B':
            if self.pc_feedback:
                cv2.rectangle(self.img, (self.xl1 - dx, self.yl1), (self.xl1, self.yr1), color, -1)
                cv2.rectangle(self.img, (self.xr1, self.yl1), (self.xr1 + dx, self.yr1), color, -1)
            if self.glass_feedback:
                self.glass.move_bar('S', dx, overlay)
        else:
            logger.error('Unknown direction %s' % dir)
        self.put_text(caption, caption_color)

    def put_text(self, txt, color='W'):
        cv2.putText(self.img, txt[:self.textlimit].center(self.textlimit, ' '), (self.text_x, self.text_y),\
                    cv2.FONT_HERSHEY_DUPLEX, self.text_size, self.color[color], 2, cv2.LINE_AA)

    def update(self):
        cv2.imshow("img", self.img)
        #time.sleep(0.0005) # needed for CV to update window -> seems ok without this line

    # Glass functions
    def glass_draw_cue(self):
        self.glass.draw_cross()

    def glass_fullbarcolor(self, color):
        self.glass.set_fullbar_color(color)
