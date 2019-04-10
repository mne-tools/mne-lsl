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
import os
import gzip
import time
import cv2
import numpy as np
import pycnbi
import pycnbi.glass.bgi_client as bgi_client
import pycnbi.utils.q_common as qc
from pycnbi import logger
from builtins import input
try:
    import cPickle as pickle  # Python 2 (cPickle = C version of pickle)
except ImportError:
    import pickle  # Python 3 (C version is the default)


def read_images(img_path, screen_size=None):
    pnglist = []
    for f in qc.get_file_list(img_path):
        if f[-4:] != '.png':
            continue

        img = cv2.imread(f)
        # fit to screen size if image is larger
        if screen_size is not None:
            screen_width, screen_height = screen_size
            rx = img.shape[1] / screen_width
            ry = img.shape[0] / screen_height
            if max(rx, ry) > 1:
                if rx > ry:
                    target_w = screen_width
                    target_h = int(img.shape[0] / rx)
                elif rx < ry:
                    target_w = int(img.shape[1] / ry)
                    target_h = screen_height
                else:
                    target_w = screen_width
                    target_h = screen_height
            else:
                target_w = img.shape[1]
                target_h = img.shape[0]
            dsize = (int(target_w), int(target_h))
            img_res = cv2.resize(img, dsize, interpolation=cv2.INTER_LANCZOS4)
            img_out = np.zeros((screen_height, screen_width, img.shape[2]), dtype=img.dtype)
            ox = int((screen_width - target_w) / 2)
            oy = int((screen_height - target_h) / 2)
            img_out[oy:oy+target_h, ox:ox+target_w, :] = img_res
        else:
            img_out = img
        pnglist.append(img_out)
        print('.', end='')
    logger.info('Done loading images.')
    return pnglist


class BodyVisual(object):
    # Default setting
    color = dict(G=(20, 140, 0), B=(255, 90, 0), R=(0, 50, 200), Y=(0, 215, 235), K=(0, 0, 0),\
                 W=(255, 255, 255), w=(200, 200, 200))
    barwidth = 100
    textlimit = 25  # maximum number of characters to show

    def __init__(self, image_path, use_glass=False, glass_feedback=True, pc_feedback=True, screen_pos=None, screen_size=None):
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
            screen_size = (screen_width, screen_height)
        else:
            screen_width, screen_height = screen_size
        if screen_pos is None:
            screen_x, screen_y = (0, 0)
        else:
            screen_x, screen_y = screen_pos

        self.text_size = 2
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
        self.xl1 = self.cx - hw
        self.xl2 = self.xl1 - self.barwidth
        self.xr1 = self.cx + hw
        self.xr2 = self.xr1 + self.barwidth
        self.yl1 = self.cy - hw
        self.yl2 = self.yl1 - self.barwidth
        self.yr1 = self.cy + hw
        self.yr2 = self.yr1 + self.barwidth

        if os.path.isdir(image_path):
            # load images
            left_image_path = '%s/left' % image_path
            right_image_path = '%s/right' % image_path
            tm = qc.Timer()
            logger.info('Reading images from %s' % left_image_path )
            self.left_images = read_images(left_image_path, screen_size)
            logger.info('Reading images from %s' % right_image_path)
            self.right_images = read_images(right_image_path, screen_size)
            logger.info('Took %.1f s' % tm.sec())
        else:
            # load pickled images
            # note: this is painfully slow in Pytohn 2 even with cPickle (3s vs 27s)
            assert image_path[-4:] == '.pkl', 'The file must be of .pkl format'
            logger.info('Loading image binary file %s ...' % image_path)
            tm = qc.Timer()
            with gzip.open(image_path, 'rb') as fp:
                image_data = pickle.load(fp)
            self.left_images = image_data['left_images']
            self.right_images = image_data['right_images']
            feedback_w = self.left_images[0].shape[1] / 2
            feedback_h = self.left_images[0].shape[0] / 2
            loc_x = [int(self.cx - feedback_w), int(self.cx + feedback_w)]
            loc_y = [int(self.cy - feedback_h), int(self.cy + feedback_h)]
            img_fit = np.zeros((screen_height, screen_width, 3), np.uint8)

            # adjust to the current screen size
            logger.info('Fitting images into the current screen size')
            for i, img in enumerate(self.left_images):
                img_fit = np.zeros((screen_height, screen_width, 3), np.uint8)
                img_fit[loc_y[0]:loc_y[1], loc_x[0]:loc_x[1]] = img
                self.left_images[i] = img_fit
            for i, img in enumerate(self.right_images):
                img_fit = np.zeros((screen_height, screen_width, 3), np.uint8)
                img_fit[loc_y[0]:loc_y[1], loc_x[0]:loc_x[1]] = img
                self.right_images[i] = img_fit

            logger.info('Took %.1f s' % tm.sec())
            logger.info('Done.')
        cv2.namedWindow("Protocol", cv2.WND_PROP_FULLSCREEN)
        cv2.moveWindow("Protocol", screen_x, screen_y)
        cv2.setWindowProperty("Protocol", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN);

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
        self.img = self.left_images[0]
        self.update()

    # draw cue with custom colors
    def draw_cue(self):
        self.img = self.left_images[0]
        self.update()

    # paints the new bar on top of the current image
    def move(self, dir, dx, overlay=False, barcolor=None, caption='', caption_color='W'):
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
                self.img = self.left_images[dx]
            if self.glass_feedback:
                self.glass.move_bar(dir, dx, overlay)
        elif dir == 'R':
            if self.pc_feedback:
                self.img = self.right_images[dx]
            if self.glass_feedback:
                self.glass.move_bar(dir, dx, overlay)
        else:
            logger.error('Unknown direction %s' % dir)
        self.put_text(caption, caption_color)
        self.update()

    def put_text(self, txt, color='W'):
        self.img = self.img.copy()
        size_wh, baseline = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, self.text_size, 2)
        pos = (int(self.cx - size_wh[0] / 2), int(self.cy - size_wh[1] / 2))
        cv2.putText(self.img, txt[:self.textlimit], pos,
                    cv2.FONT_HERSHEY_DUPLEX, self.text_size, self.color[color], 2, cv2.LINE_AA)
        self.update()

    def update(self):
        cv2.imshow("Protocol", self.img)
        #cv2.waitKey(1)  # at least 1 ms needed for CV to update window
        time.sleep(0.0005)

    # Glass functions
    def glass_draw_cue(self):
        self.glass.draw_cross()

    def glass_fullbarcolor(self, color):
        self.glass.set_fullbar_color(color)
