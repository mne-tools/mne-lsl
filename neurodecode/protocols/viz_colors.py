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
import neurodecode.glass.bgi_client as bgi_client
from neurodecode import logger


class ColorVisual(object):
    # Default setting
    color = dict(G=(20, 140, 0), B=(200, 0, 0), R=(0, 0, 200),
        Y=(0, 215, 235), K=(0, 0, 0), W=(255, 255, 255), w=(200, 200, 200))
    barwidth = 100
    textlimit = 25  # maximum number of characters to show

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
                self.screen_width = GetSystemMetrics(0)
                self.screen_height = GetSystemMetrics(1)
            else:
                screen_width = 1024
                screen_height = 768
        else:
            self.screen_width, self.screen_height = screen_size

        if screen_pos is None:
            screen_x, screen_y = (0, 0)
        else:
            screen_x, screen_y = screen_pos

        self.text_x = int(self.screen_width / 2)
        self.text_y = int(self.screen_height / 2)
        self.text_size = 2
        cv2.namedWindow("img", cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow("img", screen_x, screen_y)
        cv2.setWindowProperty("img", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN);

        self.img = np.zeros((self.screen_height, self.screen_width, 3), np.uint8)
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
        
        self.outerRadiusThickness = 5
        self.outerCircleRadius = int(self.cx/8)        
        self.innerCircleRadius = self.outerCircleRadius - self.outerRadiusThickness / 2
        
        self.xl1 = self.cx - hw  # 200
        self.xl2 = self.xl1 - self.barwidth  # 100
        self.xr1 = self.cx + hw  # 300
        self.xr2 = self.xr1 + self.barwidth  # 400
        self.yl1 = self.cy - hw
        self.yl2 = self.yl1 - self.barwidth
        self.yr1 = self.cy + hw
        self.yr2 = self.yr1 + self.barwidth
        
        self.actual_cue_color = None

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
    def draw_cue(self, dir=None):
        if dir is None:
            color =  self.color['W']
            self.draw_fixation_cross()
        elif dir == 'L':
            color =  self.color['B']
            self.actual_cue = dir
            self.put_text('Ghi')
        elif dir == 'R':
            color =  self.color['R']
            self.actual_cue = dir
            self.put_text('Fo')
        
        cv2.circle(self.img, (self.cx, self.cy), int(self.cx/8), color, self.outerRadiusThickness)
        
    def draw_fixation_cross(self):
        # cross fixation point
        cv2.rectangle( self.img, (self.cx-10,self.cy), (self.cx+10,self.cy), self.crosscol, thickness=3 )
        cv2.rectangle( self.img, (self.cx,self.cy-10), (self.cx,self.cy+10), self.crosscol, thickness=3 )
        
    # paints the new bar on top of the current image
    def move(self, dir, dx, overlay=False, barcolor=None, caption='', caption_color='W'):
        
        #if barcolor is None:
            #if dx == self.xl2:
                #c = 'G'
            #else:
                #c = 'R'
        #else:
            #c = barcolor

        #color = self.color[c]

        if dir == 'L':
            if self.pc_feedback:
                if barcolor is None:
                    color = ((dx/100) * self.color['B'][0], self.color['B'][1], self.color['B'][2])
                else:
                    color = self.color[barcolor]
                # color = ((dx/100) * self.color['B'][0], self.color['B'][1], self.color['B'][2])
                # color = ((dx/100) * color[0], color[1], color[2])
                cv2.circle(self.img, (self.cx, self.cy), int(self.cx/8) - int(self.outerRadiusThickness-2), color, -1)
                # self.draw_fixation_cross()
                if not overlay:
                    self.draw_cue(self.actual_cue)

        elif dir == 'R':
            if self.pc_feedback:
                if barcolor is None:
                    color = (self.color['R'][0], self.color['R'][1], (dx/100) * self.color['R'][2])
                else:
                    color = self.color[barcolor]                    
                # color = (color[0], color[1], (dx/100) * color[2])
                # color = (self.color['R'][0], self.color['R'][1], (dx/100) * self.color['R'][2])
                cv2.circle(self.img, (self.cx, self.cy), int(self.cx/8) - int(self.outerRadiusThickness-2), color, -1)
                # self.draw_fixation_cross()
                if not overlay:
                    self.draw_cue(self.actual_cue)
        
        #elif dir == 'U':
            #if self.pc_feedback:tas
                #color = (self.color['R'][0], self.color['R'][1], (dx/100) * self.color['R'][2])
                #cv2.circle(self.img, (self.cx, self.cy), int(self.cx/8), color, -1)
                #self.draw_fixation_cross()                
        
            
        #elif dir == 'D':
            #if self.pc_feedback:
                #color = (self.color['R'][0], self.color['R'][1], (dx/100) * self.color['R'][2])
                #cv2.circle(self.img, (self.cx, self.cy), int(self.cx/8), color, -1)
                #self.draw_fixation_cross()                
            
        else:
            logger.error('Unknown direction %s' % dir)
        self.put_text(caption, caption_color)

    #def put_text(self, txt, color='W'):
        #cv2.putText(self.img, txt[:self.textlimit].center(self.textlimit, ' '), (self.text_x, self.text_y),\
                    #cv2.FONT_HERSHEY_DUPLEX, self.text_size, self.color[color], 2, cv2.LINE_AA)
        
    
    def put_text(self, txt, color='W'):
        textsize = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 1, 2)[0]        

        text_x = int((self.screen_width - textsize[0]) / 2)
        text_y = int((self.screen_height + textsize[1]) / 2)

        cv2.putText(self.img, txt, (text_x, text_y),\
                    cv2.FONT_HERSHEY_DUPLEX, 1, self.color[color], 2)
        
    
    def update(self):
        cv2.imshow("img", self.img)
        #time.sleep(0.0005) # needed for CV to update window -> seems ok without this line

    # Glass functions
    def glass_draw_cue(self):
        self.glass.draw_cross()

    def glass_fullbarcolor(self, color):
        self.glass.set_fullbar_color(color)

        
if __name__ == '__main__':
    import cv2
    viz = ColorVisual()
    viz.draw_cue('L')
    viz.move('R', 100)
    viz.update()
    cv2.waitKey(-1)