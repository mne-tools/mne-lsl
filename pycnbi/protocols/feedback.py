from __future__ import print_function, division

"""
Visual feedback with online decoding

Kyuhwa Lee
Swiss Federal Institute of Technology Lausanne (EPFL)


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

import pycnbi
import cv2
import os
import pycnbi.utils.q_common as qc
import numpy as np
import time
import serial
import serial.tools.list_ports

# visualization
keys = {'left':81, 'right':83, 'up':82, 'down':84, 'pgup':85, 'pgdn':86, 'home':80, 'end':87, 'space':32, 'esc':27 \
    , ',':44, '.':46, 's':115, 'c':99, '[':91, ']':93, '1':49, '!':33, '2':50, '@':64, '3':51, '#':35}
color = dict(G=(20, 140, 0), B=(210, 0, 0), R=(0, 50, 200), Y=(0, 215, 235), K=(0, 0, 0), W=(255, 255, 255),
             w=(200, 200, 200))
dirs = {'L':'LEFT', 'R':'RIGHT', 'U':'UP', 'D':'DOWN', 'B':'BOTH'}


class Feedback:
    """
    Perform a classification with visual feedback
    """

    def __init__(self, cfg, viz, tdef, trigger, logfile=None):
        self.cfg = cfg
        self.tdef = tdef
        self.alpha1 = self.cfg.PROB_ACC_ALPHA
        self.alpha2 = 1.0 - self.cfg.PROB_ACC_ALPHA
        self.trigger = trigger

        self.viz = viz
        self.viz.fill()
        self.refresh_delay = 1.0 / self.cfg.REFRESH_RATE
        self.bar_step_left = self.cfg.BAR_STEP_LEFT
        self.bar_step_right = self.cfg.BAR_STEP_RIGHT
        self.bar_step_up = self.cfg.BAR_STEP_UP
        self.bar_step_down = self.cfg.BAR_STEP_DOWN
        self.bar_step_both = self.cfg.BAR_STEP_BOTH
        self.bar_bias = self.cfg.BAR_BIAS

        if hasattr(self.cfg, 'BAR_REACH_FINISH') and self.cfg.BAR_REACH_FINISH == True:
            self.premature_end = True
        else:
            self.premature_end = False

        self.tm_trigger = qc.Timer()
        self.tm_display = qc.Timer()
        self.tm_watchdog = qc.Timer()
        if logfile is not None:
            self.logf = open(logfile, 'w')
        else:
            self.logf = None

        # STIMO only
        if self.cfg.WITH_STIMO is True:
            if self.cfg.STIMO_COMPORT is None:
                atens = [x for x in serial.tools.list_ports.grep('ATEN')]
                if len(atens) == 0:
                    raise RuntimeError('No ATEN device found. Stop.')
                #for i, a in enumerate(atens):
                #    print('Found', a[0])
                try:
                    self.stimo_port = atens[0].device
                except AttributeError: # depends on Python distribution
                    self.stimo_port = atens[0][0]
            else:
                self.stimo_port = self.cfg.STIMO_COMPORT
            self.ser = serial.Serial(self.stimo_port, self.cfg.STIMO_BAUDRATE)
            print('STIMO serial port %s is_open = %s' % (self.stimo_port, self.ser.is_open))

    def __del__(self):
        # STIMO only
        if self.cfg.WITH_STIMO is True:
            self.ser.close()
            print('Closed STIMO serial port %s' % self.stimo_port)

    def classify(self, decoder, true_label, title_text, bar_dirs, state='start'):
        """
        Run a single trial
        """
        self.tm_trigger.reset()
        if self.bar_bias is not None:
            bias_idx = bar_dirs.index(self.bar_bias[0])

        if self.logf is not None:
            self.logf.write('True label: %s\n' % true_label)

        tm_classify = qc.Timer(autoreset=True)
        while True:
            self.tm_display.sleep_atleast(self.refresh_delay)
            self.tm_display.reset()
            if state == 'start' and self.tm_trigger.sec() > self.cfg.T_INIT:
                state = 'gap_s'
                self.viz.fill()
                self.tm_trigger.reset()
                self.trigger.signal(self.tdef.INIT)

            elif state == 'gap_s':
                if self.cfg.T_GAP > 0:
                    self.viz.put_text(title_text)
                state = 'gap'
                self.tm_trigger.reset()

            elif state == 'gap' and self.tm_trigger.sec() > self.cfg.T_GAP:
                state = 'cue'
                self.viz.fill()
                self.viz.draw_cue()
                self.viz.glass_draw_cue()
                self.trigger.signal(self.tdef.CUE)
                self.tm_trigger.reset()

            elif state == 'cue' and self.tm_trigger.sec() > self.cfg.T_READY:
                state = 'dir_r'

                if self.cfg.SHOW_CUE is True:
                    if self.cfg.FEEDBACK_TYPE == 'BAR':
                        self.viz.move(true_label, 100, overlay=False, barcolor='G')
                    elif self.cfg.FEEDBACK_TYPE == 'BODY':
                        self.viz.put_text(dirs[true_label], 'R')
                    if true_label == 'L':  # left
                        self.trigger.signal(self.tdef.LEFT_READY)
                    elif true_label == 'R':  # right
                        self.trigger.signal(self.tdef.RIGHT_READY)
                    elif true_label == 'U':  # up
                        self.trigger.signal(self.tdef.UP_READY)
                    elif true_label == 'D':  # down
                        self.trigger.signal(self.tdef.DOWN_READY)
                    elif true_label == 'B':  # both hands
                        self.trigger.signal(self.tdef.BOTH_READY)
                    else:
                        raise RuntimeError('Unknown direction %s' % true_label)
                self.tm_trigger.reset()
                '''
                if self.cfg.FEEDBACK_TYPE == 'BODY':
                    self.viz.set_pc_feedback(False)
                self.viz.move(true_label, 100, overlay=False, barcolor='G')

                if self.cfg.FEEDBACK_TYPE == 'BODY':
                    self.viz.set_pc_feedback(True)
                    if self.cfg.SHOW_CUE is True:
                        self.viz.put_text(dirs[true_label], 'R')

                if true_label == 'L':  # left
                    self.trigger.signal(self.tdef.LEFT_READY)
                elif true_label == 'R':  # right
                    self.trigger.signal(self.tdef.RIGHT_READY)
                elif true_label == 'U':  # up
                    self.trigger.signal(self.tdef.UP_READY)
                elif true_label == 'D':  # down
                    self.trigger.signal(self.tdef.DOWN_READY)
                elif true_label == 'B':  # both hands
                    self.trigger.signal(self.tdef.BOTH_READY)
                else:
                    raise RuntimeError('Unknown direction %s' % true_label)
                self.tm_trigger.reset()
                '''
            elif state == 'dir_r' and self.tm_trigger.sec() > self.cfg.T_DIR_CUE:
                self.viz.fill()
                self.viz.draw_cue()
                self.viz.glass_draw_cue()
                state = 'dir'

                # initialize bar scores
                bar_label = bar_dirs[0]
                bar_score = 0
                probs = [1.0 / len(bar_dirs)] * len(bar_dirs)
                self.viz.move(bar_label, bar_score, overlay=False)
                probs_acc = np.zeros(len(probs))

                if true_label == 'L':  # left
                    self.trigger.signal(self.tdef.LEFT_GO)
                elif true_label == 'R':  # right
                    self.trigger.signal(self.tdef.RIGHT_GO)
                elif true_label == 'U':  # up
                    self.trigger.signal(self.tdef.UP_GO)
                elif true_label == 'D':  # down
                    self.trigger.signal(self.tdef.DOWN_GO)
                elif true_label == 'B':  # both
                    self.trigger.signal(self.tdef.BOTH_GO)
                else:
                    raise RuntimeError('Unknown truedirection %s' % true_label)

                self.tm_watchdog.reset()
                self.tm_trigger.reset()

            elif state == 'dir':
                if self.tm_trigger.sec() > self.cfg.T_CLASSIFY or (self.premature_end and bar_score >= 100):
                    if not hasattr(self.cfg, 'SHOW_RESULT') or self.cfg.SHOW_RESULT is True:
                        # show classfication result
                        if self.cfg.WITH_STIMO is True and \
                            (self.cfg.TRIALS_RETRY is False or bar_label == true_label):
                            res_color = 'B'
                        else:
                            res_color = 'Y'
                        if self.cfg.FEEDBACK_TYPE == 'BODY':
                            self.viz.move(bar_label, bar_score, overlay=False, barcolor=res_color, caption=dirs[bar_label], caption_color=res_color)
                        else:
                            self.viz.move(bar_label, 100, overlay=False, barcolor=res_color)
                    else:
                        if self.cfg.FEEDBACK_TYPE == 'BODY':
                            self.viz.move(bar_label, bar_score, overlay=False, barcolor=res_color, caption='TRIAL END', caption_color=res_color)
                        else:
                            self.viz.move(bar_label, 0, overlay=False, barcolor=res_color)
                    self.trigger.signal(self.tdef.FEEDBACK)

                    # STIMO
                    if self.cfg.WITH_STIMO is True and \
                        (self.cfg.TRIALS_RETRY is False or bar_label == true_label):
                        if bar_label == 'L':
                            self.ser.write(b'1')
                            qc.print_c('STIMO: Sent 1', 'g')
                        elif bar_label == 'R':
                            self.ser.write(b'2')
                            qc.print_c('STIMO: Sent 2', 'g')

                    probs_acc /= sum(probs_acc)
                    if self.cfg.DEBUG_PROBS:
                        msg = 'DEBUG: Accumulated probabilities = %s' % qc.list2string(probs_acc, '%.3f')
                        print(msg)
                        if self.logf is not None:
                            self.logf.write(msg + '\n')
                    if self.logf is not None:
                        self.logf.write('%s detected as %s (%d)\n\n' % (true_label, bar_label, bar_score))
                        self.logf.flush()

                    # end of trial
                    state = 'feedback'
                    self.tm_trigger.reset()
                else:
                    # classify
                    probs_new = decoder.get_prob_unread()
                    if probs_new is None:
                        if self.tm_watchdog.sec() > 3:
                            qc.print_c('WARNING: No classification being done. Are you receiving data streams?', 'Y')
                            self.tm_watchdog.reset()
                    else:
                        self.tm_watchdog.reset()

                        # bias and accumulate
                        if self.bar_bias is not None:
                            # print('BEFORE: %.3f %.3f'% (probs_new[0], probs_new[1]) )
                            probs_new[bias_idx] += self.bar_bias[1]
                            newsum = sum(probs_new)
                            probs_new = [p / newsum for p in probs_new]
                        # print('AFTER: %.3f %.3f'% (probs_new[0], probs_new[1]) )

                        probs_acc += np.array(probs_new)
                        for i in range(len(probs_new)):
                            probs[i] = probs[i] * self.alpha1 + probs_new[i] * self.alpha2

                        ''' Original: accumulate and bias
                        # accumulate probs
                        for i in range( len(probs_new) ):
                            probs[i]= probs[i] * self.alpha1 + probs_new[i] * self.alpha2

                        # bias bar
                        if self.bar_bias is not None:
                            probs[bias_idx] += self.bar_bias[1]
                            newsum= sum(probs)
                            probs= [p/newsum for p in probs]
                        '''

                        # determine the direction
                        # TODO: np.argmax(probs)
                        max_pidx = qc.get_index_max(probs)
                        max_label = bar_dirs[max_pidx]

                        if self.cfg.POSITIVE_FEEDBACK is False or \
                                (self.cfg.POSITIVE_FEEDBACK and true_label == max_label):
                            dx = probs[max_pidx]
                            if max_label == 'R':
                                dx *= self.bar_step_right
                            elif max_label == 'L':
                                dx *= self.bar_step_left
                            elif max_label == 'U':
                                dx *= self.bar_step_up
                            elif max_label == 'D':
                                dx *= self.bar_step_down
                            elif max_label == 'B':
                                dx *= self.bar_step_both
                            else:
                                print('DEBUG: Direction %s using bar step %d' % (max_label, self.bar_step_left))
                                dx *= self.bar_step_left

                            ################################################
                            ################################################
                            # slower in the beginning?
                            if self.tm_trigger.sec() < 2.0:
                                dx *= self.tm_trigger.sec() * 0.5
                            ################################################
                            ################################################

                            # add likelihoods
                            if max_label == bar_label:
                                bar_score += dx
                            else:
                                bar_score -= dx
                                # change of direction
                                if bar_score < 0:
                                    bar_score = -bar_score
                                    bar_label = max_label
                            bar_score = int(bar_score)
                            if bar_score > 100:
                                bar_score = 100
                            if self.cfg.FEEDBACK_TYPE == 'BODY':
                                if self.cfg.SHOW_CUE:
                                    self.viz.move(bar_label, bar_score, overlay=False, caption=dirs[true_label], caption_color='G')
                                else:
                                    self.viz.move(bar_label, bar_score, overlay=False)
                            else:
                                self.viz.move(bar_label, bar_score, overlay=False)

                        if self.cfg.DEBUG_PROBS:
                            if self.bar_bias is not None:
                                biastxt = '[Bias=%s%.3f]  ' % (self.bar_bias[0], self.bar_bias[1])
                            else:
                                biastxt = ''
                            msg = '%s%s  raw %s   acc %s   bar %s%d  (%.1f ms)' % \
                                  (biastxt, bar_dirs, qc.list2string(probs_new, '%.2f'), qc.list2string(probs, '%.2f'),
                                   bar_label, bar_score, tm_classify.msec())
                            print(msg)
                            if self.logf is not None:
                                self.logf.write(msg + '\n')

            elif state == 'feedback' and self.tm_trigger.sec() > self.cfg.T_FEEDBACK:
                self.trigger.signal(self.tdef.BLANK)
                if self.cfg.FEEDBACK_TYPE == 'BODY':
                    state = 'return'
                    self.tm_trigger.reset()
                else:
                    state = 'gap_s'
                    self.viz.fill()
                    self.viz.update()
                    return bar_label

            elif state == 'return':
                self.viz.set_glass_feedback(False)
                if self.cfg.WITH_STIMO:
                    self.viz.move(bar_label, bar_score, overlay=False, barcolor='B')
                else:
                    self.viz.move(bar_label, bar_score, overlay=False, barcolor='Y')
                self.viz.set_glass_feedback(True)
                bar_score -= 5
                if bar_score <= 0:
                    state = 'gap_s'
                    self.viz.fill()
                    self.viz.update()
                    return bar_label

            self.viz.update()
            key = 0xFF & cv2.waitKey(1)
            if key == keys['esc']:
                return None
            if key == keys['space']:
                dx = 0
                bar_score = 0
                probs = [1.0 / len(bar_dirs)] * len(bar_dirs)
                self.viz.move(bar_dirs[0], bar_score, overlay=False)
                self.viz.update()
                print('RESET', probs, dx)
