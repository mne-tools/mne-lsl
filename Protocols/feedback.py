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

import pycnbi_config
import cv2
import q_common as qc
from IPython import embed

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

    def __init__(self, cfg, bar, tdef, trigger):
        self.cfg = cfg
        self.tdef = tdef
        self.alpha1 = self.cfg.PROB_ACC_ALPHA
        self.alpha2 = 1.0 - self.cfg.PROB_ACC_ALPHA
        self.trigger = trigger

        self.bar = bar
        self.bar.fill()
        self.refresh_delay = 1.0 / self.cfg.REFRESH_RATE
        self.bar_step_left = self.cfg.BAR_STEP_LEFT
        self.bar_step_right = self.cfg.BAR_STEP_RIGHT
        self.bar_bias = self.cfg.BAR_BIAS

        if hasattr(self.cfg, 'BAR_REACH_FINISH') and self.cfg.BAR_REACH_FINISH == True:
            self.premature_end = True
        else:
            self.premature_end = False

        self.tm_trigger = qc.Timer()
        self.tm_display = qc.Timer()
        self.tm_watchdog = qc.Timer()

    def classify(self, decoder, true_label, title_text, bar_dirs, state='start'):
        """
        Run a single trial
        """
        self.tm_trigger.reset()
        if self.bar_bias is not None:
            bias_idx = bar_dirs.index(self.bar_bias[0])

        tm_classify = qc.Timer()
        while True:
            self.tm_display.sleep_atleast(self.refresh_delay)
            self.tm_display.reset()
            if state == 'start' and self.tm_trigger.sec() > self.cfg.T_INIT:
                state = 'gap_s'
                self.bar.fill()
                self.tm_trigger.reset()
                self.trigger.signal(self.tdef.INIT)

            elif state == 'gap_s':
                self.bar.put_text(title_text)
                state = 'gap'
                self.tm_trigger.reset()

            elif state == 'gap' and self.tm_trigger.sec() > self.cfg.T_GAP:
                state = 'cue'
                self.bar.fill()
                self.bar.draw_cue()
                self.bar.glass_draw_cue()
                self.trigger.signal(self.tdef.CUE)
                self.tm_trigger.reset()

            elif state == 'cue' and self.tm_trigger.sec() > self.cfg.T_READY:
                state = 'dir_r'

                if self.cfg.FEEDBACK_TYPE == 'BODY':
                    self.bar.set_pc_feedback(False)
                self.bar.move(true_label, 100, overlay=False, barcolor='G')
                if self.cfg.FEEDBACK_TYPE == 'BODY':
                    self.bar.set_pc_feedback(True)
                    self.bar.put_text(dirs[true_label])

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

            elif state == 'dir_r' and self.tm_trigger.sec() > self.cfg.T_DIR_CUE:
                self.bar.fill()
                self.bar.draw_cue()
                self.bar.glass_draw_cue()
                state = 'dir'

                # initialize bar scores
                bar_label = bar_dirs[0]
                bar_score = 0
                probs = [1.0 / len(bar_dirs)] * len(bar_dirs)
                self.bar.move(bar_label, bar_score, overlay=False)

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
                    if self.cfg.FEEDBACK_TYPE == 'BODY':
                        self.bar.set_pc_feedback(False)
                    self.bar.move(bar_label, 100, overlay=False, barcolor='Y')
                    if self.cfg.FEEDBACK_TYPE == 'BODY':
                        self.bar.set_pc_feedback(True)
                        self.bar.put_text(dirs[bar_label])
                    self.trigger.signal(self.tdef.FEEDBACK)
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

                        ###################################
                        # bias and accumulate
                        if self.bar_bias is not None:
                            # print('BEFORE: %.3f %.3f'% (probs_new[0], probs_new[1]) )
                            probs_new[bias_idx] += self.bar_bias[1]
                            newsum = sum(probs_new)
                            probs_new = [p / newsum for p in probs_new]
                        # print('AFTER: %.3f %.3f'% (probs_new[0], probs_new[1]) )

                        for i in range(len(probs_new)):
                            probs[i] = probs[i] * self.alpha1 + probs_new[i] * self.alpha2
                        '''
                        # decision thresholding (confidence based)
                        max_p= max( probs_new )
                        if max_p < 0.52: ####################
                            print('Sample ignored: %.2f'% max_p )
                        else:
                            for i in range( len(probs_new) ):
                                probs[i]= probs[i] * self.alpha1 + probs_new[i] * self.alpha2
                        '''
                        ###################################


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
                        max_pidx = qc.get_index_max(probs)
                        max_label = bar_dirs[max_pidx]

                        if self.cfg.POSITIVE_FEEDBACK is False or \
                                (self.cfg.POSITIVE_FEEDBACK and true_label == max_label):
                            dx = probs[max_pidx]
                            if max_label == 'R':
                                dx *= self.bar_step_right
                            elif max_label == 'L':
                                dx *= self.bar_step_left
                            else:
                                # For now, only left and right are considered for different stepping speeds.
                                # For other directions, only BAR_STEP is assumed to be defined in the config.
                                print('DEBUG: Direction %s using bar step %d' % (max_label, self.bar_step_left))
                                dx *= self.bar_step_left

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

                            self.bar.move(bar_label, bar_score, overlay=False)

                        if self.cfg.DEBUG_PROBS:
                            if self.bar_bias is not None:
                                biastxt = '[Bias=%s%.3f]  ' % (self.bar_bias[0], self.bar_bias[1])
                            else:
                                biastxt = ''
                            print('%s%s  raw %s   acc %s   bar %s%d  (%.1f ms)' % \
                                  (biastxt, bar_dirs, qc.list2string(probs_new, '%.2f'), qc.list2string(probs, '%.2f'),
                                   bar_label, bar_score, tm_classify.msec()))
                            tm_classify.reset()

            elif state == 'feedback' and self.tm_trigger.sec() > self.cfg.T_FEEDBACK:
                self.trigger.signal(self.tdef.BLANK)
                if self.cfg.FEEDBACK_TYPE == 'BODY':
                    state = 'return'
                    self.tm_trigger.reset()
                else:
                    state = 'gap_s'
                    self.bar.fill()
                    self.bar.update()
                    return bar_label

            elif state == 'return':
                self.bar.set_glass_feedback(False)
                self.bar.move(bar_label, bar_score, overlay=False, barcolor='Y')
                self.bar.set_glass_feedback(True)
                bar_score -= 5
                if bar_score <= 0:
                    state = 'gap_s'
                    self.bar.fill()
                    self.bar.update()
                    return bar_label

            self.bar.update()
            key = 0xFF & cv2.waitKey(1)

            if key == keys['esc']:
                return None
            if key == keys['space']:
                dx = 0
                bar_score = 0
                probs = [1.0 / len(bar_dirs)] * len(bar_dirs)
                self.bar.move(bar_dirs[0], bar_score, overlay=False)
                self.bar.update()
                print('RESET', probs, dx)
