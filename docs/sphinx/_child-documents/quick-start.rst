===========
Quick-Start
===========

******************
Data communication
******************

The underlying data communication is based on LabStreamingLayer (LSL), which provides sub-millisecond time synchronization accuracy. Any signal acquisition system supported by native LSL or OpenVibe is also supported by NeuroDecode. Since the data communication is based on TCP, signals can be also transmitted wirelessly. For more information about LSL, please visit: 
`LabStreamingLayer <https://github.com/sccn/labstreaminglayer>`_.


*****************
Low-Level modules
*****************

NeuroDecode provides low-level modules for receiving, replaying, recording, visualizing and processing EEG data. It aims at providing fast and easy implementations of neurofeedback and Brain-Computer Interface protocols.


*************
Motor Imagery
*************

NeuroDecode provides a ready to use protocols for Motor Imagery experiments. The decoding performance was recognized at `Microsoft Brain Signal Decoding competition <https://github.com/dbdq/microsoft_decoding>`_ with the First Prize Award (2016) considering high decoding accuracy. It has been applied on a couple of online decoding projects based on EEG and ECoG and on various acquisition systems including AntNeuro eego, g.tec gUSBamp, BioSemi ActiveTwo, BrainProducts actiCHamp and Wearable Sensing. The decoding runs at approximately 15 classifications per second(cps) on a 4th-gen i7 laptop with 64-channel setup at 512 Hz sampling rate. High-speed decoding up to 200 cps was achieved using process-interleaving technique on 8 cores. It has been tested on both Linux and Windows using Python 3.7.

The scripts can be found `here <https://github.com/fcbg-hnp/NeuroDecode/tree/master/neurodecode/protocols/mi>`_. All protocols options can be selected in the config_files. The experiment is splitted in three parts: offline, trainer and online. 

Offline
=======

The subject is asked to perform a MI task for each condition and to repeat them over several runs composed of multiple trials. It is mandatory to collect EEG data to train a classifier. 

As an example, left-right hands MI experiment would work as follow: when the subject sees a red square on the left box, he will perform left-hand MI and vis-versa for the righ box with right hand MI. The image below depicts the bar feedback shown during a left condition.

.. image:: ../../../../_images/mi_offline_images.png
  :width: 500
  :align: center

In general, for unaccustomed subjects, a minimum of 3 runs of 15 trials per condition is necessary to achieve acceptable performance. Splitting trials into runs gives some rest to the subject and keeps the subject commitment high.

Trainer
=======

After the offline part, enough data are collected to train a classifier. Several steps needs to be performed before training the final classifier: 

- *trials epoching*: Data for each trial are extracted from raw data
- *sliding windowing*: To simulate online data acquisition, a sliding window over the trials is done
- *preprocessing*: Rereferencing, spatial, temporal, notch filtering...
- *features extraction*: Compute the Power Spectrum Density (PSD)
- *k-fold cross-validation*: Several classifiers are trained on data subsets and tested on the remaining data. Their performances are then averaged to estimate the classifier robustness to new data (optional).

The final classifier can now be trained on the whole prepared dataset and saved to a .pkl file for online applications. 

Online
======

During the online protocol, the subject will now control the feedback. The feedback displacement is linked to the classifier probabilies as follow:

- *classifier output*: The classifier returns the smoothed probabilities for each classes
- *selecting the winner*: The class with the highest probabilities is selected
- *feedback displacement*: The feedback is moved one step toward the direction associated with this class label.
- *trial end*: The trial ends when the feedback reaches full displacement to one direction or if the maximum allowed trial time is reached

At the end of each online run, the accuracy and the confusion matrix will be shown and saved to a .txt file. Moreover, all the probabilities are also saved to a .txt file. 

***
GUI
***
NeuroDecode provides a GUI for fast protocol launching, recording and eeg visualization. All the parameters from a config_file can be modified at GUI level. The GUI is ready to use for the Motor Imagery
protocol but new protocols can be added to the protocols folder (and their config files in config_files folder).

Notice: 

- The first time a protocol is used, its scripts: *offline_<protocol>.py*, *trainer_<protocol>.py* and *online_<protocol>.py* will be copied to the script folder defined during the installation. The scripts can therefore be modified and adapted outside of the NeuroDecode git folder. 
- For each new subject, the config files (found in *neurodecode/config_files/<protocol>*) will also be copied in the subject specific folder. Therefore, each subject can have his own protocol parameters.
- In *neurodecode/config_files/<protocol>*, structure config files defines all the possible values for each parameters contained in the config_file. They are only used by the GUI, not the protocols scripts.
- Save the config file (GUI file tab) to keep track of the modifications
- GUI is still in development.

.. image:: ../../../../_images/gui.png
  :width: 500
  :align: center

*************
Windows users
*************

The default timer resolution in some Windows versions is 16 ms, which can limit the precision of timings. It is recommended to run the following tool and set the resolution to 1 ms or lower before any experiments: `TimerTool <https://vvvv.org/contribution/windows-system-timer-tool>`_
