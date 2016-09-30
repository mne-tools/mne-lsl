PyCNBI provides an online brain signal decoding framework.

A motor imagery classification can be run at approx. 15 Hz on a 3rd-gen i7 2.4 GHz laptop. Although this is more than enough for myself, there are plenty of possibilities to optimize the speed.

The data communication is based on the lab streaming layer (LSL). For more information about LSL, please visit:
https://github.com/sccn/labstreaminglayer

Although default LSL servers can be used, the following customized servers are recommended:

It's because many LSL servers do not stream event channel as part of the signal stream.
The above acquisition servers support simultaneous signal+event channel streaming.
If you use OpenVibe acquisition servers, make sure to enable "LSL output".

Some important folders from the user's perspective are:

StreamViewer
StreamRecorder
Decoder
OpenVibe
Protocols
Triggers
Utils
StreamViewer
This module visualizes signals in real time with bandpass filtering and common average filtering options.
It relies on StreamReceiver module.

StreamRecorder
This is used to record signals into a file. It relies on StreamReceiver module.

Decoder
This folder contains decoder and trainer modules.
decoder.py: perform online classification and output class probabilities.
trainer.py: perform cross-validation and/or train a classifier.

OpenVibe
You can find various montage settings for OpenVibe servers, which can be loaded from OpenVibe acqisuition servers.

Protocols
Contains some basic protocols for training and testing. Google Glass visual feedback is supported through USB communication.

Triggers
Common trigger event definition files.

Utils
Contains various utilities.

It is initially developed for my personal projects and not fully cleaned up yet. It has been applied in many different online scenarios (hand imagery, leg imagery, error-related potentials) and shown to be working so far, but the code is not in the most efficient form. Any contribution is welcome. Please contact kyu.lee@epfl.ch if interested in helping me.
