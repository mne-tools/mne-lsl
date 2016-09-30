This is a general purpose configurable application to be used with shimmer and any add-on daughter-cards supplied by shimmer-research.


By default this application samples the 3-axis accelerometer at 50Hz and sends the data over the Bluetooth radio, using a data buffer size of 1.

   Data Packet Format:
          Packet Type | TimeStamp | chan1 | chan2 | ... |  chanX 
   Byte:       0      |    1-2    |  3-4  |  5-6  | ... | (x-1)-x


When the application receives an Inquiry command it respons with the following packet. The value in the channel fields indicate exactly what data from which sensor is contained in this field of the data packet:

Inquiry Response Packet Format:
          Packet Type | ADC Sampling rate | Accel Range | Config Byte 0 |Num Chans | Buf size | Chan1 | Chan2 | ... | ChanX
   Byte:       0      |         1         |      2      |       3       |    4     |     5    |   6   |   7   | ... |   x 


Currently the following parameters can be configured. This configuration is stored in the Infomem so survives a reset/power off (but does not survive reprogramming):
   - Sampling rate
   - Which sensors are sampled
   - Accelerometer range
   - The state of the 5V regulator on the AnEx board
   - The power mux
   - GSR range
      - Special cases: 
         - GSR autorange
            - When set the GSR range is controlled on the shimmer
            - The two most significant bits of the GSR channel value are overloaded to indicate which resistor is active (i.e. the range)
               - e.g. if HW_RES_1M is selected (val 2), then the GSR channel will be 10xxxxxxxxxxxxxx
         - GSRx4
            - not currently used


The following commands are available:
   - INQUIRY_COMMAND
   - GET_SAMPLING_RATE_COMMAND
   - SET_SAMPLING_RATE_COMMAND
   - TOGGLE_LED_COMMAND
   - START_STREAMING_COMMAND
   - SET_SENSORS_COMMAND
   - SET_ACCEL_RANGE_COMMAND
   - GET_ACCEL_RANGE_COMMAND
   - SET_5V_REGULATOR_COMMAND
   - SET_PMUX_COMMAND
   - SET_CONFIG_SETUP_BYTE0_COMMAND
   - GET_CONFIG_SETUP_BYTE0_COMMAND
   - SET_ACCEL_CALIBRATION_COMMAND
   - GET_ACCEL_CALIBRATION_COMMAND
   - SET_GYRO_CALIBRATION_COMMAND
   - GET_GYRO_CALIBRATION_COMMAND
   - SET_MAG_CALIBRATION_COMMAND
   - GET_MAG_CALIBRATION_COMMAND
   - STOP_STREAMING_COMMAND
   - SET_GSR_RANGE_COMMAND
   - GET_GSR_RANGE_COMMAND
   - GET_SHIMMER_VERSION_COMMAND
    


Config Setup Byte 0:
   - Bit 7: 5V regulator   # When 0 5V regulator on AnEx board is disabled, when 1 5V regulator is enabled
   - Bit 6: PMUX           # When 0 AnEx channels are read, when 1 power values are read
   - Bit 5: Not yet assigned
   - Bit 4: Not yet assigned
   - Bit 3: Not yet assigned
   - Bit 2: Not yet assigned
   - Bit 1: Not yet assigned
   - Bit 0: Not yet assigned
Config Setup Byte 1-4:
   - Not yet assigned


The format of the configuration data stored in Infomem is as follows:
   - 82 bytes starting from address 0
      Byte 0: Sampling rate
      Byte 1: Buffer Size
      Byte 2 - 11: Selected Sensors (Allows for up to 80 different sensors)
      Byte 12: Accel Range
      Byte 13 - 17: Config Bytes (Allows for 40 individual boolean settings)
      Byte 18 - 38: Accelerometers calibration values
      Byte 39 - 59: Gyroscopes calibration values
      Byte 60 - 80: Magnetometer calibration values
      Byte 81: GSR range


The assignment of the selected sensors field is a follows:
   - 1 bit per sensor. When there is a conflict priority is most significant bit -> least significant bit
      Byte2:
         Bit 7: Accel
         Bit 6: Gyro
         Bit 5: Magnetometer
         Bit 4: ECG
         Bit 3: EMG
         Bit 2: GSR
         Bit 1: AnEx ADC Channel 7
         Bit 0: AnEx ADC Channel 0
      Byte3
         Bit 7: Strain Gauge
         Bit 6: Heart Rate
         Bit 5: Not yet assigned
         Bit 4: Not yet assigned
         Bit 3: Not yet assigned
         Bit 2: Not yet assigned
         Bit 1: Not yet assigned
         Bit 0: Not yet assigned
      Byte4 - Byte11
         Not yet assigned


The GET_SHIMMER_VERSION_COMMAND returns a 1 byte value, based on the shimmer revision as follows:
   0 = shimmer1
   1 = shimmer2
   2 = shimmer2r


TODO:
   - Support for variable data buffer size
   - Real world time stamps
      - and command to initialise


Changelog:
- 10 Jan 2011
   - initial release
   - support for Accelerometer, Gyroscope, Magnetometer, ECG, EMG, AnEx sensors
   - Sampling rate, Accel range, 5V regulator and PMUX (voltage monitoring) configurable
   - save configuration to InfoMem
- 29 Mar 2011
   - support for Strain Gauge and Heart Rate sensors
   - fixed EMG problem
      - a second EMG channel was being added erroneously
   - support for transmitting 8-bit data channels instead of just 16-bit 
- 21 Apr 2011
   - Fixed bug in heart rate support
   - Fixed bug in timestamping
   - changed SampleTimer to an Alarm
- 4 May 2011
   - removed a lot of unnecessary atomic commands
   - added support for writing and reading accel, gyro and mag calibration data
- 13 May 2011
   - GSR support
   - added command to get shimmer version (revision)
- 4 July 2011
   - fixed bug which caused a command sent while shimmer was streaming data to not always receive an acknowledgement
