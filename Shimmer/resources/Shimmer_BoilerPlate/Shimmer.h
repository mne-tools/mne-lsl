/*
 * Copyright (c) 2010, Shimmer Research, Ltd.
 * All rights reserved
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:

 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above
 *       copyright notice, this list of conditions and the following
 *       disclaimer in the documentation and/or other materials provided
 *       with the distribution.
 *     * Neither the name of Shimmer Research, Ltd. nor the names of its
 *       contributors may be used to endorse or promote products derived
 *       from this software without specific prior written permission.

 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * @author Mike Healy
 * @date   November, 2010
 */

#ifndef SHIMMER_H
#define SHIMMER_H

enum {
   SAMPLING_1000HZ   = 1,
   SAMPLING_500HZ    = 2,
   SAMPLING_250HZ    = 4,
   SAMPLING_200HZ    = 5,
   SAMPLING_166HZ    = 6,
   SAMPLING_125HZ    = 8,
   SAMPLING_100HZ    = 10,
   SAMPLING_50HZ     = 20,
   SAMPLING_10HZ     = 100,
   SAMPLING_0HZ_OFF  = 255
};

// Packet Types
enum {
   DATA_PACKET                      = 0x00,
   INQUIRY_COMMAND                  = 0x01,
   INQUIRY_RESPONSE                 = 0x02,
   GET_SAMPLING_RATE_COMMAND        = 0x03,
   SAMPLING_RATE_RESPONSE           = 0x04,
   SET_SAMPLING_RATE_COMMAND        = 0x05,
   TOGGLE_LED_COMMAND               = 0x06,
   START_STREAMING_COMMAND          = 0x07,
   SET_SENSORS_COMMAND              = 0x08,
   SET_ACCEL_RANGE_COMMAND          = 0x09,
   ACCEL_RANGE_RESPONSE             = 0x0A,
   GET_ACCEL_RANGE_COMMAND          = 0x0B,
   SET_5V_REGULATOR_COMMAND         = 0x0C,
   SET_PMUX_COMMAND                 = 0x0D,
   SET_CONFIG_SETUP_BYTE0_COMMAND   = 0x0E,
   CONFIG_SETUP_BYTE0_RESPONSE      = 0x0F,
   GET_CONFIG_SETUP_BYTE0_COMMAND   = 0x10,
   SET_ACCEL_CALIBRATION_COMMAND    = 0x11,
   ACCEL_CALIBRATION_RESPONSE       = 0x12,
   GET_ACCEL_CALIBRATION_COMMAND    = 0x13,
   SET_GYRO_CALIBRATION_COMMAND     = 0x14,
   GYRO_CALIBRATION_RESPONSE        = 0x15,
   GET_GYRO_CALIBRATION_COMMAND     = 0x16,
   SET_MAG_CALIBRATION_COMMAND      = 0x17,
   MAG_CALIBRATION_RESPONSE         = 0x18,
   GET_MAG_CALIBRATION_COMMAND      = 0x19,
   STOP_STREAMING_COMMAND           = 0x20,
   SET_GSR_RANGE_COMMAND            = 0x21,
   GSR_RANGE_RESPONSE               = 0x22,
   GET_GSR_RANGE_COMMAND            = 0x23,
   GET_SHIMMER_VERSION_COMMAND      = 0x24,
   SHIMMER_VERSION_RESPONSE         = 0x25,
   ACK_COMMAND_PROCESSED            = 0xFF
};

// Maximum number of channels
enum {
   MAX_NUM_2_BYTE_CHANNELS = 11,
   MAX_NUM_1_BYTE_CHANNELS = 1,
   MAX_NUM_CHANNELS = MAX_NUM_2_BYTE_CHANNELS + MAX_NUM_1_BYTE_CHANNELS
};


// Packet Sizes
enum {
   DATA_PACKET_SIZE = 3 + (MAX_NUM_2_BYTE_CHANNELS * 2) + MAX_NUM_1_BYTE_CHANNELS,
   //RESPONSE_PACKET_SIZE = 6 + MAX_NUM_CHANNELS,    // biggest possibly required
   RESPONSE_PACKET_SIZE = 22,             // biggest possibly required (calibration responses)
   MAX_COMMAND_ARG_SIZE = 21              // maximum number of arguments for any command sent to shimmer (calibration data)    
};

// Channel contents
enum {
   X_ACCEL     = 0x00,
   Y_ACCEL     = 0x01,
   Z_ACCEL     = 0x02,
   X_GYRO      = 0x03,
   Y_GYRO      = 0x04,
   Z_GYRO      = 0x05,
   X_MAG       = 0x06,
   Y_MAG       = 0x07,
   Z_MAG       = 0x08,
   ECG_RA_LL   = 0x09,
   ECG_LA_LL   = 0x0A,
   GSR_RAW     = 0x0B,
   GSR_RES     = 0x0C,     // GSR resistance (not used in this app)
   EMG         = 0x0D,
   ANEX_A0     = 0x0E,
   ANEX_A7     = 0x0F,
   STRAIN_HIGH = 0x10,
   STRAIN_LOW  = 0x11, 
   HEART_RATE  = 0x12
};

// Infomem contents;
enum {
   NV_NUM_CONFIG_BYTES = 82
};

enum {
   NV_SAMPLING_RATE      = 0,
   NV_BUFFER_SIZE        = 1,
   NV_SENSORS0           = 2,
   NV_SENSORS1           = 3,
   NV_ACCEL_RANGE        = 12,
   NV_CONFIG_SETUP_BYTE0 = 13,
   NV_ACCEL_CALIBRATION  = 18,
   NV_GYRO_CALIBRATION   = 39,
   NV_MAG_CALIBRATION    = 60, 
   NV_GSR_RANGE          = 81
};

//Sensor bitmap
//SENSORS0
enum {
   SENSOR_ACCEL   = 0x80,
   SENSOR_GYRO    = 0x40,
   SENSOR_MAG     = 0x20,
   SENSOR_ECG     = 0x10,
   SENSOR_EMG     = 0x08,
   SENSOR_GSR     = 0x04,
   SENSOR_ANEX_A7 = 0x02,
   SENSOR_ANEX_A0 = 0x01
};
//SENSORS1
enum {
   SENSOR_STRAIN  = 0x80,
   SENSOR_HEART   = 0x40
};

// Config Byte0 bitmap
enum {
   CONFIG_5V_REG        = 0x80,
   CONFIG_PMUX          = 0x40,
};

// BoilerPlate specific extension to range values
enum {
   GSR_AUTORANGE  = 0x04,
   GSR_X4         = 0x05
};

#endif // SHIMMER_H
