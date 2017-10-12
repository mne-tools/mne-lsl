#!/usr/bin/env python

#  accel_gyro_mag publishes all shimmer imu data.
#  Copyright (C) 2013  Rafael Berkvens rafael.berkvens@uantwerpen.be
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

from math import pi, atan2, sqrt

import sys, struct, array
import bluetooth
import select
import rospy
import tf
import sensor_msgs.msg
import geometry_msgs.msg
from shimmer.msg import Heading
from numpy.ma.core import arctan2


def wait_for_ack(sock):
    ddata = ""
    ack = struct.pack('B', 0xff)
    while ddata != ack:
        ddata = sock.recv(1)
        print "received [%s]" % struct.unpack('B', ddata)
    return


def read_data(sock):
    # read incoming data
    ddata = ""
    numbytes = 0
    framesize = 21  # Packet type (1), TimeStamp (2), 3xAccel (3x2), 3xGyro (3x2), 3xMag (3x2)
    while numbytes < framesize:
        ddata += sock.recv(framesize)
        numbytes = len(ddata)

    data = ddata[0:framesize]
    ddata = ddata[framesize:]
    numbytes = len(ddata)

    packettype = struct.unpack('B', data[0:1])
    return struct.unpack('HHHHHHHhhh', data[1:framesize])


def heading_tilt_compensation(mag_x, mag_y, mag_z, accel_x, accel_y):
    norm_accel_x = accel_x / 65536.0  # Experimentally acquired max range for
    norm_accel_y = accel_y / 65536.0  # uncalibrated accelerometer data.
    try:
        x = (float(mag_x) * (1 - pow(norm_accel_x, 2)) -
             float(mag_y) * norm_accel_x * norm_accel_y -
             float(mag_z) * norm_accel_x *
             sqrt(1 - pow(norm_accel_x, 2) - pow(norm_accel_y, 2)))
        y = (float(mag_y) * sqrt(1 - pow(norm_accel_x, 2) - pow(norm_accel_y, 2)) -
             float(mag_z) * norm_accel_y)
        return atan2(y, x)
    except ValueError:
        rospy.logerr("Error: Python ValueError, probably due to negative sqrt."
                     + " Numbers were " + str(accel_x)
                     + " and " + str(accel_y) + ".")
        return 0.0


if __name__ == '__main__':
    rospy.init_node('accel_gyro_mag')
    imu_pub = rospy.Publisher('raw/imu', sensor_msgs.msg.Imu)
    imu = sensor_msgs.msg.Imu()
    imu.header.frame_id = "imu"
    imu.orientation.x = 0.0
    imu.orientation.y = 0.0
    imu.orientation.z = 0.0
    imu.orientation.w = 0.0
    imu.orientation_covariance = [0.0] * 9
    imu.orientation_covariance[0] = -1.0  # covariance unknown
    imu.angular_velocity_covariance = [0.0] * 9
    imu.angular_velocity_covariance[0] = -1.0  # covariance unknown
    imu.linear_acceleration_covariance = [0.0] * 9
    imu.linear_acceleration_covariance[0] = -1.0  # covariance unknown

    mag_pub = rospy.Publisher('raw/mag', sensor_msgs.msg.MagneticField)
    mag = sensor_msgs.msg.MagneticField()
    mag.header.frame_id = "imu"
    mag.magnetic_field_covariance = [0.0] * 9
    mag.magnetic_field_covariance[0] = -1.0  # covariance unknown

    heading_pub = rospy.Publisher('raw/heading', geometry_msgs.msg.PoseStamped)
    heading = geometry_msgs.msg.PoseStamped()
    heading.header.frame_id = "imu"
    heading.pose.position.x = 0.0
    heading.pose.position.y = 0.0
    heading.pose.position.z = 0.0

    # bd_addr = "00:06:66:43:B7:B7"
    bd_addr = "00:06:66:43:A9:0E"
    port = 1
    sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    sock.connect((bd_addr, port))

    # send the set sensors command
    sock.send(struct.pack('BBB', 0x08, 0xE0, 0x00))  # all
    wait_for_ack(sock)
    # send the set sampling rate command
    sock.send(struct.pack('BB', 0x05, 0x14))  # 51.2Hz
    # sock.send(struct.pack('BB', 0x05, 0x64))  # 10.24Hz
    wait_for_ack(sock)
    # send start streaming command
    sock.send(struct.pack('B', 0x07))
    wait_for_ack(sock)

    while not rospy.is_shutdown():
        try:
            (timestamp,
             imu.linear_acceleration.x,
             imu.linear_acceleration.y,
             imu.linear_acceleration.z,
             imu.angular_velocity.x,
             imu.angular_velocity.y,
             imu.angular_velocity.z,
             mag.magnetic_field.x,
             mag.magnetic_field.y,
             mag.magnetic_field.z) = read_data(sock)
        except bluetooth.btcommon.BluetoothError as e:
            rospy.logerr("BluetoothError during read_data: {0}".format(e.strerror))
        time = rospy.Time.now()
        imu.header.stamp = time
        imu_pub.publish(imu)
        mag.header.stamp = time
        mag_pub.publish(mag)
        # Heading expressed in degrees.
        yaw = atan2(float(mag.magnetic_field.y), float(mag.magnetic_field.x))  # *180/pi
        #     yaw = heading_tilt_compensation(mag.magnetic_field.x,
        #                                     mag.magnetic_field.y,
        #                                     mag.magnetic_field.z,
        #                                     imu.linear_acceleration.x,
        #                                     imu.linear_acceleration.y)
        heading.header.stamp = time
        quat = tf.transformations.quaternion_from_euler(0.0, 0.0, yaw)
        heading.pose.orientation.x = quat[0]
        heading.pose.orientation.y = quat[1]
        heading.pose.orientation.z = quat[2]
        heading.pose.orientation.w = quat[3]
        heading_pub.publish(heading)
        rospy.logdebug("published!")
        rospy.sleep(0.01)

        # http://www.loveelectronics.co.uk/Tutorials/13/tilt-compensated-compass-arduino-tutorial

    # send stop streaming command
    sock.send(struct.pack('B', 0x20));
    wait_for_ack(sock)
    # close the socket
    sock.close()
    print "\n"
