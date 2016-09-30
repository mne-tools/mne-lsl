%  convert_jsonIMU(fname) - read json file and returns matlab matrix of
%  size [nchan 10 nsamp]
%
%    Sample vector:
%       9DoF IMUs: TimeStamp, Accel_x, Accel_y, Accel_z, Gyro_x, Gyro_y, Gyro_z, Mag_x, Mag_y, Mag_z
%       ACC+EMG: TimeStamp, Accel_x, Accel_y, Accel_z, EMG,  [0, 0, 0, 0, 0] 
%
% Usage:
%   >> imu = convert_jsonIMU(fname)
%
% Inputs:
%  fname         - filename
%
% Outputs:
%   imu         - data matrix [nchan 10 nsamp]


% 28.10.14 rchava
%   creation

function imu = convert_jsonIMU(fname)

data = loadjson(fname);

d = cell2mat(data);
[nchan nsamp] = size(d);
imu = reshape (d,nchan,10,nsamp/10);
