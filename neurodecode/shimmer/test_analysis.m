basepath = './DATA/IMU_20141031/';
files = {'IMU_201410311444.mat'
    'IMU_201410311500.mat'};

file_i = 2;
load (strcat(basepath,files{file_i}))

t = cumsum(squeeze(imu(1,1,:))-min(imu(1,1,:)));
%% for other days
t = cumsum(squeeze(imu(1,1,:)));

plottitle = {'acc_x','acc_y','acc_z','gyr_x/trig'};
n=4;
close all
for sens_idx=1:4
    figure(order(sens_idx))
    for chan_idx=1:n
        subplot(n,1,chan_idx);plot (t,squeeze(imu(sens_idx,chan_idx+1,:))')
        ylim([0 5000])
        title(plottitle{chan_idx})
    end
end

%%
acc= [2:4];
gyr = [5:7];
mag = [8:10];

sens_ra= 3;  % right ankle
sens_la= 1;  % left ankle

clf
subplot(3,1,1);plot (t,squeeze(imu(sens_ra,acc(1),:)).^2);grid on
subplot(3,1,2);plot (t,squeeze(imu(sens_la,acc(1),:)).^2,'r');grid on
subplot(3,1,3);plot (t,squeeze(imu(4,5,:)).^2,'r');grid on