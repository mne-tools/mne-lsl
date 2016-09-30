
addpath('resources/jsonlab/jsonlab/')

%% Day 1
% basepath = './DATA/IMU_20141028/';
% files = {'IMU_201410281524.dat'
%             'IMU_201410281552.dat'
%             'IMU_201410281600.dat'
%             'IMU_201410281606.dat'};
% 
% 
% sens_order = {[1 2 3 4]
%                 [3 1 2 4]
%                 [4 2 3 1]
%                 [4 1 2 3]};

%% Day 2
% basepath = './DATA/IMU_20141029/';
% files = {'IMU_201410291618.dat'
%             'IMU_201410291623.dat'
%             'IMU_201410291627.dat'};
%                 
% sens_order = {[1 3 4 2]
%                 [2 3 4 1]
%                 [1 4 2 3]};
        

%% Day 3
% basepath = './DATA/IMU_20141030/';
% files = {'IMU_201410301519.dat'
%         'IMU_201410301550.dat'
%         'IMU_201410301636.dat'};
%                 
% sens_order = {[3 4 1 2]
%                 [1 5 3 1 4]
%                 [3 2 5 4 1]};
            
   
%% Gait- Day 5
basepath = './DATA/IMU_20141031/';
files = {'IMU_201410311444.dat'
    'IMU_201410311500.dat'};
                
sens_order = {[4 2 3 1]
                [2 3 1 4]};
%% DB + ECoG- Day 1
basepath = './DATA/IMU_20141118/';
files = {'IMU_201411181411.dat'
        'IMU_201411181418.dat'
        'IMU_201411181422.dat'
        'IMU_201411181431.dat'
        'IMU_201411181440.dat'
        'IMU_201411181447.dat'
        'IMU_201411181451.dat'};
                
sens_order = {[1 2 3]
            [1 2 3]
            [1 2 3]
            [1 2 3]
            [1 2 3]
             [1 2 3]
            [1 2 3]};
        
n_sens = 3;
%%            
      
for rec_idx=1:length(files)
  fname = strcat(basepath,files{rec_idx});
  matfname = fname;
  matfname(end-2:end) = 'mat';
  
  if isequal(exist(matfname,'file'),2) % 2 means it's a file.
    % We have a file!
    display(['File ' matfname ' already exists!']);
  else
      fprintf('Converting file: %s\n',fname)
      imu = convert_jsonIMU(fname);
      order = sens_order{rec_idx};
      save (matfname,'imu','order');
  end
end


%% Plot recording

rec_idx=5;
fname = strcat(basepath,files{rec_idx});
matfname = fname;
matfname(end-2:end) = 'mat';

load (matfname)

% if day 1 use this expression since timestamp was stored as a signed int
%t = cumsum(squeeze(imu(1,1,:))-min(imu(1,1,:)));
% for other days
t = cumsum(squeeze(imu(1,1,:)));

plottitle = {'acc_x','acc_y','acc_z','gyr_x/trig','gyr_y','gyr_z','mag_x','mag_y','mag_z'};
n=4;
close all
for sens_idx=1:n_sens
    figure(order(sens_idx))
    for rec_idx=1:n
        subplot(n,1,rec_idx);plot (t,squeeze(imu(sens_idx,rec_idx+1,:))')
        ylim([0 5000])
        title(plottitle{rec_idx})
    end
end
