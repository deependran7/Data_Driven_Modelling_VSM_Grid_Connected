simout = sim('Grid_connected_VSM.slx');
load data_vsm_ddm_10.mat;
dlmwrite('test_data_10.txt',ans); 