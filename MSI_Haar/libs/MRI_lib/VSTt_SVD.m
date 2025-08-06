function [im2] = VSTt_SVD(im1,Percentage_noise,sigma,modified)
%color_t_hosvd_denoising Summary of this function goes here
%   Detailed explanation goes here
%% first stage estimate by MNLt-SVD
% Used in older version
% if(Percentage_noise<=3)
%     ps = 3; maxK = 50;tau = 2.5;
% elseif(Percentage_noise<9)
%     ps = 4; maxK = 70;tau = 2.5;
% else
%     ps = 5; maxK = 80;tau = 2.5;
% end
% SR = ps; N_step = ps - 1;
% im2 = medical_denoising_tSVD_old(single(im1),int32(info1),int32(info2), single(info3));im2 = mat_ten(im2,1,size(im1)); 


% modified October 2017 and February 2018
if(Percentage_noise<=5)
    ps = 4; maxK = 30;tau = 1.8;
elseif(Percentage_noise<9)
    ps = 5; maxK = 80;tau = 2;
else
    ps = 5; maxK = 80;tau = 2.4;
end
SR = ps; N_step = ps - 1;

[H,W,D] = size(im1);
info1 = [H,W,D];
info2 = [ps,N_step,SR,maxK];
info3 = [tau,modified,sigma];

im2 = medical_denoising_tSVD_new(single(im1),int32(info1),int32(info2), single(info3));im2 = mat_ten(im2,1,size(im1)); 
end

