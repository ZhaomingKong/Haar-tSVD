function [im2,im3] = VSTt_SVD_H(im1,Percentage_noise,sigma,modified)
%color_t_hosvd_denoising Summary of this function goes here
%   Detailed explanation goes here
%% first stage estimate by MNLt-SVD
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

ps2 = 4; maxK2 = 30; N_step2 = ps2-1; SR2 = ps2;
info_second = [ps2, maxK2, N_step2, SR2];

[im2,im3] = medical_denoising_tSVD_h(single(im1),int32(info1),int32(info2), single(info3), int32(info_second));im2 = mat_ten(im2,1,size(im1));im3 = mat_ten(im3,1,size(im1));
end

