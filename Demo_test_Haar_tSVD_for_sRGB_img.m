addpath(genpath(pwd))
addpath(genpath('mex'))
addpath(genpath('lib'))
addpath('t_svd_lib');
addpath('Pretrained_files')
addpath(genpath('D:\Code\Denoise\GCP-ID-main\Pretrained_files'))
%% Test other real_world images (CC, PolyU, HighISO)
% % Parameter setting
disp('--------------------------------------------------------------')
ps = 8; % patch size
SR = 20; % search window size
N_step = 4; % Nstep --> could be 4/6/8. Choose a smaller size for slightly better results. Choose a larger one for faster speed. 
maxK = 32; % number of patches.  Choose a larger number for slightly better results. Choose a smaller one for faster speed. 
divide_factor = 1.2; % controls the seach scheme
modified = 1; tau = 1; global_learning = 1; 
search_strategy = 'combined';
disp([' ps: ', num2str(ps), ' maxK: ',num2str(maxK),  ' N_step: ',num2str(N_step)])
disp([' SR: ', num2str(SR), ' tau: ',num2str(tau),  ' divide_factor: ',num2str(divide_factor), ' global_learning: ',num2str(global_learning)])
disp('--------------------------------------------------------------')

% % CC15
GT_Original_image_dir = 'Data/CC15/';
GT_fpath = fullfile(GT_Original_image_dir, '*mean.png');
TT_Original_image_dir = 'Data/CC15/';
TT_fpath = fullfile(TT_Original_image_dir, '*real.png');

GT_im_dir  = dir(GT_fpath);
TT_im_dir  = dir(TT_fpath);
im_num = length(TT_im_dir);

for sigma = 20
    % % Choose pretrained models
    k = 0; psnr_count = 0; ssim_count = 0;
    for i = 1:im_num
        origin_test = single(imread(fullfile(GT_Original_image_dir, GT_im_dir(i).name)) );
        S = regexp(GT_im_dir(i).name, '\.', 'split');
        fprintf('%s :\n', GT_im_dir(i).name);
        noisy_test = single(imread(fullfile(TT_Original_image_dir, TT_im_dir(i).name)) );
               
        tic
        [Denoised] = Haar_tSVD_with_mex(noisy_test, sigma, ps, maxK, SR, N_step, tau, divide_factor);
        time = toc;

        [psnr_h, im_ssim] = calculate_index(Denoised/255, origin_test/255);

        disp([num2str(time),' ',num2str(i),' ',num2str(sigma),' ',num2str(psnr_h),' ',num2str(im_ssim)]);
        psnr_count = psnr_count + psnr_h;
        ssim_count = ssim_count + im_ssim;
        k = k + 1;

    end

    disp(['sigma = ',num2str(sigma),' maxK = ',num2str(maxK),' psnr_average = ',num2str(psnr_count/k), ' ssim_average = ',num2str(ssim_count/k)]);
    disp('-------------------------------------------------------');
    disp('-------------------------------------------------------');

end


%% Functions

function [PSNR, SSIM] = calculate_index(denoised, clean)
PSNR = 10*log10(1/mean((clean(:)-double(denoised(:))).^2));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate MMSIM value
K = [0.01 0.03];
window = fspecial('gaussian', 11, 1.5);
L = 1;
[mssim1, ~] = ssim_index(denoised(:,:,1),clean(:,:,1),K,window,L);
[mssim2, ~] = ssim_index(denoised(:,:,2),clean(:,:,2),K,window,L);
[mssim3, ~] = ssim_index(denoised(:,:,3),clean(:,:,3),K,window,L);
SSIM = (mssim1+mssim2+mssim3)/3.0;

end