%% Test Haar Transform

% Created in September,2016
% Modified in March,2017 and October, 2017.
%%
clear all

%% main options in this demo
modified = 1; % if modified = 0;then it is reduced to NL-tSVD;

estimate_noise=0;  %% estimate noise level from data using recursive algorithm with VST+Gaussian MAD

two_stage = 0; %% 0 for first stage, 1 for second stage.

addpath(genpath(pwd));
%% --------------------------------------------------------------------------------------------

%% load BrainWeb T1 phantom
name ='t1_icbm_normal_1mm_pn0_rf0.rawb';
fid = fopen(name,'r');
disp(name);
nu = reshape(fread(fid,inf,'uchar'),[181,217,181]);
fclose(fid);


%% uncomment some of the following lines to test on small subvolume
% nu=nu(:,:,91-25:91+25);
% nu=nu(91-25:91+25,120:217,26-25:26+25);
% nu=nu(1:2:end,1:2:end,1:2:end);
% nu=nu(1:end/2,1:end/2,1:end/2);
% nu=nu(60:120,60:120,60:100);
max_nu = max(nu(:));

k = 1;
for percentNoise = 1:2:5
    %% create noisy data (spatially homogeneus Rician noise)
    sigma=percentNoise*max(nu(:))/100;    % get sigma from percentNoise
    randn('seed',0);  rand('seed',0);     % fixes pseudo-random noise
    z=sqrt((nu+sigma*randn(size(nu))).^2 + (sigma*randn(size(nu))).^2);   % raw magnitude MR data
    
    %%
    disp(' ');disp(' ');disp( '---------------------------------------------------------------');
    disp(['Size of data is ', num2str(size(z,1)),'x',num2str(size(z,2)),'x',num2str(size(z,3)),'  (total ',num2str(numel(z)),' voxel)']);
    %% compute PSNR of observations
    if exist('nu','var')
        if exist('sigma','var')&&exist('percentNoise','var')
            disp(['input nu range = [',num2str(min(nu(:))),' ',num2str(max(nu(:))),'],  noise sigma = ',num2str(sigma),' (',num2str(percentNoise),'%)']);
        else
            disp(['input nu range = [',num2str(min(nu(:))),' ',num2str(max(nu(:))),']']);
        end
        
        if 1
            ind=find(nu>10);   %% compute PSNR over foreground only
        else
            ind=1:numel(nu);   %% compute PSNR over every voxel in the volume
        end
        
        range_for_PSNR = 255;
        psnr_z=10*log10(range_for_PSNR^2/(mean((z(ind)-nu(ind)).^2)));
        disp(['PSNR of noisy input z is ',num2str(psnr_z),' dB'])
    end
    
    %% noise-level estimation
    if estimate_noise||~exist('sigma','var')
        disp( '---------------------------------------------------------------');
        disp(' * Estimating noise level sigma   [ model  z ~ Rice(nu,sigma) ]');
        estimate_noise_printout=1;   %% print-out estimate at each iteration.
        
        sigma_hat=riceVST_sigmaEst(z,estimate_noise_printout);
        disp( ' --------------------------------------------------------------');
        
        
        if ~exist('sigma','var')
            disp([' sigma_hat = ',num2str(sigma_hat)]);
        else
            disp([' sigma_hat = ',num2str(sigma_hat), '  (true sigma = ',num2str(sigma),')']);
            disp([' Relative estimation accuracy (1-sigma_hat/sigma) = ',num2str(1-sigma_hat/sigma)]);
        end
        disp( '---------------------------------------------------------------');
    else
        sigma_hat=sigma;
    end
    
    %% denoising
    VST_ABC_denoising='A';  %% VST pair to be used before and after denoising (for forward and inverse transformations)
    
    
    disp(' * Applying variance-stabilizing transformation')
    fz = riceVST(z,sigma_hat,VST_ABC_denoising);   %%  apply variance-stabilizing transformation
    sigmafz = 1;                                   %%  standard deviation of noise in f(z)
    
    
    tic;
    if(two_stage == 0)
        disp(' * Denoising with  tSVD_Haar')
        [D] = VSTt_SVD_Haar(fz,percentNoise,sigmafz,modified);
    elseif(two_stage == 1)
        disp(' * Enhanced with Wiener filter');
        [D,D2] = VSTt_SVD_H(fz,percentNoise,sigmafz,modified);
    end
    
    nu_hat = riceVST_EUI(D,sigma_hat,VST_ABC_denoising);
    if(two_stage == 1)
        nu_hat2 = riceVST_EUI(D2,sigma_hat,VST_ABC_denoising);
    end
    
    
    
    %%
    
    disp(['   completed in ',num2str(toc),' seconds']);
    disp( '---------------------------------------------------------------');
    
    
    if exist('nu','var')
        if(two_stage == 0)
            psnr_nu_hat=10*log10(range_for_PSNR^2/(mean((nu_hat(ind)-nu(ind)).^2)));
            SSIM = ssim_index3d(nu_hat,nu,[1 1 1],ind);
            disp(['PSNR and SSIM of estimate nu_hat is ',num2str(psnr_nu_hat),' dB ', num2str(SSIM)]);
            if(exist('D2','var'))
                psnr_nu_hat2=10*log10(range_for_PSNR^2/(mean((nu_hat2(ind)-nu(ind)).^2)));
                SSIM2 = ssim_index3d(nu_hat2,nu,[1 1 1],ind);
                disp(['PSNR and SSIM of final estimate nu_hat is ',num2str(psnr_nu_hat2),' dB ', num2str(SSIM2)])
            end
            
        elseif(two_stage == 1)
            psnr_nu_hat=10*log10(range_for_PSNR^2/(mean((nu_hat(ind)-nu(ind)).^2)));
            SSIM = ssim_index3d(nu_hat,nu,[1 1 1],ind);
            disp(['PSNR and SSIM of basic estimate nu_hat is ',num2str(psnr_nu_hat),' dB ', num2str(SSIM)])
            
            psnr_nu_hat2=10*log10(range_for_PSNR^2/(mean((nu_hat2(ind)-nu(ind)).^2)));
            SSIM2 = ssim_index3d(nu_hat2,nu,[1 1 1],ind);
            disp(['PSNR and SSIM of final estimate nu_hat is ',num2str(psnr_nu_hat2),' dB ', num2str(SSIM2)])
            
        end
    end
    disp( '---------------------------------------------------------------');  disp(' ');
end




