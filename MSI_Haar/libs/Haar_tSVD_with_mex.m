% %% Mex with search 
function [im_denoised] = Haar_tSVD_with_mex(im1, sigma, ps, maxK, SR, N_step, tau, divide_factor)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[H,W,D] = size(im1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% compute global U and V with all reference cubes.
load U_3D; load V_3D;
load U_4D; load V_4D;
%% determine patch search scheme
count_green_mode = 0;
for i=1:N_step:H-ps+1 
    for j=1:N_step:W-ps+1
        count_green_mode = count_green_mode + 1;
    end
end

green_mode_all = zeros(count_green_mode,1);
count_green_mode = 1;

for i=1:N_step:H-ps+1 
    for j=1:N_step:W-ps+1

        patch_ij = im1(i:i+ps-1,j:j+ps-1,:);

        patch_mode = decide_mode(patch_ij, divide_factor);
        if strcmp(patch_mode, 'green') == 1
            green_mode_all(count_green_mode) = 1;
        end
        count_green_mode = count_green_mode + 1;

    end
end
%% compute local similarity with a simple mex function
info1 = [H,W,D];
info2 = [ps,N_step,SR,maxK];
info3 = [tau,sigma];
%% Perform denoising
U_3D = fft(U_3D, [], 3); V_3D = fft(V_3D, [], 3);
U_4D = fft(U_4D, [], 3); V_4D = fft(V_4D, [], 3);
Haar_mtx_input = haarmtx(maxK);
inv_Haar_mtx_input = inv(Haar_mtx_input);

im2 = GCP_CID_Haar_with_search(single(im1),single(U_3D),single(V_3D), single(U_4D),single(V_4D), single(Haar_mtx_input), single(inv_Haar_mtx_input), int32(green_mode_all), int32(info1),int32(info2),single(info3));
im_denoised = mat_ten(im2,1,size(im1));
% im_denoised = im1;
end


%% Related functions

function UV_fft_kron = kron_UV(U_fft,V_fft)
[ps,ps,D] = size(U_fft);
UV_3D_kron = zeros(ps*ps,ps*ps,D,'single');

for c = 1:D
    U_fft_c = U_fft(:,:,c);
    V_fft_c = V_fft(:,:,c);
    UV_fft_kron(:,:,c) = kron(U_fft_c', V_fft_c');
end

end

function A  = com_conj(A)
[~,~,D,~] = size(A);
k = 0;
for i = 2:floor(D/2)+1
    A(:,:,D-k,:) = conj(A(:,:,i,:));
    k = k + 1;
end
end

function A_UV_projection = cal_UV_projection(A,U,V)

A_fft = fft(A,[],3); U_fft = fft(U,[],3); V_fft = fft(V,[],3);
[~,~,D] = size(U_fft); real_count = floor(D/2) + 1;

for i = 1:real_count
    U_fft_i = U_fft(:,:,i);
    V_fft_i = V_fft(:,:,i);

    A1 = my_ttm(A_fft(:,:,i,:),U_fft_i,1,'t');
    A_12 = my_ttm(A1,V_fft_i,2,'t');
    A_fft(:,:,i,:) = A_12;
end

A_fft = com_conj(A_fft);
A_UV_projection = ifft(A_fft,[],3);

end

function patch_mode = decide_mode(refpatch, divide_factor)

patchR = refpatch(:,:,1);
patchG = refpatch(:,:,2);
patchB = refpatch(:,:,3);

R_norm = norm(patchR(:));
G_norm = norm(patchG(:));
B_norm = norm(patchB(:));

if G_norm > (B_norm/divide_factor)  && G_norm > (R_norm/divide_factor) 
    patch_mode = 'green';
else
    patch_mode = 'normal';
end

end


function noise_lvl_vectors = determin_noise_lvl_vectors(img, ps, N_step, noise_lvl_mtx)

[H,W,~] = size(img);
step_size = 128;
count_num = 1;
for i=1:N_step:H-ps+1 %why start from i=104?
    for j=1:N_step:W-ps+1 %why start from j=49?
        
        noise_lvl_ij = determine_local_noise_lvl_by_neighbours(i, j, step_size, noise_lvl_mtx);
%         noise_lvl_ij = mean(Noise_lvl_mtx(:));
        noiselevel = noise_lvl_ij;
        noise_lvl_vectors(count_num) = noiselevel;
        count_num = count_num + 1; 
    end
end

end

function noise_lvl_ij_avg = determine_local_noise_lvl_by_neighbours(i, j, step_size, noise_lvl_mtx)

[H,W] = size(noise_lvl_mtx);

idx_i = ceil(i/step_size);
idx_j = ceil(j/step_size);

sr_top = max([idx_i-2 1]);
sr_left = max([idx_j-2 1]);
sr_right = min([idx_j+2 W]);
sr_bottom = min([idx_i+2 H]);

noise_sum = 0; noise_count = 0;
for i1 = sr_top:sr_bottom
    for j1 = sr_left:sr_right
        noise_lvl_ij = noise_lvl_mtx(i1, j1);
        noise_sum = noise_sum + noise_lvl_ij;
        noise_count = noise_count + 1;
    end
end

noise_lvl_ij_avg = noise_sum/noise_count;


end

