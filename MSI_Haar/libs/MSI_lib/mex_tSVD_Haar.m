function [im2] = mex_tSVD_Haar(im1,sigma,ps,maxK,SR,N_step,tau)
% This code is a faster implementation of MNLt-SVD. It can be extended to
% efficiently handle other 3D images (color images for example).
[H,W,D] = size(im1);
im2_F = zeros(H,W,D);
%% compute global U and V with all reference cubes.

count = 0;
for i=1:N_step:H-ps+1
    for j=1:N_step:W-ps+1
        count = count + 1;
        global_indice(count,:) = [i,j];
    end
end

A_all = zeros(ps,ps,D,count,'single');
for k=1:count
    xindex = global_indice(k,1);
    yindex = global_indice(k,2);
    A_all(:,:,:,k) = im1(xindex:xindex+ps-1,yindex:yindex+ps-1,:);
end
[U,V]=NL_tSVD2(A_all);
%% compute local similarity with a simple mex function

info1 = [H,W,D];
info2 = [ps,N_step,SR,maxK];
info3 = [tau,sigma];

Haar_mtx_input = haarmtx(maxK);
inv_Haar_mtx_input = inv(Haar_mtx_input);

im1_F = fft(im1,[],3);

[im2_real,im2_imag] = MSI_denoising_Haar_efficient(single(im1),single(im1_F),single(U),single(V),single(Haar_mtx_input), single(inv_Haar_mtx_input),int32(info1),int32(info2),single(info3));
% [im2_real,im2_imag] = MSI_denoising_PCA_efficient(single(im1),single(im1_F),single(U),single(V),int32(info1),int32(info2),single(info3));

real_count = floor(D/2)+1;
im2_real = mat_ten(im2_real,1,[H,W,D]);im2_imag = mat_ten(im2_imag,1,[H,W,D]);
im2_real = im2_real(:,:,1:real_count); im2_imag = im2_imag(:,:,1:real_count);

im2_F(:,:,1:real_count) = complex(im2_real,im2_imag);
im2_F = com_conj(im2_F);im2 = ifft(im2_F,[],3);im2 = real(im2);

% im2_real = mat_ten(im2_real,1,[H,W,D]);im2_imag = mat_ten(im2_imag,1,[H,W,D]);
% im2_F = complex(im2_real,im2_imag);
% im2 = ifft(im2_F,[],3);im2 = real(im2);

end

function [U,V] = NL_tSVD2(A)
size_A = size(A);ps = size_A(1);D = size_A(3);
A_F = fft(A,[],3);U = zeros(ps,ps,D);V = U;real_count = floor(D/2) + 1;
for i = 1:real_count
    A_i = A_F(:,:,i,:);
    if(i == 1)
        A_i = real(A_i);
    end
    A1 = my_tenmat(A_i,1);A2 = my_tenmat(A_i,2);
    [Ui,~] = eig(A1*A1');[Vi,~] = eig(A2*A2');
    U(:,:,i) = Ui; V(:,:,i) = Vi;
end

U = com_conj(U); V = com_conj(V);

end

function A  = com_conj(A)
[~,~,D,~] = size(A);
k = 0;
for i = 2:floor(D/2)+1
    A(:,:,D-k,:) = conj(A(:,:,i,:));
    k = k + 1;
end
end



