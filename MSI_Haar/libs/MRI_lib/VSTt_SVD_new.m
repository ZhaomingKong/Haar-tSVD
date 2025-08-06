function [im2] = VSTt_SVD_new(im1,Percentage_noise,sigma,modified)
% This code is a faster implementation of MNLt-SVD. It can be extended to
% efficiently handle other 3D images (color images for example).

[H,W,D] = size(im1);

% modified 24/10/2018
if(Percentage_noise<=5)
    ps = [4,4,4]; maxK = 30;tau = 1.8;
elseif(Percentage_noise<9)
    ps = [5,5,5]; maxK = 80;tau = 2;
else
    ps = [5,5,5]; maxK = 80;tau = 2.4;
end
SR = max(ps);N_step = max(ps)-1;

ps1 = ps(1);N_step1 = ps1 - 1;
ps2 = ps(2);N_step2 = ps2 - 1;
ps3 = ps(3);N_step3 = ps3 - 1;
%% compute global U and V with all reference cubes.

count = 0;
for i=1:N_step1:H-ps1+1
    for j=1:N_step2:W-ps2+1
        for k = 1:N_step3:D-ps3+1
            count = count + 1;
            global_indice(count,:) = [i,j,k];
        end
    end
end

A_all = zeros(ps1,ps2,ps3,count,'single');
for k=1:count
    xindex = global_indice(k,1);
    yindex = global_indice(k,2);
    zindex = global_indice(k,3);
    A_all(:,:,:,k) = im1(xindex:xindex+ps1-1,yindex:yindex+ps2-1,zindex:zindex+ps3-1);
end
[U,V]=NL_tSVD(A_all);

%% compute local similarity with a simple mex function

info1 = [H,W,D];
info2 = [ps(1),ps(2),ps(3),N_step,SR,maxK];
info3 = [tau,modified,sigma];
im2 = medical_denoising_global_tSVD_new(single(im1),single(U),single(V),int32(info1),int32(info2), single(info3));im2 = mat_ten(im2,1,size(im1));
end

function [U,V] = NL_tSVD(A)
size_A = size(A);ps1 = size_A(1);ps2 = size_A(2);ps3 = size_A(3);
A_F = fft(A,[],3);U = zeros(ps1,ps2,ps3);V = U;real_count = floor(ps3/2) + 1;
for i = 1:real_count
    A_i = A_F(:,:,i,:);
    if(i == 1)
        A_i = real(A_i);
    end
    A1 = my_tenmat(A_i,1);A2 = my_tenmat(A_i,2);
    [Ui,~] = eig(A1*A1');[Vi,~] = eig(A2*A2');
    U(:,:,i) = Ui; V(:,:,i) = Vi;
end

U(:,:,ps3) = conj(U(:,:,2));V(:,:,ps3) = conj(V(:,:,2));

if(ps3 == 5)
    U(:,:,4) = conj(U(:,:,3));
    V(:,:,4) = conj(V(:,:,3));
end

if(ps3 == 6)
    U(:,:,5) = conj(U(:,:,3));
    V(:,:,5) = conj(V(:,:,3));
end

if(ps3 == 7)
    U(:,:,6,:) = conj(U(:,:,3,:));
    U(:,:,5,:) = conj(U(:,:,4,:));
    
    V(:,:,6,:) = conj(V(:,:,3,:));
    V(:,:,5,:) = conj(V(:,:,4,:));
end

end
    
