function [sigma_est, maxK_est, tau_scaling] = global_noise_est(im1, ps, SR, maxK, N_step)

%% compute global U and V
[H,W,D] = size(im1);

imRed = im1(:,:,1);
imGreen = im1(:,:,2);
imBlue = im1(:,:,3);

eigvec = zeros(maxK,1);
for i = 1:maxK
    eigvec_i = (-1)^(i);
    eigvec(i) = eigvec_i;
end
eigvec = eigvec/sqrt(maxK);

%% Start denoising
im1_F = (imRed + imGreen + imBlue)/3;
counter = 0;
counter_large_idx = 0;

for i=1:N_step:H-ps+1 %why start from i=104?
    for j=1:N_step:W-ps+1 %why start from j=49?

        % refpatch = get_refpatch_by_avg(im1_F, i, j, ps, 10);

        refpatch = im1_F(i:i+ps-1,j:j+ps-1,:);

        sr_top = max([i-SR 1]);
        sr_left = max([j-SR 1]);
        sr_right = min([j+SR W-ps+1]);
        sr_bottom = min([i+SR H-ps+1]);

        count = 0;
        similarity_indices = zeros((2*SR+1)^2,2);

        distvals = similarity_indices(:,1); %distance value of refpatch and each target patch.
        for i1=sr_top:sr_bottom
            for j1=sr_left:sr_right

                currpatch = im1_F(i1:i1+ps-1,j1:j1+ps-1,:);

                dist = sum((refpatch(:)-currpatch(:)).^2);
                count = count+1;
                distvals(count) = dist;
                similarity_indices(count,:) = [i1 j1];

            end
        end

        similarity_indices(1,:)=[i j];
        similarity_indices = similarity_indices(1:count,:);
        distvals = distvals(1:count);

        if count > maxK
            [~,sortedindices] = sort(distvals,'ascend');
            similarity_indices = similarity_indices(sortedindices(1:maxK),:);
            count = maxK;
        end


        A = zeros(ps,ps,3,count,'single'); % construct a 4-D tensor with count patches
        A_F = zeros(ps,ps,count,'single');

        for k=1:count
            yindex = similarity_indices(k,1);
            xindex = similarity_indices(k,2);
            A(:,:,:,k) = im1(yindex:yindex+ps-1,xindex:xindex+ps-1,:);
            A_F(:,:,k) = im1_F(yindex:yindex+ps-1,xindex:xindex+ps-1,:);
        end

        [lambda_idx] = identify_lambda_idx(A, eigvec);
        lambda_idx = min(lambda_idx);
        if(lambda_idx>10)
            counter_large_idx = counter_large_idx + 1;
            % [i,j, lambda_idx]
        end

        counter = counter + 1;
        lambda_idx_all(counter) = lambda_idx;

    end
end

% counter_large_idx

[sigma_est, maxK_est, tau_scaling] = decide_by_idx(lambda_idx_all);

end

function [lambda_idx] = identify_lambda_idx(A, eigvec)

A_group = my_tenmat(A,ndims(A));
A_circulant = Create_circulant_mtx(A_group);
A_circulant_sym = A_circulant*A_circulant';
[U,S] = eig(A_circulant_sym);
diag_S = diag(S);
eig_prevalue = eigvec'*A_circulant_sym*eigvec;
error = abs(diag_S - eig_prevalue);
lambda_idx = find(error <= 0.001);

end

function [sigma_est, maxK_est, tau_scaling] = decide_by_idx(lambda_idx_all)

maxK = max(lambda_idx_all);

mid_noise_idx = find(lambda_idx_all <= 13 & lambda_idx_all>=1);
high_noise_idx = find(lambda_idx_all <= maxK & lambda_idx_all>13);

num_mid_noise_idx = length(mid_noise_idx);
num_high_noise_idx = length(high_noise_idx);

% disp([num2str(num_mid_noise_idx), ' ',num2str(num_high_noise_idx)])

if(num_mid_noise_idx>=num_high_noise_idx) %% this indicates the strong local variance (more edges and details)
    tau_scaling = 0.8;
    sigma_est = 15;
    maxK_est = 32;
else
    tau_scaling = 1;
    sigma_est = 25;
    maxK_est = 32;
end


end