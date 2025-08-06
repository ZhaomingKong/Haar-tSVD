function A_circulant = Create_circulant_mtx(A)
[N,d] = size(A);
A_circulant = zeros(N,d*N);
circulant_idx = circulant((1:N)');
A_circ = A(circulant_idx,:);

for i = 1:N
    start_idx = 1 + (i-1) * d; end_idx = i*d;
    start_idx2 = 1 + (i-1) * N; end_idx2 = i*N;
    A_circulant(:, start_idx:end_idx) = A_circ(start_idx2:end_idx2,:);
end

end

function A_circulant = Create_circulant_mtx_fastest(A)
[N,d] = size(A);
circulant_idx = circulant((1:N)');
A_circ = A(circulant_idx,:);

A_circulant = reshape(A_circ, [N, N*d]);
end