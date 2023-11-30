clear all
clc
%% input

E = csvread('../data/example1.dat');
k = 4; % Change k to the desired number of eigenvectors

% E = csvread('../data/example2.dat');
% k = 2; % Change k to the desired number of eigenvectors


sigma = 1;


% col1 = E(:,1);
% col2 = E(:,2);
% max_ids = max(max(col1,col2));
% As= sparse(col1, col2, 1, max_ids, max_ids); 
% A = full(As);
% [v,D] = eig(A);
% sort(diag(D));



%% Adjacency matrix
col1 = E(:,1);
col2 = E(:,2);

G = graph(col1, col2);
d = distances(G);


% Aii = 0
A = exp(-(d.^2)./(2*sigma^2)) - eye(size(d));


%% diagonal matrix
D = diag(sum(A,2));
L = D^(-1/2)*A*D^(-1/2);



% Compute the eigenvectors and eigenvalues of L
[V, D] = eig(L);

% Extract eigenvalues from the diagonal matrix D
eigenvalues = diag(D);

% Sort eigenvalues in descending order
[sorted_eigenvalues, indices] = sort(eigenvalues, 'descend');

% Choose the k largest eigenvalues and corresponding eigenvectors
selected_indices = indices(1:k);
selected_eigenvectors = V(:, selected_indices);

% Make the eigenvectors orthogonal (Gram-Schmidt process)
orthogonal_eigenvectors = orth(selected_eigenvectors);

% Form the matrix X by stacking the eigenvectors in columns
X = orthogonal_eigenvectors;

%% Y renormalising
Y = X ./ sqrt(sum(X.^2, 2));


%% k-means
[idx, C] = kmeans(Y, k);


%% assign

clusters = cell(1, k);
for i = 1:k
    clusters{i} = find(idx == i);
end

%% display
colors = [
    1, 0, 0; % Red
    0, 0, 1; % Blue
    1, 1, 0; % Yellow
    0, 1, 0; % Green
];
% disp(clusters);
figure
h = plot(G,'NodeColor',colors(idx,:));


figure 
% Display the adjacency matrix as an image
imagesc(A);
% spy(A);
colormap('gray'); % Set the colormap (optional, can choose different colormaps)
colorbar; % Display colorbar (optional)

% Add title and labels
title('Adjacency Matrix');
xlabel('Nodes');
ylabel('Nodes');

% Adjust aspect ratio to make cells square
axis square;




% Sort eigenvalues in ascending order
[sorted_eigenvalues_ascend, indices_ascend] = sort(eigenvalues);

% Find the index of the second smallest eigenvalue (Fiedler eigenvalue)
fiedler_index = indices_ascend(2);

% Extract the Fiedler vector corresponding to the second smallest eigenvalue
fiedler_vector = V(:, fiedler_index);

% Sort the Fiedler vector
sorted_fiedler_vector = sort(fiedler_vector);

% Plot the sorted Fiedler vector
figure
plot(sorted_fiedler_vector);
xlabel('Index');
ylabel('Value');
title('Sorted Fiedler Vector');

