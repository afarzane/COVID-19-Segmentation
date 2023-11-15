%% Amirali Farzaneh - 301292829 %%
%% ENSC 474 - Final Project %%
% In this project, we will use the clustering of pixel intensities to
% segment "ground glass" tissue from various patient lungs.

%% Reset Everything
clear all; clc;

%% Load pictures into MATLAB from input folder
patientData = dir('Input\*.jpg'); % Find number of files = inputs
nfiles = size(patientData,1); % Number of images

% Load all CT's into a single matrix
for i = 1:nfiles
    im = im2double(rgb2gray(imread(['Input\',patientData(i).name]))); % Read images as grayscale from input folder
    
    % Crop image
    if(i ~= 1)
        m = size(patient_im,1);
        n = size(patient_im,2);
        a = size(im,1);
        b = size(im,2);

        if(m > a || n > b) % Pad
            temp = zeros(m, n);
            temp(1:a, 1:b) = im;
            im = temp;   
        elseif(m < a || n < b) % Crop
            im = im(1:m, 1:n);
        end 
    end
        
    patient_im(:,:,i) = im; % Put images into a matrix
end


%% Cropping
% Recrop images to fit the lungs
% Find centroid
cent_row = floor(m/2);
cent_col = floor(n/2);

% Find crop factors based on center
cfr = floor(cent_row/4);
cfc = floor(cent_col/4);

% Crop images based on calculated values
for i = 1:nfiles
    patient_im_cropped(:,:,i) = patient_im(cfr:end-cfr,cfc:end-cfc,i);
end

%% Segmentation
% Initiate segmentation of lungs - Otsu's Method
% Create output directory
mkdir Output\Segmented_Lungs_Tissue

% Initialize threshold matrix
T = zeros(nfiles,1);

% Find histogram threshold for each image, then save segmented lungs
for i = 1:nfiles
    T(i) = graythresh(patient_im_cropped(:,:,i));
    Seg = imbinarize(patient_im_cropped(:,:,i),T(i));
    imwrite([patient_im_cropped(:,:,i) Seg],['Output\Segmented_Lungs_Tissue\Patient', num2str(i), '.jpg']);
end

% We notice that the "ground glass" feature is very faint in terms of
% intensity and blends into the same category as the intensity of the lungs
% on the histogram. Therefore, we will move onto using another method,
% named Clustering

% for i = 1:nfiles
% imshow(patient_im_cropped(:,:,i));
% x = input('');
% end

% Intensity range of lung tissue/background
K1 = 0;
% Intensity range of ground glass feature
K2 = 0.36;
% Intensity range of other (Veins, body, etc.)
K3 = 0.6;

K = 3; % Number of clusters

m_crop = size(patient_im_cropped,1);
n_crop = size(patient_im_cropped,2);
mean_values = zeros(3, nfiles);

howsevere = zeros(nfiles, 1);

for i = 1:nfiles
    
    % Cluster 1
    mu1 = zeros(m_crop, n_crop, 1);
    [row col] = find(patient_im_cropped(:,:,i) >= K1 & patient_im_cropped(:,:,i) < K2);
    ind = sub2ind(size(mu1),row,col);
    mu1(ind) = ones(1,size(ind,1));

    % Cluster 2
    mu2 = zeros(m_crop, n_crop, 1);
    [row col] = find(patient_im_cropped(:,:,i) >= K2 & patient_im_cropped(:,:,i) < K3);
    ind = sub2ind(size(mu2),row,col);
    mu2(ind) = ones(1,size(ind,1));

    % Cluster 3
    mu3 = zeros(m_crop, n_crop, 1);
    [row col] = find(patient_im_cropped(:,:,i) >= K3 & patient_im_cropped(:,:,i) <= 1);
    ind = sub2ind(size(mu3),row,col);
    mu3(ind) = ones(1,size(ind,1));
    
    ggointen = sum(mu2(:)==1);
    lunginten = sum(mu1(:)==1);
    
    perc = (ggointen/lunginten)*100;
     
    howsevere(i,1) = perc;

    % Set Fuzzy value based off of number of mu2
    q1 = nnz(mu1)/10000;
    q2 = nnz(mu2)/10000;
    q3 = nnz(mu3)/10000;
    
    % Calculate means of each cluster
    mu1p = mu1.*patient_im_cropped(:,:,i);
    mu2p = mu2.*patient_im_cropped(:,:,i);
    mu3p = mu3.*patient_im_cropped(:,:,i);
    
    mean_values(1,i) = (sum(mu1p,'all'))/(nnz(mu1));
    mean_values(2,i) = (sum(mu2p,'all'))/(nnz(mu2));
    mean_values(3,i) = (sum(mu3p,'all'))/(nnz(mu3));
    
    % Find membership of cluster 1
    Euc_dist1 = (patient_im_cropped(:,:,i)-mean_values(1,i)).^2;
    normu1 = 1./Euc_dist1;
    normu1 = normu1 - min(normu1(:));
    normu1 = normu1./max(normu1(:));
    s1 = 1/(sum(mu1p.^q1, 'all'));
    mu1f = normu1/s1;
    
    % Find membership of cluster 2
    Euc_dist2 = (patient_im_cropped(:,:,i)-mean_values(2,i)).^2;
    normu2 = 1./Euc_dist2;
    normu2 = normu2 - min(normu2(:));
    normu2 = normu2./max(normu2(:));
    s2 = 1/(sum(mu2p.^q2, 'all'));
    mu2f = normu2/s2;
    
    % Find membership of cluster 3
    Euc_dist3 = (patient_im_cropped(:,:,i)-mean_values(3,i)).^2;
    normu3 = 1./Euc_dist3;
    normu3 = normu3 - min(normu3(:));
    normu3 = normu3./max(normu3(:));
    s3 = 1/(sum(mu3p.^q3, 'all'));
    mu3f = normu3/s3;
    
    % Concatenate all segmentations
    mu(:,:,:,i) = cat(3, mu1f, mu2f, mu3f);
    
end

% Save segmented features
mkdir Output\Segmented_Features
for i = 1:nfiles
   imwrite([mu(:,:,2,i)],['Output\Segmented_Features\Patient', num2str(i), '.jpg'])
end


