function [score_quality] = cal_QualityScore(img, type)

%% Settings
if type == "HEVC"
    block_h = 4;
    block_w = 4;
elseif type == "JPEG"
    block_h = 8;
    block_w = 8;
end

thr = 0.004; % smooth/ textured classification
T = 0.05; % maximum Q_H and Q_V
gau_size = 3;
gau_sig = 5;
G1 = fspecial('gaussian', gau_size, gau_sig);
C = 1e-5; % numerical stability
w_block = 0.9; % relative importance
w_blur = 0.1;

[frame_h, frame_w] = size(img);
num_patch_h = floor(frame_h / block_h) - 1;
num_patch_w = floor(frame_w / block_w) - 1;

%% Initialize
%img_partition = zeros(size(img_enhanced));
num_textured_patch = 0;
num_smooth_patch = 0;

score_blocky = 0;
score_blurring = 0;

%% Calculate quality score
for ite_patch_h = 1:num_patch_h
    
    start_h = ((ite_patch_h - 1) * block_h + 1) + floor(block_h / 2);
    
    for ite_patch_w = 1:num_patch_w
        
        start_w = (ite_patch_w - 1) * block_w + 1 + floor(block_w / 2);
        
        % Obtain patch
        patch = img(start_h:start_h + block_h - 1, start_w:start_w + block_w - 1);
        patch = im2double(patch); % uint8 -> double
        
        % Get Tchebichef moments
        M = TchebiFocus(patch, (block_w - 1)); % A SMALL BUG!!!!! MUST BE SQUARE!
        
        %=========== Cal SSM for block classification =============
        SSM = sum(power(M(:), 2)) - power(M(1,1), 2);
        
        if SSM > thr %====== Textured patch =========
            
            num_textured_patch = num_textured_patch + 1;
            %img_partition(start_h:start_h + block_h - 1, start_w:start_w + block_w - 1) = 1; % white
            
            %=========== Blurring evaluation =============
            blurred_patch = imfilter(patch, G1, 'same');
            %figure;imshow(patch);figure;imshow(blurred_patch);pause
            M_blurred = TchebiFocus(blurred_patch, (block_w - 1)); % A SMALL BUG!!!!! MUST BE SQUARE!
            similarity_matrix = (M .* M_blurred * 2 + C) ./ (power(M, 2) + power(M_blurred, 2) + C);
            score_blurring_patch = 1 - mean(similarity_matrix(:));
            score_blurring = score_blurring + score_blurring_patch; % bigger the similarity, smaller the 1 - mean, stronger the blurring
            
        else %======= Smooth patch ==========
            
            num_smooth_patch = num_smooth_patch + 1;
            %img_partition(start_h:start_h + block_h - 1, start_w:start_w + block_w - 1) = 0; % black
            
            %=========== Blocky effects evaluation =============
            Q_h = sum(abs(M(block_h,:))) / (sum(abs(M(:))) - abs(M(1,1)) + C); % vertical
            Q_w = sum(abs(M(:,block_w))) / (sum(abs(M(:))) - abs(M(1,1)) + C); % horizontal
            % Clip
            if Q_h > T
                Q_h = T;
            end
            if Q_w > T
                Q_w = T;
            end
            score_blocky_patch = (Q_h + Q_w) / 2;
            score_blocky_patch = log(1 - score_blocky_patch) / log(1 - T);
            score_blocky = score_blocky + score_blocky_patch; % smaller (approaches to zero) the average Q, stronger the blocky effects
            
        end
    end
end

% % Display partition results
% if test_type == "o2"
%     close all;figure;subplot(1,2,1);imshow(img);subplot(1,2,2);imshow(img_partition);
%     pause(1)
% end

% Cal two scores and the final score
if num_textured_patch ~=0
    score_blurring = score_blurring / num_textured_patch;
else
    score_blurring = 1;
end
if num_smooth_patch ~=0
    score_blocky = score_blocky / num_smooth_patch;
else
    score_blocky = 1;
end

score_quality = power(score_blocky, w_block) * power(score_blurring, w_blur);

return