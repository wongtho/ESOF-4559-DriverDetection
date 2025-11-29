% Attempt to remove noise artifact from CLAHE - adapthisteq() using fourier
% selection.
function noiselessImage = noiseFilter(image)
    img = image;
    close all;
    figure;
    imshow(img); title('Cropped + enhanced image');
    
    % Fourier Transform
    F = fft2(double(img));
    % Shift zero frequency to the center
    F_shifted = fftshift(F);

    % Compute the magnitude for visualization
    magnitude = log(abs(F_shifted) + 1);

    figure;
    subplot(1,2,1); imshow(img); title('Noisy image');
    
    % Set threshold for peaks
    threshold = max(magnitude(:)) * 0.814; %0.814
    % Find size of image
    [rows, cols] = size(img);
    % Automatically find peaks (exclude center)
    [peak_rows, peak_cols] = find(magnitude > threshold);
    % Center coordinates
    center_row = round(rows / 2);
    center_col = round(cols / 2);
    % Exclude a small radius around the center
    exclude_radius = 9;
    distance_from_center = sqrt((peak_rows - center_row).^2 + (peak_cols - center_col).^2);
    valid_peaks = distance_from_center > exclude_radius;
    % Filter valid peaks
    peak_rows = peak_rows(valid_peaks);
    peak_cols = peak_cols(valid_peaks);
    % Safety: if no peaks, return original
    if isempty(peak_rows)
        noiselessImage = img;
        return;
    end
    % Display magnitude with detected peaks
    figure;
    imshow(magnitude, []), title('Frequency Domain with Detected Peaks');
    hold on;
    % Mark peaks in red
    plot(peak_cols, peak_rows, 'ro', 'MarkerSize', 5);

    
    filtered_F5 = fourier_filtered(rows, cols, peak_cols, peak_rows, F_shifted, 5);
    % Inverse Fourier Transform
    % Shift back
    filtered_F_ishifted5 = ifftshift(filtered_F5);
    % Convert back to spatial domain
    filteredImg = real(ifft2(filtered_F_ishifted5));
    
    
    figure;
    subplot(1,2,1); imshow(mat2gray(filteredImg)), title("Spatial 5");
    subplot(1,2,2); imshow(img), title("Original");


    noiselessImage = filteredImg;
end

%% helper function
function filtered_F = fourier_filtered(rows, cols, peak_cols_in, peak_rows_in, F_shifted, mask_size_in)
    % Set size of the area around each peak
    mask_size = mask_size_in;
    % Create mask dynamically
    mask = ones(rows, cols);
    for k = 1:length(peak_rows_in)
        % Compute the boundaries of the square region to zero out
        row_start = max(peak_rows_in(k) - floor(mask_size / 2), 1);
        row_end = min(peak_rows_in(k) + floor(mask_size / 2), rows);
        col_start = max(peak_cols_in(k) - floor(mask_size / 2), 1);
        col_end = min(peak_cols_in(k) + floor(mask_size / 2), cols);
        % Zero out the specified region
        mask(row_start:row_end, col_start:col_end) = 0;
    end
    % Apply the mask
    filtered_F = F_shifted .* mask;
end