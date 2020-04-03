%
% Parameters:
% imgFile = name of the image file
% labelFile = name of the label file
% readDigits = number of digits to be read
%
% Returns:
% imgs = 20 x 20 x readDigits sized matrix of digits
% labels = readDigits x 1 matrix containing labels for each digit
%
function [imgs, labels] = readMNIST(imgFile, labelFile, readDigits)  
    % Read digits
    fid = fopen(imgFile, 'r', 'b');
    magicNr = fread(fid, 1, 'int32');
    if magicNr ~= 2051
        error('Invalid image file header');
    end
    nrItems = fread(fid, 1, 'int32');
    if nrItems < readDigits
        error('Trying to read too many digits');
    end
    
    nrows = fread(fid, 1, 'int32');
    ncols = fread(fid, 1, 'int32');
    
    imgs = zeros([nrows ncols readDigits]);
    
    for i=1:readDigits
        for y=1:nrows
            imgs(y,:,i) = fread(fid, ncols, 'uint8');
        end
    end  
    fclose(fid);

    % Read digit labels
    fid = fopen(labelFile, 'r', 'b');
    magicNr = fread(fid, 1, 'int32');
    if magicNr ~= 2049
        error('Invalid label file header');
    end
    nrItems = fread(fid, 1, 'int32');
    if nrItems < readDigits
        error('Trying to read too many digits');
    end
    
    labels = fread(fid, readDigits, 'uint8');
    fclose(fid);    
end
