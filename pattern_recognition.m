[imgsTrain, labelsTrain] = readMNIST('mnist_dataset/train-images.idx3-ubyte', ...
                                     'mnist_dataset/train-labels.idx1-ubyte', 36);
[imgsTest, labelsTest] = readMNIST('mnist_dataset/test-images.idx3-ubyte', ...
                                   'mnist_dataset/test-labels.idx1-ubyte', 36);
                      
Draw(imgsTrain, labelsTrain)
pause()
Draw(imgsTest, labelsTest)



function Draw(imgs, labels)
    close all;
    nrows = 6;
    ncols = 6;
    for i = 1:nrows
        for j = 1:ncols
            k = (i-1)*ncols + j;
            subplot(nrows, ncols, k);
            imshow(imgs(:,:,k));
            xlabel(labels(k));
        end
    end
    set(gcf, 'Position', [50,211,560,690]);
    MinGui()
end

function MinGui()
    set(gcf(), 'MenuBar', 'none');
    set(gcf(), 'MenuBar', 'none');
end
