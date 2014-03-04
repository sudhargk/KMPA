function [] = saveFile()
   saveAll('linearlySeparableData');
   saveAll('nonlinearlySeparable');
   saveAll('overlapping_data');
end

function []=saveAll(dataset)
     trainset = load(dataset,'train');
     testset = load(dataset,'test');
     valset = load(dataset,'val');
     numClass = size(trainset,1);
     save([pwd '\..\data\' dataset '\data'], 'trainset', 'testset','valset','numClass');
end

function [dataset] = load(dataset,type)
     
     path = [pwd,'\..\data\' dataset];
     files = dir([path '\class*_' type '.txt']);
     dataset = cell(size(files,1),1);
     index =1;
     for file = files'
        dataset{index}=importdata([path '\' file.name]);
        index=index+1;
     end
end