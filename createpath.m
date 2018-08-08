
%% Function creates hierarchical directory structure for RESULTS
function s=createpath(fullpath)
s=1;
list=textscan(fullpath,'%s','Delimiter','/');
partialpath='';
for i=1:length(list{1})
    partialpath=[partialpath,char(list{1}(i)),'/']; %#ok<AGROW>
    if(~exist(partialpath,'dir'))
        [s,~,~]=mkdir('./',partialpath);
    end
end
end


