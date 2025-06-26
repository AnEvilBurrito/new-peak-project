function dat = data_normalization(dat,dim)
% (1) row-wise
% (2) column-wise

if dim == 2
    dat = dat./repmat(dat(1,:),size(dat,1),1);
elseif dim == 1
    dat = dat./repmat(dat(:,1),1,size(dat,2));
end