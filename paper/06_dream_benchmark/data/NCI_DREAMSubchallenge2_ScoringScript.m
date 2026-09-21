function [weighted_cindex,pv_cindex] = NCI_DREAMsubchallenge_scoring_script(filename_input)

no_perm = 10000;
%% Read Gold standard file
[drug1 drug2 eob eob_error] = textread('NCI-DREAMSubchallenge2_GoldStandard.txt','%s%s%f%f','delimiter','\t','headerlines',1);
unique_drugs = unique(drug1);
drugpair = strcat(drug1,'_',drug2);
eob = eob*-1;
[Y,idx] = sort(eob);
eob = eob(idx);
eob_error = eob_error(idx);
drugpair = drugpair(idx);

p_matrix = probability_matrix(eob,eob_error); % Estimate probability to be used in weighted concordance index calculation

%% Read prediction file
[drugpair_pred ranks_drugpair_pred] = textread(filename_input,'%s%s','delimiter',',','headerlines',1);
ranks_drugpair_pred = str2num(char(ranks_drugpair_pred(1:91)));
drugpair_pred = drugpair_pred(1:91);
drugpair_pred = handle_fileformat(drugpair_pred);

for i = 1:length(drugpair)
    ranks_pred(i) = ranks_drugpair_pred(find(strcmp(drugpair_pred,drugpair(i))));
end

%% Computed weighted cindex,null dist and pvalue
weighted_cindex = concordance(ranks_pred,1:length(drugpair),p_matrix);
for i = 1:no_perm
    cindex_nulldist(i) = concordance(randperm(length(drugpair)),1:length(drugpair),p_matrix);
end
pv_cindex = length(find(cindex_nulldist>=weighted_cindex))/length(cindex_nulldist);
disp(['weighted cindex = ' num2str(weighted_cindex)]);
disp(['pvalue = ' num2str(pv_cindex)]);
%% Computes probability using error function
function p_matrix = probability_matrix(x,x_std)
x = x(:);
n = length(x);
p_matrix = zeros(n);
X = repmat(x,1,n);
X = X-X';
X_std = repmat(x_std,1,n);
X_std = sqrt(X_std.^2 + X_std'.^2);
p_matrix = .5*(1 + erf(X./X_std));

%% Computed probabilistic concordance index
function C = concordance(x, y,p_matrix)
assert(all(size(x)==size(y)));
x = x(:);
y = y(:);
n = length(x);
X = repmat(x,1,n);
Y = repmat(y,1,n);
C = sign(X-X')==sign(Y-Y');
C = C.*(1-p_matrix') + (1-C).*p_matrix';
C = sum(sum(tril(C,-1)))/n/(n-1)*2;

function drugpair_pred = handle_fileformat(drugpair_pred)
drugpair_pred = strrep(drugpair_pred,' & ','_');
for j = 1:length(drugpair_pred)
    a = strread(char(drugpair_pred(j)),'%s','delimiter','_');
    a = sort(a);
    drugpair_pred(j) = {[char(deblank(a(1))) '_' char(deblank(a(2)))]};
end