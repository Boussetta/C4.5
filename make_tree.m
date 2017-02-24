%% make_tree.m

%    Copyright © 2017, by Wissem Boussetta
%
%    This file is part of the C4.5 library used to generate decision tree
%    for machine learning
%
%    C4.5 is free software: you can redistribute it and/or modify
%    it under the terms of the GNU General Public License as published by
%    the Free Software Foundation, either version 3 of the License, or
%    (at your option) any later version.
%
%    This program is distributed in the hope that it will be useful,
%    but WITHOUT ANY WARRANTY; without even the implied warranty of
%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%    GNU General Public License for more details.
%
%    You should have received a copy of the GNU General Public License
%    along with this program.  If not, see <http://www.gnu.org/licenses/>.

function tree = make_tree(patterns, targets, inc_node, discrete_dim, maxNbin, base)
%Build a tree recursively

[Ni, L]    					= size(patterns);
Uc         					= unique(targets);
tree.dim					= 0;
%tree.child(1:maxNbin)	= zeros(1,maxNbin);
tree.split_loc				= inf;

if isempty(patterns),
    return
end

%When to stop: If the dimension is one or the number of examples is small
if ((inc_node > L) | (L == 1) | (length(Uc) == 1)),
    H					= hist(targets, length(Uc));
    [m, largest] 	= max(H);
    tree.Nf         = [];
    tree.split_loc  = [];
    tree.child	 	= Uc(largest);
    return
end

%Compute the node's I
for i = 1:length(Uc),
    Pnode(i) = length(find(targets == Uc(i))) / L;
end
Inode = -sum(Pnode.*log(Pnode)/log(2));

%For each dimension, compute the gain ratio impurity
%This is done separately for discrete and continuous patterns
delta_Ib    = zeros(1, Ni);
split_loc	= ones(1, Ni)*inf;

for i = 1:Ni,
    data	= patterns(i,:);
    Ud      = unique(data);
    Nbins	= length(Ud);
    if (discrete_dim(i)),
        %This is a discrete pattern
        P	= zeros(length(Uc), Nbins);
        for j = 1:length(Uc),
            for k = 1:Nbins,
                indices 	= find((targets == Uc(j)) & (patterns(i,:) == Ud(k)));
                P(j,k) 	= length(indices);
            end
        end
        Pk          = sum(P);
        P           = P/L;
        Pk          = Pk/sum(Pk);
        info        = sum(-P.*log(eps+P)/log(2));
        delta_Ib(i) = (Inode-sum(Pk.*info))/-sum(Pk.*log(eps+Pk)/log(2));
    else
        %This is a continuous pattern
        P	= zeros(length(Uc), 2);
        
        %Sort the patterns
        [sorted_data, indices] = sort(data);
        sorted_targets = targets(indices);
        
        %Calculate the information for each possible split
        I	= zeros(1, L-1);
        for j = 1:L-1,
            %for k =1:length(Uc),
            %    P(k,1) = sum(sorted_targets(1:j)        == Uc(k));
            %    P(k,2) = sum(sorted_targets(j+1:end)    == Uc(k));
            %end
            P(:, 1) = hist(sorted_targets(1:j) , Uc);
            P(:, 2) = hist(sorted_targets(j+1:end) , Uc);
            Ps		= sum(P)/L;
            P		= P/L;
            
            Pk      = sum(P);
            P1      = repmat(Pk, length(Uc), 1);
            P1      = P1 + eps*(P1==0);
            
            info	= sum(-P.*log(eps+P./P1)/log(2));
            I(j)	= Inode - sum(info.*Ps);
        end
        [delta_Ib(i), s] = max(I);
        split_loc(i) = sorted_data(s);
    end
end

%Find the dimension minimizing delta_Ib
[m, dim]    = max(delta_Ib);
dims        = 1:Ni;
tree.dim    = dim;

%Split along the 'dim' dimension
Nf		= unique(patterns(dim,:));
Nbins	= length(Nf);
tree.Nf = Nf;
tree.split_loc      = split_loc(dim);

%If only one value remains for this pattern, one cannot split it.
if (Nbins == 1)
    H				= hist(targets, length(Uc));
    [m, largest] 	= max(H);
    tree.Nf         = [];
    tree.split_loc  = [];
    tree.child	 	= Uc(largest);
    return
end

if (discrete_dim(dim)),
    %Discrete pattern
    for i = 1:Nbins,
        indices         = find(patterns(dim, :) == Nf(i));
        tree.child(i)	= make_tree(patterns(dims, indices), targets(indices), inc_node, discrete_dim(dims), maxNbin, base);
    end
else
    %Continuous pattern
    indices1		   	= find(patterns(dim,:) <= split_loc(dim));
    indices2	   		= find(patterns(dim,:) > split_loc(dim));
    if ~(isempty(indices1) | isempty(indices2))
        tree.child(1)	= make_tree(patterns(dims, indices1), targets(indices1), inc_node, discrete_dim(dims), maxNbin, base+1);
        tree.child(2)	= make_tree(patterns(dims, indices2), targets(indices2), inc_node, discrete_dim(dims), maxNbin, base+1);
    else
        H				= hist(targets, length(Uc));
        [m, largest] 	= max(H);
        tree.child	 	= Uc(largest);
        tree.dim                = 0;
    end
end
