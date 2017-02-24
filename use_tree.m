%% use_tree.m

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

function targets = use_tree(patterns, indices, tree, discrete_dim, Uc)
%Classify recursively using a tree

targets = zeros(1, size(patterns,2));

if (tree.dim == 0)
    %Reached the end of the tree
    targets(indices) = tree.child;
    return
end

%This is not the last level of the tree, so:
%First, find the dimension we are to work on
dim = tree.dim;
dims= 1:size(patterns,1);

%And classify according to it
if (discrete_dim(dim) == 0),
    %Continuous pattern
    in				= indices(find(patterns(dim, indices) <= tree.split_loc));
    targets		= targets + use_tree(patterns(dims, :), in, tree.child(1), discrete_dim(dims), Uc);
    in				= indices(find(patterns(dim, indices) >  tree.split_loc));
    targets		= targets + use_tree(patterns(dims, :), in, tree.child(2), discrete_dim(dims), Uc);
else
    %Discrete pattern
    Uf				= unique(patterns(dim,:));
    for i = 1:length(Uf),
        if any(Uf(i) == tree.Nf) %Has this sort of data appeared before? If not, do nothing
            in   	= indices(find(patterns(dim, indices) == Uf(i)));
            targets	= targets + use_tree(patterns(dims, :), in, tree.child(find(Uf(i)==tree.Nf)), discrete_dim(dims), Uc);
        end
    end
end

%END use_tree