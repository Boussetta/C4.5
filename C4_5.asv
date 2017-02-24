%% C4_5.m

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

function [test_targets, tree] = C4_5(train_patterns, train_targets, test_patterns, inc_node)

% Classify using Quinlan's C4.5 algorithm
% Inputs:
% 	training_patterns   - Train patterns
%	training_targets	- Train targets
%   test_patterns       - Test  patterns
%	inc_node            - Percentage of incorrectly assigned samples at a node
%
% Outputs
%	test_targets        - Predicted targets

%NOTE: In this implementation it is assumed that a pattern vector with fewer than 10 unique values (the parameter Nu)
%is discrete, and will be treated as such. Other vectors will be treated as continuous

[Ni, M]		= size(train_patterns);
inc_node    = inc_node*M/100;
Nu          = 10;

%Find which of the input patterns are discrete, and discretisize the corresponding
%dimension on the test patterns
discrete_dim = zeros(1,Ni);
for i = 1:Ni,
    Ub = unique(train_patterns(i,:));
    Nb = length(Ub);
    if (Nb <= Nu),
        %This is a discrete pattern
        discrete_dim(i)	= Nb;
        dist            = abs(ones(Nb ,1)*test_patterns(i,:) - Ub'*ones(1, size(test_patterns,2)));
        [m, in]         = min(dist);
        test_patterns(i,:)  = Ub(in);
    end
end

%Build the tree recursively
disp('Building tree')
tree = make_tree(train_patterns, train_targets, inc_node, discrete_dim, max(discrete_dim), 0);

%Classify test samples
disp('Classify test samples using the tree')
test_targets    = use_tree(test_patterns, 1:size(test_patterns,2), tree, discrete_dim, unique(train_targets));

%END
