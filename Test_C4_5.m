clc; clear;

load train_data;


train_patterns = TrainData';
train_patterns = train_patterns(1:4,:); 

train_targets = TrainData(:,end)';


test_patterns = [7 1 1 1 2
               7 1 2 2 1
               7 3 1 2 2];
test_patterns = test_patterns(:,1:4)';

inc_node = 4;

[test_targets, tree] = C4_5(train_patterns, train_targets, test_patterns, inc_node);

test_targets

disp_tree(tree,0,0)