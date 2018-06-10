function [ classification_error_vec ] = Decision_Tree_Cross_Validation( ...
    trainX_array, trainY_array, testX_array, testY_array, criteria_func)
% Running cross-validation of Decision Tree with the given
% parameters.
% Returns the classification vector of the 10 sets.

classification_error_vec = zeros(10,1);
for i = 1:1:10
    disp(['iteration ', num2str(i)]);
    trainSetX = transpose(squeeze(trainX_array(i,:,:)));
    trainSetY = trainY_array(i,:);
    testSetX = transpose(squeeze(testX_array(i,:, :)));
    testSetY = testY_array(i,:);
    
    tree = Decision_Tree_Train(trainSetX, trainSetY, criteria_func);
    predictedY = Decision_Tree_Test(tree, testSetX);
    accuracy = calc_accuracy( predictedY, testSetY );
    classification_error_vec(i) = 1 - accuracy/100;
end

end

