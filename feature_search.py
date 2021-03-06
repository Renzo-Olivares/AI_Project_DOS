import math
import copy
from collections import OrderedDict

def feature_search(data, selection):
    if selection == 1:
        forward_search(data)
    else:
        backward_search(data)

def forward_search(data):
    all_set_of_features = []
    current_set_of_features = OrderedDict() # Initialize an empty set
    best_accuracy = None

    for i in range(1, len(data[0])):
        print(f'On the {i}th level of the search tree')

        trace = open('trace.txt', 'a')
        trace.write(f'On the {i}th level of the search tree\n')
        trace.close()

        feature_to_add_at_this_level = None
        best_so_far_accuracy = 0

        for k in range(1,len(data[0])):
            if k not in list(current_set_of_features.values()): # Only consider adding, if not already added
                print(f'\t--Considering adding feature {k}')

                trace = open('trace.txt', 'a')
                trace.write(f'\t--Considering adding feature {k}\n')
                trace.close()

                accuracy = leave_one_out_cross_validation(data, list(current_set_of_features.values()), None, k, True)

                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_add_at_this_level = k
        
        if best_accuracy is not None:
            if best_so_far_accuracy < best_accuracy:
                print('\t(Warning, Accuracy has decreased! Continuing search in case of local maxima)')
                trace = open('trace.txt', 'a')
                trace.write('\t(Warning, Accuracy has decreased! Continuing search in case of local maxima)\n')
                trace.close()
            best_accuracy = best_so_far_accuracy
        else:
            best_accuracy = best_so_far_accuracy
            
        current_set_of_features[i] = feature_to_add_at_this_level
        all_set_of_features.append((best_so_far_accuracy, list(current_set_of_features.values())))

        print(f'\tFeature set {list(current_set_of_features.values())} was best, accuracy is {"{:.2f}".format(best_so_far_accuracy)}%')
        print(f'\tOn level {i} I added feature {feature_to_add_at_this_level} to current set\n')

        trace = open('trace.txt', 'a')
        trace.write(f'\tFeature set {list(current_set_of_features.values())} was best, accuracy is {"{:.2f}".format(best_so_far_accuracy)}%\n')
        trace.write(f'\tOn level {i} I added feature {feature_to_add_at_this_level} to current set\n\n')
        trace.close()

    print(f'Finished search!! The best feature subset is {max(all_set_of_features)[1]}, which has an accuracy of {"{:.2f}".format(max(all_set_of_features)[0])}%')

    trace = open('trace.txt', 'a')
    trace.write(f'Finished search!! The best feature subset is {max(all_set_of_features)[1]}, which has an accuracy of {"{:.2f}".format(max(all_set_of_features)[0])}%\n')
    trace.close()

def backward_search(data):
    all_set_of_features = []
    current_set_of_features = OrderedDict() # Initialize an empty set
    set_of_features_removed = set()
    best_accuracy = None

    for k in range(1, len(data[0])): # Load current set of features with all features
        current_set_of_features[k] = k

    for i in range(1, len(data[0])):
        print(f'On the {i}th level of the search tree')

        trace = open('trace.txt', 'a')
        trace.write(f'On the {i}th level of the search tree\n')
        trace.close()

        feature_to_remove_at_this_level = None
        best_so_far_accuracy = 0

        for k in range(1,len(data[0])):
            if k not in set_of_features_removed: # Only consider removing, if not already removed
                print(f'\t--Considering removing feature {k}')

                trace = open('trace.txt', 'a')
                trace.write(f'\t--Considering removing feature {k}\n')
                trace.close()

                accuracy = leave_one_out_cross_validation(data, list(current_set_of_features.values()), set_of_features_removed, k, True)

                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_remove_at_this_level = k

        if best_accuracy is not None:
            if best_so_far_accuracy < best_accuracy:
                print('\t(Warning, Accuracy has decreased! Continuing search in case of local maxima)')
                trace = open('trace.txt', 'a')
                trace.write('\t(Warning, Accuracy has decreased! Continuing search in case of local maxima)\n')
                trace.close()
            best_accuracy = best_so_far_accuracy
        else:
            best_accuracy = best_so_far_accuracy

        del current_set_of_features[feature_to_remove_at_this_level]
        set_of_features_removed.add(feature_to_remove_at_this_level)
        all_set_of_features.append((best_so_far_accuracy, list(current_set_of_features.values())))

        print(f'\tFeature set {list(current_set_of_features.values())} was best, accuracy is {"{:.2f}".format(best_so_far_accuracy)}%')
        print(f'\tOn level {i} I removed feature {feature_to_remove_at_this_level} from the current set\n')

        trace = open('trace.txt', 'a')
        trace.write(f'\tFeature set {list(current_set_of_features.values())} was best, accuracy is {"{:.2f}".format(best_so_far_accuracy)}%\n')
        trace.write(f'\tOn level {i} I removed feature {feature_to_remove_at_this_level} from the current set\n\n')
        trace.close()

    print(f'Finished search!! The best feature subset is {max(all_set_of_features)[1]}, which has an accuracy of {"{:.2f}".format(max(all_set_of_features)[0])}%')

    trace = open('trace.txt', 'a')
    trace.write(f'Finished search!! The best feature subset is {max(all_set_of_features)[1]}, which has an accuracy of {"{:.2f}".format(max(all_set_of_features)[0])}%\n')
    trace.close() 

def leave_one_out_cross_validation(data,currentFeatureSet, featuresToExclude, featureToAddOrRemove, logFlag):
    number_correct_classified = 0
    temp_data = copy.deepcopy(data)

    if featuresToExclude is None:
        for i in range(len(temp_data)): # Exclude anything not in the current feature set of the feature we are adding
            for j in range(len(temp_data[0])):
                if (j not in currentFeatureSet and j is not featureToAddOrRemove) and j != 0:
                    temp_data[i][j] = 0
    else:
        for i in range(len(temp_data)): # Exclude in our excluded features or feature being removed
            for j in range(len(temp_data[0])):
                if (j in featuresToExclude or j is featureToAddOrRemove) and j != 0:
                    temp_data[i][j] = 0

    for i in range(len(temp_data)):
        object_to_classify = temp_data[i][1:len(temp_data[i])]
        label_object_to_classify = temp_data[i][0]

        nearest_neighbor_distance = float('inf')
        nearest_neighbor_location = float('inf')

        for k in range(len(temp_data)):
            if k != i: # Don't compare to yourself!!!!
                distance = distance_calculator(object_to_classify, temp_data[k][1:len(temp_data[k])])
                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = k
                    nearest_neighbor_label = temp_data[nearest_neighbor_location][0]
        
        if label_object_to_classify == nearest_neighbor_label:
            number_correct_classified = number_correct_classified + 1

    temp_features = currentFeatureSet[:]

    if featuresToExclude is None:
        temp_features.append(featureToAddOrRemove)
    else:
        temp_features.remove(featureToAddOrRemove)

    accuracy = (number_correct_classified / len(temp_data)) * 100

    if logFlag is True:
        print(f'\t\tUsing feature(s) {temp_features} accuracy is {"{:.2f}".format(accuracy)}%')

        trace = open('trace.txt', 'a')
        trace.write(f'\t\tUsing feature(s) {temp_features} accuracy is {"{:.2f}".format(accuracy)}%\n')
        trace.close()
    
    return accuracy

def distance_calculator(origin_instance, next_instance):
    total = 0

    for i in range(len(origin_instance)):
        total = total + pow(float(origin_instance[i]) - float(next_instance[i]), 2)

    return math.sqrt(total)
