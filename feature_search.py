import random
import math
import copy

def feature_search(data):
    all_set_of_features = []
    current_set_of_features = set() # Initialize an empty set

    for i in range(1, len(data[0])):
        print(f'On the {i}th level of the search tree')
        feature_to_add_at_this_level = None
        best_so_far_accuracy = 0

        for k in range(1,len(data[0])):
            if k not in current_set_of_features: # Only consider adding, if not already added
                print(f'\t--Considering adding feature {k}')
                accuracy = leave_one_out_cross_validation(data, current_set_of_features, k) # Temporary stub function

                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_add_at_this_level = k
        
        current_set_of_features.add(feature_to_add_at_this_level)
        all_set_of_features.append((best_so_far_accuracy, list(current_set_of_features)))

        print(f'\tFeature set {list(current_set_of_features)} was best, accuracy is {best_so_far_accuracy * 100}%')
        print(f'\tOn level {i} I added feature {feature_to_add_at_this_level} to current set\n')

    print(f'Finished search!! The best feature subset is {max(all_set_of_features)[1]}, which has an accuracy of {max(all_set_of_features)[0] * 100}%')

def leave_one_out_cross_validation(data, currentFeatureSet, featureToAdd):
    number_correct_classified = 0
    temp_data = copy.deepcopy(data)

    for i in range(len(temp_data)): # Exclude anything not in the current feature set of the feature we are adding
        for j in range(len(temp_data[0])):
            if (j not in currentFeatureSet and j is not featureToAdd) and j != 0:
                temp_data[i][j] = 0

    for i in range(len(temp_data)):
        object_to_classify = temp_data[i][1:len(temp_data[i])]
        label_object_to_classify = temp_data[i][0]

        nearest_neighbor_distance = float('inf')
        nearest_neighbor_location = float('inf')

        # print(f'Looping over i at the {i + 1} location')
        # print(f'The {i + 1}th object is in class {label_object_to_classify}')

        for k in range(len(temp_data)):
            if k != i: # Don't compare to yourself!!!!
                # print(f'Ask if {i+1} is nearest neighbor with {k + 1}')
                distance = distance_calculator(object_to_classify, temp_data[k][1:len(temp_data[k])])
                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = k
                    nearest_neighbor_label = temp_data[nearest_neighbor_location][0]
        
        if label_object_to_classify == nearest_neighbor_label:
            number_correct_classified = number_correct_classified + 1

        #print(f'The {i + 1}th object is in class {label_object_to_classify}')
        #print(f'Its nearest_neighbor is {nearest_neighbor_location} which is in class {nearest_neighbor_label}')
    temp_features = list(currentFeatureSet)
    temp_features.append(featureToAdd)
    print(f'\t\tUsing feature(s) {temp_features} accuracy is {(number_correct_classified / len(temp_data)) * 100}%')
    return number_correct_classified / len(temp_data)


def distance_calculator(origin_instance, next_instance):
    total = 0

    for i in range(len(origin_instance)):
        total = total + pow(float(origin_instance[i]) - float(next_instance[i]), 2)

    return math.sqrt(total)
