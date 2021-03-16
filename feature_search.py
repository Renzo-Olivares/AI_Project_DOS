import random

def feature_search(data):
    current_set_of_features = set() # Initialize an empty set

    for i in range(len(data)):
        print(f'On the {i + 1}th level of the search tree')
        feature_to_add_at_this_level = None
        best_so_far_accuracy = 0

        for k in range(1,len(data[i])):
            if k not in current_set_of_features: # Only consider adding, if not already added
                print(f'--Considering adding feature {k}')
                accuracy = leave_one_out_cross_validation(data, current_set_of_features, k+1) # Temporary stub function

                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_add_at_this_level = k
        
        current_set_of_features.add(feature_to_add_at_this_level)
        print(f'On level {i+1} i added feature {feature_to_add_at_this_level} to current set')

def leave_one_out_cross_validation(data, currentFeatureSet, feature):
    for i in range(len(data)):
        object_to_classify = data[i][1:len(data[i])]
        label_object_to_classify = data[i][0]

        nearest_neighbor_distance = float('inf')
        nearest_neighbor_location = float('inf')

        for k in range(): # Don't compare to yourself!!!!
            if k != i + 1:
                print(f'Ask if {i+1} is the nearest neighbor with {k}')

        # print(f'Looping over i, at the {i + 1} location')
        # print(f'The {i+1}th object is in class {label_object_to_classify}')
    return random.choice([1,2,3,4,5,6,7,8,9,10])
