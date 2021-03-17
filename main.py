from feature_search import feature_search
from feature_search import leave_one_out_cross_validation

def main():
    print("Welcome to Renzo's Feature Selection Algorithm.")
    testFileName = input('Type in the name of the file to test: ')

    # Load data
    rawData = open(testFileName, 'r')
    testSet = rawData.readlines()

    for index, row in enumerate(testSet):
        testSet[index] = testSet[index].split()

    numberOfInstances = len(testSet)
    numberOfFeatures = len(testSet[0]) - 1

    rawData.close()
    # End of data load
    print(f'\nThis dataset has {numberOfFeatures} features (not including the class attribute), with {numberOfInstances} instances\n')

    feature_search(testSet)
    # leave_one_out_cross_validation(testSet, [], [])

    # print('\nType the number of the algorithm you want to run.')
    # print('\t 1) Forward Selection')
    # print('\t 2) Backward Elimination')

    # algorithmSelection = input('')

    # print(f'\nThis dataset has {numberOfFeatures} features (not including the class attribute), with {numberOfInstances} instances\n')


if __name__ == "__main__":
    main()