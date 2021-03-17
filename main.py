from feature_search import feature_search

def main():
    # Setup trace
    open('trace.txt', 'w').close()
    trace = open('trace.txt', 'a')
    # End of trace setup

    print("Welcome to Renzo's Feature Selection Algorithm.")
    testFileName = input('Type in the name of the file to test: ')

    trace.write("Welcome to Renzo's Feature Selection Algorithm.\n")
    trace.write(f'Type in the name of the file to test: {testFileName}')

    # Load data
    rawData = open(testFileName, 'r')
    testSet = rawData.readlines()

    for index, row in enumerate(testSet):
        testSet[index] = testSet[index].split()

    numberOfInstances = len(testSet)
    numberOfFeatures = len(testSet[0]) - 1

    rawData.close()
    # End of data load

    print('\nType the number of the algorithm you want to run.')
    print('\t 1) Forward Selection')
    print('\t 2) Backward Elimination')

    algorithmSelection = input('')

    print(f'\nThis dataset has {numberOfFeatures} features (not including the class attribute), with {numberOfInstances} instances\n')

    trace.write(f'{algorithmSelection}\n')
    trace.write(f'\nThis dataset has {numberOfFeatures} features (not including the class attribute), with {numberOfInstances} instances\n\n')
    trace.close()

    feature_search(testSet)


if __name__ == "__main__":
    main()