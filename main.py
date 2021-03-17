from feature_search import feature_search
import timeit

def main():
    # Setup trace
    open('trace.txt', 'w').close()
    trace = open('trace.txt', 'a')
    # End of trace setup

    print("Welcome to Renzo's Feature Selection Algorithm.")
    testFileName = input('Type in the name of the file to test: ')

    trace.write("Welcome to Renzo's Feature Selection Algorithm.\n")
    trace.write(f'Type in the name of the file to test: {testFileName}\n')

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

    trace.write('\nType the number of the algorithm you want to run.\n')
    trace.write('\t 1) Forward Selection\n')
    trace.write('\t 2) Backward Elimination\n\n')

    algorithmSelection = input('')

    print(f'\nThis dataset has {numberOfFeatures} features (not including the class attribute), with {numberOfInstances} instances\n')

    trace.write(f'{algorithmSelection}\n')
    trace.write(f'\nThis dataset has {numberOfFeatures} features (not including the class attribute), with {numberOfInstances} instances\n\n')
    trace.write('Beginning search.\n\n')
    trace.close()

    print('Beginning search.\n')
    start = timeit.default_timer()
    feature_search(testSet, int(algorithmSelection))
    stop = timeit.default_timer()

    finalExecutionTime = stop - start
    print(f'Execution Time: {finalExecutionTime}s')
    trace = open('trace.txt', 'a')
    trace.write(f'Execution Time: {finalExecutionTime}s')
    trace.close()


if __name__ == "__main__":
    main()