from feature_search import feature_search
from feature_search import leave_one_out_cross_validation
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

    default_rate = 0

    if algorithmSelection == '2':
        default_rate = leave_one_out_cross_validation(testSet, [x for x in range(1,numberOfFeatures+1)], None, None, False)
    else:
        default_rate = leave_one_out_cross_validation(testSet, [], None, None, False)

    trace.write(f'{algorithmSelection}\n')
    trace.write(f'\nThis dataset has {numberOfFeatures} features (not including the class attribute), with {numberOfInstances} instances\n\n')
    if algorithmSelection == '2':
        trace.write(f'Running the nearest neighbor with all {numberOfFeatures} features, I get an accuracy of {"{:.2f}".format(default_rate)}%\n\n')
    else:
        trace.write(f'Running the nearest neighbor with no features, I get an accuracy of {"{:.2f}".format(default_rate)}%\n\n')
    trace.write('Beginning search.\n\n')
    trace.close()

    if algorithmSelection == '2':
        print(f'Running the nearest neighbor with all {numberOfFeatures} features, I get an accuracy of {"{:.2f}".format(default_rate)}%\n')
    else:
        print(f'Running the nearest neighbor with no features, I get an accuracy of {"{:.2f}".format(default_rate)}%\n')
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