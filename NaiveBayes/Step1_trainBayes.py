import Step2_Bayes

nb = Step2_Bayes.ClassNaiveBayes()

nb.train()

test_list = ['love', 'my', 'dalmation']
print(test_list, 'classified as: ', nb.testing_nb(test_list))

test_list = ['stupid', 'garbage']
print(test_list, 'classified as: ', nb.testing_nb(test_list))

test_list = ['stop']
print(test_list, 'classified as: ', nb.testing_nb(test_list))
