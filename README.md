# AI_NN
NN Project for class

The custom dataset was found from http://archive.ics.uci.edu/ml/datasets/wiki4HE
The paper can be found here: http://openaccess.uoc.edu/webapps/o2/handle/10609/39441
It describes the participants and the results of their survey answers regarding Wikipedia and the perceived quality of the information in it.

First few features describe the participants of the survey.
Age, years of experience and university position were part of the input features.
Other features were the answers to the survey mapped from 0.2 to 1.0 (answers were 1 to 5)

The person's gender, whether the person has a PhD, and whether the person is a registered user on wikipedia were the outputs of the neural net.

Some other features were ommitted because:
	I didn't think those info were relevant enough
	I didn't know how I would put them into the neural net (besides making classes into different features and using 0/1 for true-false)

Some more info on how the data was converted can be found in data/process.awk
data/process.awk is used to format the data:
'''
cd data
./process.awk wiki4HE.csv # this will create wiki.test and wiki.train
'''
It takes 1 in every 4 and puts in wiki.test; all others go into wiki.train

The initial neural net file was generated with data/gen.py script via:
'''
cd data
./gen.py 45 10 3 > wiki.init # this will create a NN file with 45 input, 10 hidden, and 3 output nodes
'''
Browse the file for the usage

The python file (nn.py) takes as arguments all the necessary parameters (use ./nn.py -h for help), but train.sh and test.sh will prompt you for them

# Results
I failed to obtain any descent results from the dataset (or at least how I was using it)
I've tried 0.5 learning rate and 100 epochs with no changes in output between data points
Same has been true for combinations of:
	0.1 learning rate, 100 epochs, 10 hidden nodes
	0.5 learning rate, 10 epochs, 10 hidden nodes
	0.1 learning rate, 1000 epochs, 10 hidden nodes
	0.1 learning rate, 1000 epochs, 30 hidden nodes (this run trained and results files are provided)
	0.1 learning rate, 500 epochs, 15 hidden nodes (this run trained and results files are provided)

After talking to Prof.Sable and him reminding me that it is possible that the dataset cannot be learned using the given parameters, I've realized how the survey on usefulness of Wikipedia may not really be related to gender or if they are registered or not. The PhD prediction output though may be related, but it just seems like it doesn't have any defining info enough.

When a class has 0 counts and a metric calculation results in a ZeroDivisionError, it is just set to 0.
