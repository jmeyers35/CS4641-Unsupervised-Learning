The code for this project is found at https://github.com/jmeyers35/CS4641-Unsupervised-Learning.

The code is separated into wine.py and car.py, which perform the experiments and generate the visualiations for their respective datasets.
To run, make sure car.py and wine.py are in the same directory as algos.py, car.csv, and wine.csv. From there, you can run with
'python3 car.py' and 'python3 wine.py'. The bit of code in the kmeans function in algos.py that is commented out is used to generate the clusters visualization shown in Figure 5 of 
my analysis. It's commented because it causes an error with the car data. To see this visualization, uncomment this portion and run python3 wine.py.