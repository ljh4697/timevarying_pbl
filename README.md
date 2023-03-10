This code learns reward functions from human preferences in various tasks by actively generating batches of scenarios and querying a human expert.

Companion code to [CoRL 2018 paper](https://arxiv.org/abs/1810.04303):  
E Bıyık, D Sadigh. **"[Batch Active Preference-Based Learning of Reward Functions](https://arxiv.org/abs/1810.04303)"**. *Conference on Robot Learning (CoRL)*, Zurich, Switzerland, Oct. 2018.

## Dependencies
You need to have the following libraries with [Python3](http://www.python.org/downloads):
- [MuJoCo 1.50](http://www.mujoco.org/index.html)
- [NumPy](https://www.numpy.org/)
- [OpenAI Gym](https://gym.openai.com)
- [pyglet](https://bitbucket.org/pyglet/pyglet/wiki/Home)
- PYMC
- [Scikit-learn](https://scikit-learn.org)
- [SciPy](https://www.scipy.org/)
- [theano](http://deeplearning.net/software/theano/)

## Running
Throughout this demo,
- [task_name] should be selected as one of the following: Driver, LunarLander, MountainCar, Swimmer, Tosser
- [method] should be selected as one of the following: nonbatch, greedy, medoids, boundary_medoids, successive_elimination, random
For the details and positive integer parameters K, N, M, b, B; we refer to the publication.
You should run the codes in the following order:

### Sampling the input space
This is the preprocessing step, so you need to run it only once (subsequent runs will overwrite for each task). It is not interactive and necessary only if you will use batch active preference-based learning. For non-batch version and random querying, you can skip this step.

You simply run
```python
	python input_sampler.py [task_name] K
```
For quick (but highly suboptimal) results, we recommend K=1000. In the article, we used K=500000.

### Learning preference reward function
check description.txt
