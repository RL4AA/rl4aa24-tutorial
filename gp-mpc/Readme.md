# Running fast and safe shallow model-based reinforcement learning on the CERN AWAKE tutorial

Simon Hirlaender, Sabrina Pochaba, Andrea Santa Maria Garcia, Jan Kaiser,
Chenran Xu, Annika Eichler

## This is the initial implementation of the tutorial given at RL4AA'24 part II

* Run the [main.py](main.py) script. This will start the training.
* The [config.yaml](config%2Fconfig.yaml) contains all hyperparameters, which will be explaned during the tutorial.
* We use a similar environment as the one in the Meta-RL part. Here is the file with environment definition [awake.py](awake.py)

This was financed by IDA Lab.
The codebase was derived and significantly modified from the original implementation available at
[Simon Rennotte's GitHub repository](https://github.com/SimonRennotte/Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control/tree/master).
Modifications include, but are not limited to, adaptations for episodic training.
We extend our gratitude to Simon Rennotte for providing the foundational work upon which these enhancements were built.
