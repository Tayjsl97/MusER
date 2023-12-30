# MusER
This is the official implementation of **MusER (AAAI'24)**, which employs musical element-based regularization in the latent space to disentangle distinct musical elements, investigate their roles in distinguishing emotions, and further manipulate elements to alter musical emotions.
- [Paper link](https://arxiv.org/abs/2312.10307)
- Check our [demo page](https://tayjsl97.github.io/demos/aaai) and listen!<br>

<img src="img/MusER.png" width="770" height="300" alt="model"/>

## Requirements
The dependency packages can be found in requirements.txt file. One can use pip install -r requirements.txt to configure the environment. We use python 3.8 under Ubuntu 20.04.6 LTS to run the experiments. We highly recommend using the conda environment for deployment.

## Running the experiments
To run the codes, use the following command:
```{sh}
python MusER.py --data_path ./data/co-representation/emopia_data.npz --dataset emopia --model_path your_model_saving_path --log_path your_log_path 
```
Explanations on the parameters:

`data_path`: training data path.

`dataset`: which dataset to train, choices include `emopia` and `ailabs` and consistent with data_path,.


