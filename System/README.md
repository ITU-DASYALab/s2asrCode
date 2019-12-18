# ASR System

## Installation

To run the project, Python 3 and FFMPEG needs to be installed. Additionally, some packages need to be installed through Pip. 

### 1. Basic Installations

Install Python, Pip, and FFmpeg with **APT**: 

```bash
sudo apt update
sudo apt install python3-dev
sudo apt install python3-pip
sudo apt install ffmpeg
```

Install Python, Pip, and FFmpeg with **Brew**: 

```bash
brew update
brew install python3
brew install ffmpeg
```

Check that it has been installed correctly: 

```bash
python3 --version
pip3 --version
ffmpeg -version
```

### 2. Pip Packages

If the system is supposed to run **without GPUs**, the TensorFlow package should be installed with: 

```bash
pip3 install 'tensorflow==1.14.0'
```

If the system is supposed to run **with GPUs**, the TensorFlow package should be installed with: 

```bash
pip3 install 'tensorflow-gpu==1.14.0'
```

The remaining packages are installed with: 

```bash
pip3 install -r requirements.txt
```

### Notes
- Numpy version 1.16.4 is installed to avoid warnings of deprecation. 
- Building TensorFlow from source has the potential to improve CPU performance. Follow [the guides from TensorFlow](https://www.tensorflow.org/install/source) to build from source.
- If you get a numa warning, there is an option to change all the numa assignments to 0. This is done by replacing all content with 0 in the following files: /sys/bus/pci/devices*/numa_node

## Folder Structure

```bash
+-- data
|   # The folder for the datasets.
|   +-- sv-SE
|   |   # The Swedish dataset from Mozilla Common Voice.
```
More in: [./data/README.md](./data/README.md)
```bash
+-- dict
|   # The folder containing the dictionaries for the different trainings
|   # Look at SE.txt for an example
```
More in [./dict/README.md](./dict/README.md)
```bash
+-- FE_data
|   # The folder containing the features extracted from the sound files
```
More in [./FE_data/README.md](./FE_data/README.md)

```bash
+-- plots
|   # The folder containing the generated plots
```
```bash
+-- exp
|   # The folder containing execution scripts for experiments
```

```bash
+-- src
|   # The folder containing the source code.
```

## First Execution
From the System path, execute the following:

1. Get dataset. [How to get the dataset](./data/README.md).
2. Extract features from sound files by executing the following example for LibriSpeech:

    ```bash
    ./exp/FE/FE_librispeech.sh
    ```
    When using Slurm, it is necessary to create a directory to store the logs. The directory for feature extraction logs is created by executing: 

    ```bash
    mkdir -p logs/FE
    ```
3. Make a dictionary containing all the words used. This is made using only the words in the training data.

   ```bash
   python3 src/create_dictionary.py --librispeech
   ```

4. Get CTC Word Beam Search from https://github.com/Sebastian-ba/CTCWordBeamSearch and place it in the same directory as StreamSpeech: 

    ```bash
    git clone https://github.com/Sebastian-ba/CTCWordBeamSearch.git
    ```
    Call the buildTF.sh script from the CTCWordBeamSearch/cpp/proj/ directory:

    ```bash
    cd cpp/proj
    ./buildTF.sh
    ```
5. Train a model. A small model can be trained on a small dataset by doing the following: 
    ```bash
    #Training:
    # CPU (cpu small) will fit to 50 samples!
    ./exp/local_test.sh 

    # GPU (small) Using one GPU! but correct model (local_test_eval does not work with this.)
    ./exp/local_test_GPU.sh 

    #Evaluation: 
    # Evaluate on the training data. Note that this is just to test the system!
    ./exp/local_test_eval.sh
    ```

    Training a real model is done with the following scripts:

    ```bash
    # Real experiment with 4 GPUs. (Sys1$ setup)
    ./exp/paramtest21/rebel.sh

    # Setup for Slurm.
    ./exp/paramtest21/libri.sh
    ```

## Start TensorBoard

Tensorboard is used to display graphs of the execution. 
From the System path, execute the following: 

```bash
tensorboard --logdir models/
```

## External TensorBoard

The graphs from a server can be displayed on your local machine by using an SSH tunnel. Use the shell script called "port_forward.sh" on your local machine by executing the following: 

```bash
sh sh/port_forward.sh server user port
```

The script takes three arguments: *server*, *user*, and *port*.
For instance:

```bash
sh sh/port_forward.sh reb sbwr 6006
```

Remember to start Tensorboard on the server. 



## HPC

Get terminal where there is acces to GPU:

´´´bash
srun -J test -t 24:00:00 -c 48 -p gpu --gres=gpu:v100:2 --mem=190000M --pty bash -i
´´´


Get information about the GPUs on each of the nodes in HPC:

´´´bash
sinfo -o=%N--%D--%G--%m
´´´
