# Getting the Dataset

## LibriSpeech

1. Make LibriSpeech folder and cd into it.

    ```bash
    mkdir LibriSpeech
    cd LibriSpeech
    ```

2. Download dataset using wget:

    ```bash
    nohup wget http://www.openslr.org/resources/12/dev-clean.tar.gz &
    nohup wget http://www.openslr.org/resources/12/test-clean.tar.gz &
    nohup wget http://www.openslr.org/resources/12/dev-other.tar.gz &
    nohup wget http://www.openslr.org/resources/12/test-other.tar.gz &
    nohup wget http://www.openslr.org/resources/12/train-clean-100.tar.gz &
    nohup wget http://www.openslr.org/resources/12/train-clean-360.tar.gz &
    nohup wget http://www.openslr.org/resources/12/train-other-500.tar.gz &
    ```

3. Once downloaded, extract the zipped files:

    ```bash
    mkdir dev-clean
    mkdir test-clean
    mkdir dev-other
    mkdir test-other
    mkdir train-clean-100
    mkdir train-clean-360
    mkdir train-other-500
    tar xzf dev-clean.tar.gz -C ./dev-clean
    tar xzf test-clean.tar.gz -C ./test-clean
    tar xzf dev-other.tar.gz -C ./dev-other
    tar xzf test-other.tar.gz -C ./test-other
    tar xzf train-clean-100.tar.gz -C ./train-clean-100
    tar xzf train-clean-360.tar.gz -C ./train-clean-360
    tar xzf train-other-500.tar.gz -C ./train-other-500
    ```

4. Then convert all the FLAC files to Wav using the "libri_flac_to_wav" script in the data directory. 
This does not remove the original FLAC files. This can take some time.

    ```bash
    ./libri_flac_to_wav.sh
    ```

5.  If you want to use less storage space for this data, then it is possible to remove the .tar.gz at this point as well as the .flac files.

    ```bash
    # Delete the flac files:
    find . -iname "*.flac" -delete
    # Delete the tar.gz:
    rm *.tar.gz
    ```

[Back](../README.md)

## Mozilla Common Voice 

### GUI

Go to [Mozilla Common Voice](https://voice.mozilla.org/en/datasets) and download a dataset.
Move the downloaded folder into the "Data" folder.

## Terminal

Execute the following in the terminal to download the **English** dataset:

```bash
nohup wget https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/en.tar.gz &
```

The following downloads the **Swedish** dataset:

```bash
nohup wget https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/sv-SE.tar.gz &
```

Extract the dataset from the .tar.gz, into a folder appropriately named:

```bash
mkdir sv-SE
nohup tar xvzf sv-SE.tar.gz -C ./sv-SE &
```
[Back](../README.md)