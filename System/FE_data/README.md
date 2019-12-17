# TFRecord Data Format

This folder contains the TFRecords that are extracted from the original sound files.

The default extracted files each contains 50 distinct context instances. Each context instance represents a single sound file and the context has the following structure: 

```json
context:{
    feature: {
        key : "name"
        value:{
            feature:{
                bytes_list:{
                    "<file_path from root of project>"
                }
            }
        }
    }
    feature: {
        key : "string"
        value:{
            feature:{
                bytes_list:{
                    value:"target_output_string"
                    # original target string with all characters
                }
            }
        }
    }
    feature: {
        key : "label"
        value:{
            feature:{
                int64_list:{
                    value: [13,1,2,4,15]
                    # The target string as labels
                }
            }
        }
    }
    feature: {
        key : "label_length"
        value:{
            feature:{
                int64:{
                    value: 5
                    # The target length
                }
            }
        }
    }
    feature: {
        key : "target_string"
        value:{
            feature:{
                bytes_list:{
                    value: "targetoutputstring"
                    # The target length
                }
            }
        }
    }
}

feature_lists: {
    feature_list: {
        key : "mfccs"
        value: {
            feature: {
                float_list:{
                    value: [0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0]
                    # 13 values for the MFCCs.
                }
            }
        }
        ... # Repeated for every sample (10 ms standard)
    }
    feature_list: {
        key : "log_mel_spectrograms"
        value: {
            feature: {
                float_list:{
                    value: [0.0 ... 0.0]
                    # 80 values for the log mel spectrograms default.
                }
            }
        }
        ... # Repeated for every sample (10 ms standard)
    }
}
```

[Back](../README.md)