## BART

### MNLI

#### Download Data
```bash
bash data/download_mnli.sh
```

#### Run Inference
```bash
python bart_mnli.py
```


### CNN/DM
#### Download Data
```bash
bash data/download_cnn.sh
```

#### Run Inference

##### Greedy Decoding Test
```bash
python bart_test.py
```

##### Beam Search
```bash
python bart_cnn.py
```
