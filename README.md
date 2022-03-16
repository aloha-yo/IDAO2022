# IDAO2022 | Team "Предполагайте добрые намерения"

We use [GATGNN](https://github.com/superlouis/GATGNN)

1. Unzip data.  If your folder structure is different, you may need to change the corresponding paths in files.
```
IDAO2022
├── TRAINED
├── gatgnn
├── data
│   ├── dichalcogenides_private
│   │   └── structures
│   │       ├── 6141cf0efbfd4bd9ab2c2f7e.json
│   │       └── ...
│   └── dichalcogenides_public
│       ├── structures
│       │   ├── 6146900531cf3ef3d4a9f83c.json
│       │   └── ...
│       └── targets.csv
```
2. `python train.py`
3. `python predict.py`
4. profit!
