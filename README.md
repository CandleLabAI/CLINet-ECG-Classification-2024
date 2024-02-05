# CLINet-ECG-Classification-2024
Source code of "CLINet: A Novel Deep Learning Network for ECG Signal Classification", accepted in Journal of Electrocardiology 2024

If you are using this code, please cite our paper:
```bash
@article {ref199,
	
 title = "CLINet: A Novel Deep Learning Network for ECG Signal Classification",
	
 year = "2024",
	
 author = "Ananya Mantravadi and Siddharth Saini and R Sai Chandra Teja and Sparsh Mittal and Shrimay Shah and R Sri Devi and Rekha Singhal",
	
 journal = "Journal of Electrocardiology",
 }
 ```

Project Organization
------------
    ├── LICENSE                         <- The LICENSE for developers using this project.
    ├── README.md                       <- The top-level README for developers using this project.
    ├── data                            <- Data used in the project.
    │   ├── iccad                       <- Add ICCAD dataset with this path in the folder.
    │   │   ├── tinyml_contest_data_training
    │   |   │   ├──S01-AFb-1.txt
    │   |   │   ├──S01-AFb-10.txt
    │   |   │   ├──...
    │   │   ├── data-indices
    │   |   │   ├── train-indice    
    │   |   │   ├── test-indice    
    │   ├── mit-bih                     <- Add MIT-BIH dataset with this path in the folder.
    │   │   ├── mitbih_database
    │   |   │   ├──100.csv
    │   |   │   ├──100annotations.txt
    │   |   │   ├──...
    ├── src                             <- Source code for use in this project.
    │   ├── iccad_dataloader.py         <- Source code for generating data loader for ICCAD dataset.
    |   ├── mitbih_dataloader.py        <- Source code for generating data loader for MIT-BIH dataset.
    │   ├── network.py                  <- Source code for the CLINet network.
    │   ├── involution.py               <- Source code for definition of custom involution layer.
    │   ├── tsne.py                     <- Source code for plotting t-SNE.
    │   ├── main.py                     <- Source code for using CLINet on ICCAD and MIT-BIH
    └─────────────────────────────────────────────────────────────────────────────────────────────────────────────
------------

### Train model

To train CLINet, Run following command from ```/src``` directory.

```bash
python main.py
``` 
Above command will train model for 50 epochs with given configuration.

## License
MIT License
Copyright (c) 2024 CandleLabAI
