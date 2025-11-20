# MAGIN-GO

MAGIN-GO is a novel graph neural network method for protein function prediction, 
which transforms protein function prediction into a multi-label classification task by leveraging a dual-graph network architecture and learning GO semantic relationships.


# Use the MAGIN-GO model for predicting protein function
* Before making predictions, the ESM2-related model files must be loaded. The contact-regression file can be found in the data directory; the other pre-trained model file can be downloaded from https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t36_3B_UR50D.pt.
* If you want to use the trained model file to predict MF, BP, and CC functions and generate prediction results for all three categories, please prepare a FASTA file, place it in the directory where your data is stored, and run the following command:
  - `python predict.py -if data/example.fa`


# Training model
To train the models and reproduce our results:
* The dataset can be obtained via the following link:  [training-data.tar.gz](https://deepgo.cbrc.kaust.edu.sa/data/deepgo2/training-data.tar.gz)
  - The training data includes both UniProtKB/SwissProt dataset and the neXtProt
    evaluation dataset.
  - go.obo - Gene Ontology file
  - mf, bp and cc subfolders include:
    - train_data.pkl - training proteins
    - valid_data.pkl - validation proteins
    - test_data.pkl - testing proteins
    - nextprot_data.pkl - neXtProt dataset proteins (except cc folder)
    - terms.pkl - list of GO terms for each subontology
    - ppi.bin, ppi_nextprot.bin - PPI graphs saved with DGL library
* train_MAGIN_GO.py script is used to train MAGIN-GO
* Examples:
  - Train the MAGIN-GO MFO prediction model using ESM2 embeddings and save it in the data folder by entering the following command in the console(For details on the remaining required parameters of the command, please refer to the code): \
    `python train_MAGIN_GO.py -ont mf`

    
## Evaluating the predictions
The training scripts generate predictions for the test data that are used
to compute evaluation metrics.
* To evaluate the model's prediction performance for MF, please run the evaluate.py script and provide the model name and test data name. For detailed information on the specific    required parameters of the command, please refer to the code. Enter the following command in the console: \
  `python evaluate.py -m magingo -td test -on mf`

# Citation

If you use MAGIN-GO for your research, or incorporate our learning
algorithms in your work, please cite: Runxin Li, Wentao Xie, Zhenhong Shang, Xiaowu Li, Guofeng Shu, Lianyin
Jia, Wei Peng; MAGIN-GO:Protein Function Prediction Based on Dual
Graph Neural Networks and Gene Ontology Structure
>>>>>>> b698273 (Initial commit)
