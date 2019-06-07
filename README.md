# FactMap

This project utilizes the ClaimReview standard to track the social propagation of fake news and their corresponding fact checks on Reddit.

The project evolved over multiple stages:

1. Collecting Reddit posts and ClaimReviews, and organizing them in AsterixDB. This part is documented in the `./Wrangling/1. DataCollection.ipynb` and `./Wrangling/2. AsterixDB + Data Wrangling.ipynb` notebook. Additionally, the notebook `./Wrangling/3. Exploratory join analysis.ipynb` explores the resulting joint dataset.
2. Imputation of missing numerical review ratings using recurrent neural networks. This part is partially documented in the `./RNN/notebooks/1. RNN Training.ipynb`  and `./RNN/notebooks/2. Model evaluation.ipynb` notebooks. The actual model training was performed on a remote GPU cluster, however, so the complete model configuration is comprised of `./RNN/main.py` and the associated helper scripts under `./RNN/scripts/`.
3. Finally, network analysis was performed by the use of the  `NetworkX` library in `Python` and `Gephi` for visualization. The notebook `./Community Detection/community.ipynb` documents this process as closely as the interchangeable use of toolsets allows.

For demonstration purposes, then, a small notebook has been provided under `./demo/rnn_demo.ipynb` showcasing the prediction of a small subset of review ratings using the pretrained RNN.

