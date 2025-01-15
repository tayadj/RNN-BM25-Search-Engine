# RNN-BM25-Search-Engine

Advanced search engine that leverages a Recurrent Neural Network (RNN) and the BM25 relevance metric to provide highly accurate and relevant search results.

## Features

- **Recurrent Neural Network (RNN)**: Utilizes RNN for processing and understanding the context of articles.
- **Neural Network Scaling**: Supports scaling of the neural network to improve performance and accuracy.
- **BM25 Relevance Metric**: Implements the BM25 algorithm to rank articles based on their relevance to the search query.
- **Automatic Topic Expansion**: Supports automatic expansion of article topics to enhance search results.
- **Scalability**: Designed to handle large datasets and provide quick search responses.

## Installation

To install the necessary dependencies, run the following command:

```bash
pip install pytest numpy pandas
```

## Usage

To start the search engine, run the following command:

```bash
cd project
python main.py
```

## Test

To manually test the search engine, run the following command:
```bash
cd project
pytest --tb=no test.py
```

## Structure

- **articles.csv**: Contains the articles to be indexed and searched.
- **model.npz**: Compressed version of the pre-trained model.
- **main.py**: Implementation and entry point of the application, containing the `model` class for the recurrent neural network and the `engine` class for the search engine based on `model` predictions and BM25 relevance metrics.
- **test.py**: Contains unit tests for the project search engine, based on `pytest`.
- **research.ipynb**: A study on the loss and accuracy of the model over the epochs in the context of various learning strategies. 

## License

This project is licensed under the MIT License.
