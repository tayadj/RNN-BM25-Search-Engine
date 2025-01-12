from collections import Counter
import pandas as pd
import numpy as np
import warnings
import math
import re

warnings.filterwarnings("ignore")

class Model:

    def __init__(self, hidden_dimension = 64, data_path = './data/articles.csv'):

        self.activation = lambda x: np.exp(x) / sum(np.exp(x))
        self.activation_derivative = lambda x: (1 - x ** 2)
        self.loss = lambda x: -np.log(x)

        self.stops = []
        
        self.load_data(data_path)
        self.load_vocabulary()
        self.load_network(hidden_dimension)

    def normalize_text(self, text):

        text = text.lower()
        text = re.sub(r'[^\w\s\']', '', text)
        text = ' '.join([word for word in text.split() if len(word) > 2 and word not in self.stops])

        return text

    def convert_text(self, text):
        
        text = self.normalize_text(text)
        inputs = []

        for word in text.split():

            value = np.zeros((self.vocabulary_size, 1))

            try:

                value[self.word_to_index[word]] = 1

            except KeyError:

                pass

            inputs.append(value)

        return inputs

    def load_data(self, path):

        self.data_raw = pd.read_csv(path)
        self.data = self.data_raw.map(self.normalize_text)

    def load_vocabulary(self):

        self.vocabulary = sorted(list(set([word for text in self.data.values.flatten() for word in text.split()])))
        self.vocabulary_size = len(self.vocabulary)

        self.topics = sorted(list(set([word for text in self.data.values.flatten()[2::3] for word in text.split()])))
        self.topics_size = len(self.topics)

        self.stops = list(set.intersection(*[set(text.split()) for text in self.data.values.flatten()[1::3]]))

        self.word_to_index = { word : index for index, word in enumerate(self.vocabulary) }
        self.index_to_word = { index : word for index, word in enumerate(self.vocabulary) }

    def load_network(self, dimension_hidden):
        
        self.dimension_input = self.vocabulary_size
        self.dimension_output = self.topics_size
        self.dimension_hidden = dimension_hidden

        self.weights_input_to_hidden = np.random.uniform(-0.5, 0.5, (self.dimension_hidden, self.dimension_input))
        self.weights_hidden_to_output = np.random.uniform(-0.5, 0.5, (self.dimension_output, self.dimension_hidden))
        self.weights_hidden_to_hidden = np.random.uniform(-0.5, 0.5, (self.dimension_hidden, self.dimension_hidden))
        self.bias_hidden = np.zeros((self.dimension_hidden, 1))
        self.bias_output = np.zeros((self.dimension_output, 1))

    def forward(self, inputs):

        hidden = np.zeros((self.weights_hidden_to_hidden.shape[0], 1))

        self.recent_inputs = inputs
        self.recent_hidden = { 0: hidden }

        for index, vector in enumerate(inputs):

            hidden = np.tanh(self.weights_input_to_hidden @ vector + self.weights_hidden_to_hidden @ hidden + self.bias_hidden)
            
            self.recent_hidden[index + 1] = hidden

        output = self.weights_hidden_to_output @ hidden + self.bias_output

        return hidden, output
    
    def backward(self, gradient_output, learning_rate):

        size = len(self.recent_inputs)

        gradient_weights_hidden_to_output = gradient_output @ self.recent_hidden[size].T
        gradient_bias_output = gradient_output

        gradient_weights_hidden_to_hidden = np.zeros(self.weights_hidden_to_hidden.shape)
        gradient_weights_input_to_hidden = np.zeros(self.weights_input_to_hidden.shape)
        gradient_bias_hidden = np.zeros(self.bias_hidden.shape)

        gradient_hidden = self.weights_hidden_to_output.T @ gradient_output

        for time in reversed(range(size)):

            delta = self.activation_derivative(self.recent_hidden[time + 1]) * gradient_hidden
            gradient_bias_hidden += delta
            gradient_weights_hidden_to_hidden += delta @ self.recent_hidden[time].T
            gradient_weights_input_to_hidden += delta @ self.recent_inputs[time].T
            gradient_hidden = self.weights_hidden_to_hidden @ delta

        for gradient in [gradient_weights_input_to_hidden, gradient_weights_hidden_to_hidden, gradient_weights_hidden_to_output, gradient_bias_hidden, gradient_bias_output]:
            
            np.clip(gradient, -1, 1, out = gradient)

        self.weights_hidden_to_output -= learning_rate * gradient_weights_hidden_to_output
        self.weights_hidden_to_hidden -= learning_rate * gradient_weights_hidden_to_hidden
        self.weights_input_to_hidden -= learning_rate * gradient_weights_input_to_hidden
        self.bias_output -= learning_rate * gradient_bias_output
        self.bias_hidden -= learning_rate * gradient_bias_hidden 

    def process(self, data, learning_rate, learn = True):

        loss = 0
        correct = 0
        quantity = 0

        for value, label in zip(self.data_raw[['Title', 'Text']].values.flatten(), self.data[['Topic', 'Topic']].values.flatten()):

            for value_ in value.split('.'):

                quantity += 1
                inputs = self.convert_text(value_)
                
                target = self.topics.index(label)            

                hidden, output = self.forward(inputs)

                probabilities = self.activation(output)
            
                loss += self.loss(probabilities[target])[0]
                correct += int(np.argmax(probabilities) == target)

                if learn:
            
                    probabilities[target] -= 1
                    self.backward(probabilities, learning_rate)

        return loss / quantity, correct / quantity

    def learn(self, epochs = 100, learning_rate = 0.001):

        for epoch in range (epochs):

            coefficient_loss, coefficient_accuracy = self.process(self.data, learning_rate)

            if epoch % (epochs / 10) == (epochs / 10 - 1):

                print(f'Epoch {epoch+1}:')
                print(f'Loss: {round(coefficient_loss, 3)}')
                print(f'Accuracy: {round(coefficient_accuracy * 100, 3)}%')

    def predict(self, value):
        
        inputs = self.convert_text(value)
        hidden, output = self.forward(inputs)
        probabilities = self.activation(output)
        
        return probabilities

    def save(self, path = './data/model.npz'):

        np.savez(path, 
        weights_input_to_hidden = self.weights_input_to_hidden, 
        weights_hidden_to_output = self.weights_hidden_to_output, 
        weights_hidden_to_hidden = self.weights_hidden_to_hidden, 
        bias_hidden = self.bias_hidden, 
        bias_output = self.bias_output, 
        topics = self.topics,
        vocabulary = self.vocabulary,
        dimension_hidden = [self.dimension_hidden],
        stops = self.stops
        )

    def load(self, path = './data/model.npz'):

        with np.load('./data/model.npz') as loaded:

            self.weights_input_to_hidden = loaded['weights_input_to_hidden']
            self.weights_hidden_to_output = loaded['weights_hidden_to_output']
            self.weights_hidden_to_hidden = loaded['weights_hidden_to_hidden']
            self.bias_hidden = loaded['bias_hidden']
            self.bias_output = loaded['bias_output']

            self.topics = list(loaded['topics'])
            self.topics_size = len(self.topics)

            self.vocabulary = loaded['vocabulary']
            self.vocabulary_size = len(self.vocabulary)

            self.word_to_index = { word : index for index, word in enumerate(self.vocabulary) }
            self.index_to_word = { index : word for index, word in enumerate(self.vocabulary) }

            self.stops = loaded['stops']

            self.dimension_input = self.vocabulary_size
            self.dimension_output = self.topics_size
            self.dimension_hidden = loaded['dimension_hidden'][0]

                    

        


class Engine:

    def __init__(self):
        
        self.model = Model()
        self.model.load()

    def normalize_text(self, text):

        text = text.lower()
        text = re.sub(r'[^\w\s\']', '', text)

        return text

    def compute_idf(self, corpus):

        size = len(corpus)
        counter = Counter()
        idf = {}

        for document in corpus:

            unique_terms = set(document)

            for term in unique_terms:

                counter[term] += 1
        
        for term, frequency in counter.items():

            idf[term] = math.log((size - frequency + 0.5) / (frequency + 0.5) + 1)

        return idf

    def compute_bm25(self, corpus, k1 = 1.5, b = 0.75):

        idf = self.compute_idf(corpus)
        bm25 = []

        average_document_length = sum(len(document) for document in corpus) / len(corpus)       

        for document in corpus:

            document_length = len(document)
            term_frequency = Counter(document)
            score = {}

            for term, frequency in term_frequency.items():

                numerator = idf.get(term, 0) * frequency * (k1 + 1)
                denominator = frequency + k1 * (1 - b + b * (document_length / average_document_length))
                score[term] = numerator / denominator

            bm25.append(score)

        return bm25


    def display(self, content, query):
        
        print(f'===========================================\nQuery: {query}\n\n')

        for index, row in content.iterrows():

            topic = row['Title']
            text = row['Text']

            separator = "- " * int(max(content['Title'].apply(len).max(), len('Title'))/2)

            print(f"{row['Title']}\n{separator}\n{row['Text']}\n\n")

        print(f'===========================================\n')

    def search(self, query, precision = 1.75):

        tokenized_query = self.normalize_text(query).split() 

        probabilities = self.model.predict(query)
        threshold = 1 / self.model.topics_size - 0.05

        valid_indices = [index for index, probability in enumerate(probabilities) if probability > threshold]
        valid_topics = [self.model.topics[i] for i in valid_indices]

        result = self.model.data[self.model.data['Topic'].isin(valid_topics)]
        
        corpus = [document.split() for document in (result['Title'] + ' ' + result['Text'])]
        bm25_scores = self.compute_bm25(corpus)

        scores = []

        for score in bm25_scores:

            total_score = sum(score.get(term, 0) for term in tokenized_query)
            scores.append(total_score)

        result['Score'] = scores
        result = result[result['Score'] > precision]
        result = result.sort_values(by = 'Score', ascending = False)

        self.display(result, query)