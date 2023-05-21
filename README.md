* entity name: CS22M027
* project name: Assignment_3
* wandb key : ab8429eef239aaf26799fd710e5ea6aec30ab090
# cs6910_assignment_3

A Deep learning programming assignment where we have to develop a Encoder-Decoder model using Recurrent Neural Networks to train on Aksharantar  dataset (language: Telugu) so that for a given english word our Model will convert it into telugu.

This model is trained in two ways. one is without attention mechanism and one is with attention mechanism.

## 1) Without Attention Mechanism

### Parameters Used
- input_dim : number of characters presented in input data language
- hidden_dim:Dimensionality of the hidden state of the encoder.
- embedding_dim:Dimensionality of the token embeddings.
- num_layers: Number of layers in the encoder.
- output_dim : number of characters presented in target data language
- units :number of hidden units presented in each Recurrent Neural Network (default = 64)
- bidirectional: Whether to use bidirectional RNN or not ("Yes" or "No").
- learning_rate : learning rate of the model (default = 1e-3)
- num_encoders : number of encoder layers (default = 1)
- num_decoders : number of decoder layers (default = 1)
- batch_size : batch size while traininig the data (default = 64)
- cell_type : Recurrent Neural Network type  (default = 'LSTM') (options = 'LSTM','GRU','RNN')
- drop_out : drop out of the network (default = 0)
- 
### Classes and Functions 
#### MyEncoder()
MyEncoder class represents the encoder component of a sequence-to-sequence model.
Methods:
        initialize_hidden(): Initializes the initial hidden state for the encoder.
        forward(input, hidden): Performs forward pass of the encoder.
returns an Encoder model with above mentioned configurations (like num_encoders = 2  means 2 encoder layers)

#### MyDecoder()
MyDecoder class represents the decoder component of a sequence-to-sequence model.
Methods:
        forward(input, hidden): Performs forward pass of the decoder.
returns a Decoder model with above mentioned configurations (like num_decoders = 2  means 2 decoder layers)

#### evaluate_model()
 - inputs: 
    - encoder_input_data : test input dataset in one hot encoded format
    - target_data : original target test dataset
 - outputs:
  runs the model on test dataset without **teacher forcing** and returns the predictions and accuracy

#### train()
this function trains the encoder-decoder model using the provided hyperparameters and monitors the training and validation performance.
It sets up the hyperparameters and initializes the encoder and decoder models based on the provided configurations.
It iterates over the specified number of epochs and performs the training process.
After each epoch, it evaluates the model on the training and validation data and prints and logs the loss and accuracy.

## Hyperparameter configurations used
- Input embedding size: 64, 128, 256
- Number of encoder layers: 1, 2, 3 
- Number of decoder layers: 1, 2, 3 
- Hidden layer size: 64, 128, 256
- Cell type: RNN, GRU, LSTM
- Dropout values: 0, 0.1, 0.2, 0.4
- Bidirectional: Yes,No
- teach_ratio- 0.4,0.5
- Learning rate: 1e-4, 1e-5
- Batch size: 64, 128, 256

#### Training the model
 - to train the model we need to initalise the model with MyEncoder and MyDecoder functions and Train the model using train() method
 
#### Evaluating the model
 - to evaluate the model just pass the test data to evaluate model function it will display the predictions and test_accuracy without teacher forcing


## 2) With Attention Mechanism

### Classes and Functions 
- All the fuctions used above like evaluate_model,train will also be used here
#### AttentionDecoderRNN()
implements the attention mechanism in the decoder of a sequence-to-sequence model, allowing the model to focus on different parts of the input sequence during decoding
Methods:
        forward(input, hidden, encoder_outputs)
returns the output tensor representing the predictions of the decoder at the current timestep,updated hidden state of the decoder and attention weights indicating the importance of each encoder output 

### AttnDecoder()
Methods:
        forward( input, hidden,encoder_outputs,word_length,state)
returns the output tensor representing the predictions of the decoder at the current timestep,updated hidden state of the decoder and context vector computed using attention mechanism

## Hyperparameter configurations used
- All the hyperparameters used above with the parameter Attention set to 'Yes' are used

#### Training the model
 - to train the model we need to initalise the model with RnnModel function and Train the model using fit() method
 
#### Evaluating the model
 - to evaluate the model just pass the test data to evaluate model function it will display the predictions and test_accuracy without teacher forcing

