Using Recurrent Neural Networks(RNN) to predict the price of the iron_ore stock and using time lags as a feature.The model forecast iron ore prices for any given time window. Here we have considered for just 3 months/90days.
The network has a visible layer with 1 input, a hidden layer with 10/40/120 LSTM blocks or neurons, and an output layer that makes a single value prediction. The default sigmoid activation function is used for the LSTM blocks. The network is trained for 50 epochs and a batch size of 32 is used.
We can take the input of any time window for the corresponding predictions. 

               output layer ------>         Y
                                             |
                                           Dense
                                         / |   | \
                                        /  |   |  \
                Hidden layer (LSTM)    L1->L2->..->Ln
                                       |
                Input layer  ----->    X
