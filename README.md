# ML_from_scratch
Machine learning algorithms implemented from scratch in numpy

### Neural Networks:
##### Project file: binary.py

Description:
This project was meant to test my understanding of how neural networks work. Going in, my hypothesis was that if I understand backprop correctly, then if I set the input data to be numbers represented in binary, the output data to be the same number in decimal, and only give it one layer, the weights of the network should converge to: 2<sup>0</sup>, 2<sup>1</sup>, 2<sup>2</sup>, 2<sup>3</sup>, 2<sup>4</sup>, 2<sup>5</sup>, 2<sup>6</sup>, ... depending on how how many neurons I give it and the range of the training data. If you go ahead and run binary.py with enough epochs to overfit the training data, you see that this is the truth. (I think this is a pretty good experiment for people new to neural networks)

##### Project file: firstNN.py

Description:
Very simple 2 layer neural network. Only goal was to get more "fluent" with passing information through neural networks, getting the shapes of the tensors correctly, and updating the weights correctly.

### Convolutional Neural Networks:
##### Project file: firstCNN.py

Description:

Implemented simple CNN to predict whether the face represented as a numpy array is smiling, frowning, or neutral. Additionally we can check the filters after training and see that they have an interpretable state (curves that help with classify frowns or smiles).
At the end of firstCNN.py there are 3 arrays (face1, face2, face3), feel free to change around the faces and see if it still generalizes to predict them correctly. Keep in mind that if someone other than you can barely classify the emotion, the network will probably have trouble too.

### Recurrent Neural Networks:
##### Project file: firstRNN.py

Description:

Implemented recurrent neural network to predict the word "hello". The only goal was to overfit on this word and verify that these networks actually work and can learn. The concept is really interesting to me, and I've never thought of the of memory in that way. You can almost think of it as getting the information from time step t as a ball of play doh, molding it in a way that adds information to it and passing it to a friend at time step t+1. At the end of this entire pipeline, the last friend is to make a decision based on all of the molds from time steps 0...T-1





