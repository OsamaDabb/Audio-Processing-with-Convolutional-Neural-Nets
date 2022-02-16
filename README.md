# Audio-Processing-with-Convolutional-Neural-Nets
Monophonic Instrument detection done with a CNN

Dataset: Tested on the Philharmonia dataset. Total of ~12,000 audio files. Includes a single held note for every possible note in that instruments range, for a total of 17 concert instruments. 

Pre-processing: raw audio files were normalized to the first 0.5 seconds of audio measured past a certain amplitude threshold so that they could be better standardized for NN. They were also converted into spectrograms measured on the mel scale (known as mel-spectrograms), which accounts for human distinctions in hearing at higher frequencies. Finally, we created a test/train split of 80/20. 

Network: the network itself was a deep convolutional network. It took advantage of 1 dimensional convolutions along the time axis with the intent of highlighting temporal patterns including sound onset and envelope, as well as MaxPooling layers to draw out significant details and a dropout of 25% at each layer to avoid overfitting. It then flattens the inputs along the frequency axis and feeds this information into two fully convolutional layers to draw out frequency based information such as instrument brightness and harmonic frequencies. The network came out to a size of 43,000 parameters, significantly smaller than the state of the art, as it took advantage of simple 1-dimensional convolutions and avoided large fully connected layers.

Results: After running the network for 100 epochs on the given dataset, we were able to consistently reach a validation accuracy of 90% and val. loss of 0.6, though certain iterations had reached the benchmark of 96%/0.21 accuracy/loss. Further work can be done from here expanding the network and the pre-processing procedure so that the network is better capable of handling more real-world data-types such as non-monotonic or polyphonic music. 
