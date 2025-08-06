package org.deeplearning4j;

import java.io.File;

import org.datavec.api.records.reader.RecordReader; // used to read records from input file
import org.datavec.api.records.reader.impl.csv.CSVRecordReader; // read records from CSV file
import org.datavec.api.split.FileSplit; //access the file path

import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator; // used to convert csv data into a dataset iterator for training

import org.deeplearning4j.nn.api.OptimizationAlgorithm; // specifies overall strategy for training the model, minimize loss function
import org.deeplearning4j.nn.conf.MultiLayerConfiguration; // holds the entire network configuration (layer structures)
import org.deeplearning4j.nn.conf.NeuralNetConfiguration; // used to build and set up the neural net config
import org.deeplearning4j.nn.conf.layers.DenseLayer; // used to create a fully connected dense layer
import org.deeplearning4j.nn.conf.layers.OutputLayer; // used to define the final output layer of the model
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork; // used to create the neural network from configuration
import org.deeplearning4j.optimize.listeners.ScoreIterationListener; // used to print score (loss) during training
import org.deeplearning4j.nn.weights.WeightInit; // used to set the method for initializing weights

import org.nd4j.evaluation.classification.Evaluation; // used to evaluate the model’s performance on classification tasks, calculates accuracy, recall
import org.nd4j.linalg.activations.Activation; // used to set activation functions (RELU, SOFTMAX, etc.)
import org.nd4j.linalg.api.ndarray.INDArray; // ND4J’s n-dimensional array used to hold input/output data
import org.nd4j.linalg.dataset.DataSet; // used to hold one batch of input features and labels
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator; // used to iterate over the dataset during training
import org.nd4j.linalg.lossfunctions.LossFunctions; // used to set loss function (e.g. NEGATIVELOGLIKELIHOOD)


public class MLPClassifierLinear {
	public static void main(String[] args) throws Exception{
		int seed = 123; //repeat and have the same random numbers generated
		double learningRate = 0.01;
		int batchSize = 50; //determines size of bath as data is ingested
		int nEpochs = 30; //number of total passes through the data
		int numInputs = 2; 
		int numOutputs = 2; //based upon the number of labels
		int numHiddenNodes = 20; //number of nodes
		
		//load the training data
		RecordReader rr = new CSVRecordReader();
		rr.initialize(new FileSplit(new File("linear_data_train.csv")));
		DataSetIterator trainIter = new RecordReaderDataSetIterator(rr, batchSize, 0, 2);
		
		// load the test-evaluation data:
		
		RecordReader rrTest = new CSVRecordReader();
		rrTest.initialize(new FileSplit(new File("linear_data_eval.csv")));
		DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest, batchSize, 0, 2);
		
		// define network configuration
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
			.seed(seed) // same random initialization for repeatability
			.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT) // update method
			.updater(new org.nd4j.linalg.learning.config.Nesterovs(learningRate, 0.9)) // momentum helps escape shallow minima - local low point in the loss landscape where the model could get
			.list() //tart defining the network as a list of layers.
			.layer(0, new DenseLayer.Builder() // first layer (input -> hidden)
				.nIn(numInputs) // input size
				.nOut(numHiddenNodes) // output size (hidden layer size)
				.weightInit(WeightInit.XAVIER) // good default init
				.activation(Activation.RELU) // nonlinear function to introduce learning capacity
				.build())
			.layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD) // good for classification
				.weightInit(WeightInit.XAVIER)
				.activation(Activation.SOFTMAX) // used for multi-class classification
				.nIn(numHiddenNodes) // connects from hidden layer
				.nOut(numOutputs) // output size equals number of classes
				.build())
			.build();

		// initialize the model from configuration
		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init(); // build internal structures

		model.setListeners(new ScoreIterationListener(10)); // log every 10 iterations

		// training loop: fit model to training data over epochs
		for (int n = 0; n < nEpochs; n++) {
			model.fit(trainIter); // one pass over all training data
		}

		System.out.println("Evaluate model........");

		// evaluation object compares predictions vs true labels
		Evaluation eval = new Evaluation(numOutputs);
		while (testIter.hasNext()) {
			DataSet t = testIter.next(); // get batch
			INDArray features = t.getFeatures(); // input x and y
			INDArray labels = t.getLabels(); // actual label (0 or 1)
			INDArray predicted = model.output(features, false); // predicted probabilities
			eval.eval(labels, predicted); // compare prediction with actual
		}

		System.out.print(eval.stats()); // display accuracy, precision, recall, etc.
	}
}
