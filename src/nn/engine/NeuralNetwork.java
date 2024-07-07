/*
  Neural Network Example

  Copyright (c) 2024, rspber (https://github.com/rspber)

  based on: https://github.com/heldergomesramos/Neural-Network

*/

package nn.engine;

import java.util.HashSet;

public class NeuralNetwork {

	public final InputLayer inputLayer;
	public final HiddenLayer[] hiddenLayers;
	public final OutputLayer outputLayer;

	public NeuralNetwork(int iNeurons, int[] hNeurons, double learningRate, final String[] expectedValues) {
		inputLayer = new InputLayer(iNeurons);
		hiddenLayers = new HiddenLayer[hNeurons.length];
		hiddenLayers[0] = new HiddenLayer(hNeurons[0], inputLayer, 1, learningRate);
		for(int i = 1; i < hNeurons.length; i++) {
			hiddenLayers[i] = new HiddenLayer(hNeurons[i], hiddenLayers[i-1], i + 1, learningRate);
		}

		HashSet<String> possibleOutputs = new HashSet<>();

		for( final String v : expectedValues ) {
			possibleOutputs.add(v);
		}

		outputLayer = new OutputLayer(possibleOutputs.size(), hiddenLayers[hNeurons.length - 1], possibleOutputs, learningRate);
	}

	/**
	 * Debug function with an opcode for what part of the neural network to show. 0 means not show
	 * @param opcode 1XX - show neurons; X1X - show biases; XX1 - show weights
	 */
	public void debug() {
		System.out.println(inputLayer.toString());
		for (final HiddenLayer hiddenLayer : hiddenLayers) {
			System.out.println(hiddenLayer.toString());
		}
		System.out.println(outputLayer.toString());
	}

	/**
	 * Uses feed forward followed by back propagation for a given training example
	 * @param inputs array of values given as inputs
	 */
	public int trainNetwork(final double[] inputs, final String expectedVal) {
		feedForward(inputs);
		int correctIndex = outputLayer.getCorrectIndex(expectedVal);
		outputLayer.backPropagation(correctIndex);
		return outputLayer.getBestIndex() == correctIndex ? 1 : 0;
	}

	public String test(double[] testData) {
		feedForward(testData);
		return outputLayer.getResult();
	}

	public void feedForward(final double[] inputs) {
		inputLayer.setInputValues(inputs);
		for (int i = 0; i < hiddenLayers.length; ++i)
			hiddenLayers[i].setUpNeurons();
		outputLayer.setUpNeurons();
	}

	public void train(final int count, final double[][] inputMatrix, final String[] expectedValues) {
		System.out.println("TRAINING...");
		boolean stop = false;
		int iteration = 0;
		double successRatio = 0;
		while( !stop && ++iteration < count ) {
			stop = true;
			double hits = 0;
			for(int i = 0; i < inputMatrix.length; i++) {
				hits += trainNetwork(inputMatrix[i], expectedValues[i]);
				if( !outputLayer.stop )
					stop = false;
			}
			successRatio = hits / inputMatrix.length;
		}
		System.out.println(String.format("Training finished with %d iterations", iteration));
		System.out.println(String.format("Success ratio with training samples: %.2f%%", successRatio * 100));
	}

	public void test(final double[][] testMatrix) {
		System.out.println("");
		for (int i = 0; i < testMatrix.length; i++) {
			String result = test(testMatrix[i]);
			System.out.println((i + 1) + " - " + result);
		}
	}

}
