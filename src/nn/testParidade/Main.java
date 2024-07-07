/*
  Neural Network Example

  Copyright (c) 2024, rspber (https://github.com/rspber)

  based on: https://github.com/heldergomesramos/Neural-Network

*/

package nn.testParidade;

import nn.engine.NeuralNetwork;

public class Main {

	// testParidade
	static final double[][] inputMatrix = new double[][] {
		{ 1, 0, 1},  // 1
		{ 0, 0, 0},  // 0
		{ 1, 1, 1},  // 1
		{ 0, 0, 0},  // 1
		{ 1, 1, 0},  // 1
		{ 1, 1, 1},  // 0
		{ 0, 1, 0},  // 1
		{ 0, 0, 1}   // 0
	};

	static final String[] expectedValues = { "1", "0", "1", "1", "1", "0", "1", "0"};

	// settings.txt
	static final int[] hNeurons = new int[] { 4 };
	static final double learningRate = 0.05;
	static final int trainCount = 10;

	// irisTest.txt
	static final double[][] testMatrix = new double[][] { 
		{5.7, 3.8, 1.7, 0.3},
		{6.2, 2.2, 4.5, 1.5},
		{5.7, 2.5, 5.0, 2.0},
		{4.7, 3.5, 1.1, 0.2},
		{6.1, 3.0, 4.6, 1.4},
		{7.8, 3.0, 6.1, 2.4}
	};

	public static void main(String[] args) {

		final NeuralNetwork nn = new NeuralNetwork(inputMatrix[0].length, hNeurons, learningRate, expectedValues);

		nn.train(trainCount, inputMatrix, expectedValues);

		nn.debug();

		nn.test(testMatrix);

		System.out.println("And what's going on here, anybody knows ?");
	}

}
