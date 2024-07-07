/*
  Neural Network Example

  Copyright (c) 2024, rspber (https://github.com/rspber)

  based on: https://github.com/heldergomesramos/Neural-Network

*/

package nn.engine;

public class Neuron {

	public double bias;
	public double[] weights;

	public Neuron(final int numberOfSynapses) {
		bias = Math.random() * 2 - 1;
		weights = new double[numberOfSynapses];
		for( int i = 0; i < weights.length; ++i) {
			weights[i] = Math.random() * 2 - 1;
		}
	}

}
