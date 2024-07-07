/*
  Neural Network Example

  Copyright (c) 2024, rspber (https://github.com/rspber)

  based on: https://github.com/heldergomesramos/Neural-Network

*/

package nn.engine;

/**
 * Class representing a layer, from which InputLayer HiddenLayer are extended. Contains most variables and functions related to layers
 */
public abstract class Layer {

	protected final double[] outputs; /* Value for each neuron in the layer */

	public Layer(final int numberOfNeurons) {
		outputs = new double[numberOfNeurons];
	}

	public abstract void backPropagation(double[] hErrors);

	protected double weightsSqrtSum(final double[] weights) {
		double sum = 0.0;
		for (int i = 0; i < outputs.length; ++i) { 
			sum += weights[i] * outputs[i];
		}
		return sum;
	}

	public static double intermediateVal(final Neuron[] ns, final int j, final double[] hErrors) {
		double intermediateVal = 0;
		for(int i = 0; i < ns.length; i++) {
			intermediateVal += ns[i].weights[j] * hErrors[i];
		}
		return intermediateVal;
	}

	/* Calculate errors to propagate to the hidden layers (δH) */
	public double[] calcBackwardPrevErrors(final Neuron[] forwards, final double[] hErrors) { 
		double[] errors = new double[outputs.length];
		for(int j = 0; j < outputs.length; j++) {
			errors[j] = intermediateVal(forwards, j, hErrors) * (outputs[j]) * (1 - outputs[j]);
		}
		return errors;
	}

	/* Calculate deltas to update weights */
	/* Δ = η * δO * hi(E) */
	public void updateForwardingNeuronsWeights(final Neuron[] forwards, final double learningRate, final double[] hErrors) {
		for(int j = 0; j < outputs.length; j++) {
			for(int i = 0; i < forwards.length; i++) {
				double delta = learningRate * hErrors[i] * outputs[j];
				forwards[i].weights[j] += delta;
			}
		}
	}

}
