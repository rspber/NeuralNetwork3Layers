/*
  Neural Network Example

  Copyright (c) 2024, rspber (https://github.com/rspber)

  based on: https://github.com/heldergomesramos/Neural-Network

*/

package nn.engine;

import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

/**
 * Class responsible for the middle layers of the neural network, from which the output layer extends itself
 */
class HiddenLayer extends Layer {

	protected final Layer prevLayer; /* Represents the previous layer in the neural network, used to make the weights graph */
	protected final Neuron[] ns; /* Neurons in the layer, number of weights in each neuron equals number of prevLayer outputs */
	protected final int index;
	protected final double learningRate;

	/*
	 * Default constructor for random weight and bias generation
	 */
	public HiddenLayer(final int numberOfNeurons, final Layer prevLayer, final int idx, final double learningRate) {
		super(numberOfNeurons);
		this.prevLayer = prevLayer;
		ns = new Neuron[numberOfNeurons];
		for( int i = 0; i < ns.length; ++i) {
			ns[i] = new Neuron(prevLayer.outputs.length);
		}
		this.index = idx;
		this.learningRate = learningRate;
	}

	/*
	 * Constructor for specific input weights and biases
	 */
/*
	public HiddenLayer(final int numberOfNeurons, final Layer prevLayer, final int i, final double[][] weights, final double[] biases) {
		super(numberOfNeurons);
		this.prevLayer = prevLayer;
		this.weights = weights;
		this.biases = biases;
		this.index = i;
		this.learningRate = 0.0;
	}
*/
	public void readWeights(Scanner input) {
		System.out.println("Insert weights for:");
		for(int j = 0; j < prevLayer.outputs.length; j++)
			for (int i = 0; i < ns.length; i++) {
				System.out.print("[" + j + "][" + i + "] = ");
				ns[i].weights[j] = input.nextDouble();
			}
	}

	public void readBiases(Scanner input) {
		System.out.println("Insert biases for:");
		for (int i = 0; i < ns.length; i++) {
			System.out.print("[" + i + "] = ");
			ns[i].bias = input.nextDouble();
		}
	}

	/**
	 * Generates the neurons of each layer using a specific formula using weights and activations from the previous layer, as well as biases of the current one
	 */
	public void setUpNeurons() {
		for (int i = 0; i < outputs.length; ++i) {
			/* neurons[i] = sigmoid((weights[...][i] * prevNeurons[...]) + biases[i]); */
			outputs[i] = sigmoid(outputs[i] + prevLayer.weightsSqrtSum(ns[i].weights) + ns[i].bias);
		}
	}

	public static double sigmoid(double x) {
		return 1.0/(1+Math.exp(-x));
	}

	private void updateBiases(final double[] hErrors) {
		for(int i = 0; i < ns.length; i++) {
			ns[i].bias = learningRate * hErrors[i];
		}
	}

	@Override
	public void backPropagation(final double[] hErrors) {
		final double[] prevErrors = prevLayer.calcBackwardPrevErrors(ns, hErrors); 
		prevLayer.updateForwardingNeuronsWeights(ns, learningRate, hErrors);
		updateBiases(hErrors);
		prevLayer.backPropagation(prevErrors);
	}

	@Override
	public String toString() {
		final List<String> oli = new ArrayList<>(); 
		oli.add("");
		oli.add(String.format("HIDDEN LAYER, %d inputs", prevLayer.outputs.length));
		for (int i = 0; i < ns.length; ++i) {
			final Neuron n = ns[i];
			StringBuilder sb = new StringBuilder();
			sb.append(String.format("%s %d:   ", "Neuron", i));
			sb.append(String.format("Bias: %9.6f,   Weights: [", n.bias));
			String sep = "";
			for (int j = 0; j < n.weights.length; ++j) {
				sb.append(sep);
				sb.append(String.format("%9.6f", n.weights[j]));
				sep = ", ";
			}
			sb.append(String.format("],  Output: %9.6f", outputs[i]));
			oli.add(sb.toString());
		}
		return String.join("\n", oli);
	}
}
