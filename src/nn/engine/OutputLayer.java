/*
  Neural Network Example

  Copyright (c) 2024, rspber (https://github.com/rspber)

  based on: https://github.com/heldergomesramos/Neural-Network

*/

package nn.engine;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

/**
 * Class responsible for representing the last layer of the neural network, extending HiddenLayer and using its properties
 */
class OutputLayer extends HiddenLayer {

	final String[] names;
	public boolean stop;
	
	public OutputLayer(final int numberOfNeurons, final Layer prevLayer, final HashSet<String> possibleOutputs, final double learningRate) {
		super(numberOfNeurons, prevLayer, -1, learningRate);
		names = new String[numberOfNeurons];
		int i = 0;
		for(final String name: possibleOutputs) {
			names[i] = name;
			i++;
		}
	}   

	/* δOi = oi(E)(1 - oi(E))(t1(E) - oi(E)) */
	private static double doiE(final double oiE, final double t1E) {
		return oiE * (1 - oiE) * (t1E - oiE);
	}

//	@Override
	public void backPropagation(final int correctIndex) {
		/* 1 for the correct output, 0 for everything else */
		int[] target = new int[outputs.length];
		fillTarget(target, correctIndex);

		/* Calculate output errors (δO) */
		/* δOi = oi(E)(1 - oi(E))(t1(E) - oi(E)) */
		double[] outputErrors = new double[outputs.length];
		for(int i = 0; i < outputs.length; i++) {
			outputErrors[i] = doiE(outputs[i], target[i]);
			if(target[i] == 1) 
				stop = outputs[i] >= 0.95 ? true : false;   
		}
	
		/* Calculate errors to propagate to the hidden layers (δH) */
		double[] prevErrors = new double[prevLayer.outputs.length];
		/* j is the previous layer neuron index */
		for(int j = 0; j < prevLayer.outputs.length; j++) {
			double intermediateVal = 0;
			for(int i = 0; i < ns.length; i++) {
				double val = ns[i].weights[j] * outputErrors[i];
				intermediateVal += val;
			}

			/* δHj */
			prevErrors[j] = intermediateVal * (prevLayer.outputs[j]) * (1 - prevLayer.outputs[j]);
		}

		/* Calculate deltas to update weights */
		for(int j = 0; j < prevLayer.outputs.length; j++)
			for(int i = 0; i < ns.length; i++) {
				double weightDelta = learningRate * outputErrors[i] * prevLayer.outputs[j];
				ns[i].weights[j] += weightDelta;
			}

		for(int i = 0; i < ns.length; i++) {
			double biasDelta = learningRate * outputErrors[i];
			ns[i].bias = biasDelta;
		}
		prevLayer.backPropagation(prevErrors);
	}
	
	/* Returns the name of the neuron with the highest value */
	public String getResult() {
		double max = 0;
		String result = "";
		for(int i = 0; i < outputs.length; i++)
			if(outputs[i] > max) {
				max = outputs[i];
				result = names[i];
			}
		return result;
	}

	/* Returns the index of the neuron with the highest value */
	public int getBestIndex() {
		double max = 0;
		int index = -1;
		for(int i = 0; i < outputs.length; i++)
			if(outputs[i] > max) {
				max = outputs[i];
				index = i;
			}
		return index;
	}

	/* Returns the index of the neuron with the given name */
	public int getCorrectIndex(final String expectedVal) {
		for(int i = 0; i < names.length; i++)
			if(expectedVal.equals(names[i]))
				return i;
		return -1;
	}

	/* Fill the target array for the current example (1 - expectedVal | 0 - everything else) */
	public static void fillTarget(final int[] target, final int correctIndex) {
		for(int i = 0; i < target.length; i++)
			target[i] = i == correctIndex ? 1 : 0;
	}

	@Override
	public String toString() {
		final List<String> oli = new ArrayList<>();
		oli.add("");
		oli.add(String.format("OUTPUT LAYER, %d inputs", prevLayer.outputs.length));
		for (int i = 0; i < ns.length; ++i) {
			final Neuron n = ns[i];
			StringBuilder sb = new StringBuilder();
			sb.append(String.format("Neuron: %s %d:   ", names[i], i));
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
