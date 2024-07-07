/*
  Neural Network Example

  Copyright (c) 2024, rspber (https://github.com/rspber)

  based on: https://github.com/heldergomesramos/Neural-Network

*/

package nn.engine;

import java.util.ArrayList;
import java.util.List;

/**
 * Class responsible for just the first layer of the neural network
 */
class InputLayer extends Layer {

	public InputLayer(final int numberOfNeurons) {
		super(numberOfNeurons);
	}

	public void setInputValues(final double[] inputs) {
		for (int i = 0; i < outputs.length; ++i)
			outputs[i] = inputs[i];
	}

	public void backPropagation(double[] dontCare) {}

	public String toString() {
		final List<String> oli = new ArrayList<>();
		oli.add("");
		oli.add("INPUT LAYER");
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < outputs.length; ++i) {
			oli.add("Output " + i + " : " + outputs[i]);
		}
		oli.add(sb.toString());
		return String.join("\n", oli);
	}
}
