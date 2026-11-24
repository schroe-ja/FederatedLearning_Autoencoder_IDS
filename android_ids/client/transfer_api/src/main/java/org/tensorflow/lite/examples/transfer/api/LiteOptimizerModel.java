/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.lite.examples.transfer.api;

import java.io.Closeable;
import java.nio.ByteBuffer;
import java.util.Map;
import java.util.TreeMap;

/** A wrapper for TFLite optimizer model. */
public class LiteOptimizerModel implements Closeable {

  private static final int FLOAT_BYTES = 4;

  private final LiteModelWrapper modelWrapper;

  LiteOptimizerModel(LiteModelWrapper modelWrapper) {
    this.modelWrapper = modelWrapper;
  }

  /**
   * Performs a single optimizer step.
   *
   * @param currentParams current values of model trainable parameters.
   * @param gradients trainable parameter gradients.
   * @param optimizerState current mutable optimizer state.
   * @param newParams where to store new parameter values.
   * @param newOptimizerState where to store new mutable optimizer state.
   */
  void performStep(
      ByteBuffer[] currentParams,
      ByteBuffer[] gradients,
      ByteBuffer[] optimizerState,
      ByteBuffer[] newParams,
      ByteBuffer[] newOptimizerState) {
    
    int numParams = currentParams.length;
    int numStates = optimizerState.length;

    Object[] inputs = new Object[
        numParams + gradients.length + numStates
    ];

    System.arraycopy(currentParams, 0, inputs, 0, numParams);
    System.arraycopy(gradients, 0, inputs, numParams, gradients.length);

    System.arraycopy(
        optimizerState, 0, inputs,
        numParams + gradients.length, numStates
    );

    Map<Integer, Object> outputs = new TreeMap<>();

    for (int i = 0; i < newParams.length; i++) {
        outputs.put(i, newParams[i]);
    }

    for (int i = 0; i < newOptimizerState.length; i++) {
        outputs.put(
            newParams.length + i,
            newOptimizerState[i]
        );
    }

    for (Object input : inputs) {
        ((ByteBuffer) input).rewind();
    }

    for (Object output : outputs.values()) {
        ((ByteBuffer) output).rewind();
    }

    modelWrapper.getInterpreter()
        .runForMultipleInputsOutputs(inputs, outputs);

    for (Object input : inputs) {
        ((ByteBuffer) input).rewind();
    }

    for (Object output : outputs.values()) {
        ((ByteBuffer) output).rewind();
    }
  }

  /**
   * Reads the sizes of the mutable optimizer state elements.
   *
   * @return sizes of optimizer state elements.
   */
  int[] stateElementSizes() {
    // The generic optimizer model signature is:
    // *variables, *gradients, *optim_state -> *new_variables, *new_optim_state
    // There is no metadata included that would contain the number of variables
    // for the model, but we can easily infer it using the fact that
    // len(variables) == len(gradients) == len(new_variables) == number of variables.
    int numVariables =
        modelWrapper.getInterpreter().getInputTensorCount()
            - modelWrapper.getInterpreter().getOutputTensorCount();

    int[] result = new int[modelWrapper.getInterpreter().getInputTensorCount() - numVariables * 2];
    for (int inputIdx = numVariables * 2;
        inputIdx < modelWrapper.getInterpreter().getInputTensorCount();
        inputIdx++) {
      result[inputIdx - numVariables * 2] =
          modelWrapper.getInterpreter().getInputTensor(inputIdx).numElements();
    }

    return result;
  }

  @Override
  public void close() {
    modelWrapper.close();
  }
}
