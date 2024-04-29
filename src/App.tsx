import * as tf from "@tensorflow/tfjs";
import { TRAINING_DATA } from "./mnist";

const INPUTS = TRAINING_DATA.inputs;
const OUTPUTS = TRAINING_DATA.outputs;

tf.util.shuffleCombo(INPUTS, OUTPUTS);

const INPUTS_TENSOR = tf.tensor2d(INPUTS);

const OUTPUTS_TENSOR = tf.oneHot(tf.tensor1d(OUTPUTS, "int32"), 10);

const model = tf.sequential();

model.add(
  tf.layers.dense({
    inputShape: [784],
    units: 32,
    activation: "relu",
  })
);

model.add(
  tf.layers.dense({
    units: 16,
    activation: "relu",
  })
);

model.add(
  tf.layers.dense({
    units: 10,
    activation: "softmax",
  })
);

model.summary();

train();

function train() {
  model.compile({
    optimizer: "adam",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  let result = model.fit(INPUTS_TENSOR, OUTPUTS_TENSOR, {
    shuffle: true,
    validationSplit: 0.2,
    batchSize: 512,
    epochs: 50,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        console.log(
          `Epoch: ${epoch}, Loss: ${logs?.loss ?? 0}, Accuracy: ${
            logs?.acc ?? 0
          }`
        );
      },
    },
  });

  INPUTS_TENSOR.dispose();
  OUTPUTS_TENSOR.dispose();

  evaluate();
}

function evaluate() {}

function App() {
  return (
    <>
      <canvas id="canvas" width="28" height="28"></canvas>
    </>
  );
}

export default App;
