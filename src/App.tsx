import * as tf from "@tensorflow/tfjs";
import { TRAINING_DATA } from "./mnist";
import { useEffect, useState, useRef } from "react";

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

function App() {
  const [index, setIndex] = useState(0);
  const [correct, setCorrect] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);
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

  function evaluate() {
    const OFFSET = Math.floor(Math.random() * INPUTS.length);

    const answer = tf.tidy(() => {
      const newInput = tf.tensor1d(INPUTS[OFFSET]);
      const output = model.predict(newInput.expandDims()) as tf.Tensor;

      output.print();

      return output.squeeze().argMax();
    });

    answer.array().then((idx) => {
      setIndex(idx as number);
      setCorrect(idx === OUTPUTS[OFFSET]);
      answer.dispose();
      drawImage(INPUTS[OFFSET]);
    });
  }

  const drawImage = (index: number[]) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas!.getContext("2d");
    const imageData = ctx!.getImageData(0, 0, 28, 28);

    for (let i = 0; i < 784; i++) {
      const color = index[i] * 255;
      imageData.data[i * 4] = color;
      imageData.data[i * 4 + 1] = color;
      imageData.data[i * 4 + 2] = color;
      imageData.data[i * 4 + 3] = 255;
    }

    ctx!.putImageData(imageData, 0, 0);

    setTimeout(evaluate, 2000);
  };

  useEffect(() => {
    train();
  }, []);

  return (
    <div
      style={{
        border: correct ? "1px solid green" : "1px solid red",
      }}
    >
      <canvas ref={canvasRef} id="canvas" width="28" height="28"></canvas>
      <div>{index}</div>
    </div>
  );
}

export default App;
