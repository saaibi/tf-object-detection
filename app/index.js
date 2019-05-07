import * as tf from "@tensorflow/tfjs";
import { ObjectDetectionImageSynthesizer } from "./synthetic_images";

const canvas = document.getElementById("data-canvas");
const status = document.getElementById("status");
const testModel = document.getElementById("test");
const loadHostedModel = document.getElementById("load-hosted-model");
const inferenceTimeMs = document.getElementById("inference-time-ms");
const trueObjectClass = document.getElementById("true-object-class");
const predictedObjectClass = document.getElementById("predicted-object-class");

let idTimeInterval = null;

const LOCAL_MODEL_PATH = "mobilenet.json";
const HOSTED_MODEL_PATH =
  "https://storage.googleapis.com/tfjs-examples/simple-object-detection/dist/object_detection_model/model.json";

const updateUI = (isRectangle, predit) => {
  const trueClassName = isRectangle ? "rectangle" : "triangle";
  trueObjectClass.textContent = trueClassName;

  const shapeClassificationThreshold = canvas.width / 2;
  const predictClassName =
    predit > shapeClassificationThreshold ? "rectangle" : "triangle";
  predictedObjectClass.textContent = predictClassName;

  if (predictClassName === trueClassName) {
    predictedObjectClass.classList.remove("shape-class-wrong");
    predictedObjectClass.classList.add("shape-class-correct");
  } else {
    predictedObjectClass.classList.remove("shape-class-correct");
    predictedObjectClass.classList.add("shape-class-wrong");
    clearInterval(idTimeInterval);
    testModel.disabled = true;
  }
};

const generateTensorsImgTarg = (fullCanvas, boundingBox) =>
  tf.tidy(() => {
    return {
      images: [1, 224, 224, 3],
      targets: [1, 4]
    };
  });

const arrAvg = arr => arr.reduce((a, b) => a + b, 0) / arr.length;

// const t0 = tf.util.now();
// inferenceTimeMs.textContent = `${(tf.util.now() - t0).toFixed(2)}`;
const runAndVisualizeInference = async model => {
  // Sintetice una imagen de entrada y muéstrela en el lienzo.
  const synth = new ObjectDetectionImageSynthesizer(canvas);

  const { fullCanvas, boundingBox } = await synth.generateExample();

  const { images, targets } = generateTensorsImgTarg(canvas, [1, 1, 2, 2]);

  // Ejecuta inferencia con el modelo.

  // Visualice los cuadros de delimitación verdaderos y predichos.
  synth.drawBoundingBoxes([2, 3, 4, 5], [1, 2, 3, 4], 0.0);

  // Mostrar el verdadero y predecir las clases de objetos.
  // updateUI(isRectangle, modelOut[0]);

  // Tensor memory cleanup.
  // tf.dispose([images, targets]);
};

const init = async () => {
  const loadModal = async modelPath => {
    try {
      status.textContent = `Loading model ...`;
      const model = null;
      testModel.disabled = false;
      status.textContent = `Loaded model successfully. Now click "Test Model".`;
      runAndVisualizeInference(model);

      idTimeInterval = setInterval(() => {
        runAndVisualizeInference(model);
      }, 3000);
      testModel.addEventListener("click", () =>
        runAndVisualizeInference(model)
      );
    } catch (err) {
      status.textContent = `Failed to load model`;
    }
  };
  loadModal(HOSTED_MODEL_PATH);
  // loadModal(LOCAL_MODEL_PATH);
};

init();

// const catElement = document.getElementById("cat");
// const img = tf.browser.fromPixels(catElement).toFloat();
// const offset = tf.scalar(127.5);
// const normalized = img.sub(offset).div(offset);
// const batched = normalized.reshape([1, 224, 224, 3]);
// const modelOut2 = await model.predict(batched).data();
// console.log(modelOut2);
