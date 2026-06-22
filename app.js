const express = require("express");
const bodyParser = require("body-parser");
const fs = require("fs");
const path = require("path");

const app = express();
const PORT = 3000;

app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());
app.use(express.static(path.join(__dirname, "public")));

const modelPath = path.join(__dirname, "logisticRegressionModel.json");

if (!fs.existsSync(modelPath)) {
  console.error("Model file not found: logisticRegressionModel.json");
  console.error("Please run: node trainModel.js");
  process.exit(1);
}

const modelData = JSON.parse(fs.readFileSync(modelPath, "utf8"));

const featureNames = modelData.featureNames;
const means = modelData.means;
const stds = modelData.stds;
const weights = modelData.weights;
const bias = modelData.bias;

if (!featureNames || !means || !stds || !weights) {
  console.error("Model file structure is invalid.");
  process.exit(1);
}

function sigmoid(z) {
  return 1 / (1 + Math.exp(-z));
}

function normalizeInput(inputValues) {
  return inputValues.map((value, i) => {
    const std = stds[i] === 0 ? 1 : stds[i];
    return (value - means[i]) / std;
  });
}

function predictProbability(inputValues) {
  const normalized = normalizeInput(inputValues);

  let score = bias;
  for (let i = 0; i < normalized.length; i++) {
    score += normalized[i] * weights[i];
  }

  return sigmoid(score);
}

app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "index.html"));
});

app.post("/predict", (req, res) => {
  try {
    const inputValues = featureNames.map((name) => parseFloat(req.body[name]));

    for (let i = 0; i < inputValues.length; i++) {
      if (isNaN(inputValues[i])) {
        return res.status(400).json({
          error: `Invalid value for ${featureNames[i]}`
        });
      }
    }

    const probability = predictProbability(inputValues);
    const prediction = probability >= 0.5 ? 1 : 0;
    const risk = prediction === 1 ? "High" : "Low";

    res.json({
      prediction,
      risk,
      probability: Number(probability.toFixed(4)),
      input: Object.fromEntries(
        featureNames.map((name, index) => [name, inputValues[index]])
      )
    });
  } catch (error) {
    console.error("Prediction error:", error);
    res.status(500).json({
      error: "Prediction failed",
      details: error.message
    });
  }
});

app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});
