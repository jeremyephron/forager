import React, { useState, useEffect } from "react";
import {
  Button,
  Input,
  FormGroup,
  Spinner,
  Collapse,
} from "reactstrap";
import { ReactSVG } from "react-svg";

import fromPairs from "lodash/fromPairs";
import toPairs from "lodash/toPairs";

import CategoryInput from "./CategoryInput";
import FeatureInput from "./FeatureInput";

var dateFormat = require("dateformat");

const STATUS_POLL_INTERVAL = 3000;  // ms

const dnns = [
  {id: "dnn", label: "DNN w/ BG Splitting"},
];

const endpoints = fromPairs(toPairs({
  trainModel: "train_model_v2",
  modelStatus: "model_v2",
  modelInference: "model_inference_v2",
  modelInferenceStatus: "model_inference_status_v2",
  stopModelInference: "stop_model_inference_v2",
  startCluster: "start_cluster",
  clusterStatus: "cluster",
}).map(([name, endpoint]) => [name, `${process.env.REACT_APP_SERVER_URL}/api/${endpoint}`]));

const TrainPanel = ({
  datasetName,
  datasetInfo,
  modelInfo,
  isVisible,
  username,
  disabled,
  categories,
  updateModels,
}) => {
  const [dnnAdvancedIsOpen, setDnnAdvancedIsOpen] = useState(false);

  const [modelName, setModelName] = useState();
  const [dnnType, setDnnType] = useState();
  const [dnnAugmentNegs, setDnnAugmentNegs] = useState();
  const [dnnPosTags, setDnnPosTags] = useState([]);
  const [dnnNegTags, setDnnNegTags] = useState([]);

  const [dnnCheckpointModel, setDnnCheckpointModel] = useState();
  const [dnnAugmentIncludeTags, setDnnAugmentIncludeTags] = useState([]);
  const [dnnAugmentExcludeTags, setDnnAugmentExcludeTags] = useState([]);

  //
  // CLUSTER
  //
  const [clusterId, setClusterId] = useState(null);
  const [clusterReady, setClusterReady] = useState(false);

  const startCreatingCluster = async () => {
    const url = new URL(`${endpoints.startCluster}`);
    var clusterResponse = await fetch(url, {
      method: "POST",
    }).then(r => r.json());
    setClusterId(clusterResponse.cluster_id);
    setClusterReady(false);
  };

  const getClusterStatus = async () => {
    const url = new URL(`${endpoints.clusterStatus}/${clusterId}`);
    const clusterStatus = await fetch(url, {
      method: "GET",
    }).then(r => r.json());
    setClusterReady(clusterStatus.ready);
  };

  useEffect(() => {
    if (clusterId && !clusterReady) {
      const interval = setInterval(() => {
        getClusterStatus();
      }, STATUS_POLL_INTERVAL);
      return () => clearInterval(interval);
    }
  }, [clusterId, clusterReady]);

  //
  // TRAINING
  //
  const [requestDnnTraining, setRequestDnnTraining] = useState(false);
  const [trainingModelId, setTrainingModelId] = useState();
  const [trainingEpoch, setTrainingEpoch] = useState();
  const [trainingTimeLeft, setTrainingTimeLeft] = useState();

  const trainDnnOneEpoch = async () => {
    const url = new URL(`${endpoints.trainModel}/${datasetName}`);
    let body = {
      model_name: modelName,
      cluster_id: clusterId,
      bucket: "foragerml",
      index_id: datasetInfo.index_id,
      pos_tags: dnnPosTags.map(t => `${t.category}:${t.value}`),
      neg_tags: dnnNegTags.map(t => `${t.category}:${t.value}`),
      augment_negs: dnnAugmentNegs,
      aux_label_type: "imagenet",
      include: dnnAugmentIncludeTags.map(t => `${t.category}:${t.value}`),
      exclude: dnnAugmentExcludeTags.map(t => `${t.category}:${t.value}`),
    }
    if (trainingModelId) {
      body.resume = trainingModelId;
    } else if (dnnCheckpointModel) {
      body.resume = dnnCheckpointModel.latest.model_id;
    }

    const modelResponse = await fetch(url, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body),
    }).then(r => r.json());
    setTrainingModelId(modelResponse.model_id);
  };

  const getTrainingStatus = async () => {
    const url = new URL(`${endpoints.modelStatus}/${trainingModelId}`);
    const modelStatus = await fetch(url, {
      method: "GET",
    }).then(r => r.json());
    if (modelStatus.has_model) {
      // Start next epoch
      setTrainingEpoch(trainingEpoch + 1);
      trainDnnOneEpoch();
    }
    if (modelStatus.failed) {
      console.error("Model training failed", modelStatus.failure_reason);
      setRequestDnnTraining(false);
    }
    setTrainingTimeLeft(modelStatus.training_time_left);
  };

  useEffect(() => {
    if (trainingModelId) {
      const interval = setInterval(() => {
        getTrainingStatus();
      }, STATUS_POLL_INTERVAL);
      return () => clearInterval(interval);
    }
  }, [trainingModelId]);

  // Start training
  useEffect(() => {
    if (!requestDnnTraining) return;
    if (clusterReady) {
      trainDnnOneEpoch();
    } else if (!clusterId) {
      startCreatingCluster();
    }
  }, [clusterReady, clusterId, requestDnnTraining]);

  //
  // INFERENCE (automatically pipelined with training)
  //
  const [inferenceJobId, setInferenceJobId] = useState(null);
  const [inferenceEpoch, setInferenceEpoch] = useState();
  const [inferenceTimeLeft, setInferenceTimeLeft] = useState();

  const inferOneEpoch = async () => {
    const url = new URL(`${endpoints.modelInference}/${datasetName}`);
    let body = {
      model_id: trainingModelId,
      cluster_id: clusterId,
      bucket: "foragerml",
      index_id: datasetInfo.index_id,
    }
    const inferenceResponse = await fetch(url, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body),
    }).then(r => r.json());

    setInferenceEpoch(trainingEpoch);
    setInferenceJobId(inferenceResponse.job_id);
  };

  const stopInference = async () => {
    const url = new URL(`${endpoints.stopModelInference}/${inferenceJobId}`);
    await fetch(url, {
      method: "POST",
    }).then(r => r.json());
    setInferenceJobId(null);
  };

  const getInferenceStatus = async () => {
    const url = new URL(`${endpoints.modelInferenceStatus}/${inferenceJobId}`);
    const inferenceStatus = await fetch(url, {
      method: "GET",
    }).then(r => r.json());
    if (inferenceStatus.has_output) {
      inferOneEpoch();
    }
    setInferenceTimeLeft(inferenceStatus.time_left);
  };

  useEffect(() => {
    if (inferenceJobId) {
      const interval = setInterval(() => {
        getInferenceStatus();
      }, STATUS_POLL_INTERVAL);
      return () => clearInterval(interval);
    }
  }, [inferenceJobId]);

  useEffect(() => {
    if (trainingEpoch === 1) inferOneEpoch();
  }, [trainingEpoch]);

  useEffect(() => {
    if (!requestDnnTraining && inferenceJobId) stopInference();
  }, [inferenceJobId, requestDnnTraining]);

  //
  // RESET + REFRESHLOGIC
  //

  // Stop training & once at initialization time
  useEffect(() => {
    if (!requestDnnTraining && !disabled) reset();
  }, [requestDnnTraining, disabled]);

  const reset = () => {
    // Training status
    setTrainingModelId(null);
    setTrainingEpoch(0);
    setTrainingTimeLeft(undefined);

    // Inference status
    setInferenceEpoch(0);
    setInferenceTimeLeft(undefined);

    // Autofill model name
    const name = username.slice(0, username.indexOf("@"));
    const date = dateFormat(new Date(), "mm-dd-yy_HH-MM");
    setModelName(`${name}_${date}`);

    // Clear form fields
    setDnnType(dnns[0].id);
    setDnnAugmentNegs(true);
    setDnnPosTags([]);
    setDnnNegTags([]);
    setDnnCheckpointModel(null);
  };

  // Refresh model status every epoch
  useEffect(updateModels, [trainingModelId, inferenceJobId]);

  const timeLeftToString = (t) => (t && t >= 0) ? new Date(t * 1000).toISOString().substr(11, 8) : "estimating...";

  if (!isVisible) return null;
  return (
    <>
      <div className="d-flex flex-row align-items-center justify-content-between">
        {requestDnnTraining ? <>
          <div className="d-flex flex-row align-items-center">
            <Spinner color="dark" className="my-1 mr-2" />
            {clusterReady ?
              <span>
                Training model <b>{modelName} </b> &mdash;{" "}
                time left for training epoch {trainingEpoch}: {timeLeftToString(trainingTimeLeft)}
                {inferenceJobId && <span> , time left for inference epoch {inferenceEpoch}: {timeLeftToString(inferenceTimeLeft)}</span>}
              </span> :
              <b>Starting cluster</b>
            }
          </div>
          <Button
            color="danger"
            onClick={() => setRequestDnnTraining(false)}
          >Stop training</Button>
        </> : <>
          <FormGroup className="mb-0">
            <select className="custom-select mr-2" value={dnnType} onChange={e => setDnnType(e.target.value)}>
              {dnns.map((d) => <option key={d.id} value={d.id}>{d.label}</option>)}
            </select>
            <ReactSVG className="icon" src="assets/arrow-caret.svg" />
          </FormGroup>
          <Input
            className="mr-2"
            placeholder="Model name"
            value={modelName}
            onChange={(e) => setModelName(e.target.value)}
            spellcheck="false"
          />
          <CategoryInput
            id="dnn-pos-bar"
            className="mr-2"
            placeholder="Positive example tags"
            disabled={requestDnnTraining}
            categories={categories}
            selected={dnnPosTags}
            setSelected={setDnnPosTags}
          />
          <CategoryInput
            id="dnn-neg-bar"
            className="mr-2"
            placeholder="Negative example tags"
            disabled={requestDnnTraining}
            categories={categories}
            selected={dnnNegTags}
            setSelected={setDnnNegTags}
          />
          <div className="my-2 custom-control custom-checkbox">
            <input
              type="checkbox"
              className="custom-control-input"
              id="dnn-augment-negs-checkbox"
              disabled={requestDnnTraining}
              checked={dnnAugmentNegs}
              onChange={(e) => setDnnAugmentNegs(e.target.checked)}
            />
            <label className="custom-control-label text-nowrap mr-2" htmlFor="dnn-augment-negs-checkbox">
              Auto-augment negative set
            </label>
          </div>
          <Button
            color="light"
            onClick={() => setRequestDnnTraining(true)}
            disabled={disabled || dnnPosTags.length === 0 || (dnnNegTags.length === 0 && !dnnAugmentNegs)}
            >Start training
          </Button>
        </>}
      </div>
      {!requestDnnTraining && <a
        href="#"
        className="text-small text-muted mb-1"
        onClick={e => {
          setDnnAdvancedIsOpen(!dnnAdvancedIsOpen);
          e.preventDefault();
        }}
      >
        {dnnAdvancedIsOpen ? "Hide" : "Show"} advanced training options
      </a>}
      <Collapse isOpen={dnnAdvancedIsOpen && !requestDnnTraining} timeout={200}>
        <div className="d-flex flex-row align-items-center justify-content-between my-1">
          <FeatureInput
            id="checkpoint-model-bar"
            placeholder="Checkpoint to train from (optional)"
            features={modelInfo.filter(m => m.latest.has_checkpoint)}
            selected={dnnCheckpointModel}
            setSelected={setDnnCheckpointModel}
          />
          {dnnAugmentNegs && <>
            <CategoryInput
              id="dnn-augment-negs-include-bar"
              className="ml-2"
              placeholder="Tags to include in auto-negative pool"
              categories={categories}
              selected={dnnAugmentIncludeTags}
              setSelected={setDnnAugmentIncludeTags}
            />
            <CategoryInput
              id="dnn-augment-negs-exclude-bar"
              className="ml-2"
              placeholder="Tags to exclude from auto-negative pool"
              categories={categories}
              selected={dnnAugmentExcludeTags}
              setSelected={setDnnAugmentExcludeTags}
            />
          </>}
        </div>
      </Collapse>
    </>
  );
};

export default TrainPanel;
