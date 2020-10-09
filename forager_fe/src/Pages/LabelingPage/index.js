import React, { useEffect, useState, useMemo } from "react";
import { useLocation } from "react-router-dom";
import styled from "styled-components";

import { colors } from "../../Constants";
import { MainCanvas, ImageGrid } from "./Components";
import { Button, Select } from "../../Components";
import {
  BBox2D,
} from "../../assets/js/kmath.js";
import {
  ImageLabeler,
  ImageData,
  Annotation,
  PerFrameAnnotation,
  PointAnnotation,
  TwoPointBoxAnnotation,
  ExtremeBoxAnnnotation,
} from "../../assets/js/klabel.js";

const Container = styled.div`
  display: flex;
  flex-direction: column;
  background-color: white;
`;

const SubContainer = styled.div`
  display: flex;
  flex-direction: row;
  background-color: white;
  margin-left: 3vw;
  margin-top: 3vh;
`;

const ImageGridContainer = styled.div`
  width: 100%;
  height: 75vh;
  margin-top: 2vh;
  margin-right: 3vw;
  margin-left: 3vw;
  border-radius: 5px;
`;

const TitleHeader = styled.h1`
  font-family: "AirBnbCereal-Medium";
  font-size: 24px;
  color: ${colors.primary};
  padding-right: 20px;
`;

const OptionsSelect = styled(Select)`
  font-size: 13px;
  height: 28px;
  padding: 0 5px;
`;

const Slider = styled.input`
  width: 20%; /* Full-width */
  height: 25px; /* Specified height */
  border-radius: 5px;
  margin-left: 20px;
`;

function LabelingPage() {
  const location = useLocation();
  const datasetName = location.state.datasetName;
  const [paths, setPaths] = useState(location.state.paths);
  const [identifiers, setIdentifiers] = useState(location.state.identifiers);
  const [imageSize, setImageSize] = useState(150);

  const getAnnotationsUrl = "https://127.0.0.1:8000/api/get_annotations/" + datasetName;
  const addAnnotationUrl = "https://127.0.0.1:8000/api/add_annotation/" + datasetName;
  const deleteAnnotationUrl = "https://127.0.0.1:8000/api/delete_annotation/" + datasetName;
  const lookupKnnUrl = "https://127.0.0.1:8000/api/lookup_knn/" + datasetName;

  /* Klabel stuff */
  const labeler = useMemo(() => new ImageLabeler(), []);
  const main_canvas_id = 'main_canvas';

  // Annotating vs Exploring
  var forager_mode = 'forager_annotate';

  const image_data = [];
  for (let i=0; i<paths.length; i++) {
    const data = new ImageData();
    data.source_url = paths[i];
    image_data.push(data);
  }

  const onImageClick = (idx) => {
    labeler.set_current_frame_num(idx);
  }

  useEffect(() => {
    const handle_clear_boxes = () => {
      labeler.clear_boxes();
    }

    const toggle_extreme_points_display = () => {
      const button = document.getElementById("toggle_pt_viz_button");
      const new_status = !button.toggle_status;
      labeler.set_extreme_points_viz(new_status);
      button.toggle_status = new_status;

      if (new_status === false) {
        button.innerHTML = 'Show Extreme Points';
      } else {
        button.innerHTML = 'Hide Extreme Points';
      }
    }

    const toggle_letterbox = () => {
      const button = document.getElementById("toggle_letterbox_button");
      const new_status = !button.toggle_status; 
      labeler.set_letterbox(new_status);
      button.toggle_status = new_status;

      if (new_status === false) {
        button.innerHTML = 'Use Letterbox View';
      } else {
        button.innerHTML = 'Use Scaled View';
      }
    }

    const handle_mode_change = () => {
      const select = document.getElementById("select_annotation_mode");
      if (select.value.localeCompare("box_extreme_points") === 0) {
        labeler.set_annotation_mode(Annotation.ANNOTATION_MODE_EXTREME_POINTS_BBOX);
      } else if (select.value.localeCompare("box_two_points") === 0) {
        labeler.set_annotation_mode(Annotation.ANNOTATION_MODE_TWO_POINTS_BBOX);
      } else if (select.value.localeCompare("point") === 0) {
        labeler.set_annotation_mode(Annotation.ANNOTATION_MODE_POINT);
      } else if (select.value.localeCompare("per_frame") === 0) {
        labeler.set_annotation_mode(Annotation.ANNOTATION_MODE_PER_FRAME_CATEGORY);
        labeler.set_categories( { true: { idx: 1, color: "#67bf5c" }, false: {idx:2, color: "#ed665d"} } );
      }
    }

    const handle_get_annotations = () => {
      const results = labeler.get_annotations();
      console.log(results);
    }

    const handle_annotation_added = async (currFrame, annotation) => {
      const endpoint = addAnnotationUrl + '/' + currFrame.data.identifier;

      const identifer = await fetch(endpoint, {
        method: "POST",
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify(annotation)
      })
      .then(response => response.text());

      annotation.identifier = identifer;

      let url = new URL(lookupKnnUrl);
      url.search = new URLSearchParams({
        ann_identifiers:  currFrame.data.annotations.map(ann => ann.identifier)
      }).toString();
      const res = await fetch(url, {method: "GET",
        credentials: 'include',
      }).then(results => results.json());

      url = new URL(getAnnotationsUrl);
      url.search = new URLSearchParams({identifiers: res.identifiers}).toString();
      const annotations = await fetch(url, {
        method: "GET",
        credentials: 'include',
      })
      .then(results => results.json());

      const imageData = [];
      for (let i=0; i<res.paths.length; i++) {
        const data = new ImageData();
        data.source_url = res.paths[i];
        data.identifier = res.identifiers[i];

        if (data.identifier in annotations) {
          annotations[data.identifier].map(ann => {
            if (ann.type === Annotation.ANNOTATION_MODE_PER_FRAME_CATEGORY) {
              // TODO: how to handle?
            } else if (ann.type === Annotation.ANNOTATION_MODE_POINT) {
              data.annotations.push(PointAnnotation.parse(ann));
            } else if (ann.type === Annotation.ANNOTATION_MODE_TWO_POINTS_BBOX) {
              data.annotations.push(TwoPointBoxAnnotation.parse(ann));
            } else if (ann.type === Annotation.ANNOTATION_MODE_EXTREME_POINTS_BBOX) {
              data.annotations.push(ExtremeBoxAnnnotation.parse(ann));
            }
          });
        }
        imageData.push(data);
      }

      setIdentifiers(res.identifiers);
      setPaths(res.paths)

      labeler.load_image_stack(imageData);
      labeler.set_focus();
    }

    const handle_annotation_deleted = async (currFrame, annotation) => {
      const endpoint = deleteAnnotationUrl + '/' + currFrame.data.identifier + '/' + annotation.identifier;

      await fetch(endpoint, { method: "DELETE",
                              credentials: 'include',
      }).then(results => {
        if (results.status === 204) {
          // success, no content
        } else {
          // error
          console.log(results)
        }
      });
    }
    
    const handle_forager_change = () => {
      const select = document.getElementById("select_forager_mode");
      const klabeldiv = document.getElementById("klabel_wrapper");
      if (select.value.localeCompare("forager_annotate") === 0) {
        forager_mode = "forager_annotate"
        klabeldiv.style.display = "flex"
      } else if (select.value.localeCompare("forager_explore") === 0) {
        forager_mode = "forager_explore"
        klabeldiv.style.display = "none"
      } 
    }

    const klabelRun = async () => {
      const main_canvas = document.getElementById(main_canvas_id);
      labeler.init(main_canvas);

      const url = new URL(getAnnotationsUrl);
      url.search = new URLSearchParams({identifiers: identifiers}).toString();
      const annotations = await fetch(url, {
        method: "GET",
        credentials: 'include',
        headers: {
        'Content-Type': 'application/json'
        }
      })
      .then(results => results.json());

      const imageData = [];
      for (let i=0; i<paths.length; i++) {
        const data = new ImageData();
        data.source_url = paths[i];
        data.identifier = identifiers[i];

        if (data.identifier in annotations) {
          annotations[data.identifier].map(ann => {
            if (ann.type === Annotation.ANNOTATION_MODE_PER_FRAME_CATEGORY) {

            } else if (ann.type === Annotation.ANNOTATION_MODE_POINT) {
              data.annotations.push(PointAnnotation.parse(ann));
            } else if (ann.type === Annotation.ANNOTATION_MODE_TWO_POINTS_BBOX) {
              data.annotations.push(TwoPointBoxAnnotation.parse(ann));
            } else if (ann.type === Annotation.ANNOTATION_MODE_EXTREME_POINTS_BBOX) {
              data.annotations.push(ExtremeBoxAnnnotation.parse(ann));
            }
          });
        }
        imageData.push(data);
      }

      labeler.load_image_stack(imageData);
      labeler.set_focus();

      labeler.set_annotation_mode(Annotation.ANNOTATION_MODE_EXTREME_POINTS_BBOX);
      labeler.annotation_added_callback = handle_annotation_added;
      labeler.annotation_deleted_callback = handle_annotation_deleted;

      let button = document.getElementById("toggle_pt_viz_button");
      button.onclick = toggle_extreme_points_display;
      button.toggle_status = true;
      labeler.set_extreme_points_viz(button.toggle_status);

      button = document.getElementById("toggle_sound_button");
      button.toggle_status = false;
      labeler.set_play_audio(button.toggle_status);

      button = document.getElementById("toggle_letterbox_button");
      button.onclick = toggle_letterbox;
      button.toggle_status = true;
      labeler.set_letterbox(button.toggle_status);

      button = document.getElementById("clear_button");
      button.onclick = handle_clear_boxes;
      button = document.getElementById("get_annotations");
      button.onclick = handle_get_annotations;

      let select = document.getElementById("select_annotation_mode")
      select.onchange = handle_mode_change;

      select = document.getElementById("select_forager_mode")
      select.onchange = handle_forager_change;

      window.addEventListener("keydown", function(e) {
        e.preventDefault();
        labeler.handle_keydown(e);
      });

      window.addEventListener("keyup", function(e) {
        e.preventDefault();
        labeler.handle_keyup(e);
      });
    }

    klabelRun();
  }, []);

  return (
    <Container>
      <SubContainer>
        <TitleHeader>Labeling: {datasetName}</TitleHeader>
        <OptionsSelect alt="true" id="select_forager_mode">
          <option value="forager_annotate">Annotate</option>
          <option value="forager_explore">Explore</option>
        </OptionsSelect>
        <label style={{"fontSize": '25px',"marginLeft": "260px"}}>Image Size</label>
        <Slider type="range" min="50" max="300" defaultValue="100" onChange={(e) => setImageSize(e.target.value)}></Slider>
      </SubContainer>
      <SubContainer>
        <MainCanvas/>
        <ImageGridContainer>
          <ImageGrid datasetName={datasetName} onImageClick={onImageClick} imagePaths={paths} imageHeight={imageSize} />
        </ImageGridContainer>
      </SubContainer>
    </Container>
  );
};

export default LabelingPage;
