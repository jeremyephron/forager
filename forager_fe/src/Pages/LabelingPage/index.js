import React, { useEffect, useState, useMemo, useRef } from "react";
import { useSelector, useDispatch } from 'react-redux';
import { useLocation } from "react-router-dom";
import styled from "styled-components";

import { colors, baseUrl } from "../../Constants";
import { MainCanvas, ImageGrid, BuildIndex, TrainProgress } from "./Components";
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

const Divider = styled.hr`
  width: 95%;
  padding: 0px;
`;

const RowContainer = styled.div`
  display: flex;
  flex-direction: row;
  background-color: white;
  margin-top: 0.25vh;
`;

const RowWrapContainer = styled.div`
  display: flex;
  flex-direction: row;
  background-color: white;
  flex-wrap: wrap;
`;

const ColContainer = styled.div`
  display: flex;
  flex-direction: column;
  background-color: white;
  margin-right: 1vw;
`;

const BaseColContainer = styled.div`
  display: flex;
  flex-direction: column;
  background-color: white;
  margin-left: 1vw;
  margin-top: 1vh;
`;

const ImageGridContainer = styled.div`
  display: flex;
  width: 100%;
  height: 60vh;
  margin-top: 2vh;
  border-radius: 5px;
  overflow: hidden;
`;

const ImageRowContainer = styled.div`
  width: 94%;
  height: 15vh;
  margin-top: 2vh;
  margin-right: 3vw;
  margin-left: 3vw;
  border-radius: 5px;
`;

const TitleHeader = styled.h1`
  font-family: "AirBnbCereal-Medium";
  font-size: 24px;
  color: ${colors.primary};
  padding-right: 5px;
`;

const OptionsSelect = styled(Select)`
  font-size: 13px;
  height: 28px;
  padding: 0 5px;
`;

const Slider = styled.input`
  width: 100px; /* Full-width */
  height: 25px; /* Specified height */
  border-radius: 5px;
  margin-left: 20px;
`;

const FetchButton = styled(Button)`
  font-size: 13px;
  height: 28px;
  padding: 0 5px;
`;

const HideButton = styled(Button)`
  font-size: 13px;
  height: 28px;
  padding: 0 5px;
`;

const StatsContainer = styled.div`
  display: flex;
  flex-direction: row;
  justify-content: space-between;
  align-items: center;
  margin-top: 1vh;
`;

const StatsBar = (props) => {
  return (
    <StatsContainer id="annotation_stats">
      <div><p>Unlabeled images:</p>
      {props.annotationsSummary.data && Object.keys(props.annotationsSummary.data).map(cat => (
        <div key={cat}><p><b>Category: {cat}</b></p>
        {Object.keys(props.annotationsSummary.data[cat]).map(user => (
          <div key={user}>{user}: {props.annotationsSummary.data[cat][user]['unlabeled']} </div>
        ))}
        </div>
      ))}
      </div>
    </StatsContainer>
  );
}

const SummaryBar = (props) => {
  return (
    <StatsContainer>
      <div>Images {1 + (props.page - 1)*props.pageSize} to {Math.min(props.numTotalFilteredImages, props.page*props.pageSize)} out of {props.numTotalFilteredImages}</div>
    </StatsContainer>
  );
}

function LabelingPage() {
  const location = useLocation();
  const datasetName = location.state.datasetName;
  const [paths, setPaths] = useState(location.state.paths);
  const [identifiers, setIdentifiers] = useState(location.state.identifiers);
  var currIdentifiers = location.state.identifiers;
  const [imageSize, setImageSize] = useState(150);
  const [imageSubset, setImageSubset] = useState(7);
  // Same thing with imagesubset
  var currImageSubset = 7;
  const [categories, setCategories] = useState(["Hello","There"]);
  const [users, setUsers] = useState(["Kenobi"]);
  const [numTotalFilteredImages, setNumTotalFilteredImages] = useState(0);
  const [annotationsSummary, setAnnotationsSummary] = useState({});
  //var user = "";
  //var category = "";
  const  [selected, setSelected] = useState([0]);
  var currSelected = [0];
  const [keyIdentifiers, setKeyIdentifiers] = useState([]);
  const [keyPaths, setKeyPaths] = useState([]);
  var currKeyPaths = [];
  var currKeyIdentifiers = [];

  const cluster = useSelector(state => state.cluster);
  const index = useSelector(state => state.indexes[datasetName] || {});

  const clusterRef = useRef();
  const indexRef = useRef();

  useEffect(() => {
    clusterRef.current = cluster;
    indexRef.current = index;
  }, [cluster, index]);

  const getAnnotationsSummaryUrl = baseUrl + "/get_annotations_summary/" + datasetName;
  const getAnnotationsUrl = baseUrl + "/get_annotations/" + datasetName;
  const addAnnotationUrl = baseUrl + "/add_annotation/" + datasetName;
  const deleteAnnotationUrl = baseUrl + "/delete_annotation/" + datasetName;
  const lookupKnnUrl = baseUrl + "/lookup_knn/" + datasetName;
  // can modify the other urls like this
  const getConflictsUrl = baseUrl + "/get_conflicts/" + datasetName;
  const getNextImagesURL = baseUrl + "/get_next_images/" + datasetName;
  const getUsersAndCategoriesUrl = baseUrl + "/get_users_and_categories/" + datasetName;
  const setNotesUrl = baseUrl + "/set_notes/" + datasetName;
  const getNotesUrl = baseUrl + "/get_notes/" + datasetName;
  const setKeyUrl = baseUrl + "/set_marked/" + datasetName;
  const getKeyUrl = baseUrl + "/get_marked/" + datasetName;
  const querySvmUrl = baseUrl + "/query_svm/" + datasetName;
  const activeBatchUrl = baseUrl + "/active_batch/" + datasetName;
  const queryGoogleUrl = `https://www.googleapis.com/customsearch/v1`
  const getGoogleUrl = baseUrl + "/get_google/" + datasetName;

  const [PAGINATION_NUM, setPaginationNum] = useState(500);
  const [page, setPage] = useState(1);

  /* Klabel stuff */
  const labeler = useMemo(() => new ImageLabeler(), []);
  const main_canvas_id = 'main_canvas';

  // Annotating vs Exploring
  var forager_mode = 'forager_annotate';

  // Category keymap
  const filterMap = {'positive': 1, 'negative': 2, 'hard_negative': 3, 'unsure': 4}
  const labelTypeStrings = {0: 'klabel_frame', 1: 'klabel_point', 2: 'klabel_box', 3: 'klabel_extreme'}

  const image_data = [];
  for (let i=0; i<paths.length; i++) {
    const data = new ImageData();
    data.source_url = paths[i];
    image_data.push(data);
  }

  const [OnKeyImageClick, SetOnKeyImageClick] = useState(() => (e, idx) => {})

  useEffect(() => {
    SetOnKeyImageClick ( () => (e, idx) => {
      // This works, now do something useful with it
      var identifier = keyIdentifiers[idx];
      var idx = identifiers.indexOf(identifier);
      if (idx >= 0) {
        labeler.set_current_frame_num(idx);
        if (e.shiftKey) {
          labeler.current_indices.push(idx);
        } else {
          labeler.current_indices = [idx];
        }
        setSelected(currSelected);
      }
      // Decide how to handle the else
    });
  }, [keyPaths, keyIdentifiers, identifiers])

  const onImageClick = (e, idx) => {
    labeler.set_current_frame_num(idx);
    if (e.shiftKey) {
      labeler.current_indices.push(idx);
    } else {
      labeler.current_indices = [idx];
    }
    setSelected(currSelected);
  }

  const onKLabelClick = (event) => {
    const klabeldiv = document.getElementById("klabel_container");
    const button = document.getElementById("klabel_toggle")
    console.log(button.value)
    if (klabeldiv.style.display === "none") {
      klabeldiv.style.display = "flex";
      button.innerHTML = "Hide KLabel"
    } else {
      klabeldiv.style.display = "none";
      button.innerHTML = "Show KLabel"
    }
  }

  const onSummaryClick = (event) => {
    const annotationdiv = document.getElementById("annotation_stats");
    const button = document.getElementById("summary_toggle")
    if (annotationdiv.style.display === "none") {
      annotationdiv.style.display = "flex";
      button.innerHTML = "Hide Label Summary"
    } else {
      annotationdiv.style.display = "none";
      button.innerHTML = "Show Label Summary"
    }
  }

  const onKeyImageClick = (event) => {
    const keyimagediv = document.getElementById("key_grid");
    const explorediv = document.getElementById("explore_grid");
    const button = document.getElementById("keyimage_toggle")
    if (keyimagediv.style.display === "none") {
      keyimagediv.style.display = "flex";
      explorediv.style.height = "60vh";
      button.innerHTML = "Hide Key Images"
    } else {
      keyimagediv.style.display = "none";
      explorediv.style.height = "80vh";
      button.innerHTML = "Show Key Images"
    }
  }

  const onFilterCategory = (event) => {
    document.getElementById("labelCategory").value = event.target.value;
    OnLabel()
  }

  const onFilterUser = (event) => {
    document.getElementById("labelUser").value = event.target.value;
    OnLabel()
  }

  const [OnLabel, SetOnLabel] = useState(()=> async(currPage) => {})

  useEffect(() => {
    console.log("Updating onLabel")
    SetOnLabel ( () => async(currPage) => {
      console.log("onLabel")

      let labelUser = document.getElementById("labelUser").value;
      let labelCategory = document.getElementById("labelCategory").value;
      var url = new URL(getAnnotationsUrl);
      url.search = new URLSearchParams({identifiers: identifiers, user: labelUser, category: labelCategory}).toString();
      const annotations = await fetch(url, {
        method: "GET",
        credentials: 'include',
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
              data.annotations.push(PerFrameAnnotation.parse(ann));
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

      //get_notes(false);
      labeler.load_image_stack(imageData);
      labeler.set_focus();
    });
  }, [paths, identifiers])

  //const [GetNextFrame, SetGetNextFrame] = useState(()=> async(currPage) => {})

  const getNextFrame = () => {
    var nextFrame = labeler.current_frame_index + 1;
    return Math.min(nextFrame, labeler.frames.length - 1)
  }

  //const [HandleFetchImages, SetHandleFetchImages] = useState(()=> async(currPage) => {})

  const getPrevFrame = () => {
    // Move currFrame to prev, behavior dependent on mode
    var prevFrame = labeler.current_frame_index - 1;
    return Math.max(prevFrame, 0)
  }

  const [HandleFetchImages, SetHandleFetchImages] = useState(()=> async(currPage) => {})

  useEffect(() => {
    console.log("Setting handleFetchImages")
    SetHandleFetchImages ( () => async(currPage) => {
      let filter = document.getElementById("select_image_subset").value;
      let method = document.getElementById("fetch_image_mode").value;

      // If KNN fetch, then default to fetching all? We can decide whether to reset the filter later
      if (method.localeCompare("knn") === 0) {
        filter = "all";
      }

      let filterUser = document.getElementById("filterUser").value;
      let labelUser = document.getElementById("labelUser").value;
      // Fetch images by filterCategory
      let filterCategory = document.getElementById("filterCategory").value;
      let labelCategory = document.getElementById("labelCategory").value;

      var url;
      var res;
      if (method.localeCompare("knn") === 0 || method.localeCompare("spatialKnn") === 0) {
        // Get relevant frames
        if (labeler.current_indices.length === 0) {
          labeler.current_indices = [labeler.get_current_frame_num()]
        }
        console.log(labeler.current_indices)
        var filteredAnnotations = [];
        for (var j = 0; j < labeler.current_indices.length; j++) {
          var k = labeler.current_indices[j];
          for (var i = 0; i < labeler.frames[k].data.annotations.length; i++) {
            if (labeler.frames[k].data.annotations[i].type !== Annotation.ANNOTATION_MODE_PER_FRAME_CATEGORY) {
              filteredAnnotations.push(labeler.frames[k].data.annotations[i])
            }
          }
        }

        // Get augmentations
        var augmentations = []
        var augSelect = document.getElementById("augmentations");
        var augParams = document.getElementById("augmentationParam");
        for (var i = 0; i < augSelect.length; i++) {
            if (augSelect.options[i].selected) augmentations.push(augSelect.options[i].value + ":" + augParams.value);
        }

        url = new URL(lookupKnnUrl);
        let knnPayload = {
          ann_identifiers: filteredAnnotations.map(ann => ann.identifier),
          cluster_id: clusterRef.current.id,
          index_id: indexRef.current.id,
          use_full_image: (method.localeCompare("knn") === 0),
          augmentations: augmentations
        };
        if (method.localeCompare("knn") === 0) {
          knnPayload.use_full_image = true;
        }
        url.search = new URLSearchParams(knnPayload).toString();
        res = await fetch(url, {method: "GET",
          credentials: 'include',
        }).then(results => results.json());
      } else if (method.localeCompare("svmPos") === 0 || method.localeCompare("svmBoundary") === 0) {
        // Get positive image paths, positive patches, negative image paths?
        // For now makes more sense to pass the user/category, the backend should be able to find the corresponding labels and paths
        // Get augmentations
        var augmentations = []
        var augSelect = document.getElementById("augmentations");
        var augParams = document.getElementById("augmentationParam");
        for (var i = 0; i < augSelect.length; i++) {
            if (augSelect.options[i].selected) augmentations.push(augSelect.options[i].value + ":" + augParams.value);
        }

        url = new URL(querySvmUrl);
        url.search = new URLSearchParams({
          user: filterUser,
          category: filterCategory,
          cluster_id: clusterRef.current.id,
          index_id: indexRef.current.id,
          use_full_image: true,
          augmentations: augmentations,
          mode: method
        }).toString();
        res = await fetch(url, {method: "GET",
          credentials: 'include',
        }).then(results => results.json());
      } else if (method.localeCompare("google") === 0) {
        url = new URL(getGoogleUrl);
        url.search = new URLSearchParams({
          category: filterCategory,
          start: 1 + (currPage - 1)*10,
        }).toString();
        res = await fetch(url, {
          method: "GET",
          credentials: 'include',
          headers: {
          'Content-Type': 'application/json'
          }
        })
        .then(results => results.json());
      } else if (method.localeCompare("activeBatch") === 0) {
        url = new URL(activeBatchUrl);
        url.search = new URLSearchParams({
          user: filterUser,
          category: filterCategory,
          cluster_id: clusterRef.current.id,
          index_id: indexRef.current.id,
          use_full_image: true,
          augmentations: augmentations,
          start: 1 + (currPage - 1)*10,
        }).toString();
        res = await fetch(url, {
          method: "GET",
          credentials: 'include',
          headers: {
          'Content-Type': 'application/json'
          }
        })
        .then(results => results.json());
      } else {
        url = new URL(getNextImagesURL);
        url.search = new URLSearchParams({
          user: filterUser,
          category: filterCategory,
          filter: filter,
          method: method,
          num: PAGINATION_NUM,
          offset: (currPage - 1)*PAGINATION_NUM
        }).toString();
        res = await fetch(url, {
          method: "GET",
          credentials: 'include',
          headers: {
          'Content-Type': 'application/json'
          }
        })
        .then(results => results.json());
      }

      // Add in any key images not present in identifiers (add to res.identifiers and res.paths)
      for (var i = 0; i < keyIdentifiers.length; i++) {
        if (!res.identifiers.includes(keyIdentifiers[i])) {
          res.identifiers.push(keyIdentifiers[i])
          res.paths.push(keyPaths[i])
          //res.num_total += 1
        }
      }

      setNumTotalFilteredImages(res.num_total);

      // Get ann summary
      var url = new URL(getAnnotationsSummaryUrl);
      fetch(url, {
        method: "GET",
        credentials: 'include',
      }).then(results => results.json()
      ).then(results => {
        setAnnotationsSummary(results);
      });

      // Get annotations
      url = new URL(getAnnotationsUrl);
      url.search = new URLSearchParams({
        identifiers: res.identifiers, user: labelUser, category: labelCategory}).toString();
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
              data.annotations.push(PerFrameAnnotation.parse(ann));
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

      var myDiv = document.getElementById('explore_grid').firstChild;
      myDiv.scrollTop = 0;
    });
  }, [keyPaths, keyIdentifiers, PAGINATION_NUM])

  useEffect(() => {
    let button = document.getElementById("filter_button");
    var currPaginationNum = 500;
    if (document.getElementById("fetch_image_mode").value.localeCompare("google") === 0) {
      currPaginationNum = 10;
    }
    var maxPage = Math.ceil(numTotalFilteredImages/currPaginationNum)
    setPaginationNum(currPaginationNum)
    console.log("Page:")
    console.log(page)
    if (button) {
      button.onclick = (function() {
        setPage(1);
        HandleFetchImages(1);
      })
    }
    button = document.getElementById("first_button");
    if (button) {
      button.onclick = (function() {
        setPage(1);
        HandleFetchImages(1);
      })
    }
    button = document.getElementById("prev_button");
    if (button) {
      button.onclick = (function() {
        var nextPage = Math.max(page - 1,1)
        setPage(nextPage);
        HandleFetchImages(nextPage);
      })
    }
    button = document.getElementById("next_button");
    if (button) {
      button.onclick = (function() {
        var nextPage = Math.min(page + 1,maxPage)
        setPage(nextPage);
        HandleFetchImages(nextPage);
      })
    }
    button = document.getElementById("last_button");
    if (button) {
      button.onclick = (function() {
        setPage(maxPage);
        HandleFetchImages(maxPage);
      })
    }
  },[HandleFetchImages, page, numTotalFilteredImages])

  const handle_save_notes = async() => {
    let labelUser = document.getElementById("labelUser").value;
    // For saving information, use label category
    let labelCategory = document.getElementById("labelCategory").value;
    const notes = document.getElementById("user_notes").value;

    let endpoint = new URL(setNotesUrl);
    endpoint.search = new URLSearchParams({
      user: labelUser,
      category: labelCategory
    }).toString();

    let body = {
      user: labelUser,
      category: labelCategory,
      notes: notes
    }

    const res = await fetch(endpoint.toString(), {
      method: "POST",
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include',
      body: JSON.stringify(body)
    })
    .then(response => response.text());
  }

  const get_notes = async(ownNotes) => {
    let labelUser = document.getElementById("labelUser").value;
    let labelCategory = document.getElementById("labelCategory").value;

    let endpoint = new URL(getNotesUrl);
    endpoint.search = new URLSearchParams({
      user: labelUser,
      category: labelCategory
    }).toString();

    const notes = await fetch(endpoint, {method: "GET",
      credentials: 'include',
    }).then(response => response.json());

    if (ownNotes) {
      let notesDiv = document.getElementById("user_notes");
      if (labelUser in notes) {
        notesDiv.value = notes[labelUser];
      }
    }
    var otherNotes = ""
    let otherNotesDiv = document.getElementById("other_user_notes");
    // Loop through entries and build up notes field
    for (var otherUser in notes) {
      otherNotes += notes[otherUser] + "/n";
    }
    otherNotesDiv.value=otherNotes;
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
      }
    }

    const handle_get_annotations = () => {
      const results = labeler.get_annotations();
      console.log(results);
    }

    const handle_annotation_added = async (currFrame, annotation) => {
      let labelUser = document.getElementById("labelUser").value;
      let labelCategory = document.getElementById("labelCategory").value;

      if (annotation.type == Annotation.ANNOTATION_MODE_PER_FRAME_CATEGORY) {
        annotation.labeling_time = currFrame.data.labeling_time;
      }

      let endpoint = new URL(addAnnotationUrl + '/' + currFrame.data.identifier);
      endpoint.search = new URLSearchParams({
        user: labelUser,
        category: labelCategory
      }).toString();

      let body = {
        user: labelUser,
        category: labelCategory,
        annotation: annotation,
        label_type: labelTypeStrings[annotation.type]
      }

      const identifer = await fetch(endpoint.toString(), {
        method: "POST",
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify(body)
      })
      .then(response => response.text());

      annotation.identifier = identifer;
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

    const klabelRun = async () => {
      const main_canvas = document.getElementById(main_canvas_id);
      labeler.init(main_canvas);
      labeler.set_categories( { positive: { value: 1, color: "#67bf5c" }, negative: {value:2, color: "#ed665d"}, hard_negative: {value:3, color: "#ffff00"}, unsure: {value:4, color: "#ffa500"} } );

      let labelUser = document.getElementById("labelUser").value;
      let labelCategory = document.getElementById("labelCategory").value;

      let url = new URL(getAnnotationsUrl);
      url.search = new URLSearchParams({identifiers: identifiers, user: labelUser, category: labelCategory}).toString();
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
              // Use this for fast binary labeling
              data.annotations.push(PerFrameAnnotation.parse(ann));
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

      //get_notes(true); // Get own notes as well
      url = new URL(getUsersAndCategoriesUrl);
      const usersAndCategories = await fetch(
        url, {method: "GET", credentials: 'include',}
      ).then(results => results.json());

      setUsers(usersAndCategories['users']);
      setCategories(usersAndCategories['categories']);

      labeler.set_annotation_mode(Annotation.ANNOTATION_MODE_PER_FRAME_CATEGORY);
      labeler.annotation_added_callback = handle_annotation_added;
      labeler.annotation_deleted_callback = handle_annotation_deleted;

      let button = document.getElementById("toggle_pt_viz_button");
      button.onclick = toggle_extreme_points_display;
      button.toggle_status = true;
      labeler.set_extreme_points_viz(button.toggle_status);

      button = document.getElementById("toggle_letterbox_button");
      button.onclick = toggle_letterbox;
      button.toggle_status = true;
      labeler.set_letterbox(button.toggle_status);

      //button = document.getElementById("notes_button");
      //button.onclick = handle_save_notes;

      let select = document.getElementById("select_annotation_mode")
      select.onchange = handle_mode_change;

      window.addEventListener("keydown", function(e) {
        console.log("Handling keydown")
        //e.preventDefault(); // If we prevent default it stops typing, only prevent default maybe for arrow keys
        var activeElement = document.activeElement;
        if (activeElement && (activeElement.tagName.toLowerCase() === 'input')) {
          return;
        }
        var prevFrame = getPrevFrame();
        var nextFrame = getNextFrame();
        if (e.key && e.key.localeCompare("k") === 0) {
          // Mark interesting
          for (var i = 0; i < labeler.current_indices.length; i++) {
            var currFrame = labeler.current_indices[i]
            var currIdentifier = labeler.frames[currFrame].data.identifier;
            var keyIndex = currKeyIdentifiers.indexOf(currIdentifier)
            if (keyIndex >= 0) {
              currKeyIdentifiers.splice(keyIndex, 1)
              currKeyPaths.splice(keyIndex, 1)
            } else {
              currKeyIdentifiers.push(labeler.frames[currFrame].data.identifier);
              currKeyPaths.push(labeler.frames[currFrame].data.source_url);
            }
          }
          setKeyPaths(currKeyPaths.slice());
          setKeyIdentifiers(currKeyIdentifiers.slice())
        }
        if (forager_mode === "forager_annotate") {
          console.log("Passing keydown to labeler")
          labeler.handle_keydown(e, prevFrame, nextFrame);
          setSelected(labeler.current_indices)
        }
      });

      window.addEventListener("keyup", function(e) {
        e.preventDefault();
        if (forager_mode === "forager_annotate") {
          labeler.handle_keyup(e);
        }
      });
    }

    //let button = document.getElementById("filter_button");
    //button.onclick = HandleFetchImages;

    klabelRun();
  }, []);

  return (
    <BaseColContainer>
      <RowContainer id="title">
        <TitleHeader>Labeling: {datasetName}</TitleHeader>
        <BuildIndex dataset={datasetName} />
        <HideButton id="klabel_toggle" onClick={onKLabelClick}>Hide KLabel</HideButton>
        <HideButton id="summary_toggle" onClick={onSummaryClick}>Hide Label Summary</HideButton>
        <HideButton id="keyimage_toggle" onClick={onKeyImageClick}>Hide Key Images</HideButton>
        <p style={{width:"320px"}}></p>
        <Slider type="range" min="50" max="300" defaultValue="100" onChange={(e) => setImageSize(e.target.value)}></Slider>
      </RowContainer>
      <RowContainer>
        <TitleHeader>Filter: </TitleHeader>
        <input type="text" list="users" id="filterUser" onChange={onFilterUser} placeholder="FilterUser" />
        <datalist id="users">
          {users.map((item, key) =>
            <option key={key} value={item} />
          )}
        </datalist>
        <input type="text" list="categories" id="filterCategory" onChange={onFilterCategory} placeholder="FilterCategory" />
        <OptionsSelect alt="true" id="select_image_subset">
          <option value="all">All</option>
          <option value="unlabeled">Unlabeled</option>
          <option value="positive">Positive</option>
          <option value="negative">Negative</option>
          <option value="hard_negative">Hard Negative</option>
          <option value="unsure">Unsure</option>
          <option value="interesting">Interesting</option>
          <option value="conflict">Conflict</option>
          <option value="google">Googled Earlier</option>
        </OptionsSelect>
        <datalist id="categories">
          {categories.map((item, key) =>
            <option key={key} value={item} />
          )}
        </datalist>
        <OptionsSelect alt="true" id="fetch_image_mode">
          <option value="random">Random</option>
          <option value="google">Google</option>
          {index.status == 'INDEX_READY' &&
          <option value="knn">KNN</option>}
          {index.status == 'INDEX_READY' &&
          <option value="spatialKnn">Spatial KNN</option>}
          {cluster.status === 'CLUSTER_STARTED' &&
          index.status == 'INDEX_BUILT' &&
          <option value="svmPos">SVM Positive</option>}
          {cluster.status === 'CLUSTER_STARTED' &&
          index.status == 'INDEX_BUILT' &&
          <option value="svmBoundary">SVM Boundary</option>}
          {cluster.status === 'CLUSTER_STARTED' &&
          index.status == 'INDEX_BUILT' &&
          <option value="activeBatch">Active Batch</option>}
        </OptionsSelect>
        <select id="augmentations" size="1" multiple>
          <option>flip</option>
          <option>gray</option>
          <option>brightness</option>
          <option>resize</option>
          <option>rotate</option>
          <option>contrast</option>
        </select>
        <input id="augmentationParam" style={{width: "70px"}}></input>
        <FetchButton id="filter_button">Apply Filter</FetchButton>
        <p style={{width:"170px"}}></p>
        <SummaryBar id="image_summary" numTotalFilteredImages={numTotalFilteredImages} page={page} pageSize={PAGINATION_NUM}/>
        <p style={{width:"10px"}}></p>
        <FetchButton id="first_button">First</FetchButton>
        <FetchButton id="prev_button">Prev</FetchButton>
        <FetchButton id="next_button">Next</FetchButton>
        <FetchButton id="last_button">Last</FetchButton>
      </RowContainer>
      <RowContainer>
        <ColContainer id="klabel_container">
          <MainCanvas numTotalFilteredImages={numTotalFilteredImages} onCategory={OnLabel} onUser={OnLabel} annotationsSummary={annotationsSummary}/>
          <Divider/>
          <TrainProgress/>
          <Divider/>
          <StatsBar annotationsSummary={annotationsSummary}/>
        </ColContainer>
        <ColContainer id="explore_container">
          <ImageGridContainer id="explore_grid">
              <ImageGrid onImageClick={onImageClick} imagePaths={paths} imageHeight={imageSize} currentIndex={labeler.current_frame_index} selectedIndices={labeler.current_indices}/>
          </ImageGridContainer>
          <hr style={{width:"95%"}}/>
          <ImageRowContainer id="key_grid">
              <ImageGrid onImageClick={OnKeyImageClick} imagePaths={keyPaths} imageHeight={imageSize}/>
          </ImageRowContainer>
        </ColContainer>
      </RowContainer>
    </BaseColContainer>
  );
};

export default LabelingPage;
