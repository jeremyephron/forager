import React, { useEffect, useState, useMemo, useRef } from "react";
import { useSelector } from 'react-redux';
import { useLocation } from "react-router-dom";
import styled from "styled-components";

import { colors, baseUrl } from "../../Constants";
import { MainCanvas, ImageGrid, BuildIndex, TrainProgress } from "./Components";
import { Button, Select } from "../../Components";
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

const ColContainer = styled.div`
  display: flex;
  flex-direction: column;
  background-color: white;
`;

const LabelContainer = styled.div`
  display: block;
  flex-direction: column;
  background-color: white;
`;

const TextInput = styled.input`
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

function LabelingPage() {
  const location = useLocation();
  const datasetName = location.state.datasetName;
  const [paths, setPaths] = useState(location.state.paths);
  const [identifiers, setIdentifiers] = useState(location.state.identifiers);
  const [imageSize, setImageSize] = useState(150);
  // Same thing with imagesubset
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

  const [orderingMode, setOrderingMode] = useState("default");

  // SVM autolabeling
  const [svmVector, setSvmVector] = useState("");
  const [autolabelPercent, setAutolabelPercent] = useState(50);
  const [autolabelMaxVectors, setAutolabelMaxVectors] = useState(0);

  const cluster = useSelector(state => state.cluster);
  const index = useSelector(state => state.indexes[datasetName] || {});

  const clusterRef = useRef();
  const indexRef = useRef();
  const autolabelInfo = useRef({});

  useEffect(() => {
    clusterRef.current = cluster;
    indexRef.current = index;
    autolabelInfo.current = {
      prev_svm_vector: svmVector || "",
      autolabel_percent: autolabelPercent,
      autolabel_max_vectors: autolabelMaxVectors,
    };
  }, [cluster, index, svmVector, autolabelPercent, autolabelMaxVectors]);

  const getAnnotationsSummaryUrl = baseUrl + "/get_annotations_summary/" + datasetName;
  const getAnnotationsUrl = baseUrl + "/get_annotations/" + datasetName;
  const addAnnotationUrl = baseUrl + "/add_annotation/" + datasetName;
  const deleteAnnotationUrl = baseUrl + "/delete_annotation/" + datasetName;
  const lookupKnnUrl = baseUrl + "/lookup_knn/" + datasetName;
  // can modify the other urls like this
  const getNextImagesURL = baseUrl + "/get_next_images/" + datasetName;
  const getUsersAndCategoriesUrl = baseUrl + "/get_users_and_categories/" + datasetName;
  const setNotesUrl = baseUrl + "/set_notes/" + datasetName;
  const getNotesUrl = baseUrl + "/get_notes/" + datasetName;
  const querySvmUrl = baseUrl + "/query_svm/" + datasetName;
  const activeBatchUrl = baseUrl + "/active_batch/" + datasetName;
  const getGoogleUrl = baseUrl + "/get_google/" + datasetName;
  const importAnnotationsUrl = baseUrl + "/import_annotations/" + datasetName;

  const [PAGINATION_NUM, setPaginationNum] = useState(500);
  const [pageSize, setPageSize] = useState(500);
  const [page, setPage] = useState(1);
  const [isFetching, setIsFetching] = useState(false);

  /* Klabel stuff */
  const labeler = useMemo(() => new ImageLabeler(), []);
  const main_canvas_id = 'main_canvas';

  // Annotating vs Exploring
  var forager_mode = 'forager_annotate';

  // Category keymap
  const labelTypeStrings = {0: 'klabel_frame', 1: 'klabel_point', 2: 'klabel_box', 3: 'klabel_extreme'}

  const image_data = [];
  for (let i=0; i<paths.length; i++) {
    const data = new ImageData();
    data.source_url = paths[i];
    image_data.push(data);
  }

  const [OnKeyImageClick, SetOnKeyImageClick] = useState(() => (e, idx) => {})

  useEffect(() => {
    SetOnKeyImageClick ( () => (e, keyIdx) => {
      // This works, now do something useful with it
      var identifier = keyIdentifiers[keyIdx];
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

  const onDataSource = (event) => {
    var source = document.getElementById("select_data_source").value;
    if (source === "dataset") {
      // Show relevant filters
      document.getElementById("filterUserWrapper").style.display = "block"
      document.getElementById("filterCategoryWrapper").style.display = "block"
      document.getElementById("searchQueryWrapper").style.display = "none"
      document.getElementById("imageSubsetWrapper").style.display = "flex"
    } else if (source === "google") {
      // Show relevant filters
      document.getElementById("filterUserWrapper").style.display = "none"
      document.getElementById("filterCategoryWrapper").style.display = "none"
      document.getElementById("searchQueryWrapper").style.display = "block"
      document.getElementById("imageSubsetWrapper").style.display = "none"
    }
  }

  const onOrderingMode = (e) => {
    setAutolabelMaxVectors(0);
    setOrderingMode(e.target.value);
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
      const current = labeler.get_current_frame_num();
      labeler.load_image_stack(imageData);
      labeler.set_current_frame_num(current);
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

  const onImportClick = async(event) => {
    const category = document.getElementById("import_category").value;
    const path = document.getElementById("import_path").value;
    let url = new URL(importAnnotationsUrl);
    let body = {
      category: category,
      ann_file: path
    };
    // url.search = new URLSearchParams(payload).toString();
    let res = await fetch(url, {method: "POST",
      credentials: 'include',
      body: JSON.stringify(body)
    }).then(results => results.json());
    console.log(res)
  }

  const [HandleFetchImages, SetHandleFetchImages] = useState(()=> async(currPage) => {})

  useEffect(() => {
    console.log("Setting handleFetchImages")
    SetHandleFetchImages ( () => async(currPage) => {
      let filter = document.getElementById("select_image_subset").value;
      let method = document.getElementById("fetch_image_mode").value;
      let dataset = document.getElementById("select_data_source").value;

      let filterUser = document.getElementById("filterUser").value;
      let labelUser = document.getElementById("labelUser").value;
      // Fetch images by filterCategory
      let filterCategory = document.getElementById("filterCategory").value;
      let labelCategory = document.getElementById("labelCategory").value;
      let googleQuery = document.getElementById("searchQuery").value;

      var url;
      var res;

      // Get augmentations
      var augmentations = []
      var augSelect = document.getElementById("augmentations");
      var augParams = document.getElementById("augmentationParam");
      for (var i = 0; i < augSelect.length; i++) {
          if (augSelect.options[i].selected && augSelect.options[i].value !== "none") augmentations.push(augSelect.options[i].value + ":" + augParams.value);
      }

      // Get receptive field window
      var window = document.getElementById("windowSlider").value/1000.;
      console.log("Window: " + window)

      if (dataset.localeCompare("google") === 0) {
        url = new URL(getGoogleUrl);
        url.search = new URLSearchParams({
          category: googleQuery,
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
      } else if (method.localeCompare("knn") === 0 || method.localeCompare("spatialKnn") === 0) {
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

        url = new URL(lookupKnnUrl);
        let knnPayload = {
          user: filterUser,
          category: filterCategory,
          filter: filter,
          ann_identifiers: filteredAnnotations.map(ann => ann.identifier),
          cluster_id: clusterRef.current.id,
          index_id: indexRef.current.id,
          use_full_image: (method.localeCompare("knn") === 0),
          augmentations: augmentations,
          window: window
        };
        url.search = new URLSearchParams(knnPayload).toString();
        res = await fetch(url, {method: "GET",
          credentials: 'include',
        }).then(results => results.json());
      } else if (method.localeCompare("svmPos") === 0 || method.localeCompare("spatialSvmPos") === 0) {
        // Get positive image paths, positive patches, negative image paths?
        // For now makes more sense to pass the user/category, the backend should be able to find the corresponding labels and paths
        // Get augmentations

        url = new URL(querySvmUrl);
        url.search = new URLSearchParams({
          user: filterUser,
          category: filterCategory,
          filter: filter,
          cluster_id: clusterRef.current.id,
          index_id: indexRef.current.id,
          use_full_image: (method.localeCompare("svmPos") === 0),
          augmentations: augmentations,
          mode: method,
          window: window,
          ...(autolabelInfo.current),
        }).toString();
        res = await fetch(url, {method: "GET",
          credentials: 'include',
        }).then(results => results.json());

        setSvmVector(res.svm_vector);
      } else if (method.localeCompare("svmBoundary") === 0) {
        // Turned this off for now, add back to querySvmUrl call when desired
      } else if (method.localeCompare("activeBatch") === 0) {
        url = new URL(activeBatchUrl);
        url.search = new URLSearchParams({
          user: filterUser,
          category: filterCategory,
          filter: filter,
          cluster_id: clusterRef.current.id,
          index_id: indexRef.current.id,
          use_full_image: true,
          augmentations: augmentations,
          start: 1 + (currPage - 1)*10,
          window: window
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

      setPageSize(res.paths.length)

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
      url = new URL(getAnnotationsSummaryUrl);
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
        if (res.all_spatial_dists !== undefined) {
          data.spatial_dists = res.all_spatial_dists[i];
        }

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

  const fetchPage = (page) => {
    setPage(page);
    setIsFetching(true);
    HandleFetchImages(page).finally(() => setIsFetching(false));
  };

  useEffect(() => {
    let button = document.getElementById("filter_button");
    var currPaginationNum = PAGINATION_NUM;
    if (document.getElementById("select_data_source").value.localeCompare("google") === 0) {
      currPaginationNum = 10;
    }
    var maxPage = Math.ceil(numTotalFilteredImages/currPaginationNum)
    if (button) {
      button.onclick = () => fetchPage(1);
    }
    button = document.getElementById("first_button");
    if (button) {
      button.onclick = () => fetchPage(1);
    }
    button = document.getElementById("prev_button");
    if (button) {
      button.onclick = () => fetchPage(Math.max(page - 1,1));
    }
    button = document.getElementById("next_button");
    if (button) {
      button.onclick = () => fetchPage(Math.min(page + 1,maxPage));
    }
    button = document.getElementById("last_button");
    if (button) {
      button.onclick = () => fetchPage(maxPage);
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

    const handle_annotation_added = async (currFrame, annotation) => {
      let labelUser = document.getElementById("labelUser").value;
      let labelCategory = document.getElementById("labelCategory").value;

      if (annotation.type === Annotation.ANNOTATION_MODE_PER_FRAME_CATEGORY) {
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
      console.log("Delete annotation")
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

      labeler.set_annotation_mode(Annotation.ANNOTATION_MODE_EXTREME_POINTS_BBOX);
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
          // Mark to keep
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
        var activeElement = document.activeElement;
        if (activeElement && (activeElement.tagName.toLowerCase() === 'input')) {
          return;
        }
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
      </RowContainer>
      <RowContainer>
        <ColContainer id="klabel_container">
          <RowContainer>
            <LabelContainer>
              <label>Source</label><br/>
              <OptionsSelect alt="true" id="select_data_source" onChange={onDataSource}>
                <option value="dataset">Dataset</option>
                <option value="google">Google</option>
              </OptionsSelect>
            </LabelContainer>
            <LabelContainer id="filterUserWrapper">
              <label>User</label><br/>
              <TextInput type="text" list="users" id="filterUser" onChange={onFilterUser} />
            </LabelContainer>
            <datalist id="users">
              {users.map((item, key) =>
                <option key={key} value={item} />
              )}
            </datalist>
            <LabelContainer id="filterCategoryWrapper">
              <label>Category</label><br/>
              <TextInput type="text" list="categories" id="filterCategory" onChange={onFilterCategory}/>
            </LabelContainer>
            <LabelContainer id="searchQueryWrapper" style={{display: "none"}}>
              <label>Query</label><br/>
              <TextInput type="text" id="searchQuery"/>
            </LabelContainer>
            <RowContainer id="imageSubsetWrapper">
              <LabelContainer>
                <label htmlFor="select_image_subset">Image Subset</label><br/>
                <OptionsSelect alt="true" id="select_image_subset">
                  <option value="all">All</option>
                  <option value="unlabeled">Unlabeled</option>
                  <option value="positive">Positive</option>
                  <option value="negative">Negative</option>
                  <option value="hard_negative">Hard Negative</option>
                  <option value="unsure">Unsure</option>
                  <option value="interesting">Interesting</option>
                  <option value="conflict">Conflict</option>
                </OptionsSelect>
              </LabelContainer>
              <datalist id="categories">
                {categories.map((item, key) =>
                  <option key={key} value={item} />
                )}
              </datalist>
              <LabelContainer id="orderingWrapper">
                <label>Ordering</label><br/>
                <OptionsSelect alt="true" id="fetch_image_mode" value={orderingMode} onChange={onOrderingMode}>
                  <option value="default">Default</option>
                  {index.status === 'INDEX_READY' &&
                  <option value="knn">KNN</option>}
                  {index.status === 'INDEX_READY' &&
                  <option value="spatialKnn">Spatial KNN</option>}
                  {index.status === 'INDEX_READY' &&
                  <option value="svmPos">SVM</option>}
                  {index.status === 'INDEX_READY' &&
                  <option value="spatialSvmPos">Spatial SVM</option>}
                  {index.status === 'INDEX_READY' &&
                  <option value="activeBatch">Active Batch</option>}
                </OptionsSelect>
              </LabelContainer>
              <RowContainer id="knnParamWrapper" style={{display: "none"}}>  {/* Not shown */}
                <LabelContainer>
                  <label>Optic Window</label><br/>
                  <Slider id="windowSlider" type="range" min="0" max="500" defaultValue="0" ></Slider>
                </LabelContainer>
                <LabelContainer>
                  <label>Augment</label><br/>
                  <select id="augmentations" size="2" multiple>
                    <option>none</option>
                    <option>flip</option>
                    <option>gray</option>
                    <option>brightness</option>
                    <option>resize</option>
                    <option>rotate</option>
                    <option>contrast</option>
                  </select>
                </LabelContainer>
                <LabelContainer>
                  <label>AugParam</label><br/>
                  <TextInput id="augmentationParam" style={{width: "70px"}}/>
                </LabelContainer>
              </RowContainer>
              {(orderingMode === "svmPos" || orderingMode === "spatialSvmPos") &&
                <RowContainer>
                  <LabelContainer>
                    <label>Autolabel up to {autolabelMaxVectors} images...</label><br/>
                    <TextInput type="number" value={autolabelMaxVectors} onChange={(e) => setAutolabelMaxVectors(e.target.value)} ></TextInput>
                  </LabelContainer>
                  <LabelContainer>
                    <label>...from bottom {autolabelPercent}% of most recent SVM</label><br/>
                    <Slider type="range" min="0" max="100" value={autolabelPercent} onChange={(e) => setAutolabelPercent(e.target.value)} ></Slider>
                  </LabelContainer>
                </RowContainer>
              }
            </RowContainer>
            <FetchButton id="filter_button" disabled={isFetching}>Apply Filter</FetchButton>
          </RowContainer>
          <MainCanvas numTotalFilteredImages={numTotalFilteredImages} onCategory={OnLabel} onUser={OnLabel} annotationsSummary={annotationsSummary}/>
          <RowContainer id="ann_import_container">
            <input id="import_category" placeholder="Import Category"></input>
            <input id="import_path" placeholder="gs://path-to-annotations"></input>
            <FetchButton id="import_button" onClick={onImportClick}>Import Annotations</FetchButton>
          </RowContainer>
          <Divider/>
          <TrainProgress/>
          <Divider/>
          <StatsBar annotationsSummary={annotationsSummary}/>
        </ColContainer>
        <ColContainer id="explore_container">
          <RowContainer>
            <LabelContainer>
              <label>Image Size</label><br/>
              <Slider type="range" min="50" max="300" defaultValue="100" onChange={(e) => setImageSize(e.target.value)}></Slider>
            </LabelContainer>
            <LabelContainer>
              <StatsContainer>
                {(isFetching) ?
                  <progress /> :
                  <label>Images {1 + (page - 1)*pageSize} to {Math.min(numTotalFilteredImages, page*pageSize)} out of {numTotalFilteredImages}</label>
                }
              </StatsContainer>
              <RowContainer>
                <FetchButton id="first_button" disabled={isFetching}>First</FetchButton>
                <FetchButton id="prev_button" disabled={isFetching}>Prev</FetchButton>
                <FetchButton id="next_button" disabled={isFetching}>Next</FetchButton>
                <FetchButton id="last_button" disabled={isFetching}>Last</FetchButton>
              </RowContainer>
            </LabelContainer>
          </RowContainer>
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
