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

const FetchButton = styled(Button)`
  font-size: 13px;
  height: 28px;
  width: 200px;
  padding: 0 5px;
`;

function LabelingPage() {
  const location = useLocation();
  const datasetName = location.state.datasetName;
  const [paths, setPaths] = useState(location.state.paths);
  const [identifiers, setIdentifiers] = useState(location.state.identifiers);
  var currIdentifiers = location.state.identifiers;
  const [imageSize, setImageSize] = useState(150);
  const [visibility, setVisibility] = useState(new Array(location.state.paths.length).fill(true,0,location.state.paths.length))
  var currVisibility = new Array(location.state.paths.length).fill(true,0,location.state.paths.length);
  const [imageSubset, setImageSubset] = useState(7);
  // Can't get it to work with just useState(paths), and can fix later
  var currPaths = location.state.paths;
  // Same thing with imagesubset
  var currImageSubset = 7;
  const [categories, setCategories] = useState(["Hello","There"]);
  const [currentCategory, setCurrentCategory] = useState("");
  const [users, setUsers] = useState(["Kenobi"]);
  const [currentUser, setCurrentUser] = useState("");
  //var user = "";
  //var category = "";

  const getAnnotationsUrl = "https://127.0.0.1:8000/api/get_annotations/" + datasetName;
  const addAnnotationUrl = "https://127.0.0.1:8000/api/add_annotation/" + datasetName;
  const deleteAnnotationUrl = "https://127.0.0.1:8000/api/delete_annotation/" + datasetName;
  const lookupKnnUrl = "https://127.0.0.1:8000/api/lookup_knn/" + datasetName;
  // can modify the other urls like this 
  const getConflictsUrl = "https://127.0.0.1:8000/api/get_conflicts/" + datasetName;
  const getNextImagesURL = "https://127.0.0.1:8000/api/get_next_images/" + datasetName;
  const getUsersAndCategoriesUrl = "https://127.0.0.1:8000/api/get_users_and_categories/" + datasetName;

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

  const onImageClick = (idx) => {
    labeler.set_current_frame_num(idx);
    //setCurrentImage(idx);
  }

  const onCategory = async(event) => {
    // If empty, refresh list to include any new categories actually labeled
    if (event.target.value === "") {
      console.log("Handle this")
    }
    // Set currentCategory to contents of text field
    setCurrentCategory(event.target.value);
    //category = event.target.value;

    let user = document.getElementById("currUser").value;
    var url = new URL(getAnnotationsUrl);
    url.search = new URLSearchParams({identifiers: currIdentifiers, user: user, category: event.target.value}).toString();
    const annotations = await fetch(url, {
      method: "GET",
      credentials: 'include',
    })
    .then(results => results.json());

    const imageData = [];
    for (let i=0; i<currPaths.length; i++) {
      const data = new ImageData();
      data.source_url = currPaths[i];
      data.identifier = currIdentifiers[i];

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

    labeler.load_image_stack(imageData);
    labeler.set_focus();
  }

  const onUser = async(event) => {
    // If empty, refresh list to include any new categories actually labeled
    if (event.target.value === "") {
      console.log("Handle this")
    }
    // Set currentCategory to contents of text field
    setCurrentUser(event.target.value);
    //user = event.target.value;
    //console.log(user);

    let category = document.getElementById("currCategory").value;
    var url = new URL(getAnnotationsUrl);
    url.search = new URLSearchParams({identifiers: currIdentifiers, user: event.target.value, category: category}).toString();
    const annotations = await fetch(url, {
      method: "GET",
      credentials: 'include',
    })
    .then(results => results.json());

    const imageData = [];
    for (let i=0; i<currPaths.length; i++) {
      const data = new ImageData();
      data.source_url = currPaths[i];
      data.identifier = currIdentifiers[i];
      console.log(data.identifier)
      console.log(annotations)

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
    labeler.load_image_stack(imageData);
    labeler.set_focus();
  }

  const getNextFrame = () => {
    // Move currFrame to next, behavior dependent on mode
    var nextFrame = labeler.current_frame_index + 1;
    //console.log(nextFrame)
    while (nextFrame < paths.length) {
      if (currVisibility[nextFrame]) {
        // labeler.current_frame_index = nextFrame;
        // break;
        return nextFrame;
      } else {
        nextFrame += 1;
      }
    }
    return getLastFrame();
  }

  const getPrevFrame = () => {
    // Move currFrame to prev, behavior dependent on mode
    var prevFrame = labeler.current_frame_index - 1;
    while (prevFrame > 0) {
      if (currVisibility[prevFrame]) {
        //labeler.current_frame_index = prevFrame;
        //break;
        return prevFrame;
      } else {
        prevFrame -= 1;
      }
    }
    return getFirstFrame();
  }

  const getFirstFrame = () => {
    var firstFrame = 0;
    while (firstFrame < paths.length) {
      if (currVisibility[firstFrame]) {
        //labeler.current_frame_index = nextFrame;
        //break;
        return firstFrame;
      } else {
        firstFrame += 1;
      }
    }
    return 0;
  }

  const getLastFrame = () => {
    var lastFrame = paths.length - 1;
    while (lastFrame > 0) {
      if (currImageSubset & currVisibility[lastFrame]) {
        return lastFrame;
      } else {
        lastFrame -= 1;
      }
    }
    return paths.length - 1;
  }

  const handle_fetch_images = async() => {
    let filter = document.getElementById("select_image_subset").value;
    let user = document.getElementById("currUser").value;
    let category = document.getElementById("currCategory").value;
    let url = new URL(getNextImagesURL);
    url.search = new URLSearchParams({identifiers: identifiers, user: user, category: category, filter: filter}).toString();
    const res = await fetch(url, {
      method: "GET",
      credentials: 'include',
      headers: {
      'Content-Type': 'application/json'
      }
    })
    .then(results => results.json());

    url = new URL(getAnnotationsUrl);
    url.search = new URLSearchParams({identifiers: res.identifiers, user: user, category: category}).toString();
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
    currPaths = res.paths;

    labeler.load_image_stack(imageData);
    labeler.set_focus();
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
      let user = document.getElementById("currUser").value;
      let category = document.getElementById("currCategory").value;

      let endpoint = new URL(addAnnotationUrl + '/' + currFrame.data.identifier);
      endpoint.search = new URLSearchParams({
        user: user,
        category: category
      }).toString();

      let body = {
        user: user,
        category: category,
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

      // Make a copy of currFrame.data.annotations without full-frame
      var filteredAnnotations = []
      for (var i = 0; i < currFrame.data.annotations.length; i++) {
        if (currFrame.data.annotations[i].type !== Annotation.ANNOTATION_MODE_PER_FRAME_CATEGORY) {
          filteredAnnotations.push(currFrame.data.annotations[i])
        }
      }

      let url = new URL(lookupKnnUrl);
      url.search = new URLSearchParams({
        ann_identifiers:  filteredAnnotations.map(ann => ann.identifier)
      }).toString();
      const res = await fetch(url, {method: "GET",
        credentials: 'include',
      }).then(results => results.json());

      url = new URL(getAnnotationsUrl);
      url.search = new URLSearchParams({identifiers: res.identifiers, user: user, category: category}).toString();
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
      currPaths = res.paths;

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
      const explorediv = document.getElementById("explore_grid");
      if (select.value.localeCompare("forager_annotate") === 0) {
        forager_mode = "forager_annotate"
        klabeldiv.style.display = "flex"
        explorediv.style.display = "flex"
      } else if (select.value.localeCompare("forager_explore") === 0) {
        forager_mode = "forager_explore"
        klabeldiv.style.display = "none"
        explorediv.style.display = "flex"
      } 
    }

    const handle_image_subset_change = async() => {
      const select = document.getElementById("select_image_subset");
      // Calculate the desired subset of images from annotations, then pass to currVisibility
      let show = new Array(labeler.frames.length).fill(false,0,labeler.frames.length)
      let conflicts = {};
      if (select.value.localeCompare("conflict") === 0) {
        let user = document.getElementById("currUser").value;
        let category = document.getElementById("currCategory").value;

        // Assume this returns a list of conflicting identifiers
        let url = new URL(getConflictsUrl);
        url.search = new URLSearchParams({identifiers:currIdentifiers, user: user, category: category}).toString();
        conflicts = await fetch(url, {
          method: "GET",
          credentials: 'include',
          headers: {
          'Content-Type': 'application/json'
          }
        })
        .then(results => results.json());
        console.log(conflicts)
        console.log(Object.keys(conflicts))
      }
      console.log(conflicts)
      for (var i = 0; i < labeler.frames.length; i++) {
        if (select.value.localeCompare("all") === 0) {
          show[i] = true
        } else if (select.value.localeCompare("unlabeled") === 0) {
          show[i] = (labeler.frames[i].data.annotations.length === 0);
        } else if (select.value.localeCompare("conflict") === 0) {
          if (labeler.frames[i].data.identifier in conflicts) {
            show[i] = true;
          }
        } else  {
          for (var j = 0; j < labeler.frames[i].data.annotations.length; j++) {
            if (labeler.frames[i].data.annotations[j].type === Annotation.ANNOTATION_MODE_PER_FRAME_CATEGORY) {
              if (labeler.frames[i].data.annotations[j].value === filterMap[select.value]) {
                show[i] = true;
              } else  {
                show[i] = false;
              }
            }
          }
        } 
      }
      setVisibility(show);
      currVisibility = show;
      console.log(show)
      labeler.current_frame_index = getFirstFrame();
    }

    const klabelRun = async () => {
      const main_canvas = document.getElementById(main_canvas_id);
      labeler.init(main_canvas);
      labeler.set_categories( { positive: { value: 1, color: "#67bf5c" }, negative: {value:2, color: "#ed665d"}, hard_negative: {value:3, color: "#ffff00"}, unsure: {value:4, color: "#ffa500"} } );
      
      let user = document.getElementById("currUser").value;
      let category = document.getElementById("currCategory").value;

      let url = new URL(getAnnotationsUrl);
      url.search = new URLSearchParams({identifiers: identifiers, user: user, category: category}).toString();
      const annotations = await fetch(url, {
        method: "GET",
        credentials: 'include',
        headers: {
        'Content-Type': 'application/json'
        }
      })
      .then(results => results.json());

      console.log(annotations)

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

      url = new URL(getUsersAndCategoriesUrl);
      /*const usersAndCategories = await fetch(url, {method: "GET",
        credentials: 'include',
      }).then(results => results.json());*/

      labeler.load_image_stack(imageData);
      labeler.set_focus();

      labeler.set_annotation_mode(Annotation.ANNOTATION_MODE_PER_FRAME_CATEGORY);
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
      button = document.getElementById("fetch_button");
      button.onclick = handle_fetch_images;

      let select = document.getElementById("select_annotation_mode")
      select.onchange = handle_mode_change;

      select = document.getElementById("select_forager_mode")
      select.onchange = handle_forager_change;

      select = document.getElementById("select_image_subset")
      select.onchange = handle_image_subset_change;

      window.addEventListener("keydown", function(e) {
        //e.preventDefault(); // If we prevent default it stops typing, only prevent default maybe for arrow keys
        var prevFrame = getPrevFrame();
        var nextFrame = getNextFrame();
        if (forager_mode === "forager_annotate") {
          labeler.handle_keydown(e, prevFrame, nextFrame);
        } 
      });

      window.addEventListener("keyup", function(e) {
        e.preventDefault();
        if (forager_mode === "forager_annotate") {
          labeler.handle_keyup(e);
        }
      });

      currPaths = paths; // Need this for initialization when the page loads...
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
        <OptionsSelect alt="true" id="select_image_subset">
          <option value="all">All</option>
          <option value="unlabeled">Unlabeled</option>
          <option value="positive">Positive</option>
          <option value="negative">Negative</option>
          <option value="hard_negative">Hard Negative</option>
          <option value="unsure">Unsure</option>
          <option value="conflict">Conflict</option>
        </OptionsSelect>
        <input type="text" list="users" id="currUser" onChange={onUser} />
        <datalist id="users">
          {users.map((item, key) =>
            <option key={key} value={item} />
          )}
        </datalist>
        <input type="text" list="categories" id="currCategory" onChange={onCategory} />
        <datalist id="categories">
          {categories.map((item, key) =>
            <option key={key} value={item} />
          )}
        </datalist>
        <FetchButton id="fetch_button">Fetch More Images</FetchButton>
        <label style={{"fontSize": '25px',"marginLeft": "100px"}}>Image Size</label>
        <Slider type="range" min="50" max="300" defaultValue="100" onChange={(e) => setImageSize(e.target.value)}></Slider>
      </SubContainer>
      <SubContainer>
        <MainCanvas/>
        <ImageGridContainer id="explore_grid">
          <ImageGrid onImageClick={onImageClick} imagePaths={paths} imageHeight={imageSize} visibility={visibility}/>
        </ImageGridContainer>
      </SubContainer>
    </Container>
  );
};

export default LabelingPage;
