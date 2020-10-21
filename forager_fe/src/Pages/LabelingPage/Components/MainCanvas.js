import React from "react";
import styled from "styled-components";

import { Button, Select } from "../../../Components";

const OptionsContainer = styled.div`
  display: flex;
  flex-direction: row;
  justify-content: space-between;
  align-items: center;
  margin-top: 1vh;
`;

const OptionsButton = styled(Button)`
  font-size: 13px;
  height: 28px;
  padding: 0 5px;
`;

const OptionsSelect = styled(Select)`
  font-size: 13px;
  height: 28px;
  padding: 0 5px;
`;

const OptionsBar = (props) => {
  return (
    <OptionsContainer>
      <input type="text" list="categories" id="labelCategory" onChange={props.onCategory} placeholder="LabelCategory" />
      <OptionsSelect alt="true" id="select_annotation_mode">
        <option value="per_frame">Per Frame</option>
        <option value="box_extreme_points">Box (via extreme points)</option>
        <option value="box_two_points">Box (via two corner clicks)</option>
        <option value="point">Point</option>
      </OptionsSelect>
      <OptionsButton alt="true" id="notes_button">Save Notes</OptionsButton>
      <OptionsButton alt="true" id="clear_button">Clear Annotations</OptionsButton>
      <OptionsButton alt="true" id="toggle_pt_viz_button">Hide Extreme Points</OptionsButton>
      <OptionsButton alt="true" id="toggle_letterbox_button">Use Scaled View</OptionsButton>
      <OptionsButton alt="true" id="get_annotations">Print Annotations</OptionsButton>
    </OptionsContainer>
  );
}

const StatsContainer = styled.div`
  display: flex;
  flex-direction: row;
  justify-content: space-between;
  align-items: center;
  margin-top: 1vh;
`;

const StatsBar = (props) => {
  return (
    <StatsContainer>
      <div>Total selected images: {props.numTotalFilteredImages}</div>
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

const Container = styled.div`
  display: flex;
  flex-direction: column;
  width: 960px;
  margin-top: 2vh;
`;

const Canvas = styled.canvas`
  width: 960px;
  height: 500px;
  background: #000000;
  border: 4px solid #DFDFDF;
  box-shadow: 0 1px 6px 1px rgba(38,47,74,0.50);
  box-sizing: border-box;
`;

const MainCanvas = (props) => {
  return (
    <Container id="klabel_wrapper">
      <Canvas width="960" height="500" id="main_canvas" tabindex="0"/>
      <OptionsBar onCategory={props.onCategory}/>
      <div>Key bindings: "1" = positive, "2" = negative, "3" = hard negative, "4" = unsure, "k" = keep, shift-click to select multiple images</div>
      <textarea type="text" id="user_notes" placeholder="My notes: "></textarea>
      <div id="other_user_notes"></div>
      <StatsBar numTotalFilteredImages={props.numTotalFilteredImages} annotationsSummary={props.annotationsSummary}/>
    </Container>
  );
}

export default MainCanvas;
