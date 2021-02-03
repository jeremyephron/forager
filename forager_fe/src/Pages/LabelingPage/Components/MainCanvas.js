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
      <input type="text" list="users" id="labelUser" onChange={props.onUser} placeholder="LabelUser" />
      <input type="text" list="categories" id="labelCategory" onChange={props.onCategory} placeholder="LabelCategory" />
      <OptionsSelect alt="true" id="select_annotation_mode">
        <option value="box_extreme_points">Box (via extreme points)</option>
        <option value="box_two_points">Box (via two corner clicks)</option>
        <option value="point">Point</option>
      </OptionsSelect>
      <input id="frameSpeed" type="text" placeholder="ms/frame" style={{width: "70px"}}/>
      <OptionsButton alt="true" id="notes_button">Save Notes</OptionsButton>
      <OptionsButton alt="true" id="toggle_pt_viz_button">Hide Extreme Points</OptionsButton>
      <OptionsButton alt="true" id="toggle_letterbox_button">Use Scaled View</OptionsButton>
    </OptionsContainer>
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
      <OptionsBar onUser={props.onUser} onCategory={props.onCategory}/>
      <div>Key bindings: "1" = positive, "2" = negative, "3" = hard negative, "4" = unsure, "k" = keep, "a" = autolabel negative, "s" to stop autolabel</div>
      <textarea type="text" id="user_notes" placeholder="My notes: "></textarea>
      <div id="other_user_notes"></div>
    </Container>
  );
}

export default MainCanvas;
