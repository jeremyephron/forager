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

const OptionsBar = () => {
  return (
    <OptionsContainer>
      <OptionsSelect alt="true" id="select_annotation_mode">
        <option value="per_frame">Per Frame</option>
        <option value="box_extreme_points">Box (via extreme points)</option>
        <option value="box_two_points">Box (via two corner clicks)</option>
        <option value="point">Point</option>
      </OptionsSelect>
      <OptionsButton alt="true" id="notes_button">Save Notes</OptionsButton>
      <OptionsButton alt="true" id="clear_button">Clear Annotations</OptionsButton> <span>---</span> 
      <OptionsButton alt="true" id="toggle_pt_viz_button">Hide Extreme Points</OptionsButton>
      <OptionsButton alt="true" id="toggle_letterbox_button">Use Scaled View</OptionsButton> ---
      <OptionsButton alt="true" id="get_annotations">Print Annotations (to console)</OptionsButton>
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

const MainCanvas = () => {
  return (
    <Container id="klabel_wrapper">
      <Canvas width="960" height="500" id="main_canvas" tabindex="0"/>
      <OptionsBar />
      <textarea type="text" id="user_notes" placeholder="My notes: "></textarea>
      <div id="other_user_notes"></div>
    </Container>
  );
}

export default MainCanvas;