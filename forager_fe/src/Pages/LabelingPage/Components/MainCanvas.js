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
      <OptionsButton alt="true" id="clear_button">Clear Annotations</OptionsButton> <span>---</span> 
      <OptionsButton alt="true" id="toggle_pt_viz_button">Hide Extreme Points</OptionsButton>
      <OptionsButton alt="true" id="toggle_sound_button">Turn Sound On</OptionsButton>
      <OptionsButton alt="true" id="toggle_letterbox_button">Use Scaled View</OptionsButton> ---
      <OptionsButton alt="true" id="get_annotations">Print Annotations (to console)</OptionsButton>
    </OptionsContainer>
  );
}

const InstructionsContainer = styled.div`
  display: flex;
  flex-direction: column;
  font-family: "AirBnbCereal-Book";
  margin-top: 2vh;

  & > h2 {
    margin-bottom: 0.3em;
    margin-top: 1em;
  }

  h2 {
    font-family: "AirBnbCereal-Medium";
  }

  p, table {
    font-size: 13px;
  }
`;

const Instructions = () => {
  return (
    <InstructionsContainer>
      <h2>Annotation:</h2>
      <table>
        <tbody>
          <tr>
            <td width="180">Extreme points box mode:</td>
            <td>Click four times to draw a box.  (Use the <a href="https://arxiv.org/abs/1708.02750" target="_blank" rel="noopener noreferrer">extreme clicking</a> technique: leftmost, topmost, rightmost, bottommost)</td>
          </tr>
          <tr>
            <td>Two-point box mode:</td>
            <td>Two clicks to define the corners of the bounding box.</td>
          </tr> 
          <tr>
            <td>Point mode:</td>
            <td>Single click to define a point</td>
          </tr>
          <tr>
            <td>Per-frame mode:</td>
            <td>Single click anywhere to mark/unmark the frame. (spacebar also works)</td>
          </tr>
        </tbody>
      </table>
      <p style={{marginTop: 0.5 + 'em'}}>Press 'ESC' to abort current box.</p>

      <h2>Selection:</h2>
      <p>Hover over boxes/points to select them. Press DEL to delete the selected item. (Selection picks the smallest area box containing the cursor.)</p>

      <h2>Zooming:</h2>
      <p>Hold 'z' and draw a two-point box to zoom the view to that box.</p>
      <p>Press 'r' to reset zoom state (zoom all the way out).</p>

      <h2>Image Stack Navigation:</h2>
      <p>Left/right arrow keys move from frame to frame in the current batch of frames.</p>
    </InstructionsContainer>
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
      <Instructions />
    </Container>
  );
}

export default MainCanvas;