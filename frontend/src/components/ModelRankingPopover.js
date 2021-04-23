import React, { useState, useEffect } from "react";
import {
  Popover,
  PopoverBody,
} from "reactstrap";

import FeatureInput from "./FeatureInput";

const ModelRankingPopover = ({ canBeOpen, features, rankingModel, setRankingModel }) => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <Popover
      placement="bottom"
      isOpen={true}
      target="ordering-mode"
      trigger="hover"
      toggle={() => setIsOpen(!isOpen)}
      fade={false}
      popperClassName={`model-ranking-popover ${(canBeOpen && (isOpen || !!!(rankingModel[0]))) ? "visible" : "invisible"}`}
    >
      <PopoverBody>
        <FeatureInput
          id="ranking-feature-bar"
          className="my-1"
          placeholder="Model to rank by"
          features={features}
          selected={rankingModel}
          setSelected={setRankingModel}
        />
      </PopoverBody>
    </Popover>
  );
};

export default ModelRankingPopover;
