import React, { useState, useEffect } from "react";
import {
  Popover,
  PopoverBody,
} from "reactstrap";

import FeatureInput from "./FeatureInput";

const ModelRankingPopover = ({ canBeOpen, modelInfo, rankingModel, setRankingModel }) => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <Popover
      placement="bottom"
      isOpen={canBeOpen && (isOpen || !!!(rankingModel[0]))}
      target="ordering-mode"
      trigger="hover"
      toggle={() => setIsOpen(!isOpen)}
      fade={false}
      popperClassName="model-ranking-popover"
    >
      <PopoverBody>
        <FeatureInput
          id="ranking-feature-bar"
          className="my-1"
          placeholder="Model to rank by"
          features={modelInfo.filter(m => m.has_output)}
          selected={rankingModel}
          setSelected={setRankingModel}
        />
      </PopoverBody>
    </Popover>
  );
};

export default ModelRankingPopover;
