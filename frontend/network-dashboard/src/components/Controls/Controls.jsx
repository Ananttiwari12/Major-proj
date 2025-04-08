import React from "react";
import "./Controls.css";

const Controls = ({ onIntroduceAnomaly, onClearData }) => {
  return (
    <div className="controls">
      <button onClick={onIntroduceAnomaly}>Trigger Anomaly</button>
      <button onClick={onClearData}>Clear</button>
    </div>
  );
};

export default Controls;
