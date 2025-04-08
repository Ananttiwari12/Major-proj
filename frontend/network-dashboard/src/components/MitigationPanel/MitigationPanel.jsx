import React from "react";
import "./MitigationPanel.css";

const MitigationPanel = ({ mitigation }) => {
  if (!mitigation) return null;

  return (
    <div className="mitigation-panel">
      <h3>Recommended Mitigation</h3>
      <div className="mitigation-text">{mitigation}</div>
    </div>
  );
};

export default MitigationPanel;
